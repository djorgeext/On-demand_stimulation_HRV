# %%
import os
import random
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Conv1D, SpatialDropout1D, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Dense,
    Concatenate, Add, Multiply, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# %%
# =========================================================
# CONFIGURATION
# =========================================================
_HERE = os.path.dirname(os.path.abspath(__file__))  # legacy/; data files live in ../data/
TRAIN_H5 = os.path.join(_HERE, "..", "data", "hrv_dataset.h5")
VAL_H5   = os.path.join(_HERE, "..", "data", "hrv_validation.h5")
MODEL_PATH = os.path.join(_HERE, "tcn_attention_hrv_best.keras")

SEED = 211
SEQ_LEN = 30
NUM_TABULAR_FEATS = 12  # Updated to match the 12 static features
BATCH_SIZE = 128        # Smaller batch to preserve local gradient variance in custom loss
EPOCHS = 3
LEARNING_RATE = 5e-4
DELTA_HUBER = 1.0

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Feature definitions matching the new dataset exactly
static_cols = [
    'rmssd', 'rs', 'ccm', 'ccm_n5', 'guzik', 'nn20', 'porta',
    'ar_1', 'ar_2', 'ar_3', 'ar_4', 'ar_5'
]
seq_cols = [f"rr_{i}" for i in range(1, SEQ_LEN + 1)]

# %%
# =========================================================
# DATA LOADERS
# =========================================================
def decode_array(arr):
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr]

def load_entire_dataset(h5_path):
    """Loads the HDF5 dataset directly into structured NumPy arrays."""
    with h5py.File(h5_path, "r") as h5:
        df_index = pd.DataFrame({
            "interval": decode_array(h5["index"]["interval"][()]),
            "h5_path": decode_array(h5["index"]["h5_path"][()]),
            "n_samples": h5["index"]["n_samples"][()].astype(np.int64),
        })
        x_cols = decode_array(h5.attrs["x_cols"])

    seq_idx = [x_cols.index(col) for col in seq_cols]
    feat_idx = [x_cols.index(col) for col in static_cols]

    total_samples = int(df_index["n_samples"].sum())

    X_seq = np.empty((total_samples, SEQ_LEN, 1), dtype=np.float32)
    X_feat = np.empty((total_samples, len(feat_idx)), dtype=np.float32)
    y = np.empty((total_samples, 1), dtype=np.float32)

    cursor = 0
    with h5py.File(h5_path, "r") as h5:
        for _, row in df_index.iterrows():
            n_samples = int(row["n_samples"])
            subj_group = h5[row["h5_path"]]

            X_data = subj_group["X"][()]
            y_data = subj_group["y"][()]

            end = cursor + n_samples
            X_seq[cursor:end] = X_data[:, seq_idx].reshape(-1, SEQ_LEN, 1)
            X_feat[cursor:end] = X_data[:, feat_idx]
            y[cursor:end] = y_data.reshape(-1, 1)

            cursor = end

    return X_seq, X_feat, y


# =========================================================
# CAUSAL RESIDUAL BLOCK (TCN)
# =========================================================
def causal_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.1):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, padding="same")(x)

    y = Conv1D(
        filters=filters, kernel_size=kernel_size,
        dilation_rate=dilation_rate, padding="causal"
    )(x)
    y = LayerNormalization()(y)
    y = Activation("gelu")(y)
    y = SpatialDropout1D(dropout_rate)(y)

    y = Conv1D(
        filters=filters, kernel_size=kernel_size,
        dilation_rate=dilation_rate, padding="causal"
    )(y)
    y = LayerNormalization()(y)
    y = Activation("gelu")(y)
    y = SpatialDropout1D(dropout_rate)(y)

    return Add()([shortcut, y])


# =========================================================
# CUSTOM LOSS FUNCTION
# =========================================================
class RSAPhaseAwareLoss(tf.keras.losses.Loss):
    # Added **kwargs to accept 'reduction' and other base class args
    def __init__(self, delta=1.0, lambda_sign=0.2, lambda_var=0.1, name="rsa_phase_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.lambda_sign = lambda_sign
        self.lambda_var = lambda_var
        self.huber = tf.keras.losses.Huber(delta=self.delta)

    def call(self, y_true, y_pred):
        huber_loss = self.huber(y_true, y_pred)
        # Penaliza predicciones en contrafase respecto a la pendiente respiratoria
        sign_penalty = tf.reduce_mean(tf.nn.relu(-y_true * y_pred))
        # Preserva la amplitud y dispersión oscilatoria
        var_true = tf.math.reduce_variance(y_true)
        var_pred = tf.math.reduce_variance(y_pred)
        var_penalty = tf.abs(var_true - var_pred)

        return huber_loss + self.lambda_sign * sign_penalty + self.lambda_var * var_penalty

    # Added get_config so Keras knows how to serialize and deserialize the custom parameters
    def get_config(self):
        config = super().get_config()
        config.update({
            "delta": self.delta,
            "lambda_sign": self.lambda_sign,
            "lambda_var": self.lambda_var,
        })
        return config


# =========================================================
# MODEL BUILDER
# =========================================================
def build_model():
    # 1. Entrada de Secuencia Temporal de Intervalos RR (30, 1)
    input_seq = Input(shape=(SEQ_LEN, 1), name="Input_Secuencia_RR")

    # Ramas Convolucionales Paralelas Multi-Escala
    branch_micro = Conv1D(filters=16, kernel_size=3, padding="causal", activation="gelu")(input_seq)
    branch_meso  = Conv1D(filters=16, kernel_size=5, padding="causal", activation="gelu")(input_seq)
    branch_macro = Conv1D(filters=16, kernel_size=9, padding="causal", activation="gelu")(input_seq)
    multi_scale_stem = Concatenate(axis=-1)([branch_micro, branch_meso, branch_macro])

    # Pila de Bloques TCN Dilatados Causales
    x_tcn = causal_residual_block(multi_scale_stem, filters=48, kernel_size=3, dilation_rate=1)
    x_tcn = causal_residual_block(x_tcn, filters=48, kernel_size=3, dilation_rate=2)
    x_tcn = causal_residual_block(x_tcn, filters=48, kernel_size=3, dilation_rate=4)
    x_tcn = causal_residual_block(x_tcn, filters=48, kernel_size=3, dilation_rate=8)

    # Mecanismo de Auto-Atención Multicabezal Temporal
    attn_out = MultiHeadAttention(num_heads=4, key_dim=12, name="MHA_RSA_Phase")(
        query=x_tcn, key=x_tcn, value=x_tcn
    )
    x_temporal = Add()([x_tcn, attn_out])
    x_temporal = LayerNormalization()(x_temporal)
    temporal_repr = GlobalAveragePooling1D(name="GlobalPool_Temporal")(x_temporal)

    # 2. Entrada Tabular (12 Features Fisiológicas + AR)
    input_feats = Input(shape=(NUM_TABULAR_FEATS,), name="Input_Features_Fisiologicas")
    feat_dense = Dense(32, activation="gelu")(input_feats)
    feat_dense = LayerNormalization()(feat_dense)

    # 3. Modulación Afín Condicional (FiLM)
    gamma = Dense(48, activation="linear", name="FiLM_Gamma")(feat_dense)
    beta  = Dense(48, activation="linear", name="FiLM_Beta")(feat_dense)
    modulated_temporal = Add()([Multiply()([temporal_repr, gamma]), beta])

    # 4. Cabezal de Regresión Beat-to-Beat
    fused = Concatenate(name="Concatenacion_Multimodal")([modulated_temporal, feat_dense])
    h = Dense(64, activation="gelu")(fused)
    h = LayerNormalization()(h)
    h = Dense(32, activation="gelu")(h)

    output_delta = Dense(1, activation="linear", name="Delta_RR_Predict")(h)

    model = Model(
        inputs=[input_seq, input_feats],
        outputs=output_delta,
        name="TCN_Attention_FiLM_RSA_Model"
    )
    return model

# %%
# =========================================================
# EXECUTION PIPELINE
# =========================================================
if __name__ == "__main__":
    print("Loading datasets directly into RAM...")
    X_seq_train, X_feat_train, y_train = load_entire_dataset(TRAIN_H5)
    X_seq_val, X_feat_val, y_val       = load_entire_dataset(VAL_H5)

    print(f"Train samples: {X_seq_train.shape[0]} | Validation samples: {X_seq_val.shape[0]}")
    print(f"Sequence shape: {X_seq_train.shape[1:]} | Static features: {X_feat_train.shape[1:]}")

    # Transfer to GPU explicitly for training speed
    with tf.device('/GPU:0'):
        X_seq_train_tf  = tf.constant(X_seq_train)
        X_feat_train_tf = tf.constant(X_feat_train)
        y_train_tf      = tf.constant(y_train)

        X_seq_val_tf  = tf.constant(X_seq_val)
        X_feat_val_tf = tf.constant(X_feat_val)
        y_val_tf      = tf.constant(y_val)

    model_optimized = build_model()
    model_optimized.compile(
        optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4),
        loss=RSAPhaseAwareLoss(delta=DELTA_HUBER, lambda_sign=0.2, lambda_var=0.1),
        metrics=["mae", "mse"]
    )
    model_optimized.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    print("\nStarting Training...")
    history = model_optimized.fit(
        x=[X_seq_train_tf, X_feat_train_tf],
        y=y_train_tf,
        validation_data=([X_seq_val_tf, X_feat_val_tf], y_val_tf),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    print(f"\nTraining Complete. Best model saved to: {MODEL_PATH}")


