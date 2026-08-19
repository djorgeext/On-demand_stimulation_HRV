# %%
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Bidirectional, LSTM, Concatenate, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError, LogCosh, MeanAbsoluteError
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
import json
from pathlib import Path
from scipy.stats import zscore
from utils2 import random_extraction, extract_hrv_features, fast_predict, evaluate_imputation_performance

ages_table_path = 'ages_table.xlsx'
ages_table = pd.read_excel(ages_table_path)

# %%
feature_cols = [
    "mean",
    "sdsd",
    "sd2",
    "ccm",
    "guzik",
    "nn50",
    "porta",
    "std",
]

rr_cols = [f"rr_{i}" for i in range(1, 21)]


def get_rr_reference(age_years):

    seq_mean = 505 * age_years**0.122

    if age_years <= 12:
        seq_scale = 80 * age_years**0.26
    else:
        seq_scale = 290 * age_years**(-0.2)

    return float(seq_mean), float(seq_scale)


def canonical_code(x):
    s = str(x).strip()
    if s.isdigit():
        return f"{int(s):03d}" if len(s) < 3 else s
    return s

def get_subject_interval(h5_path, subject):
    subject_key = canonical_code(subject)

    with h5py.File(h5_path, "r") as h5:
        subject_ids = [
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in h5["index"]["subject_id"][()]
        ]
        intervals = [
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
            for x in h5["index"]["interval"][()]
        ]

        for sid, interval in zip(subject_ids, intervals):
            if canonical_code(sid) == subject_key:
                return interval

    raise ValueError(f"Subject {subject} not found in {h5_path}")

def load_normalization_params(
    h5_path,
    interval,
    age_years,
):
    """
    Returns:
        feats_mean : (8,)
        feats_scale : (8,)
        seq_mean : scalar
        seq_scale : scalar
        y_mean : scalar
        y_scale : scalar
    """

    with h5py.File(h5_path, "r") as h5:

        grp = h5["normalization"][str(interval)]

        cols = [
            c.decode()
            if isinstance(c, bytes)
            else str(c)
            for c in grp["columns"][:]
        ]

        mean = grp["mean"][:]
        std = grp["std"][:]

    stats = {
        col: (m, s)
        for col, m, s in zip(cols, mean, std)
    }

    # ----------------------------------
    # Static features
    # ----------------------------------

    feats_mean = np.array([
        get_rr_reference(age_years)[0],  # mean
        stats["sdsd"][0],
        stats["sd2"][0],
        stats["ccm"][0],
        stats["guzik"][0],
        stats["nn50"][0],
        stats["porta"][0],
        stats["std"][0],
    ], dtype=np.float32)

    feats_scale = np.array([
        get_rr_reference(age_years)[1],  # mean scale
        stats["sdsd"][1],
        stats["sd2"][1],
        stats["ccm"][1],
        stats["guzik"][1],
        stats["nn50"][1],
        stats["porta"][1],
        stats["std"][1],
    ], dtype=np.float32)

    # ----------------------------------
    # RR sequence
    # ----------------------------------

    seq_mean, seq_scale = get_rr_reference(age_years)

    # ----------------------------------
    # Target
    # ----------------------------------

    y_mean = float(stats["target"][0])
    y_scale = float(stats["target"][1])

    return (
        feats_mean,
        feats_scale,
        seq_mean,
        seq_scale,
        y_mean,
        y_scale,
    )

# %%
model_path = 'cnn_lstm_hrv_best.keras'
loaded_model = load_model(model_path)
series_path = 'series/'
subjects_test = [
    "18177.txt",
    "16273.txt",
    "nsr054RRcl.txt",
    "006.txt",
    "nsr048RRcl.txt",
    "19088.txt",
    "000.txt",
    "nsr010RRcl.txt",
    "nsr038RRcl.txt",
    "nsr022RRcl.txt",
    "nsr045RRcl.txt",
    "nsr016RRcl.txt",
]
# Create your array of percentages
percents_array = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

for subject in subjects_test:

    original_serie = np.loadtxt(os.path.join(series_path, subject), dtype=int).astype(float)
    print(os.path.join(series_path, subject))
    subject_name = os.path.splitext(subject)[0]
    folder = Path(subject_name)
    folder.mkdir(parents=True, exist_ok=True)

    interval = get_subject_interval("hrv_validation.h5", subject_name)
    print(interval)
    age_weeks = ages_table['age-weeks'].loc[ages_table['code'] == subject_name].values[0]
    age_years = age_weeks / 52.14

    (
        feats_mean,
        feats_scale,
        seq_mean,
        seq_scale,
        y_mean,
        y_scale,
    ) = load_normalization_params(
        "hrv_dataset.h5",
        interval,
        age_years,
    )

    # Run the function
    _ = evaluate_imputation_performance(
        original_serie=original_serie,
        percents_to_eliminate=percents_array,
        loaded_model=loaded_model,
        feats_mean=feats_mean, feats_scale=feats_scale,
        seq_mean=seq_mean, seq_scale=seq_scale,
        y_mean=y_mean, y_scale=y_scale,
        feature_cols=feature_cols, rr_cols=rr_cols,
        path_to_save = subject_name
    )

# %%



