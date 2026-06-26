import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import zscore
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@tf.function(reduce_retracing=True)
def fast_predict(seq_input, feats_input, loaded_model):
    # Llamar al modelo como una función con training=False es entre 
    # 10x y 20x más rápido que usar model.predict() para un solo elemento
    return loaded_model([seq_input, feats_input], training=False)

def cxk(rr, sint):
    """
    Calculates the scaled cross-correlation between two sequences.
    
    Parameters:
    rr (array-like): First input sequence (e.g., RR intervals).
    sint (array-like): Second input sequence.
    
    Returns:
    numpy.ndarray: Array containing the correlation values for each lag.
    """
    # Convert inputs to numpy arrays to enable element-wise math
    rr = np.asarray(rr)
    sint = np.asarray(sint)
    
    # 'bandera' translated to 'max_lag' to describe its actual purpose
    max_lag = 800
    n = len(rr)
    
    # Initialize the output array 'y' with zeros
    y = np.zeros(max_lag + 1)
    
    # Python's range(x) goes from 0 to x-1, so we use max_lag + 1
    for k in range(max_lag + 1):
        
        # Python uses 0-based indexing. 
        # rr[0:n-k] takes elements from index 0 up to (but excluding) index n-k
        rr_subset = rr[:n-k]
        
        # sint[k:n] takes elements from index k up to the end
        sint_subset = sint[k:n]
        
        # Element-wise multiplication, summation, and division
        y[k] = np.sum(rr_subset * sint_subset) / (n - k)
        
    return y

def random_extraction(original_serie, percent_to_eliminate, start_idx=40):
    """
    Randomly eliminates a percentage of values from the original series.
    
    Parameters:
    - original_serie: The original time series (numpy array or list).
    - percent_to_eliminate: The percentage of values to eliminate (float between 0 and 1).
    - start_idx: The index from which to start eliminating values (default is 40).
    
    Returns:
    - modified_serie: The modified series (as float) with NaN values in place of eliminated values.
    """
    n_total = len(original_serie)
    
    # 1. Edge Case Protections
    if start_idx >= n_total:
        raise ValueError(f"start_idx ({start_idx}) cannot be greater than or equal to the series length ({n_total}).")
        
    n_to_eliminate = int(round(n_total * percent_to_eliminate))
    available_spots = n_total - start_idx
    
    if n_to_eliminate > available_spots:
        raise ValueError(f"Cannot eliminate {n_to_eliminate} values. Only {available_spots} valid spots available after index {start_idx}.")

    # 2. CRITICAL: Copy and cast to float so NumPy accepts np.nan
    modified_serie = np.array(original_serie, dtype=float)

    # 3. Select unique, non-repeating indices using replace=False
    random_indexes = np.random.choice(
        np.arange(start_idx, n_total), 
        size=n_to_eliminate, 
        replace=False
    )
    
    # 4. Inject NaNs
    modified_serie[random_indexes] = np.nan

    return modified_serie

def extract_hrv_features(serie, window_size=20, window_size_long=40):
    """
    Toma una serie de intervalos RR y los tamaños de ventana para construir
    un DataFrame de características ortogonales y la variable objetivo (target).
    """
    if window_size_long < window_size:
        raise ValueError("window_size_long debe ser mayor o igual a window_size")

    serie = np.asarray(serie, dtype=float)

    # 1. Generar ventanas sobre la serie COMPLETA
    ventanas_long = sliding_window_view(serie, window_size_long)

    # 2. Separar variables predictoras del target
    X_ventanas_long = ventanas_long[:-1]
    y_target = serie[window_size_long:]

    # 3. Extraer ventana CORTA
    X_ventanas_short = X_ventanas_long[:, -window_size:]

    # 4. Calcular diferencias dentro de la ventana CORTA
    diffs = np.diff(X_ventanas_short, axis=1)

    # ---- MÉTRICAS TRADICIONALES ----
    n_above = np.sum(diffs > 0, axis=1)
    n_below = np.sum(diffs < 0, axis=1)
    suma_porta = n_above + n_below
    porta_index = np.divide(n_below, suma_porta, out=np.zeros_like(n_below, dtype=float), where=suma_porta!=0)

    d_above = np.sum(np.abs(diffs) * (diffs > 0), axis=1) / np.sqrt(2)
    d_total = np.sum(np.abs(diffs), axis=1) / np.sqrt(2)
    guzic_index = np.divide(d_above, d_total, out=np.zeros_like(d_above, dtype=float), where=d_total!=0)

    nn50 = np.sum(np.abs(diffs) > 50, axis=1)
    nn20 = np.sum(np.abs(diffs) > 20, axis=1)

    sdsd = np.std(diffs, axis=1)
    sd1 = np.sqrt((sdsd**2) / 2)

    mean_val = np.mean(X_ventanas_short, axis=1)
    std_val = np.std(X_ventanas_short, axis=1)
    var_val = std_val ** 2

    std_long = np.std(X_ventanas_long, axis=1)
    inner_value = 2 * std_long**2 - sd1**2
    sd2 = np.sqrt(np.maximum(inner_value, 0))
    c_n = np.pi * sd1 * sd2

    # CCM
    ventanas_4puntos = sliding_window_view(X_ventanas_short, window_shape=4, axis=1)
    rr_i, rr_i1, rr_i2, rr_i3 = ventanas_4puntos[:, 0], ventanas_4puntos[:, 1], ventanas_4puntos[:, 2], ventanas_4puntos[:, 3]
    areas = 0.5 * np.abs(rr_i * (rr_i2 - rr_i3) - rr_i1 * (rr_i1 - rr_i3) + rr_i2 * (rr_i1 - rr_i2))
    denominador_ccm = c_n * (window_size - 2)
    ccm = np.divide(np.sum(areas, axis=1), denominador_ccm, out=np.zeros_like(c_n), where=denominador_ccm!=0)
    ccm = np.where(ccm > 1, 1, ccm)

    # -------------------------------------------------------------
    # NUEVAS CARACTERÍSTICAS ORTOGONALES (NO COLINEALES)
    # -------------------------------------------------------------

    # A. Coeficiente de Variación (CV)
    cv = np.divide(std_val, mean_val, out=np.zeros_like(std_val, dtype=float), where=mean_val!=0)

    # B. Robustez a Outliers: Rango Intercuartílico (IQR) y MAD
    q75, q25 = np.percentile(X_ventanas_short, [75, 25], axis=1)
    iqr = q75 - q25

    median_val = np.median(X_ventanas_short, axis=1)
    mad = np.median(np.abs(X_ventanas_short - median_val[:, None]), axis=1)

    # C. Fragmentación del Ritmo Cardíaco (PIP - Puntos de Inflexión)
    diffs_1 = diffs[:, :-1]
    diffs_2 = diffs[:, 1:]
    inflections = (diffs_1 * diffs_2) <= 0
    pip = np.sum(inflections, axis=1) / (window_size - 2)

    # D. Asimetría (Skewness) de las diferencias
    mean_diffs = np.mean(diffs, axis=1, keepdims=True)
    std_diffs = np.std(diffs, axis=1, keepdims=True)
    std_diffs_safe = np.where(std_diffs == 0, 1e-10, std_diffs) # Evitar NaN
    skewness = np.mean(((diffs - mean_diffs) / std_diffs_safe)**3, axis=1)

    # -------------------------------------------------------------
    # EMPAQUETADO FINAL
    # -------------------------------------------------------------
    rr_columns = {f'rr_{i+1}': X_ventanas_short[:, i] for i in range(window_size)}

    stats_columns = {
        'n_above': n_above, 'n_below': n_below, 'nn20': nn20, 'nn50': nn50,
        'sdsd': sdsd, 'mean': mean_val, 'std': std_val, 'var': var_val,
        'std_long': std_long, 'sd1': sd1, 'sd2': sd2, 'c_n': c_n,
        'ccm': ccm, 'porta': porta_index, 'guzik': guzic_index,
        'cv': cv, 'iqr': iqr, 'mad': mad, 'pip': pip, 'skewness': skewness, # <--- NUEVAS
        'target': y_target
    }

    df = pd.DataFrame({**rr_columns, **stats_columns})
    return df

def evaluate_imputation_performance(original_serie, percents_to_eliminate, loaded_model, feats_mean, feats_scale, seq_mean, seq_scale, y_mean, y_scale, feature_cols, rr_cols):
    """
    Evaluates the autoregressive imputation performance of a model across 
    different percentages of missing data.
    
    Returns:
        rmse (ndarray), mae (ndarray), r2 (ndarray), correlation (ndarray)
    """
    print(f"Starting autoregressive imputation across {len(percents_to_eliminate)} thresholds...")
    
    # Initialize metric arrays
    rmse = np.zeros_like(percents_to_eliminate, dtype=float)
    mae = np.zeros_like(percents_to_eliminate, dtype=float)
    r2 = np.zeros_like(percents_to_eliminate, dtype=float)
    correlation = np.zeros_like(percents_to_eliminate, dtype=float)

    for idx, percent in enumerate(percents_to_eliminate):
        print(f"Evaluating elimination: {percent * 100:.2f}%")
        
        # CRITICAL FIX: Clear the truth/prediction lists for THIS specific percentage
        y_true = []  
        y_pred = []
        
        # Generate the series with NaNs
        modified_serie = random_extraction(original_serie, percent_to_eliminate=percent, start_idx=40)
        
        # Iterate over the series starting from index 40
        for i in range(40, len(modified_serie)):
            
            if np.isnan(modified_serie[i]):
                
                # 1. EXTRACT CLEAN HISTORY
                history_clean = modified_serie[i-40 : i]
                
                # 2. ADAPT FOR EXTRACTOR FUNCTION
                window_data_for_func = np.append(history_clean, np.nan)
                
                # 3. FEATURE EXTRACTION
                df_step = extract_hrv_features(window_data_for_func, window_size=20, window_size_long=40)
                
                X_feats_step = df_step[feature_cols].values
                X_rr_seq_step = df_step[rr_cols].values
                
                # 4. MANUAL SCALING
                X_feats_step_scaled = (X_feats_step - feats_mean) / feats_scale
                X_rr_seq_step_scaled = (X_rr_seq_step - seq_mean) / seq_scale
                
                # Reshape to 3D for the CNN-LSTM
                X_rr_seq_step_3d = X_rr_seq_step_scaled.reshape(1, 20, 1)
                
                # 5. PREDICTION (Using the precompiled fast_predict)
                y_pred_diff_scaled_tensor = fast_predict(
                    tf.convert_to_tensor(X_rr_seq_step_3d, dtype=tf.float32), 
                    tf.convert_to_tensor(X_feats_step_scaled, dtype=tf.float32),
                    loaded_model
                )
                y_pred_diff_scaled = y_pred_diff_scaled_tensor.numpy()

                # 6. DESCALING & RECONSTRUCTION
                y_pred_diff_ms = (y_pred_diff_scaled.flatten()[0] * y_scale) + y_mean
                y_pred_ms = y_pred_diff_ms + history_clean[-1] 
                
                # Store isolated values for metric calculation
                y_true.append(original_serie[i])  
                y_pred.append(y_pred_ms)  
                
                # 7. IMPUTATION
                modified_serie[i] = y_pred_ms

        # Calculate metrics ONLY if NaNs were actually generated/predicted
        if len(y_true) > 0:
            rmse[idx] = np.sqrt(mean_squared_error(y_true, y_pred))
            mae[idx] = mean_absolute_error(y_true, y_pred)
            r2[idx] = r2_score(y_true, y_pred)
        else:
            # Failsafe for very small arrays/percentages where 0 elements were removed
            rmse[idx], mae[idx], r2[idx] = np.nan, np.nan, np.nan
            
        # Overall correlation between the true series and the completely imputed series
        correlation[idx] = np.corrcoef(original_serie, modified_serie)[0, 1]
                
    print("✅ All thresholds evaluated successfully!")
    return rmse, mae, r2, correlation