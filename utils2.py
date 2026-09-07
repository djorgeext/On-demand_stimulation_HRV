import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import concurrent.futures
import itertools
from scipy.special import gamma
import math

@tf.function(reduce_retracing=True)
def fast_predict(seq_input, feats_input, loaded_model):
    # Llamar al modelo como una función con training=False es entre 
    # 10x y 20x más rápido que usar model.predict() para un solo elemento
    return loaded_model([seq_input, feats_input], training=False)

def random_extraction(original_serie, percent_to_eliminate, start_idx=40, seed=None):
    """
    Randomly eliminates a percentage of values from the original series.
    
    Parameters:
    - original_serie: The original time series (numpy array or list).
    - percent_to_eliminate: The percentage of values to eliminate (float between 0 and 1).
    - start_idx: The index from which to start eliminating values (default is 40).
    - seed: Integer to control the random number generator for reproducibility.
    
    Returns:
    - modified_serie: The modified series (as float) with NaN values in place of eliminated values.
    """
    n_total = len(original_serie)
    
    if start_idx >= n_total:
        raise ValueError(f"start_idx ({start_idx}) cannot be greater than or equal to the series length ({n_total}).")
        
    n_to_eliminate = int(round(n_total * percent_to_eliminate))
    available_spots = n_total - start_idx
    
    if n_to_eliminate > available_spots:
        raise ValueError(f"Cannot eliminate {n_to_eliminate} values. Only {available_spots} valid spots available after index {start_idx}.")

    modified_serie = np.array(original_serie, dtype=float)

    # Use NumPy's random generator with the specified seed
    rng = np.random.default_rng(seed)
    random_indexes = rng.choice(
        np.arange(start_idx, n_total), 
        size=n_to_eliminate, 
        replace=False
    )
    
    modified_serie[random_indexes] = np.nan

    return modified_serie

def get_helmert_matrix(n: int) -> np.ndarray:
    """Generates the orthonormal (N x N) Helmert transformation matrix."""
    H = np.zeros((n, n), dtype=np.float64)
    for k in range(1, n):
        H[k - 1, :k] = 1.0
        H[k - 1, k] = -float(k)
        H[k - 1, : k + 1] /= np.sqrt(k * (k + 1.0))
    H[n - 1, :] = 1.0 / np.sqrt(n)
    return H


def compute_poincare_nd(
    windows: np.ndarray, n: int = 5, tau: int = 1, ddof: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized computation of N-dimensional Poincaré standard deviations

    and hyperellipsoid volume for a 2D batch of windows (M, W).
    """
    if tau == 1:
        embeddings = sliding_window_view(
            windows, window_shape=n, axis=1
        )  # (M, L, N)
    else:
        num_pts = windows.shape[1] - (n - 1) * tau
        lag_indices = [np.arange(i * tau, i * tau + num_pts) for i in range(n)]
        embeddings = np.stack([windows[:, idx] for idx in lag_indices], axis=-1)

    H = get_helmert_matrix(n)
    projected = np.matmul(embeddings, H.T)
    sd_metrics = np.std(projected, axis=1, ddof=ddof)  # (M, N)

    vol_factor = (np.pi ** (n / 2.0)) / gamma(n / 2.0 + 1.0)
    hypervolume = vol_factor * np.prod(sd_metrics, axis=1)  # (M,)

    return sd_metrics, hypervolume


def compute_ccm_nd(
    windows: np.ndarray,
    n: int = 5,
    tau: int = 1,
    hypervolume: np.ndarray | None = None,
) -> np.ndarray:
    """Calculates the generalized N-dimensional Complex Correlation Measure (CCM_N)."""
    w = windows.shape[1]
    required_span = (2 * n - 1) * tau + 1 if tau > 1 else 2 * n
    if w < required_span:
        raise ValueError(
            f"window_size ({w}) must be at least {required_span} to form N-simplices for N={n} and tau={tau}."
        )

    # 1. Phase space embedding: shape (M, L, N)
    if tau == 1:
        embeddings = sliding_window_view(windows, window_shape=n, axis=1)
    else:
        num_pts = w - (n - 1) * tau
        lag_indices = [np.arange(i * tau, i * tau + num_pts) for i in range(n)]
        embeddings = np.stack([windows[:, idx] for idx in lag_indices], axis=-1)

    # 2. Extract consecutive (N + 1) points per simplex: shape (M, K, N+1, N)
    simplex_vertices = sliding_window_view(
        embeddings, window_shape=n + 1, axis=1
    )
    simplex_vertices = np.moveaxis(simplex_vertices, -1, 2)
    num_simplices = simplex_vertices.shape[1]

    # 3. Edge displacement vectors from vertex P_i: shape (M, K, N, N)
    d_matrices = simplex_vertices[:, :, 1:, :] - simplex_vertices[:, :, :1, :]

    # 4. Simplex volume: Vol(Delta_i) = (1 / N!) * |det(D_i)|
    simplex_volumes = (1.0 / math.factorial(n)) * np.abs(
        np.linalg.det(d_matrices)
    )
    sum_volumes = np.sum(simplex_volumes, axis=1)

    # 5. Baseline hypervolume normalization V_N
    if hypervolume is None:
        _, hypervolume = compute_poincare_nd(windows, n=n, tau=tau)

    denom = hypervolume * num_simplices
    ccm = np.divide(
        sum_volumes,
        denom,
        out=np.zeros_like(sum_volumes, dtype=float),
        where=denom > 0,
    )

    return np.clip(ccm, 0.0, 1.0)


def compute_asymmetry_features(diffs: np.ndarray) -> dict[str, np.ndarray]:
    """Computes asymmetry indices (Porta, Guzik) and difference counts (NN20, NN50)."""
    n_above = np.sum(diffs > 0, axis=1)
    n_below = np.sum(diffs < 0, axis=1)
    suma_porta = n_above + n_below
    porta_index = np.divide(
        n_below,
        suma_porta,
        out=np.zeros_like(n_below, dtype=float),
        where=suma_porta != 0,
    )

    d_above = np.sum(np.abs(diffs) * (diffs > 0), axis=1) / np.sqrt(2)
    d_total = np.sum(np.abs(diffs), axis=1) / np.sqrt(2)
    guzic_index = np.divide(
        d_above,
        d_total,
        out=np.zeros_like(d_above, dtype=float),
        where=d_total != 0,
    )

    return {
        "n_above": n_above,
        "n_below": n_below,
        "nn20": np.sum(np.abs(diffs) > 20, axis=1),
        "nn50": np.sum(np.abs(diffs) > 50, axis=1),
        "porta": porta_index,
        "guzik": guzic_index,
    }


def compute_statistical_features(
    windows: np.ndarray, diffs: np.ndarray
) -> dict[str, np.ndarray]:
    """Computes distribution moments, robust statistics (IQR, MAD),

    fragmentation (PIP, Skewness), and Rescaled Range (R/S).
    """
    mean_val = np.mean(windows, axis=1)
    std_val = np.std(windows, axis=1)
    # cv = np.divide(
    #     std_val,
    #     mean_val,
    #     out=np.zeros_like(std_val, dtype=float),
    #     where=mean_val != 0,
    # )

    q75, q25 = np.percentile(windows, [75, 25], axis=1)
    median_val = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - median_val[:, None]), axis=1)

    # Heart Rate Fragmentation: Percentage of Inflection Points (PIP)
    diffs_1 = diffs[:, :-1]
    diffs_2 = diffs[:, 1:]
    inflections = (diffs_1 * diffs_2) <= 0
    pip = np.sum(inflections, axis=1) / (windows.shape[1] - 2)

    # Differences Skewness
    # mean_diffs = np.mean(diffs, axis=1, keepdims=True)
    # std_diffs = np.std(diffs, axis=1, keepdims=True)
    # std_diffs_safe = np.where(std_diffs == 0, 1e-10, std_diffs)
    # skewness = np.mean(((diffs - mean_diffs) / std_diffs_safe) ** 3, axis=1)

    # Rescaled Range (R/S)
    # 1. Cumulative departures from the mean: X_t = sum(x_i - mean)
    cum_dev = np.cumsum(windows - mean_val[:, None], axis=1)

    # 2. Range of cumulative deviations: R_N = max(X_t) - min(X_t)
    r_n = np.max(cum_dev, axis=1) - np.min(cum_dev, axis=1)

    # 3. Rescaled Range: (R/S)_N = R_N / S_N
    rs_val = np.divide(
        r_n,
        std_val,
        out=np.zeros_like(r_n, dtype=float),
        where=std_val != 0,
    )

    return {
        "mean": mean_val,
        "std": std_val,
        # "var": std_val**2,
        # "cv": cv,
        "iqr": q75 - q25,
        # "mad": mad,
        "pip": pip,
        # "skewness": skewness,
        "rs": rs_val,
    }


def compute_phase_space_features(
    windows: np.ndarray, tau: int = 1
) -> dict[str, np.ndarray]:
    """Extracts 2D and 5D Poincaré and CCM geometric metrics."""
    # N = 2 (Classical Poincaré & CCM)
    sd_n2, vol_n2 = compute_poincare_nd(windows, n=2, tau=tau)
    ccm_n2 = compute_ccm_nd(windows, n=2, tau=tau, hypervolume=vol_n2)

    # N = 5 (Hyperellipsoid & Hyperdimensional CCM)
    sd_n5, vol_n5 = compute_poincare_nd(windows, n=5, tau=tau)
    ccm_n5 = compute_ccm_nd(windows, n=5, tau=tau, hypervolume=vol_n5)

    return {
        "sd1": sd_n2[:, 0],
        "sd2": sd_n2[:, 1],
        "c_n": vol_n2,
        "ccm": ccm_n2,
        "sd5_n5": sd_n5[:, 4],
        #"vol_n5": vol_n5,
        "ccm_n5": ccm_n5
    }


# =============================================================================
# 3. PIPELINE ORCHESTRATOR
# =============================================================================


def extract_hrv_features(
    serie: np.ndarray | list, window_size: int = 20
) -> pd.DataFrame:
    """Toma una serie de intervalos RR y extrae un conjunto de características

    estadísticas, morfológicas y geométricas ortogonales (Poincaré y CCM N=2 y N=5).
    """
    serie = np.asarray(serie, dtype=np.float64)
    if len(serie) < window_size + 1:
        raise ValueError(
            f"La longitud de la serie ({len(serie)}) debe ser al menos window_size + 1 ({window_size + 1})"
        )

    # 1. Segmentación de ventanas deslizantes (Predictores y Target)
    ventanas = sliding_window_view(serie, window_size + 1)
    X_ventanas = ventanas[:, :-1]
    y_target = ventanas[:, -1] - X_ventanas[:, -1]
    diffs = np.diff(X_ventanas, axis=1)

    # 2. Extracción modular de características
    rr_columns = {f"rr_{i+1}": X_ventanas[:, i] for i in range(window_size)}
    asymmetry_feats = compute_asymmetry_features(diffs)
    stats_feats = compute_statistical_features(X_ventanas, diffs)
    phase_space_feats = compute_phase_space_features(X_ventanas, tau=1)

    # 3. Construcción final del DataFrame
    return pd.DataFrame(
        {
            **rr_columns,
            **asymmetry_feats,
            **stats_feats,
            **phase_space_feats,
            "target": y_target,
        }
    )

def _process_single_run(seed, percent, idx, original_serie, loaded_model, feats_mean, feats_scale, seq_mean, seq_scale, y_mean, y_scale, feature_cols, rr_cols, path_to_save):
    """
    Worker function to process a single combination of seed and percentage.
    """
    seed_folder = Path(path_to_save) / str(seed)
    percent_string = f"percent_{int(percent * 100)}"
    
    # Safely append the percent subfolder
    folder = seed_folder / percent_string
    folder.mkdir(parents=True, exist_ok=True)
    
    y_true = []  
    y_pred = []
    
    # Generate the series with NaNs
    modified_serie = random_extraction(original_serie, percent_to_eliminate=percent, start_idx=0, seed=seed)
    ####################### Attached to modification later ################################################
    modified_serie[:40] = np.nan_to_num(modified_serie[:40], nan=1000.0)
    #######################################################################################################
    
    modified_serie_nan = modified_serie.copy()
    
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
            
            # 5. PREDICTION 
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
        c_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        c_mae = mean_absolute_error(y_true, y_pred)
        c_r2 = r2_score(y_true, y_pred)
    else:
        c_rmse, c_mae, c_r2 = np.nan, np.nan, np.nan
        
    c_corr = np.corrcoef(original_serie, modified_serie)[0, 1]
    
    results_df = pd.DataFrame({
        'Percent_Eliminated': [percent],
        'RMSE': [c_rmse],
        'MAE': [c_mae],
        'R2': [c_r2],
        'Correlation': [c_corr]
    })
    
    save_df = folder / 'result_df.csv'
    save_modified_serie_nan = folder / 'modified_serie_nan.txt'
    save_modified_serie_filled = folder / 'modified_serie_filled.txt'

    results_df.to_csv(save_df, index=False)
    np.savetxt(save_modified_serie_nan, modified_serie_nan, fmt='%g')
    np.savetxt(save_modified_serie_filled, modified_serie, fmt='%g')
    
    print(f"✅ Finished: Seed {seed} | Elimination: {percent_string}%")
    
    return {
        'Seed': seed, 'Percent_Eliminated': percent, 
        'RMSE': c_rmse, 'MAE': c_mae, 'R2': c_r2, 'Correlation': c_corr
    }


def evaluate_imputation_performance(original_serie, percents_to_eliminate, loaded_model, feats_mean, feats_scale, seq_mean, seq_scale, y_mean, y_scale, feature_cols, rr_cols, path_to_save):
    """
    Evaluates the autoregressive imputation performance of a model across 
    different percentages of missing data for n differents realizations in parallel.
    
    Returns:
        A aggregated Pandas DataFrame containing the results for every seed and percentage.
    """
    seeds = [7, 101, 211, 317, 421]
    print(f"Starting autoregressive imputation across {len(percents_to_eliminate)} thresholds and {len(seeds)} seeds using 4 cores...")

    # Ensure base directory exists
    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    # 1. Prepare all combinations of (seed, percent) tasks
    tasks = []
    for seed, (idx, percent) in itertools.product(seeds, enumerate(percents_to_eliminate)):
        tasks.append((
            seed, percent, idx, original_serie, loaded_model, 
            feats_mean, feats_scale, seq_mean, seq_scale, 
            y_mean, y_scale, feature_cols, rr_cols, path_to_save
        ))

    all_results = []

    # 2. Execute tasks in parallel using 4 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Map tasks to the worker function
        futures = {executor.submit(_process_single_run, *task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                # Retrieve result to catch any potential exceptions raised inside the thread
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                task_info = futures[future]
                print(f"⚠️ Task for Seed {task_info[0]}, Percent {task_info[1]} generated an exception: {exc}")

    print("🎉 All parallel tasks completed successfully!")
    
    # 3. (Optional) Return a compiled dataframe of all results instead of overwriting individual arrays
    aggregated_results_df = pd.DataFrame(all_results).sort_values(by=['Seed', 'Percent_Eliminated']).reset_index(drop=True)
    return aggregated_results_df
