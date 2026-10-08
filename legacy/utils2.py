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
from scipy.stats import zscore
import scipy.linalg as linalg
from statsmodels.tsa.arima_process import ArmaProcess
from numba import njit, prange


def get_rr_reference(age_years):

    seq_mean = 505 * age_years**0.122

    if age_years <= 12:
        seq_scale = 80 * age_years**0.26
    else:
        seq_scale = 290 * age_years**(-0.2)

    return float(seq_mean), float(seq_scale)

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
    """Computes asymmetry indices (Porta, Guzik), difference counts (NN20, NN50), and RMSSD[cite: 1]."""
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

    # Root Mean Square of Successive Differences
    rmssd = np.sqrt(np.mean(np.square(diffs), axis=1))

    return {
        "n_above": n_above,
        "n_below": n_below,
        "nn20": np.sum(np.abs(diffs) > 20, axis=1),
        "nn50": np.sum(np.abs(diffs) > 50, axis=1),
        "rmssd": rmssd,
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
    sd_n3, vol_n3 = compute_poincare_nd(windows, n=3, tau=tau)
    sd_n4, vol_n4 = compute_poincare_nd(windows, n=4, tau=tau)

    ccm_n2 = compute_ccm_nd(windows, n=2, tau=tau, hypervolume=vol_n2)
    ccm_n3 = compute_ccm_nd(windows, n=3, tau=tau, hypervolume=vol_n3)
    ccm_n4 = compute_ccm_nd(windows, n=4, tau=tau, hypervolume=vol_n4)

    # N = 5 (Hyperellipsoid & Hyperdimensional CCM)
    sd_n5, vol_n5 = compute_poincare_nd(windows, n=5, tau=tau)
    ccm_n5 = compute_ccm_nd(windows, n=5, tau=tau, hypervolume=vol_n5)

    return {
        "sd1": sd_n2[:, 0],
        "sd2": sd_n2[:, 1],
        "c_n": vol_n2,
        "ccm": ccm_n2,
        "ccm_n3": ccm_n3,
        "ccm_n4": ccm_n4,
        "sd3_n3": sd_n3[:, 2],
        "sd4_n4": sd_n4[:, 3],
        "sd5_n5": sd_n5[:, 4],
        #"vol_n5": vol_n5,
        "ccm_n5": ccm_n5
    }

# -------------------------------------------------------------------------
# OPTIMIZED BURG AR EXTRACTOR (REPLACES fit_burg_ar & compute_burg_features)
# -------------------------------------------------------------------------

@njit(fastmath=True)
def _fit_burg_ar_numba(x, order):
    """JIT-compiled core of the Burg recursive algorithm."""
    n = len(x)
    mean_x = 0.0
    for i in range(n): 
        mean_x += x[i]
    mean_x /= n
    
    f = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    sigma_sq = 0.0
    for i in range(n):
        val = x[i] - mean_x
        f[i] = val
        b[i] = val
        sigma_sq += val * val
    sigma_sq /= n
    
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    
    for m in range(1, order + 1):
        num = 0.0
        den = 0.0
        for i in range(m, n):
            num -= 2.0 * f[i] * b[i - 1]
            den += f[i] * f[i] + b[i - 1] * b[i - 1]
        
        km = 0.0
        if den > 0:
            km = num / den
            
        if km > 0.999999: km = 0.999999
        elif km < -0.999999: km = -0.999999
        
        a_prev = a.copy()
        for i in range(1, m):
            a[i] = a_prev[i] + km * a_prev[m - i]
        a[m] = km
        
        sigma_sq *= (1.0 - km * km)
        
        f_new = np.zeros(n, dtype=np.float64)
        b_new = np.zeros(n, dtype=np.float64)
        for i in range(m, n):
            f_new[i] = f[i] + km * b[i - 1]
            b_new[i] = b[i - 1] + km * f[i]
        for i in range(m, n):
            f[i] = f_new[i]
            b[i] = b_new[i]
            
    phi = np.zeros(order, dtype=np.float64)
    for i in range(order):
        phi[i] = -a[i+1]
        
    if sigma_sq < 1e-12:
        sigma_sq = 1e-12
        
    return phi, sigma_sq

@njit(parallel=True, fastmath=True)
def _compute_burg_batch_numba(windows, ar_order=5, psd_order=8, f_low=0.15, f_high=0.40, n_freq_pts=32):
    """JIT-compiled, Multithreaded AR extraction and continuous PSD integration."""
    M, W = windows.shape
    ar_coeffs = np.zeros((M, ar_order), dtype=np.float64)
    burg_powers = np.zeros(M, dtype=np.float64)
    
    f_grid = np.linspace(f_low, f_high, n_freq_pts)
    
    for i in prange(M):
        w = windows[i]
        
        # 1. AR Coefficients
        phi_ar, _ = _fit_burg_ar_numba(w, ar_order)
        for k in range(ar_order):
            ar_coeffs[i, k] = phi_ar[k]
            
        # 2. HF Power calculation
        phi_psd, sigma_sq = _fit_burg_ar_numba(w, psd_order)
        
        mean_w = 0.0
        for j in range(W):
            mean_w += w[j]
        mean_w /= W
        
        dt = mean_w / 1000.0 if mean_w > 10.0 else mean_w
        if dt < 1e-4: 
            dt = 1e-4
        
        band_power = 0.0
        psd_prev = 0.0
        for fi in range(n_freq_pts):
            freq = f_grid[fi]
            real_A = 1.0
            imag_A = 0.0
            for k in range(psd_order):
                angle = -2.0 * np.pi * freq * (k + 1) * dt
                real_A -= phi_psd[k] * np.cos(angle)
                imag_A -= phi_psd[k] * np.sin(angle)
            
            mag_sq = real_A * real_A + imag_A * imag_A
            psd_curr = (2.0 * sigma_sq * dt) / mag_sq
            
            # Trapezoidal integration step
            if fi > 0:
                band_power += 0.5 * (psd_prev + psd_curr) * (f_grid[fi] - f_grid[fi-1])
            psd_prev = psd_curr
            
        burg_powers[i] = band_power
        
    return ar_coeffs, burg_powers


def compute_burg_features(
    windows: np.ndarray,
    ar_order: int = 5,
    psd_order: int = 8,
    f_low: float = 0.15,
    f_high: float = 0.40,
) -> dict[str, np.ndarray]:
    """Extracts AR(order=5) coefficients and Burg PSD integrated power for a 2D batch of windows (M, W)."""
    windows_arr = np.asarray(windows, dtype=np.float64)
    
    # Delegates calculation entirely to the compiled C-speed engine
    ar_coeffs, burg_powers = _compute_burg_batch_numba(
        windows_arr, ar_order, psd_order, f_low, f_high, 32
    )

    feats = {f"ar_{k+1}": ar_coeffs[:, k] for k in range(ar_order)}
    feats["burg_power_hf"] = burg_powers
    return feats

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

    # 3. Burg AR(5) and HF Band Power features (0.15 - 0.40 Hz, p=8)
    burg_feats = compute_burg_features(
    X_ventanas, ar_order=5, psd_order=8, f_low=0.15, f_high=0.40
    )

    # 4. Construcción final del DataFrame
    return pd.DataFrame(
        {
            **rr_columns,
            **asymmetry_feats,
            **stats_feats,
            **phase_space_feats,
            **burg_feats,
            "target": y_target,
        }
    )

def _process_single_run(seed, percent, idx, original_serie, loaded_model, feats_mean, feats_scale, seq_mean, seq_scale, y_mean, y_scale, feature_cols, rr_cols, path_to_save, subject, phi_burg, sigma_burg, ages_table):
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
    modified_serie_nan = modified_serie.copy()

    ####################### Attached to modification later ################################################
    predictor = ExactARPacingPredictor(phi=phi_burg, sigma_sq=sigma_burg)

    ages_table_path = 'ages_table.xlsx'
    ages_table = pd.read_excel(ages_table_path)
    age_weeks = ages_table['age-weeks'].loc[ages_table['code'] == subject].values[0]
    age_years = age_weeks / 52.14
    seq_mean_ar, seq_scale_ar = get_rr_reference(age_years)

    for i in range(30):
        if i == 0 and np.isnan(modified_serie[i]):
            modified_serie[i] = seq_mean  # Replace first NaN with mean reference
            continue
        elif np.isnan(modified_serie[i]) and i >= 1:
            # Use the last observed value for imputation
            next_rr_paced = predictor.forecast_next_rr(
                modified_serie[:i], seq_mean_ar, seq_scale_ar, stochastic=True
            )
            modified_serie[i] = next_rr_paced
    #######################################################################################################
    
    # Iterate over the series starting from index 30
    for i in range(30, len(modified_serie)):
        if np.isnan(modified_serie[i]):
            
            # 1. EXTRACT CLEAN HISTORY
            history_clean = modified_serie[i-30 : i]
            
            # 2. ADAPT FOR EXTRACTOR FUNCTION
            window_data_for_func = np.append(history_clean, np.nan)
            
            # 3. FEATURE EXTRACTION
            df_step = extract_hrv_features(window_data_for_func, window_size=30)
            
            X_feats_step = df_step[feature_cols].values
            X_rr_seq_step = df_step[rr_cols].values
            
            # 4. MANUAL SCALING
            X_feats_step_scaled = (X_feats_step - feats_mean) / feats_scale

            X_rr_seq_step_scaled = (X_rr_seq_step - seq_mean) / seq_scale
            # scale the RR sequence using its own mean and scale of this window
            # X_rr_seq_step_scaled = (X_rr_seq_step - np.mean(X_rr_seq_step)) / np.std(X_rr_seq_step)
            
            # Reshape to 3D for the CNN-LSTM
            X_rr_seq_step_3d = X_rr_seq_step_scaled.reshape(1, 30, 1)
            
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


def evaluate_imputation_performance(original_serie, percents_to_eliminate, loaded_model, feats_mean, feats_scale, seq_mean, seq_scale, y_mean, y_scale, feature_cols, rr_cols, path_to_save, subject_name, phi_burg, sigma_burg, ages_table):
    """
    Evaluates the autoregressive imputation performance of a model across 
    different percentages of missing data for n differents realizations in parallel.
    
    Returns:
        A aggregated Pandas DataFrame containing the results for every seed and percentage.
    """
    seeds = [7] # , 101, 211, 317, 421
    print(f"Starting autoregressive imputation across {len(percents_to_eliminate)} thresholds and {len(seeds)} seeds using 4 cores...")

    # Ensure base directory exists
    Path(path_to_save).mkdir(parents=True, exist_ok=True)

    # 1. Prepare all combinations of (seed, percent) tasks
    tasks = []
    for seed, (idx, percent) in itertools.product(seeds, enumerate(percents_to_eliminate)):
        tasks.append((
            seed, percent, idx, original_serie, loaded_model, 
            feats_mean, feats_scale, seq_mean, seq_scale, 
            y_mean, y_scale, feature_cols, rr_cols, path_to_save,
            subject_name, phi_burg, sigma_burg, ages_table
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


def psd(serie, window_size=2048):
        overlap = window_size // 2
        quantity = len(serie) // overlap
        cutting = quantity * overlap
        serie_right = np.reshape(serie[:cutting], (quantity, overlap))
        serie_right = np.concatenate((serie_right[:-1], serie_right[1:]), axis=1)
        serie_left = np.reshape(np.flip(serie)[:cutting], (quantity, overlap))
        serie_left = np.concatenate((serie_left[:-1], serie_left[1:]), axis=1)
        serie_matrix = np.concatenate((serie_right, serie_left), axis=0)
        serie_matrix = zscore(serie_matrix, axis=1)
        serie_matrix = np.abs(np.fft.fft(serie_matrix, axis=1))**2
        return np.mean(serie_matrix, axis=0)[:window_size//2 + 1]

class ExactARPacingPredictor:
    def __init__(self, phi, sigma_sq):
        """
        phi: AR coefficients [phi_1, ..., phi_p]
        sigma_sq: Innovation variance from Burg
        """
        self.phi = np.asarray(phi, dtype=np.float64)
        self.p = len(self.phi)
        self.sigma_sq = float(sigma_sq)

        # 1. Compute exact theoretical autocovariances gamma(0) ... gamma(p)
        # In ArmaProcess, AR polynomial is 1 - phi_1*z - phi_2*z^2 ...
        ar_poly = np.r_[1.0, -self.phi]
        ma_poly = np.r_[1.0]
        arma = ArmaProcess(ar_poly, ma_poly)
        
        # Theoretical autocorrelation rho(k)
        self.autocorr = arma.acf(lags=self.p + 1)
        # Theoretical autocovariance gamma(k)
        self.gamma = self.autocorr * (self.sigma_sq / (1.0 - np.dot(self.phi, self.autocorr[1:self.p + 1])))

    def _get_optimal_weights(self, N):
        """Solves exact MMSE linear predictor weights w and error variance for history length N."""
        k = min(N, self.p)
        
        if k == self.p:
            return self.phi, self.sigma_sq

        # Exact Yule-Walker solve for finite history N < p
        R_matrix = linalg.toeplitz(self.gamma[:k])
        r_vector = self.gamma[1:k + 1]
        
        weights = linalg.solve(R_matrix, r_vector)
        # Exact conditional error variance for finite sample size N
        pred_variance = self.gamma[0] - np.dot(weights, r_vector)
        
        return weights, max(pred_variance, self.sigma_sq)

    def forecast_next_rr(self, rr_history, rr_mean_ref, rr_std_ref, stochastic=False):
        """
        Forecasts exactly 1 step ahead (RR[n+1]) from 1 to 30 past RR intervals.
        """
        rr = np.asarray(rr_history, dtype=np.float64).flatten()
        N = len(rr)
        if N == 0:
            raise ValueError("History must contain at least 1 sample.")

        # Patient baseline anchor:
        # If history is long (>=10), use the patient's current operational mean to prevent drift.
        # If history is tiny (1 to 3 beats), smoothly blend toward the age reference.
        weight_local = min(N / 10.0, 1.0)
        local_mean = np.mean(rr)
        effective_mean = weight_local * local_mean + (1.0 - weight_local) * rr_mean_ref

        # Standardize using age-matched standard deviation (scale of variability)
        z = (rr - effective_mean) / (rr_std_ref + 1e-8)

        # Get exact weights and exact variance for this specific history length
        weights, pred_var = self._get_optimal_weights(N)
        k = len(weights)

        # Most recent beats ordered backwards: [z[n], z[n-1], ...]
        z_recent = z[-1 : -k - 1 : -1]

        # 1-step exact projection
        z_next = float(np.dot(weights, z_recent))

        # Pacing innovation with exact conditional variance
        if stochastic:
            z_next += np.random.normal(loc=0.0, scale=np.sqrt(pred_var))

        # Denormalize
        rr_next = (z_next * rr_std_ref) + effective_mean
        return float(rr_next)