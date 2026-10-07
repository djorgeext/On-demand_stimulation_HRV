# %%
import os
import math
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import json
from utils2 import extract_hrv_features

ages_table_path = 'ages_table.xlsx'
ages_table = pd.read_excel("ages_table.xlsx", dtype={"code": "string"})
ages_table["code"] = ages_table["code"].str.strip()
subjects_path = 'series'

with open("subjects_train.json", "r", encoding="utf-8") as f:
    subjects_train = json.load(f)

with open("subjects_test.json", "r", encoding="utf-8") as f:
    subjects_test = json.load(f)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
OUTPUT_H5 = "hrv_dataset.h5"
TEMP_CACHE_H5 = "temp_features_cache.h5"
YEAR_TO_WEEKS = 52.14
EPS = 1e-8
WINDOW_SIZE = 30

# 12 static features
static_cols = [
    'rmssd', 'rs', 'ccm', 'ccm_n5', 'guzik', 'nn20', 'porta',
    'ar_1', 'ar_2', 'ar_3', 'ar_4', 'ar_5',
]

subject_specific_cols = [f"rr_{i}" for i in range(1, WINDOW_SIZE + 1)]

x_cols = static_cols + subject_specific_cols
y_col = "target"
features_to_stat = static_cols + [y_col]
features_to_keep = x_cols + [y_col]


def subject_to_filename(subj):
    s = str(subj).strip()
    if s.isdigit() and len(s) < 3:
        return f"{int(s):03d}.txt"
    return f"{s}.txt"


# ------------------------------------------------------------
# HELPERS & RUNNING STATS ACCUMULATORS
# ------------------------------------------------------------
def get_age_years(subj, ages_table, year_to_weeks=52.14):
    age_weeks = ages_table.loc[ages_table["code"] == subj, "age-weeks"].values
    if len(age_weeks) == 0:
        raise ValueError(f"Subject {subj} not found in ages_table.")
    return float(age_weeks[0]) / year_to_weeks


def running_stats_init(n_features):
    return {
        "sum": np.zeros(n_features, dtype=np.float64),
        "sumsq": np.zeros(n_features, dtype=np.float64),
        "n": 0,
    }


def running_stats_update(stats, x):
    if x.size == 0: return stats
    x = np.asarray(x, dtype=np.float64)
    stats["sum"] += x.sum(axis=0)
    stats["sumsq"] += np.square(x).sum(axis=0)
    stats["n"] += x.shape[0]
    return stats


def running_stats_finalize(stats, col_names):
    n = stats["n"]
    if n == 0: raise ValueError("No samples found.")
    mean = stats["sum"] / n
    if n > 1:
        var = (stats["sumsq"] - (stats["sum"] ** 2) / n) / (n - 1)
        std = np.sqrt(np.maximum(var, 0.0))
    else:
        std = np.ones_like(mean)
    std = np.where(std < EPS, 1.0, std)
    return {"columns": list(col_names), "mean": mean, "std": std, "n": n}


def running_scalar_init():
    return {"sum": 0.0, "sumsq": 0.0, "n": 0}


def running_scalar_update(stats, values):
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0: return stats
    stats["sum"] += float(np.sum(arr))
    stats["sumsq"] += float(np.sum(np.square(arr)))
    stats["n"] += int(arr.size)
    return stats


def running_scalar_finalize(stats):
    n = stats["n"]
    if n == 0: raise ValueError("No RR samples found.")
    mean = stats["sum"] / n
    if n > 1:
        var = max((stats["sumsq"] - (stats["sum"] ** 2) / n) / (n - 1), 0.0)
        std = math.sqrt(var)
    else:
        std = 1.0
    std = 1.0 if std < EPS else std
    return float(mean), float(std), int(n)


def normalize_subject_df(df, static_mean, static_std, rr_mean, rr_std, target_mean, target_std):
    df = df.copy()
    df[subject_specific_cols] = df[subject_specific_cols].astype(np.float64)
    df[static_cols] = df[static_cols].astype(np.float64)

    df[subject_specific_cols] = (df[subject_specific_cols] - rr_mean) / rr_std
    df[static_cols] = (df[static_cols] - static_mean) / static_std

    if y_col in df.columns:
        df[y_col] = (df[y_col].astype(np.float64) - target_mean) / target_std

    return df


# ------------------------------------------------------------
# PASS 1: COMPUTE STATS & CACHE RAW FEATURES
# ------------------------------------------------------------
print("Pass 1: Computing HRV features and global stats (saving to temporary cache)...")

global_feature_stats = running_stats_init(len(features_to_stat))
global_rr_stats = running_scalar_init()

with h5py.File(TEMP_CACHE_H5, "w") as f_temp:
    for interval, info in subjects_train.items():
        subjects = info["subjects"]
        grp_int = f_temp.create_group(str(interval))
        
        for subj in subjects:
            file_path = Path(subjects_path) / subject_to_filename(subj)
            serie = np.loadtxt(file_path, dtype=int)
            global_rr_stats = running_scalar_update(global_rr_stats, serie)

            # Heavy Extraction (happens ONLY once now)
            df_temp = extract_hrv_features(serie=serie, window_size=WINDOW_SIZE)
            
            missing = [c for c in features_to_keep if c not in df_temp.columns]
            if missing:
                raise KeyError(f"Missing required columns: {missing}")

            # Prune memory footprint before saving
            df_temp = df_temp[features_to_keep].copy()

            # Update Running Stats
            x_features = df_temp[features_to_stat].to_numpy(dtype=np.float64)
            global_feature_stats = running_stats_update(global_feature_stats, x_features)

            # Cache to Temp H5
            grp_int.create_dataset(str(subj), data=df_temp.to_numpy(dtype=np.float32))

final_feature_stats = running_stats_finalize(global_feature_stats, features_to_stat)
rr_global_mean, rr_global_std, _ = running_scalar_finalize(global_rr_stats)

static_mean = final_feature_stats["mean"][: len(static_cols)]
static_std = final_feature_stats["std"][: len(static_cols)]
target_mean = float(final_feature_stats["mean"][-1])
target_std = float(final_feature_stats["std"][-1])


# ------------------------------------------------------------
# PASS 2: NORMALIZE & WRITE TO FINAL HDF5
# ------------------------------------------------------------
print("Pass 2: Normalizing and writing final datasets (using fast cache)...")
string_dt = h5py.string_dtype(encoding="utf-8")

index_interval, index_subject, index_path, index_n_samples, index_age_years = [], [], [], [], []

with h5py.File(OUTPUT_H5, "w") as h5, h5py.File(TEMP_CACHE_H5, "r") as f_temp:
    # Metadata Setup
    h5.attrs["normalization_type"] = "global_zscore"
    h5.attrs["static_cols"] = np.array(static_cols, dtype=string_dt)
    h5.attrs["subject_specific_cols"] = np.array(subject_specific_cols, dtype=string_dt)
    h5.attrs["x_cols"] = np.array(x_cols, dtype=string_dt)
    h5.attrs["y_col"] = y_col
    
    intervals_group = h5.create_group("intervals")
    norm_group = h5.create_group("normalization")

    # Global Stats Tracking
    norm_group.create_dataset("columns", data=np.array(final_feature_stats["columns"], dtype=string_dt))
    norm_group.create_dataset("mean", data=final_feature_stats["mean"].astype(np.float64))
    norm_group.create_dataset("std", data=final_feature_stats["std"].astype(np.float64))
    norm_group.attrs["rr_mean"], norm_group.attrs["rr_std"] = rr_global_mean, rr_global_std
    norm_group.attrs["target_mean"], norm_group.attrs["target_std"] = target_mean, target_std

    for interval, info in subjects_train.items():
        subjects = info["subjects"]
        interval_group = intervals_group.create_group(str(interval))
        interval_samples_total = 0

        for subj in subjects:
            subj_str = str(subj)
            age_years = get_age_years(subj, ages_table, year_to_weeks=YEAR_TO_WEEKS)

            # Fast Read from Cache (replaces extract_hrv_features)
            raw_data = f_temp[str(interval)][subj_str][()]
            df_temp = pd.DataFrame(raw_data, columns=features_to_keep)

            # Normalization
            df_norm = normalize_subject_df(
                df_temp, static_mean, static_std, rr_global_mean, 
                rr_global_std, target_mean, target_std
            )

            X = df_norm[x_cols].to_numpy(dtype=np.float32)
            y = df_norm[y_col].to_numpy(dtype=np.float32)
            n_samples = X.shape[0]
            interval_samples_total += n_samples

            subject_group = interval_group.create_group(f"subject_{subj_str}")
            subject_group.create_dataset("X", data=X, compression="gzip", compression_opts=4, chunks=True)
            subject_group.create_dataset("y", data=y, compression="gzip", compression_opts=4, chunks=True)
            
            # Attributes & Indexing
            subject_group.attrs["file_id"] = subj_str
            index_interval.append(str(interval))
            index_subject.append(subj_str)
            index_path.append(f"/intervals/{interval}/subject_{subj_str}")
            index_n_samples.append(n_samples)
            index_age_years.append(float(age_years))

        interval_group.attrs["n_interval_samples"] = interval_samples_total

    # Index Table Creation
    index_group = h5.create_group("index")
    index_group.create_dataset("interval", data=np.array(index_interval, dtype=string_dt))
    index_group.create_dataset("subject_id", data=np.array(index_subject, dtype=string_dt))
    index_group.create_dataset("h5_path", data=np.array(index_path, dtype=string_dt))
    index_group.create_dataset("n_samples", data=np.array(index_n_samples, dtype=np.int64))

# Cleanup temporary file
if Path(TEMP_CACHE_H5).exists():
    os.remove(TEMP_CACHE_H5)

print(f"\nSuccessfully generated HDF5 dataset: {OUTPUT_H5}")

# %%
import os
import json
import math
from pathlib import Path
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd

# Updated to use your newly renamed utils2
from utils2 import extract_hrv_features

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TRAIN_H5 = "hrv_dataset.h5"
VALIDATION_H5 = "hrv_validation.h5"
TEST_H5 = "hrv_test.h5"

SUBJECTS_TEST_JSON = "subjects_test.json"
VALIDATION_JSON = "subjects_validation.json"
TEST_JSON = "subjects_test_split.json"

AGES_XLSX = "ages_table.xlsx"
SUBJECTS_PATH = "series"

YEAR_TO_WEEKS = 52.14
EPS = 1e-8
SPLIT_SEED = 7
WINDOW_SIZE = 30

# 12 static features
static_cols = [
    'rmssd', 'rs', 'ccm', 'ccm_n5', 'guzik', 'nn20', 'porta',
    'ar_1', 'ar_2', 'ar_3', 'ar_4', 'ar_5',
]

subject_specific_cols = [f"rr_{i}" for i in range(1, WINDOW_SIZE + 1)]

x_cols = static_cols + subject_specific_cols
y_col = "target"
features_to_keep = x_cols + [y_col]


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def canonical_code(x):
    s = str(x).strip()
    if s.isdigit():
        return f"{int(s):03d}" if len(s) < 3 else s
    return s


def subject_to_filename(subj):
    s = canonical_code(subj)
    if s.isdigit() and len(s) < 3:
        return f"{int(s):03d}.txt"
    return f"{s}.txt"


def get_age_years(subj, ages_table, year_to_weeks=52.14):
    key = canonical_code(subj)
    # Search code safely
    match = ages_table.loc[ages_table["code"] == key, "age-weeks"].values
    if len(match) == 0:
        raise ValueError(f"Subject {subj} not found in ages_table.")
    return float(match[0]) / year_to_weeks


def split_subjects_by_interval(subjects_by_interval, seed=7):
    """
    Splits subjects per interval:
    - 0 subjects -> empty validation, empty test
    - 1 subject (singleton) -> goes 100% to VALIDATION, 0 to test
    - Odd subjects (>1) -> ceil(n/2) to VALIDATION, floor(n/2) to TEST
    - Even subjects -> exact 50/50 split
    """
    rng = np.random.default_rng(seed)
    validation = OrderedDict()
    test = OrderedDict()

    for interval, payload in subjects_by_interval.items():
        if isinstance(payload, dict) and "subjects" in payload:
            subs = [str(s).strip() for s in payload["subjects"]]
        elif isinstance(payload, list):
            subs = [str(s).strip() for s in payload]
        else:
            raise TypeError(f"Unexpected format for interval {interval}")

        rng.shuffle(subs)
        n = len(subs)

        if n == 0:
            val_subs, test_subs = [], []
        elif n == 1:
            # Singleton rule
            val_subs, test_subs = subs, []
        else:
            # Gives the extra subject to validation for odd sizes
            n_val = (n + 1) // 2
            val_subs = subs[:n_val]
            test_subs = subs[n_val:]

        validation[interval] = {"subjects": val_subs}
        test[interval] = {"subjects": test_subs}

    return validation, test


def normalize_subject_df(df, static_mean, static_std, rr_mean, rr_std, target_mean, target_std):
    df = df.copy()
    df[subject_specific_cols] = df[subject_specific_cols].astype(np.float64)
    df[static_cols] = df[static_cols].astype(np.float64)

    # 1. Global RR sequence normalization
    df[subject_specific_cols] = (df[subject_specific_cols] - rr_mean) / rr_std

    # 2. Global static feature normalization
    df[static_cols] = (df[static_cols] - static_mean) / static_std

    # 3. Global target normalization
    if y_col in df.columns:
        df[y_col] = (df[y_col].astype(np.float64) - target_mean) / target_std

    return df


def save_split_h5(output_path, split_name, split_map, train_h5_path, ages_table):
    string_dt = h5py.string_dtype(encoding="utf-8")

    # Extract GLOBAL stats from the training dataset
    with h5py.File(train_h5_path, "r") as th5:
        norm_grp = th5["normalization"]
        
        # FIXED: Look up global attributes inside the normalization group
        rr_mean = float(norm_grp.attrs["rr_mean"])
        rr_std = float(norm_grp.attrs["rr_std"])
        target_mean = float(norm_grp.attrs["target_mean"])
        target_std = float(norm_grp.attrs["target_std"])
        
        # We only need the mean/std for the static cols (ignoring target at the end)
        static_mean = norm_grp["mean"][:len(static_cols)]
        static_std = norm_grp["std"][:len(static_cols)]
        static_cols_arr = norm_grp["columns"][:len(static_cols)]

    with h5py.File(output_path, "w") as out_h5:
        # File-level metadata
        out_h5.attrs["source_split_name"] = split_name
        out_h5.attrs["normalization_type"] = "global_zscore"
        out_h5.attrs["static_cols"] = np.array(static_cols, dtype=string_dt)
        out_h5.attrs["subject_specific_cols"] = np.array(subject_specific_cols, dtype=string_dt)
        out_h5.attrs["x_cols"] = np.array(x_cols, dtype=string_dt)
        out_h5.attrs["y_col"] = y_col

        intervals_group = out_h5.create_group("intervals")
        index_group = out_h5.create_group("index")
        
        # We replicate the normalization group structure for loader compatibility
        norm_group = out_h5.create_group("normalization")
        norm_group.create_dataset("columns", data=static_cols_arr)
        norm_group.create_dataset("mean", data=static_mean)
        norm_group.create_dataset("std", data=static_std)
        norm_group.attrs["rr_mean"] = rr_mean
        norm_group.attrs["rr_std"] = rr_std
        norm_group.attrs["target_mean"] = target_mean
        norm_group.attrs["target_std"] = target_std

        index_interval, index_subject, index_path, index_n_samples, index_age_years = [], [], [], [], []

        for interval, payload in split_map.items():
            subjects = [str(s).strip() for s in payload["subjects"]]
            
            interval_group = intervals_group.create_group(str(interval))
            interval_group.attrs["interval"] = str(interval)
            interval_group.attrs["split_name"] = split_name
            interval_group.attrs["n_subjects"] = len(subjects)
            
            interval_samples_total = 0

            for subj in subjects:
                subj_str = canonical_code(subj)
                file_path = Path(SUBJECTS_PATH) / subject_to_filename(subj)

                if not file_path.exists():
                    raise FileNotFoundError(f"Missing RR file: {file_path}")

                serie = np.loadtxt(file_path, dtype=int)
                df_temp = extract_hrv_features(serie=serie, window_size=WINDOW_SIZE)
                
                # Prune safely
                missing = [c for c in features_to_keep if c not in df_temp.columns]
                if missing:
                    raise KeyError(f"Missing columns in subject '{subj_str}': {missing}")
                
                df_temp = df_temp[features_to_keep].copy()
                age_years = get_age_years(subj, ages_table, year_to_weeks=YEAR_TO_WEEKS)

                # Normalize using GLOBAL attributes
                df_norm = normalize_subject_df(
                    df_temp, static_mean, static_std, rr_mean, rr_std, target_mean, target_std
                )

                X = df_norm[x_cols].to_numpy(dtype=np.float32)
                y = df_norm[y_col].to_numpy(dtype=np.float32)
                n_samples = X.shape[0]
                interval_samples_total += n_samples

                subject_group = interval_group.create_group(f"subject_{subj_str}")
                subject_group.create_dataset("X", data=X, compression="gzip", compression_opts=4, chunks=True)
                subject_group.create_dataset("y", data=y, compression="gzip", compression_opts=4, chunks=True)

                # Annotate subject
                subject_group.attrs["file_id"] = subj_str
                subject_group.attrs["interval"] = str(interval)
                subject_group.attrs["age_years"] = float(age_years)
                subject_group.attrs["n_samples"] = n_samples

                index_interval.append(str(interval))
                index_subject.append(subj_str)
                index_path.append(f"/intervals/{interval}/subject_{subj_str}")
                index_n_samples.append(n_samples)
                index_age_years.append(float(age_years))

            interval_group.attrs["n_interval_samples"] = interval_samples_total

        # Save index table
        index_group.create_dataset("interval", data=np.array(index_interval, dtype=string_dt))
        index_group.create_dataset("subject_id", data=np.array(index_subject, dtype=string_dt))
        index_group.create_dataset("h5_path", data=np.array(index_path, dtype=string_dt))
        index_group.create_dataset("n_samples", data=np.array(index_n_samples, dtype=np.int64))
        index_group.create_dataset("age_years", data=np.array(index_age_years, dtype=np.float64))
        index_group.attrs["total_samples"] = int(np.sum(index_n_samples))

        out_h5.attrs["total_subjects"] = len(index_subject)
        out_h5.attrs["total_samples"] = int(np.sum(index_n_samples))

    return len(index_subject), int(np.sum(index_n_samples))


# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Warming up Numba JIT compiler (this may take 5-15 seconds)...")
    # Feed a small array of random beats to compile the AR Burg functions
    dummy_data = np.random.normal(700, 50, size=100)
    _ = extract_hrv_features(dummy_data, window_size=WINDOW_SIZE)
    print("Numba JIT ready!\n")

    # 1. Load inputs
    ages_table = pd.read_excel(AGES_XLSX, dtype={"code": "string"})
    ages_table["code"] = ages_table["code"].str.strip()
    # Safely canonicalize the codes for easy matching
    ages_table["code"] = ages_table["code"].map(canonical_code)

    with open(SUBJECTS_TEST_JSON, "r", encoding="utf-8") as f:
        subjects_test = json.load(f)

    # 2. Split the dictionaries
    validation_split, test_split = split_subjects_by_interval(subjects_test, seed=SPLIT_SEED)

    # 3. Save JSON manifests
    with open(VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_split, f, indent=4, ensure_ascii=False)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=4, ensure_ascii=False)

    # 4. Generate HDF5 Files
    print(f"Generating Validation Data from Global Stats in {TRAIN_H5}...")
    v_subs, v_samps = save_split_h5(VALIDATION_H5, "validation", validation_split, TRAIN_H5, ages_table)
    
    print(f"Generating Test Data from Global Stats in {TRAIN_H5}...")
    t_subs, t_samps = save_split_h5(TEST_H5, "test", test_split, TRAIN_H5, ages_table)

    # 5. Summary Output
    requested_total = sum(len(v["subjects"]) if isinstance(v, dict) else len(v) for v in subjects_test.values())
    
    print("-" * 50)
    print("SPLIT SUMMARY")
    print("-" * 50)
    print(f"Total Subjects Read: {requested_total}")
    print(f"Validation Set:      {v_subs} subjects ({v_samps} samples) -> {VALIDATION_H5}")
    print(f"Test Set:            {t_subs} subjects ({t_samps} samples) -> {TEST_H5}")
    print(f"Normalization:       Inherited globally from {TRAIN_H5}")


