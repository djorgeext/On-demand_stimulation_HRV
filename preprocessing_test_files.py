import json
from pathlib import Path
from collections import OrderedDict
from utils import extract_hrv_features
import h5py
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
TRAIN_H5 = "hrv_dataset.h5"
SUBJECTS_TEST_JSON = "subjects_test.json"
AGES_XLSX = "ages_table.xlsx"

VALIDATION_H5 = "hrv_validation.h5"
TEST_H5 = "hrv_test.h5"

VALIDATION_JSON = "subjects_validation.json"
TEST_JSON = "subjects_test_split.json"

YEAR_TO_WEEKS = 52.14
EPS = 1e-8
SPLIT_SEED = 7

interval_norm_cols = ["sdsd", "sd2", "ccm", "guzik", "nn50", "porta", "std", "target"]
subject_specific_cols = ["mean"] + [f"rr_{i}" for i in range(1, 21)]
x_cols = [c for c in interval_norm_cols if c != "target"] + subject_specific_cols
y_col = "target"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def canonical_code(x):
    """
    Normalize subject codes so numeric text, numeric cells, and zero-padded
    numeric strings map to the same key.
    Examples:
        9    -> "009"
        "9"  -> "009"
        "009"-> "009"
        "4100" -> "4100"
        "nsr047RRcl" -> "nsr047RRcl"
    """
    s = str(x).strip()
    if s.isdigit():
        return f"{int(s):03d}" if len(s) < 3 else s
    return s


def subject_to_filename(subj):
    s = canonical_code(subj)
    if s.isdigit() and len(s) < 3:
        return f"{int(s):03d}.txt"
    return f"{s}.txt"


def load_ages_table(path):
    ages = pd.read_excel(path, dtype={"code": "string"})
    if "code" not in ages.columns or "age-weeks" not in ages.columns:
        raise KeyError("ages_table.xlsx must contain 'code' and 'age-weeks' columns.")

    ages["code"] = ages["code"].astype("string").str.strip()
    ages["age-weeks"] = pd.to_numeric(ages["age-weeks"], errors="coerce")

    if ages["age-weeks"].isna().any():
        bad_rows = ages.loc[ages["age-weeks"].isna(), ["code", "age-weeks"]]
        raise ValueError(f"Found non-numeric age-weeks values in ages_table:\n{bad_rows}")

    ages["_code_key"] = ages["code"].map(canonical_code)

    dup_counts = ages.groupby("_code_key")["age-weeks"].nunique()
    ambiguous = dup_counts[dup_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Some subject codes map to multiple different ages after canonicalization:\n"
            f"{ambiguous}"
        )

    age_lookup = (
        ages.drop_duplicates("_code_key")
            .set_index("_code_key")["age-weeks"]
            .to_dict()
    )
    return ages, age_lookup


def get_age_years(subj, age_lookup, year_to_weeks=52.14):
    key = canonical_code(subj)
    if key not in age_lookup:
        raise ValueError(f"Subject {subj} not found in ages_table.")
    return float(age_lookup[key]) / year_to_weeks


def get_rr_reference(age_years):
    rr_mean_ref = 505 * (age_years ** 0.122)
    if age_years <= 12:
        rr_std_ref = 80 * (age_years ** 0.26)
    else:
        rr_std_ref = 290 * (age_years ** (-0.2))
    return rr_mean_ref, rr_std_ref


def load_subjects_test_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise TypeError("subjects_test.json must be a dictionary keyed by interval.")

    out = OrderedDict()
    for interval, payload in raw.items():
        if isinstance(payload, dict) and "subjects" in payload:
            subjects = payload["subjects"]
        elif isinstance(payload, list):
            subjects = payload
        else:
            raise TypeError(
                f"Interval '{interval}' must map to a dict with 'subjects' or a list of subjects."
            )

        out[str(interval)] = [str(s).strip() for s in subjects]

    return out


def split_subjects_by_interval(subjects_by_interval, seed=7):
    """
    Strategy:
    - split independently inside each interval
    - singleton intervals -> validation only
    - even intervals -> 50/50 split
    - odd intervals -> floor(n/2) validation, ceil(n/2) test
      (keeps validation as small as possible because singleton intervals
      already force validation upward)
    """
    rng = np.random.default_rng(seed)

    validation = OrderedDict()
    test = OrderedDict()

    for interval, subjects in subjects_by_interval.items():
        subs = [str(s).strip() for s in subjects]
        rng.shuffle(subs)
        n = len(subs)

        if n == 0:
            val_subs, test_subs = [], []
        elif n == 1:
            val_subs, test_subs = subs, []
        else:
            n_val = n // 2
            val_subs = subs[:n_val]
            test_subs = subs[n_val:]

        validation[interval] = {"subjects": val_subs}
        test[interval] = {"subjects": test_subs}

    return validation, test


def normalize_subject_df(df, age_years, interval_mean, interval_std):
    """
    Applies:
    - subject-specific scaling to mean + rr_cols
    - interval z-score to the rest, including target
    """
    df = df.copy()

    rr_mean_ref, rr_std_ref = get_rr_reference(age_years)
    rr_std_ref = max(float(rr_std_ref), EPS)

    needed = subject_specific_cols + interval_norm_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataframe: {missing}")

    # Make sure these columns can store floats
    df[subject_specific_cols] = df[subject_specific_cols].astype(np.float64)
    df[interval_norm_cols] = df[interval_norm_cols].astype(np.float64)

    # Subject-specific normalization
    df.loc[:, subject_specific_cols] = (df[subject_specific_cols] - rr_mean_ref) / rr_std_ref

    # Interval-level z-score normalization
    df.loc[:, interval_norm_cols] = (df[interval_norm_cols] - interval_mean) / interval_std

    return df, rr_mean_ref, rr_std_ref


def save_split_h5(output_path, split_name, split_map, train_h5, age_lookup):
    string_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as out_h5:
        # Global metadata
        out_h5.attrs["source_h5"] = TRAIN_H5
        out_h5.attrs["source_subjects_json"] = SUBJECTS_TEST_JSON
        out_h5.attrs["source_split_name"] = split_name
        out_h5.attrs["split_seed"] = SPLIT_SEED
        out_h5.attrs["year_to_weeks"] = YEAR_TO_WEEKS
        out_h5.attrs["rr_mean_formula"] = "505 * x**0.122"
        out_h5.attrs["rr_std_formula_le_12"] = "80 * x**0.26"
        out_h5.attrs["rr_std_formula_gt_12"] = "290 * x**(-0.2)"
        out_h5.attrs["interval_norm_cols"] = np.array(interval_norm_cols, dtype=string_dt)
        out_h5.attrs["subject_specific_cols"] = np.array(subject_specific_cols, dtype=string_dt)
        out_h5.attrs["x_cols"] = np.array(x_cols, dtype=string_dt)
        out_h5.attrs["y_col"] = y_col

        intervals_group = out_h5.create_group("intervals")
        norm_group = out_h5.create_group("normalization")
        index_group = out_h5.create_group("index")

        index_interval = []
        index_subject = []
        index_path = []
        index_n_samples = []
        index_age_years = []

        if "normalization" not in train_h5:
            raise KeyError("Training H5 is missing the /normalization group.")

        for interval, payload in split_map.items():
            subjects = [str(s).strip() for s in payload["subjects"]]

            if interval not in train_h5["normalization"]:
                raise KeyError(f"Interval '{interval}' not found in training normalization data.")

            src_norm = train_h5["normalization"][interval]
            interval_mean = np.asarray(src_norm["mean"][()], dtype=np.float64)
            interval_std = np.asarray(src_norm["std"][()], dtype=np.float64)
            interval_std = np.where(interval_std < EPS, 1.0, interval_std)

            if interval_mean.shape[0] != len(interval_norm_cols):
                raise ValueError(
                    f"Normalization size mismatch for interval '{interval}': "
                    f"expected {len(interval_norm_cols)}, got {interval_mean.shape[0]}"
                )

            # Interval group in the output file
            interval_group = intervals_group.create_group(str(interval))
            interval_group.attrs["interval"] = str(interval)
            interval_group.attrs["split_name"] = split_name
            interval_group.attrs["n_subjects"] = len(subjects)
            interval_group.attrs["n_interval_samples"] = 0

            # Copy normalization metadata from training H5
            dst_norm = norm_group.create_group(str(interval))
            dst_norm.create_dataset(
                "columns",
                data=np.array(
                    [c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c)
                     for c in src_norm["columns"][()]],
                    dtype=string_dt,
                ),
            )
            dst_norm.create_dataset("mean", data=interval_mean.astype(np.float64))
            dst_norm.create_dataset("std", data=interval_std.astype(np.float64))
            dst_norm.attrs["n_samples"] = int(src_norm.attrs["n_samples"])

            if "target" in interval_norm_cols:
                target_idx = interval_norm_cols.index("target")
                dst_norm.attrs["target_mean"] = float(interval_mean[target_idx])
                dst_norm.attrs["target_std"] = float(interval_std[target_idx])

            interval_samples_total = 0

            for subj in subjects:
                subj_str = canonical_code(subj)
                file_path = Path(subjects_path) / subject_to_filename(subj)

                if not file_path.exists():
                    raise FileNotFoundError(f"Missing RR file for subject '{subj_str}': {file_path}")

                serie = np.loadtxt(file_path, dtype=int)

                df_temp = extract_hrv_features(
                    serie=serie,
                    window_size=20,
                    window_size_long=40,
                )

                required_cols = x_cols + [y_col]
                missing = [c for c in required_cols if c not in df_temp.columns]
                if missing:
                    raise KeyError(
                        f"Missing columns in subject '{subj_str}', interval '{interval}': {missing}"
                    )

                age_years = get_age_years(subj, age_lookup, year_to_weeks=YEAR_TO_WEEKS)

                df_norm, rr_mean_ref, rr_std_ref = normalize_subject_df(
                    df_temp,
                    age_years=age_years,
                    interval_mean=interval_mean,
                    interval_std=interval_std,
                )

                X = df_norm[x_cols].to_numpy(dtype=np.float32)
                y = df_norm[y_col].to_numpy(dtype=np.float32)

                n_samples = int(X.shape[0])
                interval_samples_total += n_samples

                subject_group = interval_group.create_group(f"subject_{subj_str}")
                subject_group.create_dataset(
                    "X",
                    data=X,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )
                subject_group.create_dataset(
                    "y",
                    data=y,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )

                subject_group.attrs["file_id"] = subj_str
                subject_group.attrs["interval"] = str(interval)
                subject_group.attrs["age_years"] = float(age_years)
                subject_group.attrs["rr_mean_ref"] = float(rr_mean_ref)
                subject_group.attrs["rr_std_ref"] = float(rr_std_ref)
                subject_group.attrs["n_samples"] = n_samples
                subject_group.attrs["x_cols"] = np.array(x_cols, dtype=string_dt)
                subject_group.attrs["y_col"] = y_col

                index_interval.append(str(interval))
                index_subject.append(subj_str)
                index_path.append(f"/intervals/{interval}/subject_{subj_str}")
                index_n_samples.append(n_samples)
                index_age_years.append(float(age_years))

            interval_group.attrs["n_interval_samples"] = int(interval_samples_total)

        # Save index table
        index_group.create_dataset("interval", data=np.array(index_interval, dtype=string_dt))
        index_group.create_dataset("subject_id", data=np.array(index_subject, dtype=string_dt))
        index_group.create_dataset("h5_path", data=np.array(index_path, dtype=string_dt))
        index_group.create_dataset("n_samples", data=np.array(index_n_samples, dtype=np.int64))
        index_group.create_dataset("age_years", data=np.array(index_age_years, dtype=np.float64))
        index_group.attrs["total_samples"] = int(np.sum(index_n_samples))

        # File-level summary
        out_h5.attrs["total_subjects"] = len(index_subject)
        out_h5.attrs["total_samples"] = int(np.sum(index_n_samples))
        out_h5.attrs["split_subjects"] = split_name

    return {
        "output_path": output_path,
        "total_subjects": len(index_subject),
        "total_samples": int(np.sum(index_n_samples)),
    }


# ------------------------------------------------------------
# LOAD INPUTS
# ------------------------------------------------------------
subjects_path = 'series'
subjects_by_interval = load_subjects_test_json(SUBJECTS_TEST_JSON)
_, age_lookup = load_ages_table(AGES_XLSX)

with h5py.File(TRAIN_H5, "r") as train_h5:
    validation_split, test_split = split_subjects_by_interval(subjects_by_interval, seed=SPLIT_SEED)

    # Save split manifests
    with open(VALIDATION_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_split, f, indent=4, ensure_ascii=False)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_split, f, indent=4, ensure_ascii=False)

    # Build validation and test HDF5 files
    val_info = save_split_h5(VALIDATION_H5, "validation", validation_split, train_h5, age_lookup)
    test_info = save_split_h5(TEST_H5, "test", test_split, train_h5, age_lookup)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
# requested_total = sum(len(v["subjects"]) for v in subjects_by_interval.values())
requested_total = sum(len(v) for v in subjects_by_interval.values())
requested_half = requested_total / 2.0

val_total = sum(len(v["subjects"]) for v in validation_split.values())
test_total = sum(len(v["subjects"]) for v in test_split.values())

print(f"Input subjects from JSON: {requested_total}")
print(f"Requested split target: ~{requested_half:.1f} / ~{requested_half:.1f}")
print(f"Validation subjects: {val_total}")
print(f"Test subjects: {test_total}")
print(f"Validation H5: {VALIDATION_H5}")
print(f"Test H5: {TEST_H5}")
print(f"Validation JSON: {VALIDATION_JSON}")
print(f"Test JSON: {TEST_JSON}")

if val_total != int(round(requested_half)):
    print(
        "Note: exact 50/50 is not possible under the singleton-to-validation rule, "
        "so the split is the closest interval-aware split that preserves it."
    )