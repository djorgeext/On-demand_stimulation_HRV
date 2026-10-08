"""
tools/make_test_plots.py - closed-loop imputation plots on the subjects of one split (default: the 10 test
subjects -> results/test/; --split val writes the 16 validation subjects to results/validation/), for every model.

For each subject x model, on the ENTIRE series: drop 80% of the beats (seed 7, the semantics of
utils2.random_extraction with start_idx=0; the same mask for every model), fill them in closed loop
as on the pacemaker (correlation_study_v3: Burg-AR cold start for the first 30 beats, then the
model, whose fills feed later windows), and save, one folder per subject and model:
  results/test/<subject>/<model>/<model>_psd.png                 log-log PSD of the entire series, original
                                                         vs filled (psd() verbatim from fft_visualizations.ipynb)
  results/test/<subject>/<model>/<model>_series_0-1500.png       beats 0-1500, original vs filled
  results/test/<subject>/<model>/<model>_series_10000-11500.png  beats 10000-11500, original vs filled
  results/test/<subject>/<model>/<model>_filled.npy              the filled series (float32), to re-plot
  results/test/<subject>/<model>/<model>_filled.txt              same, np.savetxt '%g' (--export_txt; TCNs)
  results/test/<subject>/dropped_nan.txt                         original with NaN at the dropped beats
plus results/test/summary.csv (per subject x model) and results/test/summary_by_model.csv.
Diagnostic only: the model selection is final (made on validation; the single final test is done).

Run (CPU; one closed loop per process, single-threaded):
  .venv/bin/python tools/make_test_plots.py [--split test|val] [--out DIR] [--workers 24] [--int8 selected|all]
                                           [--models a b ...] [--subjects c d ...]
  ... --replot                  redraw the plots from the saved filled series (no closed loop)
  ... --export_txt [MODEL ...]  write the saved filled series as .txt (default: the TCN models)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_k, "1")  # many single-threaded workers beat one multi-threaded one here
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")  # CPU: per-beat calls are faster than on the ROCm GPU

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# output root; spawned workers re-import this module, so main() passes --out through the environment
OUT = Path(os.environ.get("DRR_PLOTS_OUT", ROOT / "results" / "test"))
sys.path.insert(0, str(ROOT / "src"))

PERCENT, SEED = 0.8, 7
WIN_PSD = 4096       # as fft_visualizations.ipynb calls psd()
SELECTED = "tcn_lite_ft"
SLOW = {"gbdt", "gbdt_all", "tcn_mha_current"}  # scheduled first so the pool ends evenly


def psd(serie, window_size=2048):  # verbatim from fft_visualizations.ipynb
    from scipy.stats import zscore
    overlap = window_size // 2
    quantity = len(serie) // overlap
    cutting = quantity * overlap
    serie_right = np.reshape(serie[:cutting], (quantity, overlap))
    serie_right = np.concatenate((serie_right[:-1], serie_right[1:]), axis=1)
    serie_left = np.reshape(np.flip(serie)[:cutting], (quantity, overlap))
    serie_left = np.concatenate((serie_left[:-1], serie_left[1:]), axis=1)
    serie_matrix = np.concatenate((serie_right, serie_left), axis=0)
    serie_matrix = zscore(serie_matrix, axis=1)
    serie_matrix = np.abs(np.fft.fft(serie_matrix, axis=1)) ** 2
    return np.mean(serie_matrix, axis=0)[:window_size // 2 + 1]


def drop_mask(n, percent=PERCENT, seed=SEED):
    """utils2.random_extraction(serie, percent, start_idx=0, seed) as a boolean mask."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(0, n), size=int(round(n * percent)), replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def model_specs(int8):
    specs = [("persistence", "baseline"), ("ema_mean", "baseline"), ("rls_ar", "baseline")]
    for run in sorted(p.name for p in (ROOT / "runs").iterdir() if (p / "config.json").exists()):
        specs.append((run, "run"))
        if (ROOT / "runs" / run / "model_int8.tflite").exists() and (int8 == "all" or run == SELECTED):
            specs.append((run + "_int8", "int8"))
    specs.append(("tcn_mha_current", "legacy"))
    return specs


_CACHE = {}


def get_predictor(name, kind):
    if name not in _CACHE:
        import utils as U
        if kind == "baseline":
            pred = {"persistence": U.Persistence, "ema_mean": U.EMAMean, "rls_ar": U.RLSAR}[name]()
        elif kind == "run":
            pred = U.RunPredictor(ROOT / "runs" / name)
        elif kind == "int8":
            pred = U.TFLitePredictor(ROOT / "runs" / name[:-len("_int8")])
        else:
            pred = U.LegacyPredictor(str(U.LEGACY_DIR / "tcn_attention_hrv_best.keras"), str(U.DATA_DIR / "hrv_dataset.h5"))
        _CACHE[name] = pred
    return _CACHE[name]


SERIES_WINDOWS = [(0, 1500), (10000, 11500)]  # beat ranges of the series plots


def make_plots(name, code, rr, filled, mask):
    """results/test/<code>/<name>/<name>_psd.png (fft_visualizations.ipynb format) and one series plot per window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = OUT / code / name
    d.mkdir(parents=True, exist_ok=True)
    tag = f"Subject: {code}, Model: {name}, Percent: {int(PERCENT * 100)}%, Seed: {SEED}"

    freqs = np.linspace(0, 0.5, WIN_PSD // 2 + 1)
    fig = plt.figure(figsize=(14, 7))
    plt.loglog(freqs, psd(rr, window_size=WIN_PSD), label="Original", color="blue")
    plt.loglog(freqs, psd(filled, window_size=WIN_PSD), label=f"Filled ({name})", color="green")
    plt.title(tag)
    plt.xlabel("Frequency (cycles/beat)")
    plt.ylabel("Power Spectral Density")
    plt.ylim(1e2, 1e7)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(d / f"{name}_psd.png")
    plt.close(fig)

    for a, b in SERIES_WINDOWS:
        x = np.arange(a, min(b, len(rr)))
        fig = plt.figure(figsize=(20, 10))
        plt.plot(x, rr[x], label="Original", color="blue")
        plt.plot(x, filled[x], label=f"Filled ({name})", color="green")
        plt.xlim(a, b)
        plt.title(f"{tag}, samples {a}-{b}, missing values in this range: {int(mask[x].sum())}, "
                  f"in the first 30 elements: {int(mask[:30].sum())}")
        plt.xlabel("Sample Index")
        plt.ylabel("RR Interval Value (ms)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(d / f"{name}_series_{a}-{b}.png")
        plt.close(fig)


def replot(name, code):
    """Redraw the plots from the saved filled series (no closed loop)."""
    import utils as U
    rr = U.load_series(U.DATA_DIR / U.SERIES_DIR / f"{code}.txt")
    filled = np.load(OUT / code / name / f"{name}_filled.npy").astype(np.float64)
    mask = drop_mask(len(rr))
    assert len(filled) == len(rr) and np.allclose(filled[~mask], rr[~mask]), (name, code)
    make_plots(name, code, rr, filled, mask)
    return name, code


def _replot_task(args):
    return replot(*args)


def export_txt(names, subjects):
    """results/test/<code>/<model>/<model>_filled.txt (np.savetxt fmt '%g', as utils2 writes modified_serie_filled.txt)
    and results/test/<code>/dropped_nan.txt (original with NaN at the dropped beats, as modified_serie_nan.txt)."""
    import utils as U
    n_written = 0
    for _, code in subjects:
        d = OUT / code
        rr = U.load_series(U.DATA_DIR / U.SERIES_DIR / f"{code}.txt")
        mask = drop_mask(len(rr))
        nan_serie = rr.astype(np.float64).copy()
        nan_serie[mask] = np.nan
        np.savetxt(d / "dropped_nan.txt", nan_serie, fmt="%g")
        for name in names:
            src = d / name / f"{name}_filled.npy"
            if src.exists():
                filled = np.load(src).astype(np.float64)
                assert len(filled) == len(rr) and np.allclose(filled[~mask], rr[~mask]), (name, code)
                np.savetxt(d / name / f"{name}_filled.txt", filled, fmt="%g")
                n_written += 1
    return n_written


def run_task(task):
    name, kind, code, interval = task
    os.chdir(ROOT)
    from threadpoolctl import threadpool_limits
    import utils as U

    rr = U.load_series(U.DATA_DIR / U.SERIES_DIR / f"{code}.txt")
    mask = drop_mask(len(rr))
    ar = U.ColdStartAR.for_interval(interval)
    ref_mean, ref_std = U.rr_reference(U.interval_age_years(interval))
    t0 = time.time()
    with threadpool_limits(limits=1):
        t, p, filled = U.closed_loop_subject(get_predictor(name, kind), rr, mask,
                                             lambda h: ar.forecast(h, ref_mean, ref_std))
    secs = time.time() - t0
    # runnable check: kept beats untouched, exactly 80% dropped, every model-filled beat scored
    assert np.array_equal(filled[~mask], rr[~mask]) and mask.sum() == int(round(len(rr) * PERCENT))
    assert len(t) == mask[U.WINDOW:].sum() and np.all(np.isfinite(filled)), (name, code)

    d = OUT / code / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}_filled.npy", filled.astype(np.float32))
    make_plots(name, code, rr, filled, mask)

    e = p - t
    return dict(model=name, subject=code, interval=interval, n_beats=len(rr), n_imputed=len(t),
                rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                r2=float(1.0 - np.sum(e ** 2) / (np.sum((t - t.mean()) ** 2) + U.EPS)),
                corr_series=float(np.corrcoef(rr, filled)[0, 1]), seconds=round(secs, 1),
                **U.spectral_fidelity(rr, filled))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["test", "val"], default="test", help="subjects to run (default test)")
    ap.add_argument("--out", default=None, help="output folder (default: results/test or results/validation)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--int8", choices=["selected", "all"], default="selected",
                    help="int8 TFLite versions: only the deployed model (default) or every Keras run")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names (default: all)")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subject codes (default: all 10)")
    ap.add_argument("--replot", action="store_true",
                    help="only redraw the plots from the saved results/<split>/<subject>/<model>/<model>_filled.npy")
    ap.add_argument("--export_txt", nargs="*", default=None, metavar="MODEL",
                    help="write the saved filled series of these models as .txt (default: every TCN model)")
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out) if args.out else ROOT / "results" / ("test" if args.split == "test" else "validation")
    OUT = OUT if OUT.is_absolute() else ROOT / OUT
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["DRR_PLOTS_OUT"] = str(OUT)  # inherited by the spawned workers
    import pandas as pd
    import utils as U
    subjects = U.read_manifest(U.DATA_DIR / U.MANIFESTS[args.split])
    specs = model_specs(args.int8)
    if args.models:
        specs = [s for s in specs if s[0] in args.models]
    if args.subjects:
        subjects = [s for s in subjects if s[1] in args.subjects]
    if args.export_txt is not None:
        names = args.export_txt or [n for n, _ in specs if "tcn" in n]  # TCN family, incl. TCN_MHA
        print(f"exported {export_txt(names, subjects)} filled series as .txt: {', '.join(names)}", flush=True)
        return
    if args.replot:
        jobs = [(n, c) for n, _ in specs for _, c in subjects if (OUT / c / n / f"{n}_filled.npy").exists()]
        with get_context("spawn").Pool(args.workers) as pool:
            done = list(pool.imap_unordered(_replot_task, jobs))
        print(f"replotted {len(done)} subject x model pairs", flush=True)
        return
    tasks = [(n, k, c, iv) for n, k in specs for iv, c in subjects]
    tasks.sort(key=lambda x: x[0] not in SLOW)
    print(f"{len(specs)} models x {len(subjects)} subjects = {len(tasks)} closed loops, "
          f"{args.workers} workers -> {OUT}/", flush=True)

    rows, t0 = [], time.time()
    with get_context("spawn").Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            rows.append(row)
            print(f"[{i}/{len(tasks)}] {time.time() - t0:6.0f}s  {row['subject']:>11} {row['model']:<18} "
                  f"rmse {row['rmse']:6.2f}  invented {row['rsa_invented']:.0f}  ({row['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows).sort_values(["subject", "model"])
    df.to_csv(OUT / "summary.csv", index=False)
    cols = ["rmse", "mae", "r2", "corr_series", "hf_power_log2", "rsa_invented", "rsa_displaced", "rsa_lost"]
    by = df.groupby("model")[cols].mean().sort_values("rmse")
    by.to_csv(OUT / "summary_by_model.csv")
    print(by.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
