"""
tools/crosscorr.py - correlation between each ENTIRE original test series and the series filled in closed
loop by each model (80% dropped, seed 7; the filled series saved by make_test_plots.py), computed as
    np.corrcoef(original_serie, modified_serie)[0, 1]
(as utils2._process_single_run computes 'Correlation'). The same function on the beat-to-beat
changes, np.corrcoef(np.diff(original), np.diff(filled))[0, 1], is reported next to it: it isolates
the short-term dynamics, where the models differ most.

Writes
  results/test/<subject>/<model>/<model>_crosscorr.csv   corrcoef_rr, corrcoef_drr for that subject and model
  results/test/crosscorr.csv                             all subjects x models
  results/test/crosscorr_by_model.csv                    mean / min / max over the 10 test subjects
Run: .venv/bin/python tools/crosscorr.py [--out results/test|results/validation]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "test"
sys.path.insert(0, str(ROOT / "src"))


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/test", help="folder written by make_test_plots.py (results/test or results/validation)")
    OUT = ROOT / ap.parse_args().out
    import utils as U
    rows = []
    for subj in sorted(p for p in OUT.iterdir() if p.is_dir() and (p / "dropped_nan.txt").exists()):
        original_serie = U.load_series(U.DATA_DIR / U.SERIES_DIR / f"{subj.name}.txt").astype(np.float64)
        for mdir in sorted(p for p in subj.iterdir() if p.is_dir()):
            m = mdir.name
            modified_serie = np.load(mdir / f"{m}_filled.npy").astype(np.float64)
            assert len(modified_serie) == len(original_serie), (subj.name, m)
            row = dict(subject=subj.name, model=m, n_beats=len(original_serie),
                       corrcoef_rr=np.corrcoef(original_serie, modified_serie)[0, 1],
                       corrcoef_drr=np.corrcoef(np.diff(original_serie), np.diff(modified_serie))[0, 1])
            pd.DataFrame([row]).to_csv(mdir / f"{m}_crosscorr.csv", index=False, float_format="%.6f")
            rows.append(row)
    df = pd.DataFrame(rows)
    # runnable check: same numbers as the closed-loop summary's corr_series (computed in float64 there)
    summ = pd.read_csv(OUT / "summary.csv", dtype={"subject": str}).set_index(["subject", "model"])
    chk = df.set_index(["subject", "model"]).join(summ["corr_series"])
    assert np.allclose(chk.corrcoef_rr, chk.corr_series, atol=1e-5), (chk.corrcoef_rr - chk.corr_series).abs().max()
    df.to_csv(OUT / "crosscorr.csv", index=False, float_format="%.6f")
    g = df.groupby("model")
    by = pd.DataFrame(dict(corrcoef_rr_mean=g.corrcoef_rr.mean(), corrcoef_rr_min=g.corrcoef_rr.min(),
                           corrcoef_rr_max=g.corrcoef_rr.max(), corrcoef_drr_mean=g.corrcoef_drr.mean(),
                           corrcoef_drr_min=g.corrcoef_drr.min(), corrcoef_drr_max=g.corrcoef_drr.max(),
                           n_subjects=g.size())).sort_values("corrcoef_rr_mean", ascending=False)
    by.to_csv(OUT / "crosscorr_by_model.csv", float_format="%.6f")
    print(f"{len(df)} subject x model correlations (np.corrcoef, entire series)")
    print(by.round(4).to_string())


if __name__ == "__main__":
    main()
