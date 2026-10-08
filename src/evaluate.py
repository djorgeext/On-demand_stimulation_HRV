"""
evaluate.py - compare baselines, trained runs, their int8 versions and your current TCN_MHA
on held-out subjects (test split by default; use --split val for model selection).

Examples
  python src/evaluate.py --runs runs/*
  python src/evaluate.py --runs runs/* --int8 --legacy_model legacy/tcn_attention_hrv_best.keras
  python src/evaluate.py --runs runs/micro_tcn runs/gru --closed_loop --percents 0.1 0.3 0.5 --seeds 7 101

Open loop  : one-step-ahead dRR on the true history (RMSE, MAE, P95, skill vs persistence,
             r, sign accuracy with a 5 ms deadband, amplitude ratio, subject-bootstrap 95% CIs,
             Wilcoxon p-value of per-subject RMSE vs --ref).
Closed loop: your streaming imputation test - random beats are dropped and filled with the
             model's own predictions, which then feed the following windows (RR-level error).
Outputs go to results/evaluation/<split>_*.csv by default (use --out results/evaluation/<round>).
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import logging
from pathlib import Path

import pandas as pd

import utils as U


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories created by train.py")
    ap.add_argument("--split", choices=["val", "test"], required=True,
                    help="val for model selection; test only for the single approved final evaluation")
    ap.add_argument("--data_dir", default=str(U.DATA_DIR))
    ap.add_argument("--out", default=str(U.RESULTS_DIR / "evaluation"))
    ap.add_argument("--int8", action="store_true", help="also evaluate model_int8.tflite of each run")
    ap.add_argument("--legacy_model", default=None, help="current TCN_MHA .keras (needs utils2.py)")
    ap.add_argument("--legacy_h5", default=str(U.DATA_DIR / "hrv_dataset.h5"), help="h5 with the legacy z-score stats")
    ap.add_argument("--ref", default="rls_ar", help="reference model for the paired Wilcoxon test")
    ap.add_argument("--closed_loop", action="store_true")
    ap.add_argument("--percents", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7])
    ap.add_argument("--closed_loop_max_beats", type=int, default=5000,
                    help="beats per subject in the closed loop (it runs one prediction per dropped beat)")
    ap.add_argument("--max_beats", type=int, default=None, help="truncate each series (quick tests)")
    return ap.parse_args()


def main():
    args = parse_args()
    logging.getLogger("tensorflow").setLevel(logging.ERROR)  # per-model predict() retracing notices
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{args.split}' subjects and running the streaming feature engine ...")
    subjects = U.load_split(args.split, args.data_dir, max_beats=args.max_beats)
    print(f"{len(subjects)} subjects, {sum(len(s['y']) for s in subjects)} windows")

    p = U.ENGINE["ar_order"]
    predictors = [U.Persistence(), U.EMAMean(), U.RLSAR()]
    complexity = [dict(model="persistence", params=0, macs_stream=0),
                  dict(model="ema_mean", params=0, macs_stream=U.WINDOW),
                  dict(model="rls_ar", params=0, macs_stream=2 * p * p + 3 * p)]  # ~FLOPs per beat
    for run in sorted(map(Path, args.runs)):
        if not (run / "config.json").exists():
            print(f"skip {run}: no config.json")
            continue
        rp = U.RunPredictor(run)
        predictors.append(rp)
        c = dict(model=rp.name, **rp.cfg.get("complexity", {}))
        complexity.append(c)
        if args.int8 and (run / "model_int8.tflite").exists():
            predictors.append(U.TFLitePredictor(run))
            complexity.append({**c, "model": rp.name + "_int8"})
    if args.legacy_model:
        lp = U.LegacyPredictor(args.legacy_model, args.legacy_h5)
        predictors.append(lp)
        complexity.append(dict(model=lp.name, **U.keras_complexity(lp.model)))

    cx = pd.DataFrame(complexity)
    if "params" in cx:
        cx["int8_weights_kb"] = (cx["params"] / 1024).round(1)

    # ---------------- open loop ----------------
    print("\nOpen-loop evaluation ...")
    summary, per_subj, per_int = U.evaluate_open_loop(predictors, subjects, ref=args.ref)
    summary = summary.merge(cx, on="model", how="left")
    summary.to_csv(out / f"{args.split}_open_loop_summary.csv", index=False)
    per_subj.to_csv(out / f"{args.split}_open_loop_per_subject.csv", index=False)
    per_int.to_csv(out / f"{args.split}_open_loop_per_interval.csv", index=False)

    cols = [c for c in ["model", "rmse", "rmse_lo", "rmse_hi", "mae", "p95", "skill", "r", "sign_acc",
                        "amp_ratio", f"p_vs_{args.ref}", "params", "macs_window", "macs_stream",
                        "int8_weights_kb", "trees", "compares_per_pred"] if c in summary]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\nRMSE (ms) per interval:")
        print(per_int.pivot(index="model", columns="interval", values="rmse")
              .loc[summary["model"]].to_string(float_format=lambda v: f"{v:.2f}"))

    # ---------------- closed loop ----------------
    if args.closed_loop:
        print(f"\nClosed-loop imputation (first {args.closed_loop_max_beats} beats per subject) ...")
        summ, per = U.evaluate_closed_loop(predictors, subjects, args.percents, args.seeds,
                                           args.closed_loop_max_beats)
        summ.to_csv(out / f"{args.split}_closed_loop_summary.csv", index=False)
        per.to_csv(out / f"{args.split}_closed_loop_per_subject.csv", index=False)
        with pd.option_context("display.width", 200):
            print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
