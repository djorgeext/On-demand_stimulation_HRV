"""
train.py - train candidate dRR predictors on your subject-disjoint splits
(subjects_train.json / subjects_validation.json from preprocessing_v2.py).

Examples
  # all candidates, default settings (residual on the RLS forecast where it applies)
  python src/train.py --models linear mlp gbdt micro_tcn tcn_lite gru

  # batch-size study (same model, same wall-clock budget, LR scaled to the batch)
  python src/train.py --models micro_tcn --batch_size 512  --lr 1e-3 --tag tcn_b512
  python src/train.py --models micro_tcn --batch_size 2048 --lr 2e-3 --warmup_steps 2000 --tag tcn_b2048

  # ablations
  python src/train.py --models micro_tcn --residual off --tag micro_tcn_noresid
  python src/train.py --models gbdt --features all --tag gbdt_all      # importance of every candidate

  # knowledge distillation from your current TCN_MHA (utils2.py must be importable)
  python src/train.py --models micro_tcn gru --teacher legacy/tcn_attention_hrv_best.keras

Scale (~20M windows): the (N, 30, 2) input tensor is never built. Batches are gathered on
the fly by tf.data from the concatenated RR series, each "epoch" is a fresh random draw of
--epoch_samples windows, and Keras runs --steps_per_execution steps per call to remove the
per-step overhead that dominates small models at small batch sizes.

Each run writes runs/<tag>/ with config.json (features, z-score stats, engine settings, data
hashes, ms/step, validation metrics), the model, history.csv, model_int8.tflite (Keras) and,
for GBDT, feature_importance.csv + feature_correlation.csv.
"""
import os

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import gc
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np

import utils as U


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=list(U.MODELS))
    ap.add_argument("--reexport_int8", nargs="+", default=None, metavar="RUN_DIR",
                    help="re-export model_int8.tflite + val_int8_check of existing runs (no training)")
    ap.add_argument("--data_dir", default=str(U.DATA_DIR), help="folder with series/ and the JSON manifests")
    ap.add_argument("--out", default=str(U.RUNS_DIR))
    ap.add_argument("--tag", default=None, help="run name (single model only); default = model name")
    ap.add_argument("--features", nargs="+", default=None,
                    help=f"a set name {list(U.FEATURE_SETS)} or explicit feature names")
    ap.add_argument("--residual", choices=["auto", "on", "off"], default="auto",
                    help="predict dRR - RLS forecast (auto = model default)")
    ap.add_argument("--teacher", default=None, help="current TCN_MHA .keras for distillation")
    ap.add_argument("--teacher_h5", default=str(U.DATA_DIR / "hrv_dataset.h5"), help="h5 holding the teacher's z-score stats")
    ap.add_argument("--kd_alpha", type=float, default=0.5, help="target = (1-a)*true + a*teacher")
    ap.add_argument("--epochs", type=int, default=100, help="max epochs of --epoch_samples windows each")
    ap.add_argument("--epoch_samples", type=int, default=2_000_000,
                    help="windows drawn per epoch (0 = number of training windows)")
    ap.add_argument("--val_samples", type=int, default=200_000,
                    help="fixed random validation windows monitored each epoch (final metrics use all)")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--steps_per_execution", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_steps", type=int, default=0, help="linear LR warm-up (use with large batches)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=211)
    ap.add_argument("--max_beats", type=int, default=None, help="truncate each series (quick tests)")
    ap.add_argument("--no_int8", action="store_true", help="skip int8 TFLite export")
    ap.add_argument("--init_from", default=None, metavar="RUN_DIR",
                    help="fine-tune: start from RUN_DIR/model.keras, reuse its features, residual flag and norm")
    ap.add_argument("--dad_filler", default=None, metavar="RUN_DIR",
                    help="closed-loop (DAD) training: run whose float model fills the dropped beats")
    ap.add_argument("--dad_share", type=float, default=0.0,
                    help="probability that a training window is closed-loop (0 = off; > 0 needs --dad_filler)")
    ap.add_argument("--dad_seed", type=int, default=0, help="seed of the closed-loop drop masks (val uses +1000)")
    args = ap.parse_args()
    if not args.models and not args.reexport_int8:
        ap.error("--models or --reexport_int8 is required")
    if args.tag and args.models and len(args.models) > 1:
        ap.error("--tag only works with a single model")
    if not 0.0 <= args.dad_share < 1.0:
        ap.error("--dad_share must be in [0, 1)")
    if (args.dad_share > 0) != bool(args.dad_filler):
        ap.error("--dad_filler and --dad_share > 0 go together")
    if args.dad_filler and args.teacher:
        ap.error("--dad_filler cannot be combined with --teacher (two different targets)")
    if (args.init_from or args.dad_filler) and args.models and len(args.models) != 1:
        ap.error("--init_from / --dad_filler work with a single model")
    return args


# -----------------------------------------------------------------------------
# Compact training data: features/target per window + concatenated RR + window starts
# -----------------------------------------------------------------------------
def compact_arrays(cfg, subjects, y_list):
    """RAM per window: n_features + 1 floats + one int64 start, instead of a (30, 2) tensor."""
    X = np.concatenate([U.normalize_feats(cfg, s["F"]) for s in subjects])
    z = np.concatenate([U.to_z(cfg, y, s["F"]) for s, y in zip(subjects, y_list)])
    R = np.concatenate([s["rr"] for s in subjects]).astype(np.float32)
    base = np.cumsum([0] + [len(s["rr"]) for s in subjects[:-1]])
    start = np.concatenate([b + s["offset"] + np.arange(len(s["y"])) for b, s in zip(base, subjects)])
    return dict(X=X, z=z, R=R, start=start.astype(np.int64))


MIX_PERIOD = 10000  # batch-mixing resolution: dad_share is applied in steps of 1e-4


def merge_arrays(arr_tf, arr_cl, share):
    """One array set: teacher-forced windows [0, n_tf), then closed-loop windows [n_tf, n_tf + n_cl)."""
    m = {k: np.concatenate([arr_tf[k], arr_cl[k]]) for k in ("X", "z", "R")}
    m["start"] = np.concatenate([arr_tf["start"], arr_cl["start"] + len(arr_tf["R"])])
    m.update(n_tf=len(arr_tf["z"]), n_cl=len(arr_cl["z"]), thr=int(round(MIX_PERIOD * share)))
    return m


def mix_np(arr, i):
    """Raw Dataset.random int64 -> window index (NumPy twin of the mapping inside make_gather_tf).
    Plain arrays: i mod n, as before. Merged arrays: closed-loop iff (i mod 10000) < round(10000 *
    share), so each window is closed-loop with probability share whatever the two sizes are."""
    i = np.asarray(i, dtype=np.int64)
    if not arr.get("n_cl"):
        return np.mod(i, len(arr["z"]))
    u, j = np.mod(i, MIX_PERIOD), np.floor_divide(i, MIX_PERIOD)
    return np.where(u < arr["thr"], arr["n_tf"] + np.mod(j, arr["n_cl"]), np.mod(j, arr["n_tf"]))


def gather_np(cfg, arr, idx):
    """Reference (NumPy) batch for window indices `idx`, built with utils.make_seq."""
    x = {"feats": arr["X"][idx]}
    if cfg["kind"] == "seq":
        x["seq"] = U.make_seq(cfg, arr["R"][arr["start"][idx][:, None] + np.arange(U.WINDOW)])
    return x, arr["z"][idx]


def model_order(cfg, x):
    return [x["seq"], x["feats"]] if cfg["kind"] == "seq" else x["feats"]


def make_gather_tf(cfg, arr):
    """TensorFlow twin of gather_np, used inside tf.data. The arrays live in non-trainable
    CPU variables: captured by reference, so no 2 GB graph-constant limit and no GPU copy."""
    import tensorflow as tf
    nm = cfg["norm"]
    with tf.device("/CPU:0"):
        X, z, R, st = (tf.Variable(arr[k], trainable=False) for k in ("X", "z", "R", "start"))
    n = len(arr["z"])
    offs = tf.range(U.WINDOW, dtype=tf.int64)

    n_cl = arr.get("n_cl", 0)

    def gather(i):
        i = tf.cast(i, tf.int64)
        if n_cl:  # same mapping as mix_np
            u, j = tf.math.floormod(i, MIX_PERIOD), tf.math.floordiv(i, MIX_PERIOD)
            i = tf.where(u < arr["thr"], arr["n_tf"] + tf.math.floormod(j, n_cl),
                         tf.math.floormod(j, arr["n_tf"]))
        else:
            i = tf.math.floormod(i, n)
        x = {"feats": tf.gather(X, i)}
        if cfg["kind"] == "seq":
            win = tf.gather(R, tf.gather(st, i)[:, None] + offs)  # (B, W) raw ms
            drr = tf.concat([tf.zeros_like(win[:, :1]), win[:, 1:] - win[:, :-1]], axis=1)
            x["seq"] = tf.stack([(win - nm["rr_mean"]) / nm["rr_std"],
                                 (drr - nm["drr_mean"]) / nm["drr_std"]], axis=-1)
        return x, tf.gather(z, i)

    return gather


# -----------------------------------------------------------------------------
# Fitters
# -----------------------------------------------------------------------------
def make_callbacks(args, steps, run_dir):
    from tensorflow import keras

    class WarmUp(keras.callbacks.Callback):
        """Linear LR ramp over the first `warmup_steps` optimizer steps, then hands over to
        ReduceLROnPlateau (with steps_per_execution it is applied once per execution)."""

        def on_train_batch_begin(self, batch, logs=None):
            it = int(keras.ops.convert_to_numpy(self.model.optimizer.iterations))
            if it < args.warmup_steps:
                self.model.optimizer.learning_rate = args.lr * (it + 1) / args.warmup_steps
            elif it < args.warmup_steps + args.steps_per_execution:
                self.model.optimizer.learning_rate = args.lr

    class StepTimer(keras.callbacks.Callback):
        """Training ms/step per epoch (validation excluded): the throughput number to optimise."""

        def __init__(self):
            super().__init__()
            self.ms = []

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.perf_counter()

        def on_test_begin(self, logs=None):
            self.ms.append(1000.0 * (time.perf_counter() - self.t0) / steps)

    timer = StepTimer()
    cbs = [WarmUp()] if args.warmup_steps > 0 else []
    cbs += [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, args.patience // 3), min_lr=1e-5),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        timer,
    ]
    return cbs, timer


def warm_start_expand(name, cfg, parent_dir, parent_cfg, val, n_check=256):
    """--init_from with a parent whose features are a subset of cfg's: every parent weight is
    copied; in the Dense that consumes `feats`, the parent's columns go to their features' new
    positions and the new features' input weights are zero, so at init the model computes exactly
    the parent's function (checked below on real val windows)."""
    from tensorflow import keras
    old, new = parent_cfg["features"], cfg["features"]
    pm = keras.models.load_model(parent_dir / "model.keras", compile=False)
    m = U.build_keras_model(name, len(new))
    assert len(pm.layers) == len(m.layers), "parent and new model differ in structure"
    off, f_in = 0, m.get_layer("feats").output  # offset of the feats block in the concatenation
    for layer in m.layers:
        if isinstance(layer, keras.layers.Concatenate):
            ins = layer.input
            assert any(t is f_in for t in ins), "a Concatenate that does not take feats"
            off = sum(t.shape[-1] for t in ins[:[t is f_in for t in ins].index(True)])
    moved = 0
    for lp, ln in zip(pm.layers, m.layers):
        wp, wn = lp.get_weights(), ln.get_weights()
        if len(wp) == len(wn) and all(a.shape == b.shape for a, b in zip(wp, wn)):
            ln.set_weights(wp)
            continue
        assert isinstance(ln, keras.layers.Dense) and moved == 0, f"unexpected shape change in {ln.name}"
        k = np.zeros_like(wn[0])
        k[:off] = wp[0][:off]
        for i, f in enumerate(old):
            k[off + new.index(f)] = wp[0][off + i]
        k[off + len(new):] = wp[0][off + len(old):]
        ln.set_weights([k] + wp[1:])
        moved += 1
    assert moved == 1, "no feature-consuming Dense found"
    # the one check: same predictions as the parent at init, on real val windows (new features included)
    per = max(1, n_check // min(4, len(val)))
    rr = np.concatenate([U.subject_windows(s)[:per] for s in val[:4]]).astype(np.float32)
    F = np.concatenate([s["F"][:per] for s in val[:4]])
    z_par = U.RunPredictor(parent_dir).predict_z(U.make_inputs(parent_cfg, rr, F))
    z_new = np.asarray(m(U.make_inputs(cfg, rr, F), training=False)).ravel()
    err = float(np.max(np.abs(z_new - z_par)))
    assert err <= 1e-5, f"warm start differs from the parent by {err:.2e} (z) at init"
    print(f"warm start from {parent_dir}: + {[f for f in new if f not in old]} (zero weights), "
          f"max |z - parent z| at init = {err:.1e} on {len(rr)} val windows")
    cfg["init_from"]["init_max_abs_diff_z"] = err
    return m


def fit_keras(name, cfg, arr_tr, val_xy, args, run_dir, init_model=None):
    import tensorflow as tf
    from tensorflow import keras
    n = len(arr_tr["z"])
    epoch_samples = n if args.epoch_samples <= 0 else min(args.epoch_samples, n)
    steps = max(1, epoch_samples // args.batch_size)
    spe = max(1, min(args.steps_per_execution, steps))

    gather = make_gather_tf(cfg, arr_tr)
    # the one check: on-the-fly TF batches == utils.make_inputs path used at inference
    n_cl = arr_tr.get("n_cl", 0)
    if n_cl:  # raw int64 draws as Dataset.random yields them, so the mapping is checked too
        probe = np.random.default_rng(args.seed).integers(-2 ** 62, 2 ** 62, size=256)
        idx = mix_np(arr_tr, probe)
        assert (idx < arr_tr["n_tf"]).any() and (idx >= arr_tr["n_tf"]).any(), "probe missed one half"
    else:
        probe = np.random.default_rng(args.seed).integers(0, n, size=64)
        idx = mix_np(arr_tr, probe)  # == probe
    x_tf, z_tf = gather(tf.constant(probe))
    x_np, z_np = gather_np(cfg, arr_tr, idx)
    for k in x_np:
        assert np.allclose(x_tf[k].numpy(), x_np[k], atol=1e-5), f"tf.data '{k}' differs from utils.make_inputs"
    assert np.allclose(z_tf.numpy(), z_np)

    # ponytail: windows are drawn uniformly WITH replacement, so every epoch is a fresh random
    # subset. Ceiling: a few repeats inside an epoch (harmless: neighbouring windows share 29/30
    # beats anyway). Upgrade: Dataset.range(n).shuffle(n) for exact without-replacement passes.
    def draws():
        """Raw int64 draws. Dataset.random only yields values in [0, 2^32): fine for i mod n, but
        the merged-array mapping uses j = i // 10000 (<= 429,496 -> only the first 429k windows of
        each half, the 2026-10-08 tcn_lite_dad bug). For merged arrays, two draws make a 63-bit i."""
        if not n_cl:
            return tf.data.Dataset.random(seed=args.seed)
        return tf.data.Dataset.zip(tf.data.Dataset.random(seed=args.seed),
                                   tf.data.Dataset.random(seed=args.seed + 1)).map(
            lambda hi, lo: tf.math.floormod(hi, 2 ** 31) * 2 ** 32 + lo)

    ds = (draws().batch(args.batch_size, drop_remainder=True)
          .map(gather, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))

    # the check that catches an index stream which cannot reach every window: the first 100 batches
    # (51,200 uniform draws) must reach the last 10% of each half (miss probability ~ 0.9^25600)
    raw = np.concatenate([b.numpy() for b in draws().batch(args.batch_size, drop_remainder=True).take(100)])
    idx = mix_np(arr_tr, raw)
    n_tf = arr_tr.get("n_tf", n)
    halves = [("teacher-forced", idx[idx < n_tf], n_tf)]
    if n_cl:
        halves.append(("closed-loop", idx[idx >= n_tf] - n_tf, n_cl))
    for label, h, size in halves:
        assert len(h) and h.max() >= 0.9 * size, \
            f"{label} draws reach only window {h.max() if len(h) else -1} of {size}: index stream too narrow"
    realised = None
    if n_cl:  # the share actually drawn by the first 100 batches of the (identically seeded) stream
        realised = float(np.mean(idx >= n_tf))
        print(f"closed-loop share over the first 100 batches: {realised:.4f} (target {args.dad_share})")

    if init_model is not None:  # warm start with extra inputs (warm_start_expand)
        model = init_model
    elif args.init_from:  # fine-tune: parent's weights, fresh optimizer (compile below)
        model = keras.models.load_model(Path(args.init_from) / "model.keras", compile=False)
    else:
        model = U.build_keras_model(name, len(cfg["features"]))
    # ponytail: plain Huber. Your RSA loss was dropped: with shuffled batches its variance term
    # mostly measures between-subject spread and pulls predictions away from the MSE optimum.
    # Upgrade path: deadbanded sign penalty on the de-residualised output.
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-4),
                  loss=keras.losses.Huber(delta=1.0), steps_per_execution=spe)
    callbacks, timer = make_callbacks(args, steps, run_dir)
    print(f"{steps} steps/epoch x batch {args.batch_size} ({steps * args.batch_size} of {n} windows), "
          f"steps_per_execution={spe}, lr={args.lr}, warmup_steps={args.warmup_steps}")
    model.fit(ds, steps_per_epoch=steps, epochs=args.epochs, validation_data=val_xy,
              validation_batch_size=4096, verbose=2, callbacks=callbacks)
    model.save(run_dir / "model.keras")
    train_info = dict(steps_per_epoch=steps, steps_per_execution=spe, epoch_samples=steps * args.batch_size,
                      ms_per_step_median=float(np.median(timer.ms[1:] if len(timer.ms) > 1 else timer.ms)),
                      epochs_run=len(timer.ms))
    if realised is not None:
        train_info["dad_share_realised_100_batches"] = realised
    return model, train_info


def export_int8(model, X_rep, path):
    """Full-integer post-training quantisation (int8 in/out), the CMSIS-NN / ST Edge AI format.
    ponytail: PTQ only. If the int8 validation gap grows noticeably, upgrade to QAT."""
    import contextlib
    import io
    import tensorflow as tf
    X_rep = X_rep if isinstance(X_rep, list) else [X_rep]  # same order as model.inputs
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x[i:i + 1] for x in X_rep] for i in range(len(X_rep[0])))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # converter prints the SavedModel signature + benign notices
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(conv.convert())
    os.replace(tmp, path)  # atomic: a running evaluate.py keeps its mmap of the old file


# ponytail: deterministic best-of-K PTQ calibration. Full-int8 TFLite PTQ only does min/max, so
# every range follows the most extreme windows of the calibration draw, and the draw (not its size)
# dominated the int8 gap (calibration study 2026-10-08: 0.12-1.17% for tcn_lite on held-out training
# windows; 500 windows in the 18:14 version saturated, 20k widened the step). K fixed draws (2k and
# 5k windows x 4 seeds) are exported, and the one with the smallest int8-vs-float RMSE gap on
# held-out TRAINING subjects (teacher-forced + closed-loop windows, never validation) is kept.
# Ceiling: the choice can fit noise of the 40k-window score. Upgrade path: QAT.
CALIB_SIZES = (2000, 5000)
CALIB_SEEDS = (0, 1, 2, 3)
CALIB_KEY = 0xCA1               # draw rng = default_rng([CALIB_KEY, size, seed])
CALIB_REF = 20000               # plain 20k min/max export (the previous default): the check's reference
CALIB_HOLDOUT_SUBJECTS = 20     # training subjects excluded from every calibration draw
CALIB_HOLDOUT_WINDOWS = 20000   # per kind: teacher-forced and closed-loop
CALIB_HOLDOUT_SEED = 2026
CALIB_CL_SEED = 4242            # masks of the held-out closed-loop windows (DAD uses 0 / 1000)


def window_subjects(train, segments=None):
    """Training-subject index of every window of compact_arrays(train) [+ its closed-loop segments]."""
    subj = [np.repeat(np.arange(len(train), dtype=np.int32), [len(s["y"]) for s in train])]
    if segments is not None:
        ci = {s["code"]: i for i, s in enumerate(train)}
        assert len(ci) == len(train), "subject codes are not unique"
        subj.append(np.repeat(np.array([ci[g["code"]] for g in segments], dtype=np.int32),
                              [len(g["y"]) for g in segments]))
    return np.concatenate(subj)


def calib_holdout(cfg, train, arr_tr, run_dir):
    """Fixed held-out TRAINING windows that score the calibration candidates: teacher-forced windows
    of CALIB_HOLDOUT_SUBJECTS subjects + one-step closed-loop windows of the same subjects, filled
    by the run's own float model (as the evaluator's closed loop does)."""
    import tensorflow as tf
    rng = np.random.default_rng(CALIB_HOLDOUT_SEED)
    hold = np.sort(rng.choice(len(train), size=min(CALIB_HOLDOUT_SUBJECTS, len(train) // 2), replace=False))
    n_tf = arr_tr.get("n_tf", len(arr_tr["z"]))
    i_tf = np.flatnonzero(np.isin(arr_tr["subj"][:n_tf], hold))
    x1, z1 = gather_np(cfg, arr_tr, rng.choice(i_tf, size=min(CALIB_HOLDOUT_WINDOWS, len(i_tf)), replace=False))
    with tf.device("/CPU:0"):  # the per-beat fill is faster on CPU (see main)
        cl = U.closed_loop_dataset(U.RunPredictor(run_dir), [train[i] for i in hold], seed=CALIB_CL_SEED)
    arr_cl = compact_arrays(cfg, cl, [g["y"] for g in cl])
    x2, z2 = gather_np(cfg, arr_cl, rng.choice(len(arr_cl["z"]), size=min(CALIB_HOLDOUT_WINDOWS, len(arr_cl["z"])),
                                               replace=False))
    return hold, {k: np.concatenate([x1[k], x2[k]]) for k in x1}, np.concatenate([z1, z2]), len(z1)


def calib_draw(arr_tr, pool_tf, pool_cl, size, seed):
    """`size` calibration windows from the non-held-out subjects; merged DAD arrays keep the
    training closed-loop share."""
    rng = np.random.default_rng([CALIB_KEY, size, seed])
    k_cl = int(round(size * arr_tr["thr"] / MIX_PERIOD)) if len(pool_cl) else 0
    parts = [rng.choice(pool_tf, size=min(size - k_cl, len(pool_tf)), replace=False)]
    if k_cl:
        parts.append(rng.choice(pool_cl, size=min(k_cl, len(pool_cl)), replace=False))
    return np.concatenate(parts)


def int8_export_and_check(model, cfg, arr_tr, checks, run_dir, pred, train):
    """Best-of-K int8 export (see the ponytail note above; arr_tr needs "subj" from window_subjects),
    then compare int8 vs float, in ms, on the first 20k windows of each validation set in `checks`
    ({config key: (x, z)}): reporting only, validation never takes part in the choice."""
    import shutil
    n_tf = arr_tr.get("n_tf", len(arr_tr["z"]))
    ys = cfg["norm"]["y_std"]

    def rmse(a, b):
        return float(ys * np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))

    hold, xh, zh, nh = calib_holdout(cfg, train, arr_tr, run_dir)
    fh = pred.predict_z(model_order(cfg, xh))

    def score(d):  # d holds config.json, model.keras and model_int8.tflite
        q = U.TFLitePredictor(d)
        zi = q.predict_z(model_order(cfg, xh))
        s, zp = q.out["quantization"]
        gap = {k: 100 * (rmse(zi[sl], zh[sl]) / rmse(fh[sl], zh[sl]) - 1)
               for k, sl in (("gap", slice(None)), ("gap_tf", slice(0, nh)), ("gap_cl", slice(nh, None)))}
        return dict(**gap, out_scale=float(s), out_zp=int(zp),
                    out_range_z=[float(s * (-128 - zp)), float(s * (127 - zp))])

    pool = ~np.isin(arr_tr["subj"], hold)
    pool_tf, pool_cl = np.flatnonzero(pool[:n_tf]), n_tf + np.flatnonzero(pool[n_tf:])
    tmp = run_dir / "_calib_tmp"
    shutil.rmtree(tmp, ignore_errors=True)

    def candidate(name, rep):
        d = tmp / name
        d.mkdir(parents=True)
        for f in ("config.json", "model.keras"):
            os.symlink((run_dir / f).resolve(), d / f)
        export_int8(model, model_order(cfg, gather_np(cfg, arr_tr, rep)[0]), d / "model_int8.tflite")
        return d, dict(name=name, n=len(rep), n_closed_loop=int(np.sum(rep >= n_tf)), **score(d))

    try:
        previous = score(run_dir) if (run_dir / "model_int8.tflite").exists() else None
        _, ref = candidate("ref_20k_minmax", calib_draw(arr_tr, pool_tf, pool_cl, CALIB_REF, 0))
        cands = [candidate(f"n{n}_s{sd}", calib_draw(arr_tr, pool_tf, pool_cl, n, sd))
                 for n in CALIB_SIZES for sd in CALIB_SEEDS]
        best_dir, best = min(cands, key=lambda dc: dc[1]["gap"])
        # the one check: best-of-K must not be worse than plain 20k min/max on the held-out windows
        assert best["gap"] <= ref["gap"] + 1e-9, \
            f"best-of-K held-out gap {best['gap']:.3f}% > 20k min/max {ref['gap']:.3f}%"
        os.replace(best_dir / "model_int8.tflite", run_dir / "model_int8.tflite")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    cfg["int8_calib"] = dict(
        method="best_of_K_minmax", chosen=best, ref_20k_minmax=ref, previous_file=previous,
        candidates=[c for _, c in cands], holdout_subjects=[train[i]["code"] for i in hold],
        n_holdout_tf=nh, n_holdout_cl=len(zh) - nh, holdout_cl_seed=CALIB_CL_SEED)

    q = U.TFLitePredictor(run_dir)
    scale, zp = q.out["quantization"]
    for key, (xv, zv) in checks.items():
        k = min(20000, len(zv))  # error in ms = error in z * y_std (the residual base cancels)
        xs = {kk: v[:k] for kk, v in xv.items()}
        z_int8 = q.predict_z(model_order(cfg, xs))
        z_flt = pred.predict_z(model_order(cfg, xs))
        cfg[key] = dict(
            n=k, rmse_ms_float=rmse(z_flt, zv[:k]), rmse_ms_int8=rmse(z_int8, zv[:k]),
            calib_windows=best["n"], calib_closed_loop=best["n_closed_loop"], calib_draw=best["name"],
            out_scale=float(scale), out_zp=int(zp),
            out_range_z=[float(scale * (-128 - zp)), float(scale * (127 - zp))])


def fit_gbdt(Xtr, ztr, Xva, zva, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=500, max_depth=6, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0, early_stopping=True, n_iter_no_change=20,
        random_state=seed)
    try:
        model.fit(Xtr, ztr, X_val=Xva, y_val=zva)  # early stopping on the subject-disjoint val split
    except TypeError:
        warnings.warn("scikit-learn without X_val support: early stopping uses a random 10% of "
                      "training rows (same-subject windows -> optimistic stopping). Upgrade sklearn.")
        model.fit(Xtr, ztr)
    return model


def fit_linear(Xtr, ztr, ridge=1e-3, chunk=1_000_000):
    """Ridge via float64 normal equations accumulated in chunks (no 20M x p float64 copy)."""
    p = Xtr.shape[1] + 1
    AtA = np.zeros((p, p))
    Atz = np.zeros(p)
    for a in range(0, len(Xtr), chunk):
        A = np.c_[Xtr[a:a + chunk].astype(np.float64), np.ones(len(Xtr[a:a + chunk]))]
        AtA += A.T @ A
        Atz += A.T @ ztr[a:a + chunk].astype(np.float64)
    reg = ridge * np.eye(p)
    reg[-1, -1] = 0.0  # do not shrink the intercept
    w = np.linalg.solve(AtA + reg, Atz)
    return w[:-1], w[-1]


def feature_report(model, features, Xva, zva, train_subjects, run_dir, seed, n_max=50000):
    """Permutation importance (val) + Spearman correlation of all candidate features (train).
    Correlated features share importance, so read both tables together before dropping any."""
    import pandas as pd
    from sklearn.inspection import permutation_importance
    rng = np.random.default_rng(seed)
    iv = rng.choice(len(Xva), size=min(n_max, len(Xva)), replace=False)
    pi = permutation_importance(model, Xva[iv], zva[iv], n_repeats=5, random_state=seed,
                                scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame(dict(feature=features, importance=pi.importances_mean,
                            importance_std=pi.importances_std)).sort_values("importance", ascending=False)
    imp.to_csv(run_dir / "feature_importance.csv", index=False)
    frac = min(1.0, n_max / sum(len(s["F"]) for s in train_subjects))
    F_sample = np.concatenate([s["F"][rng.random(len(s["F"])) < frac] for s in train_subjects])
    corr = pd.DataFrame(F_sample, columns=U.FEATURE_NAMES).corr(method="spearman")
    corr.to_csv(run_dir / "feature_correlation.csv")
    print("\nPermutation importance (val, RMSE increase in z units):")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(U.FEATURE_NAMES)
             for b in U.FEATURE_NAMES[i + 1:] if abs(corr.loc[a, b]) > 0.9]
    if pairs:
        print("Highly correlated candidate pairs (|Spearman| > 0.9):")
        for a, b, r in pairs:
            print(f"  {a:>10s} ~ {b:<10s} {r:+.3f}")


# -----------------------------------------------------------------------------
# One run
# -----------------------------------------------------------------------------
def train_one(name, args, train, val, y_fit, data_info, dad=None):
    spec = U.MODELS[name]
    fspec = args.features or spec["feats"]
    if isinstance(fspec, list) and len(fspec) == 1 and fspec[0] in U.FEATURE_SETS:
        fspec = fspec[0]
    feats = U.resolve_features(fspec)
    residual = spec["residual"] if args.residual == "auto" else args.residual == "on"
    parent = None
    if args.init_from or dad:
        assert spec["framework"] == "keras", "--init_from / --dad_filler apply to Keras models only"
    if args.init_from:  # fine-tune: the parent's inputs, residual flag and norm, unchanged
        parent = json.loads((Path(args.init_from) / "config.json").read_text())
        assert parent["model"] == name, f"--init_from {args.init_from} is a {parent['model']}, not {name}"
        assert parent["engine"] == U.ENGINE, f"{args.init_from}: trained with another ENGINE"
        assert all(U.RSA_TRACKER.get(k) == v for k, v in parent.get("rsa_tracker", {}).items()), \
            f"{args.init_from}: other RSA_TRACKER"
        assert args.residual == "auto" or residual == parent["residual"], "--residual differs from the parent's"
        residual = parent["residual"]
        expand = bool(args.features) and feats != parent["features"]
        if expand:  # warm start with extra inputs: the parent's features must all be kept
            missing = sorted(set(parent["features"]) - set(feats))
            assert not missing, f"--init_from: the new features must contain the parent's; missing {missing}"
        else:
            feats = parent["features"]
    tag = args.tag or (name + ("_kd" if args.teacher else ""))
    run_dir = Path(args.out) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    import sklearn
    cfg = dict(tag=tag, model=name, kind=spec["kind"], framework=spec["framework"], features=feats,
               residual=residual, engine=U.ENGINE, rsa_tracker=U.RSA_TRACKER,
               rls_warmup=U.RLS_WARMUP, window=U.WINDOW,
               seed=args.seed, args=vars(args), data=data_info,
               versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    if parent:
        cfg["init_from"] = dict(run=str(args.init_from), model_md5=U.file_md5(Path(args.init_from) / "model.keras"))
        if not expand:
            cfg["norm"] = parent["norm"]
        else:  # the parent's stats for its inputs and target; train-fitted stats for the new features only
            nm = U.fit_norm(train, feats, residual)
            pn = parent["norm"]
            for k in ("rr_mean", "rr_std", "drr_mean", "drr_std", "y_mean", "y_std"):
                nm[k] = pn[k]
            for i, f in enumerate(parent["features"]):
                j = feats.index(f)
                nm["feat_mean"][j], nm["feat_std"][j] = pn["feat_mean"][i], pn["feat_std"][i]
            cfg["norm"] = nm
            cfg["init_from"].update(mode="expand", new_features=[f for f in feats if f not in parent["features"]])
    else:
        cfg["norm"] = U.fit_norm(train, feats, residual)
    arr_tr = compact_arrays(cfg, train, y_fit)
    if dad:
        cfg["dad"] = dict(dad["info"])
        arr_tr = merge_arrays(arr_tr, compact_arrays(cfg, dad["tr"], [g["y"] for g in dad["tr"]]), args.dad_share)
    arr_tr["subj"] = window_subjects(train, dad["tr"] if dad else None)  # int8 calibration hold-out
    arr_va = compact_arrays(cfg, val, [s["y"] for s in val])
    rng = np.random.default_rng(args.seed)
    if not dad:
        iv = rng.choice(len(arr_va["z"]), size=min(args.val_samples, len(arr_va["z"])), replace=False)
        xv, zv = gather_np(cfg, arr_va, iv)  # fixed monitoring subset, built with the inference path
        int8_sets = {"val_int8_check": (xv, zv)}
    else:  # monitor (and early-stop on) a shuffled 50/50 mix of teacher-forced and closed-loop val windows
        arr_vc = compact_arrays(cfg, dad["va"], [g["y"] for g in dad["va"]])
        h = args.val_samples // 2
        iv = rng.choice(len(arr_va["z"]), size=min(h, len(arr_va["z"])), replace=False)
        ic = rng.choice(len(arr_vc["z"]), size=min(h, len(arr_vc["z"])), replace=False)
        (xt, zt), (xc, zc) = gather_np(cfg, arr_va, iv), gather_np(cfg, arr_vc, ic)
        perm = rng.permutation(len(zt) + len(zc))
        xv = {k: np.concatenate([xt[k], xc[k]])[perm] for k in xt}
        zv = np.concatenate([zt, zc])[perm]
        int8_sets = {"val_int8_check": (xt, zt), "val_int8_check_dad": (xc, zc)}
        del arr_vc

    print(f"\n=== {tag}: {name}, {len(feats)} features, residual={residual} ===")
    t0 = time.time()
    if spec["framework"] == "keras":
        import tensorflow as tf
        cfg["versions"]["tensorflow"] = tf.__version__
        init_model = warm_start_expand(name, cfg, Path(args.init_from), parent, val) if parent and expand else None
        model, cfg["train_info"] = fit_keras(name, cfg, arr_tr, (xv, zv), args, run_dir, init_model)
        cfg["complexity"] = U.keras_complexity(model)
    elif spec["framework"] == "sklearn":
        import joblib
        model = fit_gbdt(arr_tr["X"], arr_tr["z"], xv["feats"], zv, args.seed)
        joblib.dump(model, run_dir / "model.joblib")
        cfg["complexity"] = dict(trees=int(model.n_iter_), max_depth=6,
                                 compares_per_pred=int(model.n_iter_) * 6)
        feature_report(model, feats, xv["feats"], zv, train, run_dir, args.seed)
    else:
        coef, b = fit_linear(arr_tr["X"], arr_tr["z"])
        cfg["linear"] = dict(coef=coef.tolist(), intercept=float(b))
        cfg["complexity"] = dict(params=len(coef) + 1, macs_window=len(coef), macs_stream=len(coef))
    cfg["train_seconds"] = round(time.time() - t0, 1)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # round trip: reload the saved artifact and score it in ms on ALL validation windows
    pred = U.RunPredictor(run_dir)
    y_va = np.concatenate([s["y"] for s in val])
    cfg["val_metrics"] = U.point_metrics(y_va, U.predict_subjects(pred, val))
    cfg["val_metrics_rls_ar"] = U.point_metrics(y_va, U.predict_subjects(U.RLSAR(), val))
    pp = U.RunPredictor(args.init_from) if parent else None
    if pp:
        cfg["val_metrics_parent"] = U.point_metrics(y_va, U.predict_subjects(pp, val))
    if dad:  # all closed-loop val windows (filled by the filler, masks seeded dad_seed + 1000)
        y_vc = np.concatenate([g["y"] for g in dad["va"]])
        cfg["val_metrics_dad"] = U.point_metrics(y_vc, U.predict_subjects(pred, dad["va"]))
        cfg["val_metrics_dad_rls_ar"] = U.point_metrics(y_vc, U.predict_subjects(U.RLSAR(), dad["va"]))
        if pp:
            cfg["val_metrics_dad_parent"] = U.point_metrics(y_vc, U.predict_subjects(pp, dad["va"]))
        cfg["dad"]["share_realised_100_batches"] = cfg["train_info"].get("dad_share_realised_100_batches")

    if spec["framework"] == "keras" and not args.no_int8:
        try:
            int8_export_and_check(model, cfg, arr_tr, int8_sets, run_dir, pred, train)
        except Exception as exc:  # a failed export must not lose the trained model
            warnings.warn(f"int8 export failed for {tag}: {exc!r}")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    vm, rm = cfg["val_metrics"], cfg["val_metrics_rls_ar"]
    msg = (f"{tag}: val RMSE {vm['rmse']:.2f} ms (RLS {rm['rmse']:.2f}), skill {vm['skill']:.3f}, "
           f"sign acc {vm['sign_acc']:.3f}")
    if "val_int8_check" in cfg:
        c = cfg["val_int8_check"]
        msg += f" | int8 {c['rmse_ms_int8']:.2f} vs float {c['rmse_ms_float']:.2f} ms (subset)"
    if "val_int8_check_dad" in cfg:
        c = cfg["val_int8_check_dad"]
        msg += f" | closed-loop int8 {c['rmse_ms_int8']:.2f} vs float {c['rmse_ms_float']:.2f} ms"
    if "val_metrics_parent" in cfg:
        msg += f"\n  parent {args.init_from}: val RMSE {cfg['val_metrics_parent']['rmse']:.2f} ms"
    if "val_metrics_dad" in cfg:
        msg += (f"\n  closed-loop val ({cfg['val_metrics_dad']['n']} windows): run {cfg['val_metrics_dad']['rmse']:.2f}"
                f" | parent {cfg.get('val_metrics_dad_parent', {}).get('rmse', float('nan')):.2f}"
                f" | rls_ar {cfg['val_metrics_dad_rls_ar']['rmse']:.2f} ms")
    if "train_info" in cfg:
        msg += f" | {cfg['train_info']['ms_per_step_median']:.1f} ms/step"
    print(msg)
    del arr_tr, arr_va
    gc.collect()


def reexport_int8(run_dirs, data_dir):
    """Re-export int8 for existing Keras runs from model.keras, without retraining: rebuilds the
    training arrays (DAD runs: the closed-loop fill from cfg["dad"], on CPU) and the same seeded
    validation subsets as train_one, then reruns the best-of-K export and the int8 checks."""
    import tensorflow as tf
    from tensorflow import keras
    runs = [Path(r) for r in run_dirs]
    cfgs = [json.loads((r / "config.json").read_text()) for r in runs]
    assert all(c["framework"] == "keras" for c in cfgs), "int8 export applies to Keras runs only"
    max_beats = {c["data"]["max_beats"] for c in cfgs}
    assert len(max_beats) == 1, "runs trained on different --max_beats: re-export them separately"
    max_beats = max_beats.pop()
    train = U.load_split("train", data_dir, max_beats=max_beats)
    val = U.load_split("val", data_dir, max_beats=max_beats)
    U.check_disjoint(train=train, val=val)
    for run_dir, cfg in zip(runs, cfgs):
        pred = U.RunPredictor(run_dir)  # refuses runs whose ENGINE / RSA tracker no longer match
        arr_tr = compact_arrays(cfg, train, [s["y"] for s in train])  # rep uses inputs only
        arr_va = compact_arrays(cfg, val, [s["y"] for s in val])
        rng = np.random.default_rng(cfg["seed"])  # same draw order as train_one
        vs = cfg["args"].get("val_samples", 200_000)
        dad = cfg.get("dad")
        if not dad:
            iv = rng.choice(len(arr_va["z"]), size=min(vs, len(arr_va["z"])), replace=False)
            checks = {"val_int8_check": gather_np(cfg, arr_va, iv)}
            arr_tr["subj"] = window_subjects(train)
        else:
            fdir = Path(dad["filler"])
            assert U.file_md5(fdir / "model.keras") == dad["filler_md5"], f"{fdir} changed since {run_dir} was trained"
            print(f"{cfg['tag']}: rebuilding the closed-loop fill with {fdir} (CPU) ...", flush=True)
            with tf.device("/CPU:0"):
                filler = U.RunPredictor(fdir)
                dad_tr = U.closed_loop_dataset(filler, train, seed=dad["seed"])
                dad_va = U.closed_loop_dataset(filler, val, seed=dad["seed"] + 1000)
            arr_tr = merge_arrays(arr_tr, compact_arrays(cfg, dad_tr, [g["y"] for g in dad_tr]), dad["share"])
            arr_tr["subj"] = window_subjects(train, dad_tr)
            arr_vc = compact_arrays(cfg, dad_va, [g["y"] for g in dad_va])
            h = vs // 2
            iv = rng.choice(len(arr_va["z"]), size=min(h, len(arr_va["z"])), replace=False)
            ic = rng.choice(len(arr_vc["z"]), size=min(h, len(arr_vc["z"])), replace=False)
            checks = {"val_int8_check": gather_np(cfg, arr_va, iv), "val_int8_check_dad": gather_np(cfg, arr_vc, ic)}
            del dad_tr, dad_va, arr_vc
        model = keras.models.load_model(run_dir / "model.keras", compile=False)
        old = {k: cfg.get(k) for k in ("val_int8_check", "val_int8_check_dad", "int8_calib") if k in cfg}
        int8_export_and_check(model, cfg, arr_tr, checks, run_dir, pred, train)
        cfg["int8_reexport"] = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), previous=old)
        tmp = run_dir / "config.json.tmp"
        tmp.write_text(json.dumps(cfg, indent=2))
        os.replace(tmp, run_dir / "config.json")
        ic8 = cfg["int8_calib"]
        b, r, pv = ic8["chosen"], ic8["ref_20k_minmax"], ic8["previous_file"]
        line = (f"{cfg['tag']}: held-out train gap {b['gap']:+.2f}% (tf {b['gap_tf']:+.2f}, cl {b['gap_cl']:+.2f}) "
                f"with {b['name']}, range {b['out_range_z'][0]:+.2f}..{b['out_range_z'][1]:+.2f} z | "
                f"20k min/max {r['gap']:+.2f}%" + (f" | previous file {pv['gap']:+.2f}%" if pv else ""))
        for k in checks:
            c = cfg[k]
            line += f" | {k}: {100 * (c['rmse_ms_int8'] / c['rmse_ms_float'] - 1):+.2f}%"
        print(line, flush=True)
        del arr_tr, arr_va
        gc.collect()


def main():
    args = parse_args()
    if args.reexport_int8:
        return reexport_int8(args.reexport_int8, Path(args.data_dir))
    random.seed(args.seed)
    np.random.seed(args.seed)
    if any(U.MODELS[m]["framework"] == "keras" for m in args.models) or args.teacher:
        from tensorflow import keras
        keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    print("Loading train/val subjects and running the streaming feature engine ...")
    train = U.load_split("train", data_dir, max_beats=args.max_beats)
    val = U.load_split("val", data_dir, max_beats=args.max_beats)
    U.check_disjoint(train=train, val=val)
    n_tr, n_va = sum(len(s["y"]) for s in train), sum(len(s["y"]) for s in val)
    print(f"train: {len(train)} subjects / {n_tr} windows | val: {len(val)} subjects / {n_va} windows")

    y_fit = [s["y"] for s in train]
    data_info = dict(manifests={k: U.file_md5(data_dir / v) for k, v in U.MANIFESTS.items()
                                if k != "test" and (data_dir / v).exists()},
                     n_train_subjects=len(train), n_train_windows=int(n_tr),
                     n_val_subjects=len(val), n_val_windows=int(n_va), max_beats=args.max_beats)
    if args.teacher:
        print(f"Distillation: computing teacher predictions with {args.teacher} ...")
        # ponytail: teacher outputs are recomputed on every invocation (slow on big data).
        # Train all KD students in one call; cache to .npy if this becomes a bottleneck.
        teacher = U.LegacyPredictor(args.teacher, args.teacher_h5)
        y_teacher = [teacher.predict(U.subject_windows(s).astype(np.float32), s["F"]) for s in train]
        tm = U.point_metrics(np.concatenate(y_fit), np.concatenate(y_teacher))
        print(f"teacher on train: RMSE {tm['rmse']:.2f} ms, skill {tm['skill']:.3f}")
        y_fit = [(1.0 - args.kd_alpha) * y + args.kd_alpha * t for y, t in zip(y_fit, y_teacher)]
        data_info.update(teacher=str(args.teacher), teacher_md5=U.file_md5(args.teacher), kd_alpha=args.kd_alpha)

    dad = None
    if args.dad_filler:
        import tensorflow as tf
        filler_dir = Path(args.dad_filler)
        print(f"Closed-loop (DAD) data: filling train + val with {filler_dir} (on CPU) ...")
        t0 = time.time()
        # the fill is one tiny batched call per beat: ~5x faster on CPU than on the ROCm GPU
        # (kernel launches + MIOpen search per new shape), so the filler's variables and ops stay
        # on CPU while training itself still runs on the GPU
        with tf.device("/CPU:0"):
            filler = U.RunPredictor(filler_dir)  # float model; refuses another ENGINE / RSA_TRACKER
            dad_tr = U.closed_loop_dataset(filler, train, seed=args.dad_seed, progress=True)
            dad_va = U.closed_loop_dataset(filler, val, seed=args.dad_seed + 1000)
        art = next(filler_dir / f for f in ("model.keras", "model.joblib", "config.json") if (filler_dir / f).exists())
        info = dict(filler=str(filler_dir), filler_md5=U.file_md5(art), share=args.dad_share, seed=args.dad_seed,
                    val_seed=args.dad_seed + 1000, seg_beats=U.DAD_SEG_BEATS, drop_range=list(U.DAD_DROP_RANGE),
                    n_train_segments=len(dad_tr), n_val_segments=len(dad_va),
                    n_train_windows=int(sum(len(g["y"]) for g in dad_tr)),
                    n_val_windows=int(sum(len(g["y"]) for g in dad_va)), fill_seconds=round(time.time() - t0, 1))
        print(f"closed-loop windows: train {info['n_train_windows']} ({len(dad_tr)} segments), "
              f"val {info['n_val_windows']}, {info['fill_seconds']} s")
        dad = dict(tr=dad_tr, va=dad_va, info=info)
        del filler

    for name in args.models:
        train_one(name, args, train, val, y_fit, data_info, dad=dad)


if __name__ == "__main__":
    main()
