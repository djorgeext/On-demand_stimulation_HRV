# HRV dRR predictor - project rules (loaded by every agent)

## Goal
Predict dRR[n+1] = RR[n+1] - RR[n] (ms) from the last 30 RR intervals, beat by beat, for a pacemaker.
First hardware: STM32H750VBT6 (Cortex-M7, 128 KB internal flash, 1 MB RAM). The implant will be far
weaker, so the budget below is a hard gate. Inside it, precision comes first (user, 2026-10-07): a
model that is significantly more accurate on a paired per-subject test wins even if it is larger;
size only breaks true ties. Precision is judged in the closed loop of the int8 model (user,
2026-10-08: beats dropped and filled by the model, the real pacemaker use); open loop is context.

## Files (single sources of truth; layout reorganised 2026-10-08, see README.md)
- `src/utils.py`: streaming feature engine (`ENGINE`, `FEATURE_NAMES`), data loading, z-score
  normalisation, models (`MODELS`, builders), predictors, metrics, and the project paths
  (`ROOT`, `DATA_DIR`, `RUNS_DIR`, `RESULTS_DIR`, `LEGACY_DIR`). `python src/utils.py` runs the
  self-test and must pass after any edit.
- `src/train.py`: training on train + validation only. `src/evaluate.py`: comparisons (`--split val`).
- `data/`: `series/*.txt` (RR in ms) + `subjects_train.json`, `subjects_validation.json`,
  `subjects_test_split.json` (split by subject), `ar_model_parameters.json`, legacy `hrv_*.h5`.
- `legacy/`: `utils2.py`, `TCN_MHA.py`/`.ipynb`, `tcn_attention_hrv_best.keras`. Used only through
  `LegacyPredictor` (comparison or distillation teacher). Build nothing new on it.
- `runs/<tag>/config.json`: everything about a run. `results/evaluation/<round>/`: evaluate.py outputs.
- `docs/EXPERIMENTS.md`: the shared log. Agents hand work to each other only through it.
- `docs/RESEARCH.md`: literature notes and design hypotheses (RSA fidelity, closed-loop training, MCU).
  `docs/SUMMARY.md`: user-facing summary.
- `docs/architectures/`: one PDF per architecture family + overview + feature engineering, rebuilt by
  `tools/make_architecture_pdfs.py` (acronyms in `tools/glossary.py`); Spanish copies in
  `docs/architectures/architectures-spanish/` by `tools/make_architecture_pdfs_es.py` (same facts, translated
  text; rebuild both together); the per-feature formulas of 01_feature_engineering live in `tools/feature_details.py`,
  whose NumPy re-implementation must match the engine or the build fails. `docs/training/`: how each model family was
  trained, `<nn>_<family>_{en,es}.pdf`, by `tools/make_training_pdfs.py`. `results/test/` and
  `results/validation/`: closed-loop plots (80% dropped, seed 7, whole series), filled series and
  correlations of every model on the test / validation subjects (`tools/make_test_plots.py --split
  test|val`, `tools/crosscorr.py --out results/<split>`), diagnostic only.
- `notebooks/correlation_study_v3.ipynb`: how the model is used on the pacemaker (streaming imputation
  of dropped beats, Burg-AR fill for the first 30 beats); `evaluate_closed_loop` mirrors it.

## Hard rules
1. **Test split is opened once**, for the final evaluation, after the user creates
   `.claude/FINAL_TEST_APPROVED`. A hook blocks earlier access. Never work around it.
2. **No leakage**: splits are by subject (`check_disjoint`), singleton subjects go to validation,
   convolutions are strictly causal, normalisation stats come from train only (`fit_norm`).
3. **One feature implementation**: features come only from the numba engine in `src/utils.py`; it is
   the reference for the C port. Changing `ENGINE` values invalidates every run (`RunPredictor`
   refuses them), so it needs the user's OK and a full retrain.
4. **Baselines first**: every result is reported against `rls_ar` and `persistence`. A model that
   does not beat `rls_ar` is not progress, however complex it is.
5. **MCU-native ops only**: Conv1D, SeparableConv1D, Dense, BatchNormalization (folds into the
   conv), ReLU, Add, ZeroPadding1D (causal), last-timestep readout, GRU. Not allowed: attention,
   LayerNormalization, GELU, pooling over time, bidirectional RNNs, Lambda layers.
   Receptive field <= 30 beats, so the window model equals the streaming model.
6. **Budget**: `macs_stream` <= 10k per beat (prefer <= 3k), int8 weights <= 32 KB, int8 RMSE
   within 2% of float. GBDT counts ~8 bytes per node, so it is deployable only if
   nodes x 8 B <= 32 KB (about 60 trees at depth 6); otherwise it is an accuracy reference.

## Lazy-MLE principles (adapted from ponytail.md)
Stop at the first rung that holds:
1. Does a baseline (`rls_ar`, `linear`) already do it? Then don't train anything.
2. Does `src/utils.py` already compute it? Reuse it; never add a second implementation.
3. Vectorised NumPy, a TensorFlow/scikit-learn built-in, or one `@njit`. No Python loops over
   samples (the loops inside the `@njit` engine are intentional: they mirror the C code).
4. Can the model be smaller? Halve filters, drop a block, drop features. Pruning over scaling.
5. Only then write the minimum new logic.

- Trace tensor shapes first: seq `(B, 30, 2)` [z-RR, z-dRR] + feats `(B, n_feats)` -> `(B, 1)`
  z-target (residual over the RLS forecast when `residual=True`).
- Root cause over symptom. Example: slow epochs come from per-step overhead, so raise
  `--steps_per_execution`; a bigger batch only hides it and changes the optimisation.
- `model.fit` with standard callbacks; no custom training loops.
- No new dependencies beyond numpy, scipy, pandas, h5py, numba, tensorflow, scikit-learn.
  The user allows installing a library into `.venv` when it is really necessary (pin the core
  packages with `pip install -c`, never upgrade them). Installed so far: `statsmodels` (2026-10-07),
  only so the legacy `legacy/utils2.py` imports and TCN_MHA can be evaluated; `matplotlib` (2026-10-08) for the
  results/ plots; `reportlab` + `pypdfium2` (2026-10-08) for the docs/architectures/ PDFs. Model code must not
  use them.
- Numerical stability: epsilon in denominators, zero variance -> std = 1 (`fit_norm` does it).
- Mark deliberate shortcuts with a `ponytail:` comment naming the ceiling and the upgrade path.
- Every non-trivial change leaves ONE runnable check (an assert, a self-test entry, or a smoke
  run with `--max_beats`).
- Not lazy about: leakage, numerical stability, hardware limits (int8/TFLite, Colab RAM, ROCm if
  training on AMD), and anything the user explicitly asked for.

## Workflow
Main session coordinates: architect -> trainer -> evaluator -> architect ...
- One hypothesis per experiment, one changed variable vs its parent, same seed when comparing.
- Run tags: `<model>_<change>`, e.g. `micro_tcn_f24`, `gru_b2048_lr2e-3`.
- Report metrics in ms on the validation split, always next to `rls_ar`.

## Job queue (unattended runs: survive Claude usage limits)
Every full training run and every evaluation goes through `jobs/` instead of being launched directly:
`jobs/submit.sh '<full command>'` appends it; a detached runner (`jobs/start_runner.sh`) executes
the queue one job at a time (so one GPU job at a time), logs to `jobs/logs/`, and records
START/END/rc in `jobs/status.log`. `jobs/status.sh` shows the state. The runner refuses test-split
jobs. Direct runs are only for smoke tests, on CPU (`HIP_VISIBLE_DEVICES=-1`) while the queue is
busy. Submit the pipeline in dependency order (train candidates, then the evaluation of them), then
hand back: nobody needs to stay awake for it. `jobs/RESUME.md` says how to pick up after a pause.

## Commands
`python` below means `.venv/bin/python` (TF 2.19 ROCm, sklearn 1.9.1, the env every run was trained
in). The bare `python` on PATH is /opt/venv: no TensorFlow, older sklearn. Training: GPU 0 only
(`HIP_VISIBLE_DEVICES=0`), one GPU job at a time. Evaluation: CPU (`HIP_VISIBLE_DEVICES=-1`).
All commands run from the project root.
- Self-test: `python src/utils.py` (real validation series only; no synthetic data)
- Smoke run: `python src/train.py --models <model> --max_beats 3000 --epochs 1 --epoch_samples 20000 --no_int8 --tag smoke_<model>`
- Train: `python src/train.py --models <model> [--features ...] [--batch_size B --lr LR --warmup_steps S] --tag <tag>`
- Compare: `python src/evaluate.py --runs runs/<a> runs/<b> --split val --int8 [--closed_loop] --out results/evaluation/<round>`
