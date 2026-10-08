# HRV dRR predictor

Beat-by-beat prediction of the change of the next RR interval, dRR[n+1] = RR[n+1] - RR[n] (ms), from
the last 30 RR intervals, to fill missing beats in real time on a **pacemaker microcontroller**
(first target: STM32H750, Arm Cortex-M7; the implant will be weaker). The model must stay inside a hard
budget: at most 10k multiply-accumulate operations (MACs) per beat, at most 32 KB of int8 weights, int8
accuracy within 2% of float, and only microcontroller-native layers.

## Result

**Selected model: `tcn_lite_ft_int8`**, a causal temporal convolutional network (24 filters, dilations
1/2/4/7, receptive field 29 beats) on the last 30 RR intervals plus 9 engineered features. It predicts a
correction on top of an adaptive RLS AR(8) forecast. Cost: **8,048 MACs per beat, 8.4 KB int8**.

RMSE in ms. Closed loop = the pacemaker use: a share of beats is dropped and filled by the model, whose
fills feed the following predictions.

| | open loop | closed loop 10% | 30% | 50% | 80% dropped |
|---|---|---|---|---|---|
| **tcn_lite_ft_int8, test (10 subjects, run once)** | **28.59** | **28.25** | **30.20** | **33.07** | **41.55** |
| rls_ar baseline, test | 31.15 | 29.94 | 32.54 | 36.09 | 54.36 |
| previous production model TCN_MHA, test | 29.11 | 28.58 | 30.63 | 33.90 | 42.53 |
| tcn_lite_ft_int8, validation (16 subjects) | 32.99 | 25.87 | 29.12 | 33.13 | 41.66 |

It never invents or displaces a respiratory (RSA) peak in the filled series on the test subjects; the
previous TCN_MHA does, and it cannot run on the microcontroller. Full story: `docs/SUMMARY.md`; one PDF
per architecture in `docs/architectures/` (start with `00_overview.pdf`).

## Layout

```
README.md, requirements.txt
CLAUDE.md               rules for the Claude Code agents that ran the experiments (+ .claude/agents/)
src/                    the pipeline
  utils.py              streaming feature engine, data loading, models, predictors, metrics, self-test
  train.py              training (train + validation splits only)
  evaluate.py           open- and closed-loop evaluation against the baselines
tools/                  report and diagnostic scripts
  make_test_plots.py    closed-loop plots / filled series for every model (--split test|val)
  crosscorr.py          np.corrcoef(original, filled) for every subject and model
  make_architecture_pdfs.py, glossary.py   the PDFs in docs/architectures/
data/
  series/*.txt          one RR series per subject (ms, one value per line)
  subjects_*.json       train / validation / test split, by subject and age interval
  ar_model_parameters.json   Burg AR(15) used to fill the first 30 beats (as on the pacemaker)
  hrv_dataset.h5, hrv_validation.h5   legacy datasets: only needed for TCN_MHA (1.2 GB)
runs/<tag>/             trained models: config.json, model.keras, model_int8.tflite, history.csv
results/
  evaluation/<round>/   evaluate.py outputs: round1-4 (validation), final_test (test split, run once)
  test/, validation/    per subject/model closed-loop plots (80% dropped, seed 7), filled series, correlations
  experiments/          side experiments (round-4 RLS-freeze traces)
docs/
  SUMMARY.md            results summary          EXPERIMENTS.md   full experiment log (every decision)
  RESEARCH.md           literature notes         architectures/   one PDF per architecture + glossary
legacy/                 the previous pipeline: utils2.py, TCN_MHA.py/.ipynb, tcn_attention_hrv_best.keras
notebooks/              correlation_study_v3.ipynb (pacemaker use case), fft_visualizations.ipynb
logs/                   training logs of the first runs
jobs/                   unattended job queue used during the experiments (see CLAUDE.md)
```

## Getting started

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # see the TensorFlow note inside
.venv/bin/python src/utils.py      # self-test on real validation series; must print "Self-test passed."
```

All commands run from the project root; every default path is anchored to it (`src/utils.py`:
`DATA_DIR`, `RUNS_DIR`, `RESULTS_DIR`, `LEGACY_DIR`).

| task | command |
|---|---|
| train a model | `.venv/bin/python src/train.py --models tcn_lite --tag my_run` (GPU; add `--max_beats 3000 --epochs 1` for a quick check) |
| compare on validation | `.venv/bin/python src/evaluate.py --runs runs/tcn_lite_ft runs/micro_tcn --split val --int8 --closed_loop --out results/evaluation/my_round` |
| closed-loop plots | `.venv/bin/python tools/make_test_plots.py --split val` (then `--export_txt` and `tools/crosscorr.py --out results/validation`) |
| rebuild the PDFs | `.venv/bin/python tools/make_architecture_pdfs.py` |

Evaluation runs on CPU (`HIP_VISIBLE_DEVICES=-1` on AMD, `CUDA_VISIBLE_DEVICES=-1` on NVIDIA): per-beat
calls are faster there than on the GPU.

## Rules the results depend on

- **Splits are by subject** (142 training, 16 validation, 10 test); normalisation statistics come from the
  training subjects only; convolutions are causal.
- **Model selection used validation only.** The test split was opened once, for the final evaluation, after
  the project owner approved it (approval file in `.claude/`, enforced by a hook; see `CLAUDE.md`). The
  test-subject plots in `results/test/` are diagnostics made afterwards and were not used to choose anything.
- **One feature implementation**: all features come from the streaming engine in `src/utils.py`, which is
  the reference for a C port.

## Notes for sharing

- Do not share `.venv/` (3.1 GB); recreate it from `requirements.txt`.
- `data/hrv_*.h5` (1.2 GB) are only needed to evaluate or retrain the legacy TCN_MHA.
- `results/test/` and `results/validation/` (about 740 MB) hold plots and filled series; the CSV summaries
  in those folders are small if you only need the numbers.
- The notebooks read from `../data` and `../legacy`, so open them from `notebooks/`.
  `correlation_study_v3.ipynb` also needs `ages_table.xlsx`, which is not in this project: put it in
  `notebooks/`.
