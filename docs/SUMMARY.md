# Best features and architecture so far

*Paths: see the README at the project root (layout reorganised 2026-10-08). Detailed per-architecture PDFs: `docs/architectures/`.*

*2026-10-07, 23:55. Validation split: 16 subjects, 1.42M windows. Closed loop = the pacemaker use
(random beats dropped and filled by the model, as in `correlation_study_v3.ipynb`), mean of 3 seeds.*

> ## FINAL TEST RESULT (2026-10-08 14:05, held-out test split, run once, 10 subjects, 942,526 windows)
>
> **`tcn_lite_ft_int8` passes every rule on the test subjects.** RMSE in ms; validation in brackets.
>
> | | open loop | closed 10% | closed 30% | closed 50% | closed 80% |
> |---|---|---|---|---|---|
> | **tcn_lite_ft_int8** | **28.59** (32.99) | **28.25** (25.87) | **30.20** (29.12) | **33.07** (33.13) | **41.55** (41.66) |
> | TCN_MHA (current production) | 29.11 | 28.58 | 30.63 | 33.90 | 42.53 |
> | rls_ar | 31.15 | 29.94 | 32.54 | 36.09 | 54.36 |
> | persistence | 36.80 | 33.77 | 36.74 | 39.37 | 46.72 |
>
> - **vs rls_ar:** better in 10/10 subjects open loop (-8.2%) and pooled closed loop; at 80% dropped
>   -9.0 ms (-23.6%, 10/10). At 10% dropped the gain (-0.56 ms, 8/10) is no longer significant.
> - **vs TCN_MHA:** better in 10/10 subjects open loop and 9/10 pooled closed loop (p = 0.01), with
>   ~250x fewer operations, and it fits the MCU.
> - **RSA:** invents and displaces **no** RSA peak at any drop rate (rls_ar invents 16/30 at 80%,
>   TCN_MHA 2/30); at 80% it damps RSA power (the information limit).
> - **int8 costs only +0.53%** (float 28.44 ms).
> - **Validation to test:** the lead over rls_ar shrinks (open loop -11.2% to -8.2%; closed 10-50% by
>   3-5 points; 80% holds at -23%). TCN_MHA, never selected on validation, shrank about as much, and
>   rls_ar is relatively stronger on these 10 subjects (persistence +18% vs +12%), so the likelier cause
>   is the different subject mix; mild optimism from selection on validation can't be ruled out.

> ## FINAL SELECTION (round 4, 2026-10-08 07:40): `tcn_lite_ft_int8`, firm
>
> | | |
> |---|---|
> | Model | causal TCN (`tcn_lite` + 1 fine-tune epoch), 30-beat window + 9 features, correction on top of RLS AR(8) |
> | MCU cost | 8,048 MACs per beat, 8.4 KB int8 weights (hard limit 10k / 32 KB) |
> | Open loop (validation) | 32.62 ms float, 32.99 ms int8 (-11.2% vs rls_ar) |
> | Closed loop int8, 10 / 30 / 50 / 80% dropped | **25.9 / 29.1 / 33.1 / 41.7 ms** (rls_ar: 28.9 / 33.1 / 37.5 / 53.7) |
> | RSA | invents no peak at 10-50% dropped (rls_ar: 9/48); at 80% it damps RSA, the information limit |
> | int8 export | deterministic best-of-K calibration (draw n2000_s2), md5 6219d6ae3450b6ec0422a11c641b1e57 |
>
> - It loses to none of the 8 other candidates in the int8 closed loop (36 paired comparisons). It beats
>   micro_tcn at 10% and 30% dropped (p <= 0.003, float agrees) and beats tcn_lite. It ties only
>   tcn_lite_dad, which is the same size.
> - Rounds 1-4 tested capacity up and down, MLP variants, dropping `bp`, GRU, closed-loop (DAD) training,
>   RSA features (plain, gated, warm-started), a tracker redesign and int8 calibration. Nothing left
>   has a realistic chance of a significant gain, so development has converged.
> - **Your current TCN_MHA** loses to it in 16/16 subjects open loop and 14/16 closed loop, invents
>   RSA peaks 3-6x more often, and cannot run on the MCU.
>
> The single final test on the held-out test subjects has since been run once (results at the top of
> this file); `runs/tcn_lite_ft`, `ENGINE` and `RSA_TRACKER` stay frozen.

*The rest of this file is the history up to round 2.*

## Best model now: `tcn_lite` (causal TCN). Runner-up and much cheaper: `micro_tcn`

| | tcn_lite | micro_tcn |
|---|---|---|
| Inputs | last 30 beats (z-RR, z-dRR) + 9 engineered features | same |
| Output | correction on top of the RLS AR(8) forecast | same |
| Cost per beat | 8,048 MACs (under the 10k hard limit) | 2,080 MACs |
| int8 weights | 8.4 KB | 2.4 KB |
| int8 loss | +1.1% | +1.3% |
| Open loop | **32.64 ms** (-12.1% vs rls_ar) | 32.93 ms (-11.3%) |
| Closed loop, 10 / 30 / 50 / 80% dropped | 25.7 / 28.9 / 33.0 / 41.6 ms | 25.8 / 29.1 / 33.1 / **41.4** ms |
| RSA peaks invented at 80% dropped | 2% of subject-runs | 4% |

**Why tcn_lite leads:** under your rule (precision first, paired per-subject test), tcn_lite beats
micro_tcn in the open loop in 15/16 subjects (p = 0.003). In the closed loop the two are an exact tie
(8/16, p = 0.46), so micro_tcn gives up nothing in actual pacemaker use for a quarter of the compute.
That trade-off is the one question left for you (see the end).

## Round 2 results (all models)

| model | open loop | vs rls_ar | closed loop at 80% | RSA invented at 80% | RSA power at 80% | MACs/beat | int8 gap |
|---|---|---|---|---|---|---|---|
| tcn_lite | 32.64 | -12.1% | 41.60 | 2% | x0.52 | 8,048 | +1.1% |
| micro_tcn | 32.93 | -11.3% | 41.38 | 4% | x0.55 | 2,080 | +1.3% |
| TCN_MHA (current production) | 33.72 | -9.2% | 43.88 | 12% | x0.74 | about 2M | - |
| mlp_rsa | 34.21 | -7.9% | 44.46 | 6% | x0.59 | 1,264 | +0.4% |
| mlp_nobp | 34.45 | -7.2% | 43.00 | 10% | x0.61 | 1,072 | +0.3% |
| mlp | 34.60 | -6.9% | 42.11 | 17% | x0.61 | 1,136 | +0.4% |
| micro_tcn_f8 | 34.73 | -6.5% | 43.59 | 17% | x0.62 | 944 | +0.6% |
| rls_ar (baseline) | 37.14 | 0 | 53.73 | **75%** | x1.32 | 152 | - |
| persistence | 41.70 | +12.3% | 46.42 | 0% | x0.55 | 0 | - |

RMSE in ms. "RSA power" = RSA-band power of the filled series relative to the original
(x1 = unchanged). "Invented" = the filled series has an RSA peak at least 2x above the trend that
the original did not have.

### What round 2 taught us

1. **Causal TCNs are the best family.** They are the most accurate and invent the fewest RSA peaks.
2. **Your current TCN_MHA is beaten by tcn_lite** in 16/16 subjects open loop and 14/16 closed loop.
   It invents RSA peaks 3-6x more often than the TCNs, which matches what you saw. It also breaks the
   MCU rules (attention, LayerNorm, about 75 KB).
3. **Adaptive RSA features (mlp_rsa):** better one step ahead (15/16 subjects vs mlp), but WORSE with
   gaps (closed loop worse in 13/16), and they did not fix the RSA damping. Not adopted as they are.
4. **The fixed RSA band-pass (`bp`) does help with gaps:** removing it (mlp_nobp) hurts the closed
   loop (p = 0.02).
5. **Halving the conv filters (micro_tcn_f8)** costs 1.8 ms: capacity matters.
6. **Every model damps RSA at high drop rates** (power roughly halved at 80%). That is the main
   open problem. The next idea is training on closed-loop-filled inputs (see `RESEARCH.md`).

## Features (gbdt_all permutation importance, strongest first)

1. `rls_pred`, the RLS AR(8) forecast. By far the strongest: "correction on top of RLS" is the
   backbone of every good model.
2. `rr_dev`, last RR minus the window mean.
3. `drr_lag0`, last dRR.
4. `guzik`, HRV asymmetry.
5. `bp`, the fixed 0.30 cycles/beat RSA band-pass (it helps in the closed loop, see above).

Next come `ccm`, `rls_err` and `rmssd`. The 8 RLS weights add nothing, and all 29 features are no
better than 19.

## Problems found and fixed today

- **RLS stability guard lost** in the 18:14 copy of `utils.py` (now `src/utils.py`): rls_ar had gone to 203 ms. Restored, and
  every trained run reproduces its stored number exactly. The self-test now catches it on real data.
- **int8 calibration too small** in the new `train.py` (now `src/train.py`; 500 windows): the output range was clipped.
  Now 20,000 windows; the round-2 int8 gaps are 0.3-0.6% (they were 2.4-3.0%).
- **Closed-loop test aligned with the notebook:** beats are dropped from beat 0, the Burg-AR fill
  covers the first 30 beats, drop rates go up to 80%, R² is reported, and RSA fidelity is measured.
- **Evaluation 10-20x faster** (compiled per-beat calls, CPU, single thread).
- **Unattended job queue** (`jobs/`): training and evaluation continue through usage-limit pauses.

## Running now / next

1. Finished just now: the int8 re-check of the 3 round-2 runs (all within 2% now).
2. In the queue: `gru` training on GPU 0 (about 2.5 h max), then its evaluation.
3. Next: the evaluator's formal round-2 verdict, then architect round 3. Likely directions:
   closed-loop-aware training for the TCNs (to fix RSA damping and the closed-loop gap), and
   possibly more TCN capacity inside the 10k MAC limit, since precision comes first.

## Decided (2026-10-08): the closed loop decides

Precision is judged on the closed loop of the int8 model (what the pacemaker runs). Result for round 2:
**tcn_lite stays selected.** Pooled over 10-80% dropped it ties micro_tcn (11/16 subjects, p = 0.27), but its
int8 version is significantly better at 10% and 30% dropped (after correcting for testing 4 rates) and
never worse. micro_tcn would win only if heavy loss (50-80%) were the only case that mattered. Both TCNs
beat every MLP variant in the closed loop. Round 3 trains closed-loop-aware versions of both.
