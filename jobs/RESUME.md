# Resuming after a pause (usage limit, disconnect, new session)

Layout (2026-10-08): code in src/ and tools/, data in data/, reports in docs/, training logs in logs/, results in
results/{evaluation,test,validation}/; README.md at the root explains it.

The queue runner works without Claude. After a pause, do this before anything else:

1. `jobs/status.sh 20`: is the runner alive? What finished (END rc=0), what failed (rc != 0), what is
   still queued? If the runner is dead: `jobs/start_runner.sh` (the queue file keeps its jobs).
2. Read `docs/EXPERIMENTS.md`: rows `queued` / `trained`, the Decisions and Open requests.
3. Continue the loop (main session coordinates; agents run at effort high):
   - Training jobs finished -> trainer fills in their rows (val RMSE, int8 gap, ms/step).
   - Evaluation job finished (`results/evaluation/<round>/`) -> evaluator applies the decision rules and
     writes the verdict and the requests for the architect.
   - Verdict written -> architect proposes the next round -> trainer submits it to the queue.
4. Post a short timestamped log for the user (they asked for logs of everything).

## User rules to remember (also in CLAUDE.md and memory)
- Train only on GPU 0, sequentially (the queue guarantees it). Evaluation on CPU.
- Validate on the real series of subjects_validation.json only, never synthetic data. The test
  split stays locked until the user approves the single final evaluation.
- Work autonomously inside the project; ask only for actions outside it.
- Engine feature changes for the RSA problem are approved (MCU budget still applies).

## State update 2026-10-08 14:05: FINAL TEST DONE, project complete
- The single final evaluation on the test split ran once (results/evaluation/final_test/, docs/EXPERIMENTS.md "FINAL"
  section, docs/SUMMARY.md top). tcn_lite_ft_int8 passes every rule. Never re-run the test split, and
  never select or tune using it. Any future work is a new project phase: ask the user first.

## (previous) State update 2026-10-08 07:40: CONVERGED, waiting for the user's final-test approval
- Final selection: tcn_lite_ft_int8 (firm; see docs/EXPERIMENTS.md round-4 verdict and docs/SUMMARY.md).
  Freeze runs/tcn_lite_ft, ENGINE and RSA_TRACKER: no re-exports and no engine edits.
- Do NOT start new rounds. On each check-in, only check whether the user has created the
  final-test approval file in .claude/ (ls .claude/). If it exists, start the evaluator for the
  single final test: selected model + its int8, rls_ar / persistence (ema_mean comes automatically),
  and TCN_MHA via --legacy_model legacy/tcn_attention_hrv_best.keras; round-4
  protocol on CPU (closed loop 10/30/50/80%, seeds 7 101 211), output results/evaluation/final_test. The queue
  runner refuses test-split jobs, so the evaluator runs it directly, once. No selection or tuning
  after it. Then update docs/SUMMARY.md with the test numbers.

## State update 2026-10-07 23:55
- Round 2 verdict: tcn_lite selected (rule 3 is now precision first, paired per-subject test).
  micro_tcn ties it in the closed loop; whether the closed loop should decide is open with the user.
- Round 3: the architect builds closed-loop-aware training (DAD: fill training series in closed
  loop with the parent model, true targets) -> trainer adds the src/train.py option -> queue
  tcn_lite_dad and micro_tcn_dad -> evaluator. gru is training in the queue (started 23:46), and
  its evaluation is queued behind it.
- docs/SUMMARY.md is the user-facing summary; keep it current after each verdict.

## State when this file was written (2026-10-07 21:45)
- Round 1 evaluated: mlp selected (rule 3, unpaired); micro_tcn better on a paired test. That
  question is open with the user.
- Round 2: the architect is adding the RLS freeze on imputed beats plus 3 candidates (mlp_nobp,
  micro_tcn_f8, mlp_rsa with adaptive RSA features). The trainer queues them, then the round-2
  evaluation (results/evaluation/round2, 3 seeds), then gru (cap about 3 h) and its evaluation
  (results/evaluation/round2_gru).
- Open questions for the user: the tie rule (paired vs CI overlap); statsmodels for TCN_MHA.
