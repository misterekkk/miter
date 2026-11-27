# Pre-registration: E1, binary features out of the oblique pool (`oblique_min_distinct=3`)

Written and hashed before any E1 run.

## Change
- **What:** `oblique_min_distinct=3`. Numeric features with fewer than 3 distinct training values
  (binary features, e.g. one-hot columns) no longer enter oblique projections. They stay available
  for axis splits.
- **Effect on the candidate count:** `m = m_mult · ceil(sqrt(p_eligible))` is computed over the
  remaining features, so it also drops.
- **Everything else** is the v1.0 default (`BASE`: symmetric, depth 6, `min_data_in_leaf=1`,
  `lambda_l2=3`, `m_mult=4`).
- **Hypothesis:** much faster training on data with many binary features; no loss of quality,
  because a projection term on a 0/1 feature only shifts the threshold, which axis splits cover.

## Measurements (fixed in advance)
1. **Held-out quality.** `benchmarks/bench_exp_speedups.py`, variants `BASE` and `E1`.
   - **Data:** the 32 OpenML datasets of `exp_scaling_defaults` part 3, with no TabArena overlap
     (asserted), 5 folds, early stopping 50, one thread.
   - **Metrics:** nRMSE for regression, log-loss for classification.
   - **Comparison:** per-dataset mean over folds, relative error of E1 vs BASE.
   - **Sanity check:** BASE must reproduce the `S6_M4` metrics of `exp_scaling_defaults` part 3
     exactly. A mismatch is reported and the run investigated before any conclusion.
2. **TabArena.** `benchmarks/tabarena/run_tabarena_speedup_51.py E1`: 51 tasks, `lite`, one
   configuration (no tuning), compared per task with `MITER_DEFAULT_V1`
   (`tabarena_default_v1_51_summary.md`).
3. **Speed.** `bench_exp_speedups.py speed --variants BASE E1`: seconds for 100 trees on covertype
   (581k × 54, 44 binary columns) and on synthetic 500k × 50 (no binary columns, where E1 is a
   no-op), 1 and 10 threads, median of 3. Run on a quiet machine.

## Decision rule (all must hold to adopt `oblique_min_distinct=3` as the default)
- **Held-out (A):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (E1 worse) p ≥ 0.05;
  - worst dataset ≤ +5%.
- **TabArena (B):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (E1 worse) p ≥ 0.05.
- **Speed (C):** covertype, one thread, E1 time ≤ 0.8 × BASE time.

Otherwise E1 stays an option (default 0), and the result is reported either way. Datasets with no
binary features give identical models (ties); they are kept in all statistics.
