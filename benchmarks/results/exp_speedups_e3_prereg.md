# Pre-registration: E3, exclusive feature bundling (`efb=True`)

Written and hashed before any E3 run.

## Change
- **What:** for symmetric trees (the default), sparse numeric features are bundled into one
  histogram pass per bundle. A feature is sparse when at most 25% of its rows are off its most
  frequent bin and it has at most 64 bins. Features enter a bundle greedily, and only when none of
  their non-default rows are already used by the bundle (zero conflicts). A bundle holds at most
  256 bins. One-hot columns are the typical case.
- **Splits:** each member feature's histogram is recovered exactly from its bundle's: its own bins,
  and its default bin as the leaf total minus those bins. Split search, oblique projections (binary
  features stay in them) and the model format are unchanged.
- **Expected effect on models:** identical up to floating-point summation order.
- **Baseline:** the current default (`DEF2`: symmetric, depth 6, `m_mult=4`,
  `oblique_sample_rows=65536`).
- **Hypothesis:** much faster training on data with many binary or sparse columns; no change in
  quality.

## Measurements (fixed in advance)
1. **Held-out quality, `E3` vs `BASE`.** `bench_exp_speedups.py`: 32 OpenML datasets, 5 folds,
   early stopping 50, one thread. These datasets have fewer than 65,536 rows, so `BASE` equals the
   current default there. Also reported: the number of folds with bit-identical metrics.
2. **TabArena, `E3` vs `DEF2`.** `run_tabarena_speedup_51.py`: both variants run now, 51 tasks,
   one configuration each.
3. **Speed, `DEF2` vs `E3`.** Seconds for 100 trees on covertype (44 binary columns) and synthetic
   500k × 50 (no sparse columns, where E3 is a no-op); 1 and 10 threads; median of 3; quiet machine.

## Decision rule (all must hold to adopt `efb=True` as the default)
- **Held-out (A):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (worse) p ≥ 0.05;
  - worst dataset ≤ +5%.
- **TabArena (B):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (worse) p ≥ 0.05.
- **Speed (C):** covertype, one thread, E3 time ≤ 0.8 × DEF2.
- **No regression where E3 is a no-op:** synthetic 500k × 50, one thread, E3 time ≤ 1.05 × DEF2.
