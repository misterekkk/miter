# Pre-registration: E6, do the remaining options earn their place?

Written and hashed before any E6 run. Rule from the maintainer: an option stays only if it brings
a statistically supported benefit (quality, or for a speed option, speed). An effect on one dataset
out of dozens is chance, not a reason to keep it.

## Options under test

1. **E6-T, the tuning dimensions CTR and quantizer.** Native CTR splits (`max_ctr_complexity`,
   `ctr_n_permutations`, `ctr_prior_weight`, `ctr_model_size_reg`, `ctr_max_borders`) and the
   quantizer modes (`quantizer="auto"` / `"quantile"`). Both failed their pre-registered tests as
   defaults (exp_native_ctr_v2, exp_ctr_auto_confirm, exp_quantizer_auto). What was never tested is
   whether they help **as tuning dimensions**, where the choice is made on validation data.
2. **E6-A, `preset="accurate"`:** twice the default number of oblique candidates (`m_mult` 8 for
   symmetric trees, 4 for leaf-wise). Never tested.
3. **E6-F, `preset="fast"`:** oblique splits only at the first level (`oblique_max_depth=1`).
   Never tested. It is a speed option, so it is judged on speed with a bounded quality cost.

## Arms and measurements (fixed in advance)

### E6-T: tuned TabArena, full space vs lean space
- **Benchmark:** TabArena `lite`, all 51 tasks, `run_tabarena_e6_51.py`, same protocol as
  `MITER_V12_T25` (8-fold bagging, tuned and tuned + ensemble over the configurations).
- **`FULL`:** the default configuration plus 24 random configurations from the V12 space
  (`generate_configs_miter_v12`, seed 1234). 25 configurations.
- **`LEAN`:** the default configuration plus 24 random configurations from the same space without
  the CTR dimensions (`max_ctr_complexity` fixed at 0) and without `quantizer` (fixed at
  `"linear"`), seed 1234. 25 configurations.
- Both arms run now, on the same code (branch `cleanup/dead-options`).
- **Compared:** per-task error of `LEAN` vs `FULL`, for tuned + ensemble (primary) and tuned
  (secondary).
- No held-out check: the held-out harness evaluates single configurations, not tuning.

### E6-A and E6-F: single configuration, against the current default
- **Held-out:** `bench_exp_speedups.py`, 32 OpenML datasets, 5 folds, early stopping 50, one
  thread; reference `BASE` (cached; the current default is bit-identical to it on these datasets).
- **TabArena:** `run_tabarena_speedup_51.py`, 51 tasks, one configuration; reference `DEF2`
  (cached; bit-identical to the current default).
- **Speed (E6-F, and reported for E6-A):** seconds for 100 trees on covertype and synthetic
  500k × 50, one thread, `benchmarks/ci/speed_ab.py`, 5 interleaved runs per setting.

## Decision rules

- **E6-T, remove CTR and the quantizer modes** if `LEAN` is not worse than `FULL` on TabArena
  tuned + ensemble:
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (`LEAN` worse) p ≥ 0.05.
  - Otherwise both stay. Tuned (secondary) and the per-task worst are reported either way.
- **E6-A, keep `accurate`** only if it improves on the default (the only quality-improvement
  test in E6, so no multiplicity correction):
  - held-out: median relative error < 0 and one-sided Wilcoxon (better) p < 0.05;
  - TabArena: median ≤ +0.25% and one-sided Wilcoxon (worse) p ≥ 0.05.
  - Otherwise `preset="accurate"` is removed (`m_mult` remains for tuning).
- **E6-F, keep `fast`** only if both hold:
  - time ≤ 0.67 × default on both speed datasets (one thread);
  - held-out median relative error ≤ +1.0%.
  - Otherwise `preset="fast"` is removed. If both `accurate` and `fast` are removed, `preset` goes.

Results are reported whatever the outcome.
