# Pre-registration: E2, oblique directions chosen on a row sample (`oblique_sample_rows`)

Written and hashed before any E2 run.

## Change
- **What:** for symmetric trees (the default), when a level has more rows than
  `oblique_sample_rows`, the oblique candidates are scored on a strided sample of that size.
- **The sample:** every `ceil(n / oblique_sample_rows)`-th row in the level's row order. It is
  deterministic and independent of the thread count.
- **Picking the split:** the sample only chooses the direction, which is the candidate with the
  best sampled gain. That direction's 64-bin histogram is then rebuilt on all rows, and threshold,
  missing-value direction and gain are found exactly as in the full search. So the comparison with
  axis, categorical and CTR splits uses exact gains.
- **Other parameters:** the candidates (features, weights) are drawn as before; everything else is
  the v1.0 default (BASE).
- **Hypothesis:** large data trains much faster (the oblique search is ~70% of training time), with
  no meaningful loss of quality.

## Settings
- **Quality is tested on the stress setting `E2` (`oblique_sample_rows=4096`).** The 32 held-out
  datasets have at most ~20k training rows, so the candidate default would rarely trigger there. A
  smaller sample chooses directions less accurately, so passing at 4096 is taken as evidence for any
  larger value.
- **Candidate default `E2_64K` (`oblique_sample_rows=65536`).** Adopted if E2 passes.

## Measurements (fixed in advance)
1. **Held-out quality, `E2` vs `BASE`.** `benchmarks/bench_exp_speedups.py`: the 32 OpenML datasets
   of `exp_scaling_defaults` part 3, 5 folds, early stopping 50, one thread. Per-dataset mean over
   folds, relative error.
2. **TabArena, `E2` vs `MITER_DEFAULT_V1`.** `benchmarks/tabarena/run_tabarena_speedup_51.py E2`:
   51 tasks, `lite`, one configuration.
3. **Speed, `BASE` vs `E2_64K`** (`bench_exp_speedups.py speed`). Seconds for 100 trees on synthetic
   500k × 50 and covertype 581k × 54, 1 and 10 threads, median of 3, quiet machine.

## Decision rule (all must hold to adopt `oblique_sample_rows=65536` as the default)
- **Held-out (A):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (E2 worse) p ≥ 0.05;
  - worst dataset ≤ +5%.
- **TabArena (B):**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (E2 worse) p ≥ 0.05.
- **Speed (C):** synthetic 500k × 50, one thread, `E2_64K` time ≤ 0.7 × BASE.

Otherwise E2 stays an option (default 0), and the result is reported either way. E2 applies to
symmetric trees only; leaf-wise trees are unchanged.
