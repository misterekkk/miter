# Pre-registration: E4, gain penalties for oblique splits

Written and hashed before any E4 run. Unlike E1–E3, E4 aims at quality, not speed. E1 showed that
binary features in projections help on some datasets (`cars`) and hurt on others (`fps_benchmark`,
`cylinder-bands`). A soft penalty could keep the first effect and remove the second.

## Arms (each on top of the current default: symmetric, depth 6, `m_mult=4`, `oblique_sample_rows=65536`, `efb=True`)
- **E4A `oblique_binary_penalty=0.25`:** an oblique candidate's gain is multiplied by
  `max(0, 1 − 0.25 · b)`, where `b` is the number of binary features (≤ 2 distinct values) in its
  projection. The penalty applies when choosing among oblique candidates and when comparing them
  with axis, categorical and CTR splits.
- **E4B `oblique_gain_margin=0.05`:** every oblique gain is divided by 1.05, so an oblique split
  must beat the other split types by 5%.
- The values are fixed here and are not tuned afterwards. Symmetric trees only.

## Measurements (fixed in advance)
1. **Held-out:** `bench_exp_speedups.py`, `E4A` and `E4B` vs `BASE`.
   - 32 OpenML datasets, 5 folds, early stopping 50, one thread.
   - Per-dataset mean over folds, relative error.
   - BASE equals the current default on these datasets: fewer than 65,536 rows, and efb is
     bit-identical.
2. **TabArena:** `run_tabarena_speedup_51.py`, `E4A` and `E4B` vs `DEF2` (current default; efb is
   bit-identical on all 51 tasks), 51 tasks, one configuration.

## Decision rule (per arm; Holm correction over the 2 arms at α = 0.05)
An arm is adopted as the new default only if all of the following hold.
- **Held-out improvement:**
  - median relative error < 0;
  - one-sided Wilcoxon (arm better) p below its Holm threshold (smaller p of the two < 0.025, the
    other < 0.05);
  - worst dataset ≤ +5%.
- **TabArena, not worse:**
  - median relative error ≤ +0.25%;
  - one-sided Wilcoxon (arm worse) p ≥ 0.05.
- **Both arms pass:** the one with the lower held-out median is adopted. Combining them would need
  a new pre-registration.

Otherwise both parameters stay options (default 0), and the result is reported either way.
