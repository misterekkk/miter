---
title: Determinism and exactness
description: Why the same seed gives the same model on every machine and thread count.
---

miter gives three guarantees, all checked by its test suites:

1. **Same seed, same model.** Training with the same data, parameters and `random_state` produces a
   bit-identical model for any `n_jobs`, on every platform.
2. **Prediction equals training.** A row's path through a tree at prediction time is exactly the
   path it took during training.
3. **Saved equals live.** A model loaded from pickle or JSON predicts bit-identically to the model
   that was saved.

## How

- **Integer splits.** Numeric features are quantized once to 16-bit codes with a fixed formula
  (`floor(v + 0.5)`, no rounding-mode dependence). Axis thresholds are codes. Oblique splits use
  integer weights and int32 projections of the codes. The quantizer is stored in the model, so
  prediction recomputes the same codes from raw values.
- **Counter-based randomness.** Every random draw comes from SplitMix64 keyed by
  `(seed, tree, node, candidate, purpose)`, not from a shared generator. Threads cannot change the
  order of draws.
- **Fixed-order sums.** Floating-point sums that threads compute (histograms, gradients, losses)
  are reduced over fixed row chunks in chunk order, never in the order threads finish.
- **Deterministic ties.** Equal gains are broken by a fixed order (axis, categorical, oblique, then
  by index).
- **One prediction rule.** Training partitions rows with the same comparison prediction uses.
  miter's own evaluator and the lignum kernel are tested to agree bit for bit on every model shape.

There is no `-ffast-math` and no reliance on FMA contraction.
