---
title: Oblique splits
description: How miter draws, scores and stores random sparse oblique splits.
---

A regular decision tree splits on one feature at a time: `x₃ ≥ 2.5`. When the boundary between
classes runs diagonally, an axis-aligned tree needs a staircase of many splits to follow it. An
**oblique split** tests a linear combination instead: `0.8·x₁ − 1.3·x₄ ≥ t`, which can follow such
a boundary in one step.

Searching for the best linear combination is expensive. miter does not search: at each node it
draws a few **random sparse** combinations, scores them like ordinary features, and lets them
compete with the axis and categorical splits. Boosting does the rest. Many trees, each with a few
good random directions, add up to a model that follows diagonal structure without the cost of
optimising directions.

## How a candidate is built

At every node, miter tries `m = m_mult · ceil(sqrt(p))` candidates (`m_mult = 2` by default):

1. Pick `k = 2` or `3` distinct numeric features at random, with Gaussian weights.
2. Turn the weights into integers: scale so the largest is `2¹⁴` and round. Only the integer weights
   are ever used, in training and in prediction.
3. Project every row of the node onto the direction: `P = Σ wq · c`. Here `c` are the 16-bit
   feature codes, so this is exact int32 arithmetic.
4. Split the node's own range `[Pmin, Pmax]` into 64 equal-width bins, and scan the thresholds,
   sending missing values left and right.

The node takes the best split among all candidates, axis, categorical and oblique.

## Why it is built this way

- **Local bins.** Bins on the node's own range keep thresholds available in the tails of the
  projection. In our measurements, global or quantile bins for projections lost up to 18% on some
  datasets.
- **Random directions.** Directions guided by gradients never beat random ones in our experiments,
  and they cost more. More random candidates than the default did not help either: doubling
  them gave a median of 0.00% on 32 held-out datasets.
- **Integer arithmetic.** It removes floating-point differences between training and prediction,
  between platforms and between thread counts. A row goes left in prediction exactly when it went
  left in training.

## Cost

Oblique candidates make training slower than an axis-only GBDT, roughly 1.5–9× LightGBM per tree
depending on the data. Prediction runs through [lignum](https://github.com/misterekkk/lignum) and is faster than
LightGBM's own `predict` for the same number of trees.

## Symmetric trees

With `grow_policy="symmetric"` (the default), every level of a tree uses one split for all its
nodes, as in CatBoost. Oblique candidates compete for that level split too. On 32 held-out OpenML
datasets, symmetric trees with oblique splits beat leaf-wise ones on 27, which is why they are the
default. The two tree shapes also complement each other in ensembles.
