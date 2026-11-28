# Architecture

miter is a gradient-boosting library whose trees, besides axis-aligned thresholds, use **random
sparse oblique splits**: `w₁·x₁ + w₂·x₂ (+ w₃·x₃) ≥ t` on two or three numeric features. Everything a
split depends on is integer arithmetic on 16-bit feature codes, so training and prediction agree bit
for bit on every platform and for any number of threads.

## Pipeline

```
fit(X, y)
  │
  ├─ data        Q16r quantizer (numeric → 16-bit codes), K1 encoder (categorical → codes),
  │              axis binner (codes → ≤ 255 bins per feature)               src/data/
  │
  ├─ boosting    objective → gradients/hessians → grow one tree → update scores,
  │              early stopping on eval_set                                 src/core/booster.cpp
  │
  │    tree      leaf-wise (best-first)            src/core/tree_learner.cpp
  │              or symmetric (oblivious)           src/core/symmetric_tree.cpp
  │                level-wise split searches:       src/core/symmetric_{axis,oblique,ctr,efb}.cpp
  │      node    best of: axis (histograms)         src/core/histogram.cpp
  │                       categorical (K1)          src/core/categorical.cpp
  │                       oblique (m random candidates, 64 local bins)  src/core/oblique.cpp
  │                       CTR (symmetric trees only) src/core/ctr.cpp
  │
  └─ model       Model: quantizers, encoders, trees      include/miter/model.hpp
                   ├─ JSON (save_model / pickle)         src/model/json_io.cpp
                   ├─ own exact evaluator                src/model/evaluator.cpp
                   └─ lignum model for predict           src/model/lignum_bridge.cpp
```

The C++20 core (`src/`, public headers in `include/miter/`) has no dependencies besides OpenMP and,
for prediction, [lignum](https://github.com/misterekkk/lignum), fetched by CMake. The Python package
(`python/miter/`) wraps it with nanobind (`python/bindings.cpp`).

## Input quantization (src/data/quantizer.*)

Every numeric feature is quantized once, at the start of training, to a 16-bit code. The same codes
feed axis splits, oblique splits and CTRs, in training and in prediction.

- **Linear (Q16r, default).** On the non-NaN training values, `lo` and `hi` are the 0.01% and 99.99%
  quantiles (type 7). `u = clamp(floor((x − lo) · scale + 0.5), 0, 65534)` with
  `scale = 65534 / (hi − lo)`; NaN is 65535. `floor(v + 0.5)` is written out so the result does not
  depend on the rounding mode. Values outside `[lo, hi]` saturate, which makes the codes robust to
  outliers.
- **Quantile.** 257 quantile knots, piecewise linear between them, for heavily skewed features.
- **Auto.** Per feature: Linear, unless some 1/16 quantile slice of the data gets fewer than 32
  linear codes (strong skew), then Quantile.

Arithmetic uses the centred code `c = u − 32768`. The model stores each feature's quantizer, and
prediction applies the identical formula to raw values.

## Splits

All candidates of a node are scored with the usual second-order gain
`G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`, and the best one wins. Ties are broken in a fixed order
(axis < categorical < oblique, then by index), so the choice never depends on threading.

### Axis (src/data/axis_binner.*, src/core/histogram.*)

Up to `max_bin = 255` quantile bins per feature, built on the codes, plus a NaN bin. Histograms hold
`(G, H, count)` per bin. The smaller child's histogram is built and the larger one is obtained by
subtraction from the parent. NaN is tried on both sides and the better side is stored as the
default direction.

**Exclusive feature bundling** (`efb`, symmetric trees, on by default; Ke et al., 2017):
- **Which features are bundled.** Sparse numeric features are grouped greedily: at most 25% of a
  feature's rows may be off its most frequent bin, and its non-default rows may not overlap those of
  any other feature in the group. One-hot columns are the typical case.
- **How a bundle is histogrammed.** Each bundle is histogrammed in one pass, and every member's
  histogram is recovered from it. Its default bin is the leaf total minus its other bins.
- **What stays the same.** Split search, projections and the model are unchanged.

### Oblique (src/core/oblique.*)

At each node, `m = m_mult · ceil(sqrt(p))` candidates (`m_mult` = 4 for symmetric trees, 2 for leaf-wise):

1. Draw `k ∈ {2, 3}` distinct numeric features and Gaussian weights. Quantize the weights to int16
   with `max |wq| = 2¹⁴`. Only the integer weights are stored and used.
2. Project every row in the node: `P = Σ wq · c` in int32 (it cannot overflow for k ≤ 3). A row with
   NaN in any of the k features is missing for this candidate.
3. Build 64 equal-width bins on the node's own range `[Pmin, Pmax]` in int64 arithmetic, and scan
   them with missing rows sent left and right.
4. The threshold after bin `b` is `Pmin + ceil((b + 1) · R / 64)`, so `bin(P) ≤ b ⇔ P < t`. This is
   exhaustively tested.

Rows are partitioned by recomputing `P` with the stored weights and comparing it with the stored
threshold, exactly the prediction rule. Training and prediction therefore cannot disagree. Bins on
the node's own range keep thresholds available in the tails. In measurements they were both better
and cheaper than global or quantile bins. All projections of a node are computed in one pass over
its rows.

Two optimisations keep large data fast, both for symmetric trees and on by default:
- **`oblique_sample_rows` (65,536).** On a level with more rows, the oblique candidates are scored
  on a strided sample of that size. Only the best direction is then evaluated on all rows: its
  histogram is rebuilt and scanned exactly as in the full search, in parallel over fixed row chunks
  reduced in chunk order. The split itself, and its comparison with the other split types, use
  exact statistics.

### Categorical: K1 (src/data/cat_encoder.*, src/core/categorical.*)

LightGBM-style. Categories with at least `min_data_per_group` rows get codes by frequency, rare and
unseen ones share `OTHER`, and NaN is `MISSING`. With few categories in a node it tries one category
against the rest. Otherwise it sorts categories by `G / (H + cat_smooth)` and scans prefixes of at
most `max_cat_threshold`. The left set is stored as a bitset. Categorical features never enter
oblique projections.

### Categorical: ordered CTRs (src/core/ctr.*, symmetric trees)

CatBoost-style online target statistics, enabled with `max_ctr_complexity` (0 = off):

- **Ordered statistics.** For each permutation (`ctr_n_permutations`, cycled over trees), a row's
  statistic uses only the rows before it. This avoids target leakage.
- **Combinations.** Splits can use single categories or combinations of up to `max_ctr_complexity`
  categorical features. The combinations are built greedily from the features already used in the
  tree, and `ctr_model_size_reg` penalises larger ones.
- **Hashing.** Combinations are hashed with CityHash64, the same function CatBoost and lignum use.
- **`"auto"`.** Picks the complexity from the data:
  - with a small sample (`n < 1500` or `n < 300·classes`), CTRs are on only if some feature has
    more than 16 categories;
  - with one categorical feature, single CTRs only;
  - otherwise complexity 2, or 3 with at least 3 categorical features and `n ≥ 8000`.

## Tree growth

- **Leaf-wise (`grow_policy="leafwise"`).** Best-first up to `num_leaves`. Both children of
  a split are evaluated immediately, and a priority queue picks the next leaf by gain. Ties go to
  the lower leaf id.
- **Symmetric (`grow_policy="symmetric"`, the default of the Python estimators).** Oblivious trees: every level uses one split for all of
  its nodes, chosen by the gain summed over the level. Oblique and CTR candidates compete like axis
  ones. Depth is `max_depth` (1–8, default 6), so a tree has `2^depth` leaves addressed by a bit
  index. Histograms use pairwise subtraction across the level.

The Python estimators resolve `min_data_in_leaf`, `lambda_l2` and `m_mult` from the tree shape when
they are left as `None`: symmetric 1 / 3.0 / 4, leaf-wise 20 / 1.0 / 2. The C++ `Params` struct
keeps the leaf-wise values as its defaults.

## Boosting (src/core/booster.*, src/core/objective.*)

- **Objectives:** L2 regression, binary log-loss, and multiclass softmax (one tree per class per
  iteration).
- **Training loop:** gradients and hessians come from the current scores. Each iteration grows a
  tree and adds `learning_rate · leaf value` to the train and validation scores.
- **Stopping:** early stopping keeps `best_iteration` and truncates the model to it. `time_limit` is
  checked between trees.
- **Options** (all off by default): `feature_fraction` (per tree; also the pool for oblique
  features), `bagging_fraction` (an exact-size sample without replacement), and `extra_trees`. With
  bagging, out-of-bag rows are scored by walking the tree.

## Determinism (src/core/rng.hpp, src/util/threading.hpp)

- **Randomness:** all random draws come from a counter-based SplitMix64 keyed by
  `(seed, tree, node, candidate, purpose)`. No RNG state is shared between threads, and the node id
  is its creation index, which is itself deterministic.
- **Sums:** floating-point sums whose order could depend on threading are reduced over fixed row
  chunks, in chunk order.
- **Tests:** the same `random_state` gives an identical model with 1, 2 or 8 threads, and the test
  suites check this.

## Prediction

Every miter model is built into a lignum model in memory (`src/model/lignum_bridge.cpp`) and
predicted with lignum's binned kernel:

- **Numeric columns:** passed raw. lignum stores miter's quantizers and oblique nodes (format 0.10),
  so the codes are recomputed with the same formula.
- **Categorical columns:** passed as miter's category codes.
- **Equality:** raw scores equal miter's own evaluator (`src/model/evaluator.cpp`) bit for bit, and
  the test suites check this on every model shape.
- **Fallback:** a model lignum cannot hold is predicted by the own evaluator (`k_proj` is limited
  to 2 or 3 features, which lignum always holds).

## Model format (src/model/json_io.*)

`save_model` writes JSON with:
- the quantizers, as doubles with 17 significant digits;
- the category maps;
- the CTR tables;
- the trees, with the integer weights and thresholds of oblique nodes.

Loading restores the exact model, and pickling goes through the same JSON. The lignum model used for
prediction is rebuilt from it after loading.

## Python (python/miter/)

- **`booster.py`:** `MiterRegressor` and `MiterClassifier`. They are scikit-learn estimators when
  sklearn is installed (they pass `check_estimator`) and work without it.
  - **Input:** NumPy arrays and pandas DataFrames, where `category` columns are categorical.
  - **Validation:** input is checked with sklearn's error messages, and feature names are checked
    at prediction time.
- **`tabarena.py`:** `MiterModel` for AutoGluon / TabArena, with the search spaces used in the
  benchmarks.

## Known limitations

- No sparse input.
- No ranking or custom objectives.
- Training is slower than LightGBM because of the oblique candidates; see the
  [benchmarks](README.md#benchmarks).

## References

- G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. *LightGBM: A Highly Efficient
  Gradient Boosting Decision Tree.* NeurIPS 2017.
  - Used here: exclusive feature bundling, histogram subtraction.
  - The K1 categorical splits follow LightGBM's implementation.
- L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin. *CatBoost: unbiased boosting with
  categorical features.* NeurIPS 2018. Used here: ordered target statistics (CTRs), feature
  combinations, oblivious (symmetric) trees.
- L. Breiman, J. Friedman, R. Olshen, C. Stone. *Classification and Regression Trees.* 1984. The
  origin of oblique (linear-combination) splits; miter draws sparse random directions instead of
  optimising them.
- T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016. The second-order gain
  and L2-regularised leaf values.
