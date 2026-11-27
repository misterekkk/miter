---
title: Parameters
description: Every constructor parameter of MiterRegressor and MiterClassifier.
---

Both estimators take the same keyword arguments.

## Boosting

| parameter | default | meaning |
|---|---|---|
| `n_estimators` | 5000 | maximum number of boosting iterations |
| `learning_rate` | 0.05 | shrinkage of every tree |
| `early_stopping_rounds` | 50 | stop when `eval_set` has not improved for this many rounds (`None` = off) |
| `time_limit` | `None` | stop adding trees after this many seconds |
| `random_state` | 0 | seed; the same seed gives the same model for any `n_jobs` |
| `n_jobs` | -1 | threads (-1 = all); the result does not depend on it |

## Trees

| parameter | default | meaning |
|---|---|---|
| `grow_policy` | `"leafwise"` | `"leafwise"` (best-first) or `"symmetric"` (oblivious: one split per level) |
| `num_leaves` | 31 | leaves per tree, leaf-wise growth |
| `max_depth` | -1 | leaf-wise: depth limit (-1 = none); symmetric: depth 1–8 (default 6) |
| `min_data_in_leaf` | 20 | minimum rows per leaf |
| `min_sum_hessian_in_leaf` | 1e-3 | minimum hessian per leaf |
| `lambda_l2` | 1.0 | L2 regularisation of leaf values |
| `min_gain_to_split` | 0.0 | minimum gain to split |
| `max_bin` | 255 | axis bins per numeric feature |

## Oblique splits

| parameter | default | meaning |
|---|---|---|
| `preset` | `"default"` | `"accurate"` sets `m_mult=4`; `"fast"` uses oblique splits only at depth 0–1 |
| `m_mult` | 2 | oblique candidates per node = `m_mult · ceil(sqrt(p))` |
| `k_proj` | 0 | features per candidate (0 = random 2 or 3; above 3 predicts via miter's own evaluator) |

See [Oblique splits](../../concepts/oblique-splits/) for how candidates are drawn and scored.

## Numeric features

| parameter | default | meaning |
|---|---|---|
| `quantizer` | `"linear"` | 16-bit codes: `"linear"` (Q16r), `"quantile"`, or `"auto"` (quantile only for strongly skewed features) |

## Categorical features

| parameter | default | meaning |
|---|---|---|
| `max_categories` | 253 | categories kept per feature; the rest share `OTHER` |
| `min_data_per_group` | 100 | rarer categories share `OTHER` |
| `max_cat_to_onehot` | 4 | up to this many categories: one-vs-rest splits |
| `max_cat_threshold` | 32 | most categories on the left side of a split |
| `cat_smooth` | 10.0 | smoothing of the category order `G / (H + cat_smooth)` |
| `cat_l2` | 10.0 | extra L2 for categorical splits |
| `max_ctr_complexity` | 0 | symmetric trees: ordered CTR splits on up to 1–3 combined features; `"auto"` picks from the data; 0 = off |
| `ctr_n_permutations` | 1 | permutations for the ordered statistics, cycled over trees |
| `ctr_prior_weight` | 1.0 | prior weight of the CTR smoothing |
| `ctr_max_borders` | 15 | thresholds tried per CTR |
| `ctr_model_size_reg` | 0.5 | penalty on larger combinations |
| `cat_oblique_ts` | False | experimental: out-of-fold target statistics of categorical features may enter oblique projections |
| `cat_ts_smooth` | 10.0 | smoothing of those statistics |
| `cat_mode` | `"k1"` | categorical split mode (only `"k1"` for now) |

See [Categorical features](../categorical/).

## Reserved

`robust` (default False) is reserved for a robust-scaling mode; `True` raises `ValueError` for now.

## Sampling (all off by default)

| parameter | default | meaning |
|---|---|---|
| `feature_fraction` | 1.0 | fraction of features per tree (also the pool for oblique splits) |
| `bagging_fraction` | 1.0 | fraction of rows per tree, sampled without replacement |
| `extra_trees` | False | one random valid threshold per feature instead of the best one |

## A strong configuration

On TabArena and on held-out OpenML data, symmetric trees with the auto quantizer and CTRs were the
best single configuration we measured (see [Benchmarks](../../benchmarks/)):

```python
MiterClassifier(grow_policy="symmetric", max_depth=6, min_data_in_leaf=1, lambda_l2=3.0,
                quantizer="auto", max_ctr_complexity="auto", ctr_n_permutations=4)
```
