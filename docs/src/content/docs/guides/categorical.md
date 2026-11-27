---
title: Categorical features
description: K1 splits, ordered CTRs, and how unseen categories are handled.
---

miter handles categorical features natively. There is no need to one-hot or target-encode them.

## Marking features as categorical

```python
import pandas as pd
df["city"] = df["city"].astype("category")       # picked up automatically
clf.fit(df, y)

clf.fit(X, y, categorical_feature=[0, 3])         # or by index / column name
```

Values can be strings or integers. NaN is its own category (`MISSING`). Categories seen fewer than
`min_data_per_group` times in training, and categories never seen in training, share one `OTHER`
code. At prediction time both are handled exactly as in training.

## K1 splits (both tree shapes)

LightGBM-style splits on a set of categories:

- **Few categories in a node** (at most `max_cat_to_onehot`): miter tries each category against the
  rest.
- **Otherwise:** categories are sorted by `G / (H + cat_smooth)` and miter scans prefixes of at most
  `max_cat_threshold` categories. The gain uses `lambda_l2 + cat_l2`, which damps overfitting on
  rare categories.
- **Categories not in the node:** they go to the side with more data.

## Ordered CTRs (symmetric trees)

With `grow_policy="symmetric"` and `max_ctr_complexity` > 0, miter also uses CatBoost-style online
target statistics (CTRs):

- **Ordered statistics.** Each row's statistic is computed only from rows before it in a random
  permutation, so the model never sees its own target. `ctr_n_permutations` permutations are cycled
  across trees.
- **Combinations.** Splits can use combinations of up to `max_ctr_complexity` categorical features,
  built greedily from the features already used in the tree. `ctr_model_size_reg` penalises larger
  combinations.
- **`max_ctr_complexity="auto"`.** Picks the complexity from the data:
  - with a small sample (under 1500 rows, or under 300 rows per class), CTRs are on only if some
    feature has more than 16 categories;
  - with one categorical feature, single CTRs only;
  - otherwise 2, or 3 with at least 3 categorical features and 8000+ rows.

CTRs help most on high-cardinality features (IDs, ZIP codes, products) and their interactions. On
TabArena, adding them to symmetric trees improved binary classification the most.

## Categorical features and oblique splits

Oblique projections use numeric features only. `cat_oblique_ts=True` (experimental, off by
default) lets out-of-fold target statistics of categorical features enter projections. It did not
pass its pre-registered test, so it stays opt-in.
