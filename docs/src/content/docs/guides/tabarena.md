---
title: TabArena and AutoGluon
description: miter as an AutoGluon model, with the search spaces used in the benchmarks.
---

`miter.tabarena` (needs `autogluon.tabular`) provides `MiterModel`, an AutoGluon model for
regression, binary and multiclass tasks. It is the wrapper used for the
[TabArena results](../../benchmarks/).

```python
from autogluon.tabular import TabularPredictor
from miter.tabarena import MiterModel

predictor = TabularPredictor(label="target").fit(
    train_df,
    hyperparameters={MiterModel: [{}, {"grow_policy": "symmetric", "max_depth": 6}]},
)
```

- **Inside AutoGluon.** `MiterModel` passes AutoGluon's validation fold as `eval_set`, so early
  stopping works. It uses AutoGluon's categorical columns, and its time limit becomes miter's
  `time_limit`.
- **`generate_configs_miter_v12(n, seed)`.** Returns `n` random configurations from the search space
  of our latest TabArena run. That space covers grow policy, depth / leaves, learning rate, L2,
  `m_mult`, the quantizer and CTR settings.
- **`generate_configs_miter(n, seed)`.** The earlier search space, without quantizer and CTR
  settings.
