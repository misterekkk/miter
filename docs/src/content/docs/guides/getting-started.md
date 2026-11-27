---
title: Getting started
description: Install miter and train your first model.
---

## Install

```bash
pip install miter                      # numpy only
pip install "miter[sklearn,pandas]"    # with scikit-learn and pandas support
```

Wheels exist for Linux (x86_64, aarch64), macOS (arm64) and Windows (x64), Python 3.10+. To build
from source you need a C++20 compiler and CMake 3.20+; on macOS also `brew install libomp`.

## Regression

```python
from miter import MiterRegressor

reg = MiterRegressor()                                # up to 5000 trees, early stopping after 50 rounds
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # without eval_set: no early stopping
pred = reg.predict(X_test)

reg.best_iteration_          # trees kept
reg.feature_importances_     # split counts per feature
```

## Classification

```python
from miter import MiterClassifier

clf = MiterClassifier().fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = clf.predict_proba(X_test)     # (n, n_classes): binary log-loss or multiclass softmax
labels = clf.predict(X_test)          # original labels, strings included
```

Two classes train one log-loss model; more classes train a native softmax model (one tree per class
and iteration).

## Input

- `X` is a NumPy array (float32 or float64, C or Fortran order) or a pandas DataFrame.
- NaN is a missing value; every split learns where missing values go.
- pandas `category` columns are categorical automatically, or pass
  `categorical_feature=[...]` (indices or names) to `fit`. See [Categorical features](../categorical/).
- With a DataFrame, column names are stored and checked at prediction time.

## What to try next

- `grow_policy="symmetric"`: symmetric (oblivious) trees of depth 6, often better on small and
  medium data. Combined with `quantizer="auto"` and `max_ctr_complexity="auto"` it is the strongest
  single configuration in our [benchmarks](../../benchmarks/).
- `preset="accurate"`: twice as many oblique candidates per node, about 2× the oblique cost.
- `preset="fast"`: oblique splits only near the root, for large data.

All parameters are on the [Parameters](../parameters/) page.
