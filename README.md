# miter

Gradient-boosted decision trees with **random sparse oblique splits**. Besides the usual axis-aligned
thresholds, every tree node also tries a few random 2–3-feature linear combinations
(`w₁·x₁ + w₂·x₂ (+ w₃·x₃) ≥ t`). These are computed in integer arithmetic, so training and prediction
agree **bit for bit** on every platform. The core is C++20 and the estimators are scikit-learn
compatible. Predictions run on the [lignum](https://github.com/misterekkk/lignum) inference engine.

- Native missing values (NaN) and native categorical features (pandas `category`, LightGBM-style K1 splits).
- Deterministic: the same `random_state` gives the same model for any number of threads.
- Regression (L2), binary classification (log-loss), and multiclass classification (native softmax).

## Installation

```bash
# not on PyPI yet; once released:
pip install miter                 # wheels: Linux x86_64/aarch64, macOS arm64, Windows x64; Python ≥ 3.10
pip install "miter[sklearn,pandas]"
```

To build from source, see [Building from source](#building-from-source).

## Quick start

```python
from miter import MiterRegressor, MiterClassifier

reg = MiterRegressor()                                # 5000 trees max, early stopping after 50 rounds
reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])  # without eval_set: no early stopping
pred = reg.predict(X_test)
reg.best_iteration_, reg.feature_importances_

clf = MiterClassifier(preset="accurate").fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = clf.predict_proba(X_test)                     # shape (n, 2)
```

- `X` can be a NumPy array (float32/float64, C or Fortran order) or a pandas DataFrame.
- `category` columns are treated as categorical automatically; you can also pass
  `categorical_feature=[...]` (indices or names) to `fit`.
- Unseen categories and NaN at prediction time are handled the same way as during training.
- When `X` is a DataFrame, feature names are checked at prediction time.

### Main parameters

| parameter | default | meaning |
|---|---|---|
| `learning_rate` | 0.05 | shrinkage |
| `num_leaves` | 31 | leaves per tree (leaf-wise growth) |
| `grow_policy` | `"leafwise"` | tree growth policy (`"leafwise"` or `"symmetric"` oblivious trees) |
| `max_depth` | -1 | max depth (`-1` = unlimited for leafwise; `1..8` with default `6` for symmetric) |
| `min_data_in_leaf` | 20 | minimum rows per leaf |
| `lambda_l2` | 1.0 | L2 regularisation of leaf values |
| `preset` | `"default"` | `"accurate"` tries 2× more oblique candidates (`m_mult=4`), about 2× the oblique cost; `"fast"` uses oblique splits only at depth 0–1 |
| `m_mult` | 2 | oblique candidates per node = `m_mult · ceil(sqrt(p))` |
| `k_proj` | 0 | features per oblique candidate (0 = random 2 or 3) |
| `quantizer` | `"linear"` | feature bin quantizer (`"linear"` Q16r, `"quantile"`, or `"auto"` per-feature skewed) |
| `max_ctr_complexity` | 0 | native C++20 online ordered CTR combinations for `symmetric` trees (`0` = off, `1..3`, or `"auto"`) |
| `early_stopping_rounds` | 50 | patience on `eval_set` (`None` = off) |
| `time_limit` | `None` | stop adding trees after this many seconds |
| `n_jobs` | -1 | threads (the result does not depend on it) |
| `random_state` | 0 | seed |
| `feature_fraction` | 1.0 | fraction of features per tree (also the pool for oblique splits) |
| `bagging_fraction` | 1.0 | fraction of rows per tree (sampled without replacement) |
| `extra_trees` | False | one random valid threshold per feature instead of the best one |

All parameters, including the categorical ones (`max_cat_to_onehot`, `min_data_per_group`,
`cat_smooth`, `cat_l2`, `max_cat_threshold`, `max_categories`), are documented in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) §5.2.

### scikit-learn

With scikit-learn installed, `MiterRegressor` and `MiterClassifier` are regular sklearn estimators.
They pass `sklearn.utils.estimator_checks` and work with `clone`, `Pipeline`, `GridSearchCV` and
`cross_val_score(n_jobs=...)`.

### Saving models

```python
import pickle
blob = pickle.dumps(reg)                 # exact: predictions after loading are bit-identical
reg.save_model("model.json")             # miter's JSON format, also exact
reg2 = MiterRegressor.load_model("model.json")
```

`load_model` restores the model and its parameters. It does not restore the class labels of a
classifier: they come back as 0/1. If you need the original labels, use pickle.

### Prediction speed

`predict` runs on lignum's binned kernel. It gives the same bits as miter's own evaluator and is
about 3× faster than it was before lignum.

The table shows one thread on an Apple M4, 300 trees, in ns per row and tree:

| | ns |
|---|---|
| LightGBM's own `Booster.predict` | 19.3 |
| miter `predict` (lignum, binned) | 15.8 |
| miter in lignum's compiled mode (C++ only for now) | 4.3 |

A LightGBM model of the same size, also run through lignum, is about 2× faster than miter. It has
only axis splits, while miter's oblique nodes cost more to evaluate.

### TabArena / AutoGluon

```python
from miter.tabarena import MiterModel, generate_configs_miter_v12   # needs autogluon.tabular
```

`MiterModel` is an AutoGluon model for regression, binary classification, and native softmax
multiclass. `generate_configs_miter` / `generate_configs_miter_v12` return random search spaces
for TabArena's "tuned" and "tuned + ensemble" settings.

## Benchmarks

The results below come from the TabArena protocol on 13 regression datasets (8-fold bagging, all
TabArena splits). Only miter was trained; the other methods' results are TabArena's own published
runs. miter was run with its **default** configuration only.

Relative RMSE of miter's default configuration against other methods, computed per dataset
(positive = miter is better):

| miter default vs | mean | median | wins | Wilcoxon p |
|---|---|---|---|---|
| LightGBM default | +2.36% | +1.87% | 13/13 | 0.0002 |
| XGBoost default | +3.10% | +2.20% | 13/13 | 0.0002 |
| CatBoost default | +0.42% | −0.03% | 5/13 | 0.74 |
| CatBoost tuned + ensemble | −1.67% | −0.63% | 2/13 | 0.008 |

So miter's default config:
- beats the LightGBM and XGBoost defaults on every dataset;
- is level with CatBoost's default;
- trails CatBoost after TabArena's tuning plus ensembling, a setting miter has not been run in yet.

Classification and tuned miter runs are not part of this table yet. See
[benchmarks/results/](benchmarks/results/) for the full protocols, including the Elo tables, the
parity checks with the research prototype and the training-speed measurements.

## Limitations

- No sparse input; convert with `X.toarray()`.
- `robust=True` is not available yet.
- Training is slower than LightGBM because of the oblique candidates. The table shows one thread,
  in ms per tree:

  | data | LightGBM | miter default | miter `fast` |
  |---|---|---|---|
  | 20k × 16 | 1.5 | 3.6 | 1.8 |
  | 20k × 100 | 8.9 | 13.5 | 7.9 |
  | 500k × 50 | 47 | 240 | — |
  | covertype 566k × 10 | 12.5 | 117 | — |

  On large datasets, use `preset="fast"` or several threads.
- Models using multiclass softmax, native CTR splits (`max_ctr_complexity`), `quantizer="quantile"`
  (or `"auto"` quantile features) with oblique splits, or `k_proj > 3` predict via miter's own exact
  evaluator instead of lignum until those extensions are added to lignum. The results are bit-identical,
  but inference is slower than lignum's binned kernel.

## Building from source

miter builds against a patched local copy of lignum in `extern/lignum/` (gitignored). The sdist
already contains it. From a git checkout:

```bash
git clone --branch v0.4.3 https://github.com/misterekkk/lignum extern/lignum
git -C extern/lignum checkout -b miter-integration
git -C extern/lignum am "$PWD"/lignum_patches/*.patch
```

macOS needs `brew install libomp` for multithreading.

Python package and tests:

```bash
pip install ".[test]"
pytest tests/python -v
```

C++ library and tests:

```bash
cmake -S . -B build -DMITER_BUILD_TESTS=ON -DMITER_WARNINGS_AS_ERRORS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Design and decisions: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md),
[docs/DECISIONS.md](docs/DECISIONS.md), [docs/LIGNUM_INTEGRATION.md](docs/LIGNUM_INTEGRATION.md).
