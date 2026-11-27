---
title: Python API
description: The public classes and functions of the miter package.
---

Generated from miter 0.1.0.dev0 by `docs/scripts/api_reference.py`.

## `miter.MiterRegressor`

Gradient-boosted decision tree regressor (L2 loss).

```python
miter.MiterRegressor(
    *,
    n_estimators: int = 5000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    grow_policy: str = 'symmetric',
    min_data_in_leaf: int | None = None,
    min_sum_hessian_in_leaf: float = 0.001,
    lambda_l2: float | None = None,
    min_gain_to_split: float = 0.0,
    max_bin: int = 255,
    max_cat_to_onehot: int = 4,
    min_data_per_group: int = 100,
    max_categories: int = 253,
    cat_smooth: float = 10.0,
    cat_l2: float = 10.0,
    max_cat_threshold: int = 32,
    m_mult: int | None = None,
    k_proj: int = 0,
    preset: str = 'default',
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    extra_trees: bool = False,
    robust: bool = False,
    cat_mode: str = 'k1',
    cat_oblique_ts: bool = False,
    cat_ts_smooth: float = 10.0,
    max_ctr_complexity: int | str = 0,
    ctr_prior_weight: float = 1.0,
    ctr_n_permutations: int = 1,
    ctr_max_borders: int = 15,
    ctr_model_size_reg: float = 0.5,
    quantizer: str = 'linear',
    early_stopping_rounds: int | None = 50,
    time_limit: float | None = None,
    n_jobs: int = -1,
    random_state: int = 0,
) -> None
```

See [Parameters](../../guides/parameters/) for what each parameter does.

### `MiterRegressor.fit`

```python
MiterRegressor.fit(
    X: Any,
    y: Any,
    *,
    sample_weight: Any = None,
    eval_set: Sequence[tuple[Any, Any]] | None = None,
    categorical_feature: str | Sequence[int | str] | None = 'auto',
) -> MiterRegressor
```

Train the model. `eval_set=[(X_val, y_val)]` enables early stopping (`early_stopping_rounds`); without it all `n_estimators` trees are built. `categorical_feature` takes indices or column names; `"auto"` uses pandas `category` columns.

### `MiterRegressor.predict`

```python
MiterRegressor.predict(X: Any, num_iteration: int | None = None) -> NDArray[np.float64]
```

Predictions (regression: values; classification: class labels). `num_iteration` uses the first trees only (default: the best iteration).

### `MiterRegressor.save_model`

```python
MiterRegressor.save_model(path: str) -> None
```

Write the model as miter JSON. Loading it gives bit-identical predictions.

### `MiterRegressor.load_model`

```python
MiterRegressor.load_model(path: str) -> MiterRegressor
```

Class method: read a model written by `save_model`. A classifier's labels come back as `0..K-1`; pickle keeps the original labels.

### `MiterRegressor.get_params`

```python
MiterRegressor.get_params(deep=True)
```

scikit-learn: the constructor parameters.

### `MiterRegressor.set_params`

```python
MiterRegressor.set_params(**params)
```

scikit-learn: set constructor parameters.

### `MiterRegressor.score`

```python
MiterRegressor.score(X, y, sample_weight=None)
```

scikit-learn: R² (regressor) or accuracy (classifier).

### Fitted attributes

| attribute | meaning |
|---|---|
| `best_iteration_` | Number of trees kept (the best validation iteration with early stopping). |
| `feature_importances_` | Split counts per input feature (oblique splits count for each feature they use). |
| `feature_importances_gain_` | Total gain per input feature. |
| `n_features_in_` | Number of features seen in `fit`. |
| `feature_names_in_` | Column names seen in `fit` (DataFrame input only). |
| `train_loss_` | Training loss after each iteration. |
| `val_loss_` | Validation loss after each iteration (with `eval_set`). |

## `miter.MiterClassifier`

Gradient-boosted decision tree classifier (binary log-loss and multiclass softmax).

```python
miter.MiterClassifier(
    *,
    n_estimators: int = 5000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    grow_policy: str = 'symmetric',
    min_data_in_leaf: int | None = None,
    min_sum_hessian_in_leaf: float = 0.001,
    lambda_l2: float | None = None,
    min_gain_to_split: float = 0.0,
    max_bin: int = 255,
    max_cat_to_onehot: int = 4,
    min_data_per_group: int = 100,
    max_categories: int = 253,
    cat_smooth: float = 10.0,
    cat_l2: float = 10.0,
    max_cat_threshold: int = 32,
    m_mult: int | None = None,
    k_proj: int = 0,
    preset: str = 'default',
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    extra_trees: bool = False,
    robust: bool = False,
    cat_mode: str = 'k1',
    cat_oblique_ts: bool = False,
    cat_ts_smooth: float = 10.0,
    max_ctr_complexity: int | str = 0,
    ctr_prior_weight: float = 1.0,
    ctr_n_permutations: int = 1,
    ctr_max_borders: int = 15,
    ctr_model_size_reg: float = 0.5,
    quantizer: str = 'linear',
    early_stopping_rounds: int | None = 50,
    time_limit: float | None = None,
    n_jobs: int = -1,
    random_state: int = 0,
) -> None
```

See [Parameters](../../guides/parameters/) for what each parameter does.

### `MiterClassifier.fit`

```python
MiterClassifier.fit(
    X: Any,
    y: Any,
    *,
    sample_weight: Any = None,
    eval_set: Sequence[tuple[Any, Any]] | None = None,
    categorical_feature: str | Sequence[int | str] | None = 'auto',
) -> MiterClassifier
```

Train the model. `eval_set=[(X_val, y_val)]` enables early stopping (`early_stopping_rounds`); without it all `n_estimators` trees are built. `categorical_feature` takes indices or column names; `"auto"` uses pandas `category` columns.

### `MiterClassifier.predict`

```python
MiterClassifier.predict(X: Any, num_iteration: int | None = None) -> NDArray[Any]
```

Predictions (regression: values; classification: class labels). `num_iteration` uses the first trees only (default: the best iteration).

### `MiterClassifier.predict_proba`

```python
MiterClassifier.predict_proba(X: Any, num_iteration: int | None = None) -> NDArray[np.float64]
```

Class probabilities, shape `(n, n_classes)`.

### `MiterClassifier.save_model`

```python
MiterClassifier.save_model(path: str) -> None
```

Write the model as miter JSON. Loading it gives bit-identical predictions.

### `MiterClassifier.load_model`

```python
MiterClassifier.load_model(path: str) -> MiterClassifier
```

Class method: read a model written by `save_model`. A classifier's labels come back as `0..K-1`; pickle keeps the original labels.

### `MiterClassifier.get_params`

```python
MiterClassifier.get_params(deep=True)
```

scikit-learn: the constructor parameters.

### `MiterClassifier.set_params`

```python
MiterClassifier.set_params(**params)
```

scikit-learn: set constructor parameters.

### `MiterClassifier.score`

```python
MiterClassifier.score(X, y, sample_weight=None)
```

scikit-learn: R² (regressor) or accuracy (classifier).

### Fitted attributes

| attribute | meaning |
|---|---|
| `best_iteration_` | Number of trees kept (the best validation iteration with early stopping). |
| `feature_importances_` | Split counts per input feature (oblique splits count for each feature they use). |
| `feature_importances_gain_` | Total gain per input feature. |
| `n_features_in_` | Number of features seen in `fit`. |
| `feature_names_in_` | Column names seen in `fit` (DataFrame input only). |
| `classes_` | Class labels (classifier only). |
| `train_loss_` | Training loss after each iteration. |
| `val_loss_` | Validation loss after each iteration (with `eval_set`). |

## Build information

| function | returns |
|---|---|
| `miter.has_openmp()` | whether the library was built with OpenMP |
| `miter.openmp_max_threads()` | the number of threads `n_jobs=-1` uses |
| `miter.lignum_linked()` | whether predictions run through lignum |
| `miter.__version__` | the package version |
