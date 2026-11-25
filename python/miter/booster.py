"""Scikit-learn-style GBDT estimators for miter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from miter import _core
from miter.data import (
    Dataset,
    _factorize_categorical_values,
    _is_pandas_dataframe,
    _prepare_numeric_1d,
)


def _extract_mixed_predict_inputs(
    X: Any,
    model: _core.Model,
    *,
    fitted_on_dataframe: bool,
) -> tuple[
    int,
    list[NDArray[np.float64] | NDArray[np.float32]],
    list[list[str]],
    list[NDArray[np.int32]],
]:
    is_cat = model.is_categorical
    expected_features = model.n_features
    expected_feature_names = model.feature_names
    is_df = _is_pandas_dataframe(X)

    if is_df:
        n_rows, n_features = X.shape
        if n_features != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {n_features}")
        if fitted_on_dataframe:
            cols = [str(c) for c in X.columns]
            if cols != expected_feature_names:
                raise ValueError(
                    f"Feature names mismatch: expected {expected_feature_names}, got {cols}"
                )
        num_cols: list[NDArray[np.float64] | NDArray[np.float32]] = []
        cat_vocabs: list[list[str]] = []
        cat_indices: list[NDArray[np.int32]] = []
        for j in range(n_features):
            col = X.iloc[:, j]
            if is_cat[j]:
                vocab, codes = _factorize_categorical_values(col, is_pandas=True)
                cat_vocabs.append(vocab)
                cat_indices.append(codes)
            else:
                num_cols.append(_prepare_numeric_1d(col, is_pandas=True))
        return n_rows, num_cols, cat_vocabs, cat_indices

    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    n_rows, n_features = arr.shape
    if n_features != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {n_features}")
    num_cols = []
    cat_vocabs = []
    cat_indices = []
    for j in range(n_features):
        col = arr[:, j]
        if is_cat[j]:
            vocab, codes = _factorize_categorical_values(col, is_pandas=False)
            cat_vocabs.append(vocab)
            cat_indices.append(codes)
        else:
            num_cols.append(_prepare_numeric_1d(col, is_pandas=False))
    return n_rows, num_cols, cat_vocabs, cat_indices


def _to_numeric_matrix_view(
    X: Any,
    *,
    expected_features: int,
    fitted_on_dataframe: bool,
    expected_feature_names: list[str],
) -> NDArray[np.float64] | NDArray[np.float32]:
    if _is_pandas_dataframe(X):
        n_rows, n_features = X.shape
        if n_features != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {n_features}")
        if fitted_on_dataframe:
            cols = [str(c) for c in X.columns]
            if cols != expected_feature_names:
                raise ValueError(
                    f"Feature names mismatch: expected {expected_feature_names}, got {cols}"
                )
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        n_rows, n_features = arr.shape
        if n_features != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {n_features}")

    if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
        return arr
    if arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


class _BaseMiterEstimator:
    """Shared training, prediction, parameter, and JSON I/O logic."""

    _objective: _core.Objective

    def __init__(
        self,
        *,
        n_estimators: int = 5000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_data_in_leaf: int = 20,
        min_sum_hessian_in_leaf: float = 1e-3,
        lambda_l2: float = 1.0,
        min_gain_to_split: float = 0.0,
        max_bin: int = 255,
        max_cat_to_onehot: int = 4,
        min_data_per_group: int = 100,
        max_categories: int = 253,
        cat_smooth: float = 10.0,
        cat_l2: float = 10.0,
        max_cat_threshold: int = 32,
        m_mult: int = 2,
        k_proj: int = 0,
        preset: str = "default",
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        extra_trees: bool = False,
        robust: bool = False,
        cat_mode: str = "k1",
        early_stopping_rounds: int | None = 50,
        time_limit: float | None = None,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.max_bin = max_bin
        self.max_cat_to_onehot = max_cat_to_onehot
        self.min_data_per_group = min_data_per_group
        self.max_categories = max_categories
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.max_cat_threshold = max_cat_threshold
        self.m_mult = m_mult
        self.k_proj = k_proj
        self.preset = preset
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.extra_trees = extra_trees
        self.robust = robust
        self.cat_mode = cat_mode
        self.early_stopping_rounds = early_stopping_rounds
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_params(self) -> _core.Params:
        if self.preset not in ("default", "fast", "accurate"):
            raise ValueError(f"Unknown preset: {self.preset!r}")
        if self.cat_mode != "k1":
            raise ValueError(
                f"Unsupported cat_mode={self.cat_mode!r} in v1.0; only 'k1' is supported"
            )
        p = _core.Params()
        p.n_estimators = int(self.n_estimators)
        p.learning_rate = float(self.learning_rate)
        p.num_leaves = int(self.num_leaves)
        p.max_depth = int(self.max_depth)
        p.min_data_in_leaf = int(self.min_data_in_leaf)
        p.min_sum_hessian_in_leaf = float(self.min_sum_hessian_in_leaf)
        p.lambda_l2 = float(self.lambda_l2)
        p.min_gain_to_split = float(self.min_gain_to_split)
        p.max_bin = int(self.max_bin)
        p.max_cat_to_onehot = int(self.max_cat_to_onehot)
        p.min_data_per_group = int(self.min_data_per_group)
        p.max_categories = int(self.max_categories)
        p.cat_smooth = float(self.cat_smooth)
        p.cat_l2 = float(self.cat_l2)
        p.max_cat_threshold = int(self.max_cat_threshold)
        if self.preset == "fast":
            p.m_mult = int(self.m_mult)
            p.oblique_max_depth = 1
        elif self.preset == "accurate":
            p.m_mult = 4 if self.m_mult == 2 else int(self.m_mult)
            p.oblique_max_depth = -1
        else:
            p.m_mult = int(self.m_mult)
            p.oblique_max_depth = -1
        p.k_proj = int(self.k_proj)
        p.early_stopping_rounds = 0 if self.early_stopping_rounds is None else int(self.early_stopping_rounds)
        p.time_limit = 0.0 if self.time_limit is None else float(self.time_limit)
        p.n_jobs = int(self.n_jobs)
        p.random_state = int(self.random_state)
        return p

    def _check_fitted(self) -> _core.Model:
        model = getattr(self, "_model", None)
        if model is None:
            raise RuntimeError("This estimator instance is not fitted yet. Call 'fit' first.")
        return model

    def _encode_targets(self, y: Any, *, is_fit: bool) -> NDArray[np.float64]:
        arr = np.asarray(y)
        if arr.ndim != 1:
            arr = arr.ravel()
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _fit_impl(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Sequence[tuple[Any, Any]] | None = None,
        categorical_feature: str | Sequence[int | str] | None = "auto",
    ) -> _core.TrainResult:
        params = self._build_params()
        self._fitted_on_dataframe = _is_pandas_dataframe(X)
        ds = Dataset(
            X,
            categorical_feature=categorical_feature,
            max_bin=self.max_bin,
            min_data_per_group=self.min_data_per_group,
            max_categories=self.max_categories,
        )
        y_arr = self._encode_targets(y, is_fit=True)
        sw_arr = (
            None
            if sample_weight is None
            else np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
        )

        valid_ds_inner = None
        y_val_arr = None
        if eval_set:
            if len(eval_set) != 1:
                raise ValueError("eval_set supports a single (X_val, y_val) pair in v1.0")
            X_val, y_val = eval_set[0]
            valid_ds_inner = ds.transform(X_val)._inner
            y_val_arr = self._encode_targets(y_val, is_fit=False)

        result = _core.fit_booster(
            ds._inner,
            self._objective,
            y_arr,
            sw_arr,
            valid_ds_inner,
            y_val_arr,
            None,
            params,
        )
        self._model = result.model
        self._lignum_cache: dict[int, _core.LignumPredictor | None] = {}
        self.best_iteration_ = result.model.best_iteration
        self.n_features_in_ = result.model.n_features
        self.feature_names_in_ = np.array(result.model.feature_names, dtype=object)
        self.feature_importances_ = result.model.feature_importances_split()
        self.feature_importances_gain_ = result.model.feature_importances_gain()
        self.train_loss_ = list(result.train_loss)
        self.val_loss_ = list(result.valid_loss)
        self._train_scores_ = result.final_train_scores()
        return result

    def _lignum(self, n_iter: int) -> _core.LignumPredictor | None:
        """The lignum model for `n_iter` trees (0 = best iteration), built once per fitted model.

        None when lignum cannot hold the model (oblique splits with k_proj > 3): miter's own
        evaluator predicts it then, with the same results.
        """
        model = self._check_fitted()
        cache: dict[int, _core.LignumPredictor | None] | None = getattr(self, "_lignum_cache", None)
        if cache is None:
            cache = {}
            self._lignum_cache = cache
        if n_iter not in cache:
            try:
                cache[n_iter] = _core.LignumPredictor(model, n_iter)
            except ValueError:
                cache[n_iter] = None
        return cache[n_iter]

    def _predict_raw(self, X: Any, num_iteration: int | None = None) -> NDArray[np.float64]:
        """Raw scores through lignum (M6); bitwise equal to miter's own evaluator."""
        model = self._check_fitted()
        n_iter = 0 if num_iteration is None else int(num_iteration)
        predictor = self._lignum(n_iter)
        fitted_on_df = getattr(self, "_fitted_on_dataframe", False)
        if any(model.is_categorical):
            n_rows, num_cols, cat_vocabs, cat_indices = _extract_mixed_predict_inputs(
                X,
                model,
                fitted_on_dataframe=fitted_on_df,
            )
            if predictor is None:
                return _core.predict_raw_mixed(model, n_rows, num_cols, cat_vocabs, cat_indices, n_iter)
            return predictor.predict_raw_mixed(n_rows, num_cols, cat_vocabs, cat_indices)
        arr = _to_numeric_matrix_view(
            X,
            expected_features=model.n_features,
            fitted_on_dataframe=fitted_on_df,
            expected_feature_names=model.feature_names,
        )
        if predictor is None:
            if arr.dtype == np.float32:
                return _core.predict_raw_f32(model, cast(NDArray[np.float32], arr), n_iter)
            return _core.predict_raw_f64(model, cast(NDArray[np.float64], arr), n_iter)
        if arr.dtype == np.float32:
            return predictor.predict_raw_f32(cast(NDArray[np.float32], arr))
        return predictor.predict_raw_f64(cast(NDArray[np.float64], arr))

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lignum_cache", None)  # rebuilt on the next prediction
        return state

    def save_model(self, path: str) -> None:
        model = self._check_fitted()
        _core.save_model_json(model, str(path))

    @classmethod
    def _from_core_model(cls, model: _core.Model) -> Any:
        p = model.params
        est = cls(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            num_leaves=p.num_leaves,
            max_depth=p.max_depth,
            min_data_in_leaf=p.min_data_in_leaf,
            min_sum_hessian_in_leaf=p.min_sum_hessian_in_leaf,
            lambda_l2=p.lambda_l2,
            min_gain_to_split=p.min_gain_to_split,
            max_bin=p.max_bin,
            max_cat_to_onehot=p.max_cat_to_onehot,
            min_data_per_group=p.min_data_per_group,
            max_categories=p.max_categories,
            cat_smooth=p.cat_smooth,
            cat_l2=p.cat_l2,
            max_cat_threshold=p.max_cat_threshold,
            m_mult=p.m_mult,
            k_proj=p.k_proj,
            early_stopping_rounds=None if p.early_stopping_rounds == 0 else p.early_stopping_rounds,
            time_limit=None if p.time_limit <= 0.0 else p.time_limit,
            n_jobs=p.n_jobs,
            random_state=p.random_state,
        )
        est._model = model
        est._lignum_cache = {}
        est._fitted_on_dataframe = False
        est.best_iteration_ = model.best_iteration
        est.n_features_in_ = model.n_features
        est.feature_names_in_ = np.array(model.feature_names, dtype=object)
        est.feature_importances_ = model.feature_importances_split()
        est.feature_importances_gain_ = model.feature_importances_gain()
        return est

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "max_bin": self.max_bin,
            "max_cat_to_onehot": self.max_cat_to_onehot,
            "min_data_per_group": self.min_data_per_group,
            "max_categories": self.max_categories,
            "cat_smooth": self.cat_smooth,
            "cat_l2": self.cat_l2,
            "max_cat_threshold": self.max_cat_threshold,
            "m_mult": self.m_mult,
            "k_proj": self.k_proj,
            "preset": self.preset,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "extra_trees": self.extra_trees,
            "robust": self.robust,
            "cat_mode": self.cat_mode,
            "early_stopping_rounds": self.early_stopping_rounds,
            "time_limit": self.time_limit,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> _BaseMiterEstimator:
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self


class MiterRegressor(_BaseMiterEstimator):
    """Gradient-boosted decision tree regressor (L2 loss)."""

    _objective = _core.Objective.Regression

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Sequence[tuple[Any, Any]] | None = None,
        categorical_feature: str | Sequence[int | str] | None = "auto",
    ) -> MiterRegressor:
        self._fit_impl(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            categorical_feature=categorical_feature,
        )
        return self

    def predict(self, X: Any, num_iteration: int | None = None) -> NDArray[np.float64]:
        return self._predict_raw(X, num_iteration=num_iteration)

    @classmethod
    def load_model(cls, path: str) -> MiterRegressor:
        model = _core.load_model_json(str(path))
        if model.objective != _core.Objective.Regression:
            raise ValueError("Loaded model is not a regression model")
        est: MiterRegressor = cls._from_core_model(model)
        return est


class MiterClassifier(_BaseMiterEstimator):
    """Gradient-boosted decision tree binary classifier (log-loss)."""

    _objective = _core.Objective.Binary

    def _encode_targets(self, y: Any, *, is_fit: bool) -> NDArray[np.float64]:
        arr = np.asarray(y)
        if arr.ndim != 1:
            arr = arr.ravel()
        if is_fit:
            classes = np.unique(arr)
            if len(classes) != 2:
                raise ValueError(
                    f"MiterClassifier v1.0 supports binary classification only; got {len(classes)} classes"
                )
            self.classes_ = classes
        if np.array_equal(self.classes_, np.array([0, 1])) or np.array_equal(
            self.classes_, np.array([0.0, 1.0])
        ):
            return np.ascontiguousarray(arr, dtype=np.float64)
        mapping = {self.classes_[0]: 0.0, self.classes_[1]: 1.0}
        return np.array([mapping[v] for v in arr], dtype=np.float64)

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        eval_set: Sequence[tuple[Any, Any]] | None = None,
        categorical_feature: str | Sequence[int | str] | None = "auto",
    ) -> MiterClassifier:
        self._fit_impl(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            categorical_feature=categorical_feature,
        )
        return self

    def predict_proba(self, X: Any, num_iteration: int | None = None) -> NDArray[np.float64]:
        # Raw scores through lignum, then miter's sigmoid (bitwise equal to miter's evaluator).
        raw = self._predict_raw(X, num_iteration=num_iteration)
        return _core.proba_from_raw(np.ascontiguousarray(raw, dtype=np.float64))

    def predict(self, X: Any, num_iteration: int | None = None) -> NDArray[Any]:
        proba = self.predict_proba(X, num_iteration=num_iteration)
        classes = getattr(self, "classes_", np.array([0, 1]))
        return classes[np.argmax(proba, axis=1)]

    @classmethod
    def load_model(cls, path: str) -> MiterClassifier:
        model = _core.load_model_json(str(path))
        if model.objective != _core.Objective.Binary:
            raise ValueError("Loaded model is not a binary classification model")
        est: MiterClassifier = cls._from_core_model(model)
        est.classes_ = np.array([0, 1])
        return est
