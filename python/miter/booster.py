"""Scikit-learn-style GBDT estimators for miter."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from miter import _core

try:  # scikit-learn is optional: with it the estimators are real sklearn estimators
    from sklearn.base import BaseEstimator as _SklearnBase
    from sklearn.base import ClassifierMixin as _ClassifierMixin
    from sklearn.base import RegressorMixin as _RegressorMixin

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only without scikit-learn
    _HAS_SKLEARN = False

    class _SklearnBase:  # type: ignore[no-redef]
        pass

    class _ClassifierMixin:  # type: ignore[no-redef]
        pass

    class _RegressorMixin:  # type: ignore[no-redef]
        pass

try:
    from sklearn.exceptions import DataConversionWarning as _DataConversionWarning
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover

    class NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        """Raised when predicting with an estimator that has not been fitted."""

    class _DataConversionWarning(UserWarning):  # type: ignore[no-redef]
        pass


from miter.cat_ts import CatObliqueTSEncoder, resolve_categorical_indices_for_X
from miter.data import (
    Dataset,
    _factorize_categorical_values,
    _is_pandas_dataframe,
    _prepare_numeric_1d,
)


def _check_input_X(X: Any, *, name: str) -> tuple[int, int]:
    """sklearn-style checks shared by fit and predict; returns (n_rows, n_features).

    The input itself is not converted here (DataFrames keep their columns and categoricals).
    """
    if X is None:
        raise ValueError(f"{name} requires X to be passed, but X is None")
    module = type(X).__module__
    if module.startswith("scipy.sparse") or hasattr(X, "tocsr"):
        raise TypeError(
            f"Sparse input is not supported by {name}; convert it with X.toarray() first"
        )
    if _is_pandas_dataframe(X):
        if any(str(dt).startswith("complex") for dt in X.dtypes):
            raise ValueError("Complex data not supported")
        shape = tuple(X.shape)
    else:
        arr = np.asarray(X)
        if np.iscomplexobj(arr):
            raise ValueError("Complex data not supported")
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {arr.ndim}D array instead. Reshape your data either "
                "using array.reshape(-1, 1) if your data has a single feature or "
                "array.reshape(1, -1) if it contains a single sample."
            )
        shape = tuple(arr.shape)
    if shape[0] == 0:
        raise ValueError(f"Found array with 0 sample(s) (shape={shape}) while a minimum of 1 is required.")
    if shape[1] == 0:
        raise ValueError(f"Found array with 0 feature(s) (shape={shape}) while a minimum of 1 is required.")
    return int(shape[0]), int(shape[1])


def _check_target(y: Any, n_rows: int, *, name: str) -> NDArray[Any]:
    if y is None:
        raise ValueError(f"{name} requires y to be passed, but the target y is None")
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. Please change the shape "
            "of y to (n_samples,), for example using ravel().",
            _DataConversionWarning,
            stacklevel=4,
        )
        arr = arr.ravel()
    elif arr.ndim != 1:
        raise ValueError(f"y should be a 1d array, got an array of shape {arr.shape} instead.")
    if arr.shape[0] != n_rows:
        raise ValueError(f"X has {n_rows} rows but y has {arr.shape[0]} values")
    if np.iscomplexobj(arr):
        raise ValueError("Complex data not supported")
    return arr


def _check_sample_weight(sample_weight: Any, n_rows: int) -> NDArray[np.float64] | None:
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64)
    if sw.ndim == 0:
        sw = np.full(n_rows, float(sw))
    if sw.ndim != 1:
        raise ValueError("Sample weights must be 1D array or scalar")
    if sw.shape != (n_rows,):
        raise ValueError(f"sample_weight.shape == {sw.shape}, expected {(n_rows,)}!")
    if not np.all(np.isfinite(sw)) or np.any(sw < 0):
        raise ValueError("sample_weight must be finite and non-negative")
    if not np.any(sw > 0):
        raise ValueError("Sample weights must contain at least one non-zero weight")
    return np.ascontiguousarray(sw)


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
            raise ValueError(
                f"X has {n_features} features, but the model is expecting {expected_features} features as input."
            )
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
        raise ValueError(
                f"X has {n_features} features, but the model is expecting {expected_features} features as input."
            )
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
            raise ValueError(
                f"X has {n_features} features, but the model is expecting {expected_features} features as input."
            )
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
            raise ValueError(
                f"X has {n_features} features, but the model is expecting {expected_features} features as input."
            )

    if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
        return arr
    if arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


class _BaseMiterEstimator(_SklearnBase):
    """Shared training, prediction, parameter, and JSON I/O logic."""

    _objective: _core.Objective

    def __init__(
        self,
        *,
        n_estimators: int = 5000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        grow_policy: str = "symmetric",
        min_data_in_leaf: int | None = None,
        min_sum_hessian_in_leaf: float = 1e-3,
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
        preset: str = "default",
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        extra_trees: bool = False,
        oblique_min_distinct: int = 0,
        oblique_sample_rows: int = 65536,
        efb: bool = True,
        oblique_binary_penalty: float = 0.0,
        oblique_gain_margin: float = 0.0,
        robust: bool = False,
        cat_mode: str = "k1",
        cat_oblique_ts: bool = False,
        cat_ts_smooth: float = 10.0,
        max_ctr_complexity: int | str = 0,
        ctr_prior_weight: float = 1.0,
        ctr_n_permutations: int = 1,
        ctr_max_borders: int = 15,
        ctr_model_size_reg: float = 0.5,
        quantizer: str = "linear",
        early_stopping_rounds: int | None = 50,
        time_limit: float | None = None,
        n_jobs: int = -1,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.grow_policy = grow_policy
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
        self.oblique_min_distinct = oblique_min_distinct
        self.oblique_sample_rows = oblique_sample_rows
        self.efb = efb
        self.oblique_binary_penalty = oblique_binary_penalty
        self.oblique_gain_margin = oblique_gain_margin
        self.robust = robust
        self.cat_mode = cat_mode
        self.cat_oblique_ts = cat_oblique_ts
        self.cat_ts_smooth = cat_ts_smooth
        self.max_ctr_complexity = max_ctr_complexity
        self.ctr_prior_weight = ctr_prior_weight
        self.ctr_n_permutations = ctr_n_permutations
        self.ctr_max_borders = ctr_max_borders
        self.ctr_model_size_reg = ctr_model_size_reg
        self.quantizer = quantizer
        self.early_stopping_rounds = early_stopping_rounds
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _build_params(self) -> _core.Params:
        if self.preset not in ("default", "fast", "accurate"):
            raise ValueError(f"Unknown preset: {self.preset!r}")
        if self.cat_mode != "k1":
            raise ValueError(
                f"Unsupported cat_mode={self.cat_mode!r}; only 'k1' is supported"
            )
        if self.robust:
            raise ValueError("robust=True is not available yet")
        for name in ("feature_fraction", "bagging_fraction"):
            value = float(getattr(self, name))
            if not (0.0 < value <= 1.0):  # also rejects NaN
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)!r}")
        smooth = float(self.cat_ts_smooth)
        if not np.isfinite(smooth) or smooth <= 0.0:
            raise ValueError(f"cat_ts_smooth must be finite and > 0, got {self.cat_ts_smooth!r}")
        p = _core.Params()
        p.n_estimators = int(self.n_estimators)
        p.learning_rate = float(self.learning_rate)
        p.num_leaves = int(self.num_leaves)
        p.max_depth = int(self.max_depth)
        if self.grow_policy == "leafwise":
            p.grow_policy = _core.GrowPolicy.Leafwise
        elif self.grow_policy == "symmetric":
            p.grow_policy = _core.GrowPolicy.Symmetric
        else:
            raise ValueError(
                f"Unknown grow_policy: {self.grow_policy!r} (expected 'leafwise' or 'symmetric')"
            )
        # None = the default of the tree shape: symmetric trees (the default) use the
        # S6_M4 setting (min_data_in_leaf 1, lambda_l2 3, m_mult 4); leaf-wise trees keep 20 / 1 / 2.
        symmetric = self.grow_policy == "symmetric"
        p.min_data_in_leaf = (1 if symmetric else 20) if self.min_data_in_leaf is None else int(self.min_data_in_leaf)
        p.min_sum_hessian_in_leaf = float(self.min_sum_hessian_in_leaf)
        p.lambda_l2 = (3.0 if symmetric else 1.0) if self.lambda_l2 is None else float(self.lambda_l2)
        p.min_gain_to_split = float(self.min_gain_to_split)
        p.max_bin = int(self.max_bin)
        p.max_cat_to_onehot = int(self.max_cat_to_onehot)
        p.min_data_per_group = int(self.min_data_per_group)
        p.max_categories = int(self.max_categories)
        p.cat_smooth = float(self.cat_smooth)
        p.cat_l2 = float(self.cat_l2)
        p.max_cat_threshold = int(self.max_cat_threshold)
        base_m_mult = (4 if symmetric else 2) if self.m_mult is None else int(self.m_mult)
        if self.preset == "fast":
            p.m_mult = base_m_mult
            p.oblique_max_depth = 1
        elif self.preset == "accurate":
            # Twice the default number of oblique candidates, unless m_mult is set explicitly.
            p.m_mult = 2 * base_m_mult if self.m_mult is None else base_m_mult
            p.oblique_max_depth = -1
        else:
            p.m_mult = base_m_mult
            p.oblique_max_depth = -1
        p.k_proj = int(self.k_proj)
        p.early_stopping_rounds = 0 if self.early_stopping_rounds is None else int(self.early_stopping_rounds)
        p.time_limit = 0.0 if self.time_limit is None else float(self.time_limit)
        p.n_jobs = int(self.n_jobs)
        p.random_state = int(self.random_state)
        p.feature_fraction = float(self.feature_fraction)
        p.bagging_fraction = float(self.bagging_fraction)
        p.extra_trees = bool(self.extra_trees)
        if (
            isinstance(self.oblique_min_distinct, bool)
            or not isinstance(self.oblique_min_distinct, (int, np.integer))
            or int(self.oblique_min_distinct) < 0
        ):
            raise ValueError(f"oblique_min_distinct must be an int >= 0, got {self.oblique_min_distinct!r}")
        p.oblique_min_distinct = int(self.oblique_min_distinct)
        if (
            isinstance(self.oblique_sample_rows, bool)
            or not isinstance(self.oblique_sample_rows, (int, np.integer))
            or int(self.oblique_sample_rows) < 0
        ):
            raise ValueError(f"oblique_sample_rows must be an int >= 0, got {self.oblique_sample_rows!r}")
        p.oblique_sample_rows = int(self.oblique_sample_rows)
        if not isinstance(self.efb, (bool, np.bool_)):
            raise ValueError(f"efb must be a bool, got {self.efb!r}")
        p.efb = bool(self.efb)
        for name in ("oblique_binary_penalty", "oblique_gain_margin"):
            v = float(getattr(self, name))
            if not np.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
        p.oblique_binary_penalty = float(self.oblique_binary_penalty)
        p.oblique_gain_margin = float(self.oblique_gain_margin)
        if isinstance(self.max_ctr_complexity, str):
            if self.max_ctr_complexity != "auto":
                raise ValueError(
                    "max_ctr_complexity must be 'auto' or an int in [-1, 3], "
                    f"got {self.max_ctr_complexity!r}"
                )
            p.max_ctr_complexity = _core.CTR_COMPLEXITY_AUTO
        elif isinstance(self.max_ctr_complexity, bool) or not isinstance(
            self.max_ctr_complexity, (int, np.integer)
        ):
            raise ValueError(
                "max_ctr_complexity must be 'auto' or an int in [-1, 3], "
                f"got {self.max_ctr_complexity!r}"
            )
        else:
            mcc = int(self.max_ctr_complexity)
            if mcc < -1 or mcc > 3:
                raise ValueError(f"max_ctr_complexity must be -1 (auto) or in [0, 3], got {mcc}")
            p.max_ctr_complexity = mcc
        p.ctr_prior_weight = float(self.ctr_prior_weight)
        p.ctr_n_permutations = int(self.ctr_n_permutations)
        p.ctr_max_borders = int(self.ctr_max_borders)
        reg = float(self.ctr_model_size_reg)
        if not np.isfinite(reg) or reg < 0.0:
            raise ValueError(
                f"ctr_model_size_reg must be finite and >= 0, got {self.ctr_model_size_reg!r}"
            )
        p.ctr_model_size_reg = reg
        if self.quantizer == "linear":
            p.quantizer_mode = _core.QuantizerMode.Linear
        elif self.quantizer == "quantile":
            p.quantizer_mode = _core.QuantizerMode.Quantile
        elif self.quantizer == "auto":
            p.quantizer_mode = _core.QuantizerMode.Auto
        else:
            raise ValueError(
                f"quantizer must be 'linear', 'quantile', or 'auto', got {self.quantizer!r}"
            )
        return p

    def _check_fitted(self) -> _core.Model:
        model = getattr(self, "_model", None)
        if model is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
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
        name = type(self).__name__
        n_rows, _ = _check_input_X(X, name=name)
        y = _check_target(y, n_rows, name=name)
        sw_arr = _check_sample_weight(sample_weight, n_rows)
        self._fitted_on_dataframe = _is_pandas_dataframe(X)
        y_arr = self._encode_targets(y, is_fit=True)

        encoder: CatObliqueTSEncoder | None = None
        X_fit: Any = X
        ds_cat_arg: str | Sequence[int | str] | None = categorical_feature
        if self.cat_oblique_ts:
            cat_idx = resolve_categorical_indices_for_X(X, categorical_feature)
            if cat_idx:
                n_classes = (
                    int(len(self.classes_))
                    if self._objective == _core.Objective.Multiclass
                    else (2 if self._objective == _core.Objective.Binary else 1)
                )
                encoder = CatObliqueTSEncoder(
                    smooth=float(self.cat_ts_smooth),
                    n_folds=5,
                    random_state=int(self.random_state),
                )
                X_fit = encoder.fit_transform(
                    X,
                    y_arr,
                    sw_arr,
                    cat_idx,
                    is_binary=(self._objective == _core.Objective.Binary),
                    n_classes=n_classes,
                )
                ds_cat_arg = cat_idx

        ds = Dataset(
            X_fit,
            categorical_feature=ds_cat_arg,
            max_bin=self.max_bin,
            min_data_per_group=self.min_data_per_group,
            max_categories=self.max_categories,
            n_jobs=int(self.n_jobs),
            quantizer_mode=self.quantizer,
        )

        valid_ds_inner = None
        y_val_arr = None
        if eval_set:
            if len(eval_set) != 1:
                raise ValueError("eval_set supports a single (X_val, y_val) pair")
            X_val, y_val = eval_set[0]
            n_val, _ = _check_input_X(X_val, name=name)
            y_val = _check_target(y_val, n_val, name=name)
            X_val_fit = encoder.transform(X_val) if encoder is not None else X_val
            valid_ds_inner = ds.transform(X_val_fit)._inner
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
        self._cat_ts_encoder = encoder
        self._lignum_cache: dict[int, _core.LignumPredictor | None] = {}
        self.best_iteration_ = result.model.best_iteration
        if encoder is not None:
            self.n_features_in_ = encoder.n_input_features_
            self.feature_names_in_ = np.array(encoder.input_feature_names_, dtype=object)
        else:
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
        _, n_features = _check_input_X(X, name=type(self).__name__)
        enc: CatObliqueTSEncoder | None = getattr(self, "_cat_ts_encoder", None)
        if enc is not None:
            X = enc.transform(X)
            n_features = model.n_features
        elif n_features != model.n_features:
            raise ValueError(
                f"X has {n_features} features, but {type(self).__name__} is expecting "
                f"{model.n_features} features as input."
            )
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
        base = getattr(super(), "__getstate__", None)
        state = dict(base()) if _HAS_SKLEARN and base is not None else self.__dict__.copy()
        state.pop("_lignum_cache", None)  # rebuilt on the next prediction
        model = state.pop("_model", None)
        if model is not None:
            # The fitted model travels as miter's JSON, which round-trips bit for bit.
            state["_model_json"] = _core.model_to_json(model)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        model_json = state.pop("_model_json", None)
        base = getattr(super(), "__setstate__", None)
        if _HAS_SKLEARN and base is not None:
            base(state)
        else:
            self.__dict__.update(state)
        if model_json is not None:
            self._model = _core.model_from_json(model_json)
        self._lignum_cache = {}

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
        return {"allow_nan": True}

    def save_model(self, path: str) -> None:
        model = self._check_fitted()
        if getattr(self, "_cat_ts_encoder", None) is not None:
            raise NotImplementedError("save_model JSON does not yet support cat_oblique_ts=True; use pickle")
        _core.save_model_json(model, str(path))

    @classmethod
    def _from_core_model(cls, model: _core.Model) -> Any:
        p = model.params
        est = cls(
            n_estimators=p.n_estimators,
            learning_rate=p.learning_rate,
            num_leaves=p.num_leaves,
            max_depth=p.max_depth,
            grow_policy="symmetric" if p.grow_policy == _core.GrowPolicy.Symmetric else "leafwise",
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
            feature_fraction=p.feature_fraction,
            bagging_fraction=p.bagging_fraction,
            extra_trees=p.extra_trees,
            oblique_min_distinct=p.oblique_min_distinct,
            oblique_sample_rows=p.oblique_sample_rows,
            efb=p.efb,
            oblique_binary_penalty=p.oblique_binary_penalty,
            oblique_gain_margin=p.oblique_gain_margin,
            max_ctr_complexity=("auto" if p.max_ctr_complexity == -1 else p.max_ctr_complexity),
            ctr_prior_weight=p.ctr_prior_weight,
            ctr_n_permutations=p.ctr_n_permutations,
            ctr_max_borders=p.ctr_max_borders,
            ctr_model_size_reg=p.ctr_model_size_reg,
            quantizer=(
                "quantile"
                if p.quantizer_mode == _core.QuantizerMode.Quantile
                else ("auto" if p.quantizer_mode == _core.QuantizerMode.Auto else "linear")
            ),
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

if not _HAS_SKLEARN:  # pragma: no cover - minimal parameter API without scikit-learn
    import inspect

    _PARAM_NAMES = [n for n in inspect.signature(_BaseMiterEstimator.__init__).parameters if n != "self"]

    def _get_params(self: _BaseMiterEstimator, deep: bool = True) -> dict[str, Any]:
        del deep
        return {n: getattr(self, n) for n in _PARAM_NAMES}

    def _set_params(self: _BaseMiterEstimator, **params: Any) -> _BaseMiterEstimator:
        for key, value in params.items():
            if key not in _PARAM_NAMES:
                raise ValueError(f"Invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    _BaseMiterEstimator.get_params = _get_params  # type: ignore[attr-defined]
    _BaseMiterEstimator.set_params = _set_params  # type: ignore[attr-defined]


class MiterRegressor(_RegressorMixin, _BaseMiterEstimator):
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


class MiterClassifier(_ClassifierMixin, _BaseMiterEstimator):
    """Gradient-boosted decision tree classifier (binary log-loss and multiclass softmax)."""

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = True
        return tags

    def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
        return {"allow_nan": True}

    _objective = _core.Objective.Binary

    def _encode_targets(self, y: Any, *, is_fit: bool) -> NDArray[np.float64]:
        arr = np.asarray(y)
        if arr.ndim != 1:
            arr = arr.ravel()
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            raise ValueError("Input y contains NaN.")
        if arr.dtype.kind == "O" and any(v is None or (isinstance(v, float) and np.isnan(v)) for v in arr):
            raise ValueError("Input y contains NaN.")
        if arr.dtype.kind == "f" and not np.array_equal(arr, np.round(arr)):
            raise ValueError("Unknown label type: continuous. Discrete class labels are expected.")
        if is_fit:
            classes = np.unique(arr)
            if len(classes) < 2:
                unit = "class" if len(classes) == 1 else "classes"
                raise ValueError(
                    f"MiterClassifier requires at least 2 distinct classes; got {len(classes)} {unit}"
                )
            self.classes_ = classes
            self._objective = _core.Objective.Binary if len(classes) == 2 else _core.Objective.Multiclass
        if len(self.classes_) == 2:
            if np.array_equal(self.classes_, np.array([0, 1])) or np.array_equal(
                self.classes_, np.array([0.0, 1.0])
            ):
                if not is_fit and not np.all(np.isin(arr, self.classes_)):
                    raise ValueError("Validation target y contains unseen class labels")
                return np.ascontiguousarray(arr, dtype=np.float64)
            if not is_fit and not np.all(np.isin(arr, self.classes_)):
                raise ValueError("Validation target y contains unseen class labels")
            mapping = {self.classes_[0]: 0.0, self.classes_[1]: 1.0}
            return np.array([mapping[v] for v in arr], dtype=np.float64)
        if not is_fit and not np.all(np.isin(arr, self.classes_)):
            raise ValueError("Validation target y contains unseen class labels")
        idx = np.searchsorted(self.classes_, arr)
        return np.ascontiguousarray(idx, dtype=np.float64)

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
        # Raw scores through lignum; transformed via miter sigmoid or softmax.
        model = self._check_fitted()
        raw = self._predict_raw(X, num_iteration=num_iteration)
        if model.objective == _core.Objective.Multiclass:
            return _core.proba_softmax_from_raw(np.ascontiguousarray(raw, dtype=np.float64))
        return _core.proba_from_raw(np.ascontiguousarray(raw, dtype=np.float64))

    def predict(self, X: Any, num_iteration: int | None = None) -> NDArray[Any]:
        proba = self.predict_proba(X, num_iteration=num_iteration)
        classes = getattr(self, "classes_", np.arange(proba.shape[1]))
        return classes[np.argmax(proba, axis=1)]

    @classmethod
    def load_model(cls, path: str) -> MiterClassifier:
        model = _core.load_model_json(str(path))
        if model.objective not in (_core.Objective.Binary, _core.Objective.Multiclass):
            raise ValueError("Loaded model is not a classification model")
        est: MiterClassifier = cls._from_core_model(model)
        est._objective = model.objective
        k = int(model.n_classes) if model.objective == _core.Objective.Multiclass else 2
        est.classes_ = np.arange(k)
        return est
