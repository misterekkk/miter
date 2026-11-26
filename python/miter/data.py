"""Dataset and input preprocessing helpers for miter."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from miter import _core


def _is_pandas_dataframe(obj: Any) -> bool:
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(obj, pd.DataFrame)


def _canonical_cat_str(val: Any) -> str:
    if isinstance(val, (float, np.floating)) and float(val).is_integer():
        return str(int(val))
    return str(val)


def _factorize_categorical_values(values: Any, *, is_pandas: bool) -> tuple[list[str], NDArray[np.int32]]:
    if is_pandas:
        import pandas as pd

        if isinstance(values.dtype, pd.CategoricalDtype):
            return (
                [_canonical_cat_str(v) for v in values.cat.categories.tolist()],
                np.ascontiguousarray(values.cat.codes.to_numpy(dtype=np.int32)),
            )

        isna = pd.isna(values).to_numpy(dtype=bool)
        raw = values.to_numpy()
    else:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.floating):
            isna = np.isnan(arr)
        else:
            isna = np.array([v is None or (isinstance(v, float) and math.isnan(v)) for v in arr], dtype=bool)
        raw = arr

    vocab: list[str] = []
    seen: dict[str, int] = {}
    codes = np.empty(len(raw), dtype=np.int32)
    for i in range(len(raw)):
        if isna[i]:
            codes[i] = -1
        else:
            key = _canonical_cat_str(raw[i])
            idx = seen.get(key)
            if idx is None:
                idx = len(vocab)
                seen[key] = idx
                vocab.append(key)
            codes[i] = idx
    return vocab, codes


def _resolve_categorical_indices(
    n_features: int,
    feature_names: list[str],
    categorical_feature: str | Sequence[int | str] | None,
    auto_indices: list[int],
) -> list[int]:
    if categorical_feature == "auto":
        return sorted(set(auto_indices))
    if categorical_feature is None:
        return []
    if isinstance(categorical_feature, str):
        if categorical_feature in feature_names:
            return [feature_names.index(categorical_feature)]
        raise ValueError(f"Unknown categorical_feature string: {categorical_feature!r}")

    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    resolved: set[int] = set()
    for item in categorical_feature:
        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0 or idx >= n_features:
                raise ValueError(f"Categorical feature index {idx} out of range [0, {n_features})")
            resolved.add(idx)
        elif isinstance(item, str):
            if item not in name_to_idx:
                raise ValueError(f"Unknown categorical feature name: {item!r}")
            resolved.add(name_to_idx[item])
        else:
            raise TypeError(f"Invalid categorical feature specifier: {item!r}")
    return sorted(resolved)


def _prepare_numeric_1d(col: Any, *, is_pandas: bool) -> NDArray[np.float64] | NDArray[np.float32]:
    arr = col.to_numpy() if is_pandas else np.asarray(col)
    if arr.dtype == np.float64 or arr.dtype == np.float32:
        if arr.ndim == 1 and (len(arr) <= 1 or arr.strides[0] > 0):
            return arr
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr, dtype=np.float64)


def _parse_quantizer_mode(mode: str | _core.QuantizerMode) -> _core.QuantizerMode:
    if isinstance(mode, _core.QuantizerMode):
        return mode
    if mode == "linear":
        return _core.QuantizerMode.Linear
    if mode == "quantile":
        return _core.QuantizerMode.Quantile
    if mode == "auto":
        return _core.QuantizerMode.Auto
    raise ValueError(
        f"quantizer_mode must be 'linear', 'quantile', or 'auto', got {mode!r}"
    )


class TransformedData:
    """Quantized codes and binned features returned by ``Dataset.transform``."""

    def __init__(self, inner: _core.TransformedData) -> None:
        self._inner = inner

    @property
    def n_rows(self) -> int:
        return self._inner.n_rows

    @property
    def n_features(self) -> int:
        return self._inner.n_features

    @property
    def n_num_features(self) -> int:
        return self._inner.n_num_features

    @property
    def codes(self) -> NDArray[np.int16]:
        return self._inner.codes()

    def unsigned_codes(self) -> NDArray[np.uint16]:
        return self._inner.unsigned_codes()

    @property
    def bins(self) -> NDArray[np.uint8]:
        return self._inner.bins()


class Dataset:
    """Preprocessed tabular dataset with Q16r codes and axis/categorical bins."""

    def __init__(
        self,
        X: Any,
        *,
        categorical_feature: str | Sequence[int | str] | None = "auto",
        max_bin: int = 255,
        min_data_per_group: int = 100,
        max_categories: int = 253,
        feature_names: Sequence[str] | None = None,
        n_jobs: int = 1,
        quantizer_mode: str | _core.QuantizerMode = "linear",
    ) -> None:
        is_df = _is_pandas_dataframe(X)
        self._fitted_on_dataframe = is_df
        self._n_jobs = int(n_jobs)
        qmode = _parse_quantizer_mode(quantizer_mode)

        if is_df:
            n_rows, n_features = X.shape
            resolved_names = (
                [str(name) for name in feature_names]
                if feature_names is not None
                else [str(c) for c in X.columns]
            )
            if len(resolved_names) != n_features:
                raise ValueError("feature_names length must match X.shape[1]")
            auto_cats = [j for j in range(n_features) if str(X.dtypes.iloc[j]) == "category"]
            cat_idx = _resolve_categorical_indices(
                n_features, resolved_names, categorical_feature, auto_cats
            )
            if not cat_idx:
                arr = X.to_numpy()
                if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                    self._inner = _core.fit_numeric_f64(arr, max_bin, resolved_names, self._n_jobs, qmode)
                elif arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                    self._inner = _core.fit_numeric_f32(arr, max_bin, resolved_names, self._n_jobs, qmode)
                else:
                    arr_f64 = np.ascontiguousarray(arr, dtype=np.float64)
                    self._inner = _core.fit_numeric_f64(arr_f64, max_bin, resolved_names, self._n_jobs, qmode)
                return

            cat_set = set(cat_idx)
            is_cat = [j in cat_set for j in range(n_features)]
            num_cols: list[NDArray[np.float64] | NDArray[np.float32]] = []
            cat_vocabs: list[list[str]] = []
            cat_indices: list[NDArray[np.int32]] = []
            for j in range(n_features):
                col = X.iloc[:, j]
                if j in cat_set:
                    vocab, codes = _factorize_categorical_values(col, is_pandas=True)
                    cat_vocabs.append(vocab)
                    cat_indices.append(codes)
                else:
                    num_cols.append(_prepare_numeric_1d(col, is_pandas=True))
            self._inner = _core.fit_mixed(
                n_rows,
                is_cat,
                resolved_names,
                num_cols,
                cat_vocabs,
                cat_indices,
                max_bin,
                min_data_per_group,
                max_categories,
                self._n_jobs,
                qmode,
            )
            return

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        n_rows, n_features = arr.shape
        resolved_names = (
            [str(name) for name in feature_names]
            if feature_names is not None
            else [f"f{j}" for j in range(n_features)]
        )
        if len(resolved_names) != n_features:
            raise ValueError("feature_names length must match X.shape[1]")

        cat_idx = _resolve_categorical_indices(n_features, resolved_names, categorical_feature, [])
        if not cat_idx:
            if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                self._inner = _core.fit_numeric_f64(arr, max_bin, resolved_names, self._n_jobs, qmode)
            elif arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                self._inner = _core.fit_numeric_f32(arr, max_bin, resolved_names, self._n_jobs, qmode)
            else:
                arr_f64 = np.ascontiguousarray(arr, dtype=np.float64)
                self._inner = _core.fit_numeric_f64(arr_f64, max_bin, resolved_names, self._n_jobs, qmode)
            return

        cat_set = set(cat_idx)
        is_cat = [j in cat_set for j in range(n_features)]
        num_cols = []
        cat_vocabs = []
        cat_indices = []
        for j in range(n_features):
            col = arr[:, j]
            if j in cat_set:
                vocab, codes = _factorize_categorical_values(col, is_pandas=False)
                cat_vocabs.append(vocab)
                cat_indices.append(codes)
            else:
                num_cols.append(_prepare_numeric_1d(col, is_pandas=False))
        self._inner = _core.fit_mixed(
            n_rows,
            is_cat,
            resolved_names,
            num_cols,
            cat_vocabs,
            cat_indices,
            max_bin,
            min_data_per_group,
            max_categories,
            self._n_jobs,
            qmode,
        )

    @property
    def n_rows(self) -> int:
        return self._inner.n_rows

    @property
    def n_features(self) -> int:
        return self._inner.n_features

    @property
    def n_num_features(self) -> int:
        return self._inner.n_num_features

    @property
    def n_cat_features(self) -> int:
        return self._inner.n_cat_features

    @property
    def feature_names(self) -> list[str]:
        return self._inner.feature_names

    @property
    def is_categorical(self) -> list[bool]:
        return self._inner.is_categorical

    @property
    def codes(self) -> NDArray[np.int16]:
        return self._inner.codes()

    def unsigned_codes(self) -> NDArray[np.uint16]:
        return self._inner.unsigned_codes()

    @property
    def bins(self) -> NDArray[np.uint8]:
        return self._inner.bins()

    def quantizer_params(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint8]]:
        return self._inner.quantizer_params()

    def bin_edges(self, feature_idx: int) -> NDArray[np.uint16]:
        return self._inner.bin_edges(feature_idx)

    def num_bins_for_feature(self, feature_idx: int) -> int:
        return self._inner.num_bins_for_feature(feature_idx)

    def cat_categories(self, feature_idx: int) -> list[str]:
        return self._inner.cat_categories(feature_idx)

    def cat_other_code(self, feature_idx: int) -> int:
        return self._inner.cat_other_code(feature_idx)

    def cat_missing_code(self, feature_idx: int) -> int:
        return self._inner.cat_missing_code(feature_idx)

    def transform(self, X: Any) -> TransformedData:
        is_df = _is_pandas_dataframe(X)
        if is_df:
            n_rows, n_features = X.shape
            if n_features != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {n_features}")
            if self._fitted_on_dataframe:
                cols = [str(c) for c in X.columns]
                if cols != self.feature_names:
                    raise ValueError(
                        f"Feature names mismatch: expected {self.feature_names}, got {cols}"
                    )
            if self.n_cat_features == 0:
                arr = X.to_numpy()
                if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                    return TransformedData(self._inner.transform_numeric_f64(arr, self._n_jobs))
                if arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                    return TransformedData(self._inner.transform_numeric_f32(arr, self._n_jobs))
                return TransformedData(
                    self._inner.transform_numeric_f64(np.ascontiguousarray(arr, dtype=np.float64), self._n_jobs)
                )

            is_cat = self.is_categorical
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
            return TransformedData(
                self._inner.transform_mixed(n_rows, num_cols, cat_vocabs, cat_indices, self._n_jobs)
            )

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {arr.shape}")
        n_rows, n_features = arr.shape
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n_features}")
        if self.n_cat_features == 0:
            if arr.dtype == np.float64 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                return TransformedData(self._inner.transform_numeric_f64(arr, self._n_jobs))
            if arr.dtype == np.float32 and (n_rows <= 1 or arr.strides[0] > 0) and (n_features <= 1 or arr.strides[1] > 0):
                return TransformedData(self._inner.transform_numeric_f32(arr, self._n_jobs))
            return TransformedData(
                self._inner.transform_numeric_f64(np.ascontiguousarray(arr, dtype=np.float64), self._n_jobs)
            )

        is_cat = self.is_categorical
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
        return TransformedData(
            self._inner.transform_mixed(n_rows, num_cols, cat_vocabs, cat_indices, self._n_jobs)
        )
