"""Out-of-fold smoothed Target Statistics for categoricals in oblique splits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from miter.data import (
    _factorize_categorical_values,
    _is_pandas_dataframe,
    _resolve_categorical_indices,
)

_CAT_TS_SEED_TAG = 0x54534F4F  # "TSOO"


@dataclass(frozen=True)
class _CatColSpec:
    orig_index: int
    out_name: str
    mapping: dict[str, float]
    missing_value: float


def resolve_categorical_indices_for_X(
    X: Any,
    categorical_feature: str | Sequence[int | str] | None,
) -> list[int]:
    """Resolve categorical feature indices matching ``miter.data.Dataset`` semantics."""
    if _is_pandas_dataframe(X):
        n_features = int(X.shape[1])
        names = [str(c) for c in X.columns]
        auto_cats = [j for j in range(n_features) if str(X.dtypes.iloc[j]) == "category"]
        return _resolve_categorical_indices(n_features, names, categorical_feature, auto_cats)

    arr = np.asarray(X)
    n_features = int(arr.shape[1])
    names = [f"f{j}" for j in range(n_features)]
    return _resolve_categorical_indices(n_features, names, categorical_feature, [])


class CatObliqueTSEncoder:
    """Fit deterministic OOF smoothed target statistics and append them as numeric columns.

    Original categorical columns stay in place so miter retains native K1 splits alongside
    oblique-eligible Q16r target-statistic columns.
    """

    def __init__(self, *, smooth: float = 10.0, n_folds: int = 5, random_state: int = 0) -> None:
        self.smooth = float(smooth)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.prior_: float = 0.0
        self.n_input_features_: int = 0
        self.fitted_on_dataframe_: bool = False
        self.input_feature_names_: list[str] = []
        self.cat_indices_: list[int] = []
        self.specs_: list[_CatColSpec] = []

    def fit_transform(
        self,
        X: Any,
        y: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None,
        cat_indices: list[int],
        *,
        is_binary: bool,
    ) -> Any:
        is_df = _is_pandas_dataframe(X)
        self.fitted_on_dataframe_ = is_df
        self.cat_indices_ = list(cat_indices)

        if is_df:
            n_rows, n_features = int(X.shape[0]), int(X.shape[1])
            orig_names = [str(c) for c in X.columns]
        else:
            arr = np.asarray(X)
            n_rows, n_features = int(arr.shape[0]), int(arr.shape[1])
            orig_names = [f"f{j}" for j in range(n_features)]

        self.n_input_features_ = n_features
        self.input_feature_names_ = orig_names

        w = np.ones(n_rows, dtype=np.float64) if sample_weight is None else sample_weight
        total_w = float(np.sum(w))
        prior = float(np.sum(y * w) / total_w) if total_w > 0.0 else 0.0
        self.prior_ = prior

        k = min(self.n_folds, n_rows)
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.random_state) & 0xFFFFFFFF, _CAT_TS_SEED_TAG])
        )
        fold_ids = np.empty(n_rows, dtype=np.int32)
        if k >= 2:
            if is_binary:
                for cls_val in (0.0, 1.0):
                    cls_idx = np.flatnonzero(y == cls_val)
                    if len(cls_idx) > 0:
                        perm = cls_idx[rng.permutation(len(cls_idx))]
                        fold_ids[perm] = np.arange(len(perm), dtype=np.int32) % k
            else:
                perm = rng.permutation(n_rows)
                fold_ids[perm] = np.arange(n_rows, dtype=np.int32) % k
        else:
            fold_ids.fill(0)

        used_names = set(orig_names)
        specs: list[_CatColSpec] = []
        oof_cols: list[NDArray[np.float64]] = []
        smooth = self.smooth

        for j in self.cat_indices_:
            raw_col = X.iloc[:, j] if is_df else arr[:, j]
            vocab, codes = _factorize_categorical_values(raw_col, is_pandas=is_df)
            n_vocab = len(vocab)
            n_buckets = n_vocab + 1
            buckets = np.where(codes < 0, n_vocab, codes)

            tot_w = np.bincount(buckets, weights=w, minlength=n_buckets)
            tot_wy = np.bincount(buckets, weights=w * y, minlength=n_buckets)

            ts_col = np.empty(n_rows, dtype=np.float64)
            if k < 2:
                ts_col.fill(prior)
            else:
                for f in range(k):
                    mask = fold_ids == f
                    if not np.any(mask):
                        continue
                    fb = buckets[mask]
                    fw = np.bincount(fb, weights=w[mask], minlength=n_buckets)
                    fwy = np.bincount(fb, weights=w[mask] * y[mask], minlength=n_buckets)
                    oof_vals = ((tot_wy - fwy) + smooth * prior) / ((tot_w - fw) + smooth)
                    ts_col[mask] = oof_vals[fb]

            full_vals = (tot_wy + smooth * prior) / (tot_w + smooth)
            mapping = {vocab[b]: float(full_vals[b]) for b in range(n_vocab)}
            missing_val = float(full_vals[n_vocab])

            base = f"__miter_cat_ts_{j}_{orig_names[j]}"
            out_name = base
            suffix = 1
            while out_name in used_names:
                out_name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(out_name)

            specs.append(_CatColSpec(orig_index=j, out_name=out_name, mapping=mapping, missing_value=missing_val))
            oof_cols.append(ts_col)

        self.specs_ = specs
        return self._append_columns(X, oof_cols, is_df=is_df)

    def transform(self, X: Any) -> Any:
        is_df = _is_pandas_dataframe(X)
        n_features = int(X.shape[1]) if is_df else int(np.asarray(X).shape[1])
        if n_features != self.n_input_features_:
            raise ValueError(
                f"X has {n_features} features, but the model is expecting {self.n_input_features_} features as input."
            )
        if is_df and self.fitted_on_dataframe_:
            cols = [str(c) for c in X.columns]
            if cols != self.input_feature_names_:
                raise ValueError(
                    f"Feature names mismatch: expected {self.input_feature_names_}, got {cols}"
                )

        arr = None if is_df else np.asarray(X)
        prior = self.prior_
        out_cols: list[NDArray[np.float64]] = []
        for spec in self.specs_:
            raw_col = X.iloc[:, spec.orig_index] if is_df else arr[:, spec.orig_index]  # type: ignore[index]
            vocab, codes = _factorize_categorical_values(raw_col, is_pandas=is_df)
            lut = np.empty(len(vocab) + 1, dtype=np.float64)
            for idx, key in enumerate(vocab):
                lut[idx] = spec.mapping.get(key, prior)
            lut[len(vocab)] = spec.missing_value
            lut_idx = np.where(codes < 0, len(vocab), codes)
            out_cols.append(lut[lut_idx])

        return self._append_columns(X, out_cols, is_df=is_df)

    def _append_columns(self, X: Any, extra_cols: list[NDArray[np.float64]], *, is_df: bool) -> Any:
        if not extra_cols:
            return X
        if is_df:
            import pandas as pd

            extra_df = pd.DataFrame(
                {spec.out_name: col for spec, col in zip(self.specs_, extra_cols, strict=True)},
                index=X.index,
            )
            return pd.concat([X, extra_df], axis=1)
        arr = np.asarray(X)
        extra_mat = np.column_stack(extra_cols)
        return np.column_stack([arr, extra_mat])
