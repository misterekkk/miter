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
class _SingleTargetSpec:
    out_name: str
    mapping: dict[str, float]
    missing_value: float
    prior: float


@dataclass(frozen=True)
class _SingleColBundle:
    orig_index: int
    stats: list[_SingleTargetSpec]


@dataclass
class _FactorizedCol:
    orig_index: int
    vocab: list[str]
    codes: NDArray[np.int32]
    buckets: NDArray[np.int32]
    n_vocab: int
    n_buckets: int


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


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    suffix = 1
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def _oof_encode_buckets(
    buckets: NDArray[np.int32],
    n_buckets: int,
    Y: NDArray[np.float64],
    w: NDArray[np.float64],
    fold_ids: NDArray[np.int32],
    n_folds_eff: int,
    priors: NDArray[np.float64],
    smooth: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute OOF smoothed target statistics and full-data bucket lookup tables for (N, T)."""
    n_rows, n_targets = Y.shape
    tot_w = np.bincount(buckets, weights=w, minlength=n_buckets).astype(np.float64)
    tot_wy = np.empty((n_buckets, n_targets), dtype=np.float64)
    for t in range(n_targets):
        tot_wy[:, t] = np.bincount(buckets, weights=w * Y[:, t], minlength=n_buckets)

    oof = np.empty((n_rows, n_targets), dtype=np.float64)
    if n_folds_eff < 2:
        oof[:] = priors[None, :]
    else:
        for f in range(n_folds_eff):
            mask = fold_ids == f
            if not np.any(mask):
                continue
            fb = buckets[mask]
            fw = np.bincount(fb, weights=w[mask], minlength=n_buckets).astype(np.float64)
            denom = (tot_w - fw) + smooth
            wm = w[mask]
            Ym = Y[mask]
            for t in range(n_targets):
                fwy = np.bincount(fb, weights=wm * Ym[:, t], minlength=n_buckets)
                vals = ((tot_wy[:, t] - fwy) + smooth * priors[t]) / denom
                oof[mask, t] = vals[fb]

    full_lut = (tot_wy + smooth * priors[None, :]) / (tot_w[:, None] + smooth)
    return oof, full_lut


class CatObliqueTSEncoder:
    """Fit deterministic OOF smoothed target statistics and append them as numeric columns.

    Original categorical columns stay in place so miter retains native K1 splits alongside
    oblique-eligible Q16r target-statistic columns.
    """

    def __init__(
        self,
        *,
        smooth: float = 10.0,
        n_folds: int = 5,
        random_state: int = 0,
        max_multiclass_targets: int = 8,
    ) -> None:
        if max_multiclass_targets < 1:
            raise ValueError(
                f"max_multiclass_targets must be >= 1, got {max_multiclass_targets!r}"
            )
        self.smooth = float(smooth)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.max_multiclass_targets = int(max_multiclass_targets)

        self.prior_: float = 0.0
        self.priors_: list[float] = []
        self.n_input_features_: int = 0
        self.fitted_on_dataframe_: bool = False
        self.input_feature_names_: list[str] = []
        self.cat_indices_: list[int] = []
        self.single_bundles_: list[_SingleColBundle] = []
        self.out_names_: list[str] = []

    def _build_target_matrix(
        self,
        y: NDArray[np.float64],
        *,
        n_classes: int,
    ) -> tuple[NDArray[np.float64], list[int]]:
        """Build (N, T) target matrix for reg/binary (T=1) or multiclass OvR (T<=8)."""
        if n_classes < 3:
            return np.ascontiguousarray(y, dtype=np.float64)[:, None], []

        y_int = np.asarray(y, dtype=np.int64)
        counts = np.bincount(y_int, minlength=n_classes)
        # Most frequent classes first, tie-breaking by smallest class index.
        ranked = sorted(range(n_classes), key=lambda c: (-int(counts[c]), c))
        selected = ranked[: min(n_classes, self.max_multiclass_targets)]
        Y = np.empty((len(y_int), len(selected)), dtype=np.float64)
        for t, cls_id in enumerate(selected):
            Y[:, t] = (y_int == cls_id).astype(np.float64)
        return Y, selected

    def fit_transform(
        self,
        X: Any,
        y: NDArray[np.float64],
        sample_weight: NDArray[np.float64] | None,
        cat_indices: list[int],
        *,
        is_binary: bool,
        n_classes: int = 1,
    ) -> Any:
        is_df = _is_pandas_dataframe(X)
        self.fitted_on_dataframe_ = is_df
        self.cat_indices_ = list(cat_indices)

        if is_df:
            n_rows, n_features = int(X.shape[0]), int(X.shape[1])
            orig_names = [str(c) for c in X.columns]
            arr = None
        else:
            arr = np.asarray(X)
            n_rows, n_features = int(arr.shape[0]), int(arr.shape[1])
            orig_names = [f"f{j}" for j in range(n_features)]

        self.n_input_features_ = n_features
        self.input_feature_names_ = orig_names

        w = (
            np.ones(n_rows, dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        Y, selected_cls = self._build_target_matrix(y, n_classes=n_classes)
        n_targets = Y.shape[1]

        total_w = float(np.sum(w))
        total_wy = np.sum(w[:, None] * Y, axis=0)
        priors = (total_wy / total_w) if total_w > 0.0 else np.zeros(n_targets, dtype=np.float64)
        self.prior_ = float(priors[0])
        self.priors_ = [float(p) for p in priors]

        k = min(self.n_folds, n_rows)
        oof_rng = np.random.default_rng(
            np.random.SeedSequence([int(self.random_state) & 0xFFFFFFFF, _CAT_TS_SEED_TAG])
        )
        fold_ids = np.empty(n_rows, dtype=np.int32)
        if k >= 2:
            if is_binary:
                for cls_val in (0.0, 1.0):
                    cls_idx = np.flatnonzero(y == cls_val)
                    if len(cls_idx) > 0:
                        perm = cls_idx[oof_rng.permutation(len(cls_idx))]
                        fold_ids[perm] = np.arange(len(perm), dtype=np.int32) % k
            elif n_classes >= 3:
                y_int = np.asarray(y, dtype=np.int64)
                for cls_val in range(n_classes):
                    cls_idx = np.flatnonzero(y_int == cls_val)
                    if len(cls_idx) > 0:
                        perm = cls_idx[oof_rng.permutation(len(cls_idx))]
                        fold_ids[perm] = np.arange(len(perm), dtype=np.int32) % k
            else:
                perm = oof_rng.permutation(n_rows)
                fold_ids[perm] = np.arange(n_rows, dtype=np.int32) % k
        else:
            fold_ids.fill(0)

        used_names = set(orig_names)
        single_bundles: list[_SingleColBundle] = []
        extra_cols: list[NDArray[np.float64]] = []
        out_names: list[str] = []

        for j in self.cat_indices_:
            raw_col = X.iloc[:, j] if is_df else arr[:, j]  # type: ignore[index]
            vocab, codes = _factorize_categorical_values(raw_col, is_pandas=is_df)
            n_vocab = len(vocab)
            n_buckets = n_vocab + 1
            buckets = np.where(codes < 0, n_vocab, codes).astype(np.int32)
            fc = _FactorizedCol(
                orig_index=j,
                vocab=vocab,
                codes=codes,
                buckets=buckets,
                n_vocab=n_vocab,
                n_buckets=n_buckets,
            )

            train_mat, full_lut = _oof_encode_buckets(
                fc.buckets, fc.n_buckets, Y, w, fold_ids, k, priors, self.smooth
            )

            stats_specs: list[_SingleTargetSpec] = []
            for t in range(n_targets):
                base = (
                    f"__miter_cat_ts_{j}_{orig_names[j]}"
                    if n_targets == 1
                    else f"__miter_cat_ts_{j}_{orig_names[j]}_c{selected_cls[t]}"
                )
                col_name = _unique_name(base, used_names)
                mapping = {fc.vocab[b]: float(full_lut[b, t]) for b in range(fc.n_vocab)}
                missing_val = float(full_lut[fc.n_vocab, t])
                stats_specs.append(
                    _SingleTargetSpec(
                        out_name=col_name,
                        mapping=mapping,
                        missing_value=missing_val,
                        prior=float(priors[t]),
                    )
                )
                extra_cols.append(np.ascontiguousarray(train_mat[:, t], dtype=np.float64))
                out_names.append(col_name)

            single_bundles.append(_SingleColBundle(orig_index=j, stats=stats_specs))

        self.single_bundles_ = single_bundles
        self.out_names_ = out_names
        return self._append_columns(X, extra_cols, is_df=is_df)

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
        out_cols: list[NDArray[np.float64]] = []
        for bundle in self.single_bundles_:
            raw_col = X.iloc[:, bundle.orig_index] if is_df else arr[:, bundle.orig_index]  # type: ignore[index]
            vocab, codes = _factorize_categorical_values(raw_col, is_pandas=is_df)
            n_vocab = len(vocab)
            lut_idx = np.where(codes < 0, n_vocab, codes)

            for spec in bundle.stats:
                lut = np.empty(n_vocab + 1, dtype=np.float64)
                for idx, key in enumerate(vocab):
                    lut[idx] = spec.mapping.get(key, spec.prior)
                lut[n_vocab] = spec.missing_value
                out_cols.append(lut[lut_idx])

        return self._append_columns(X, out_cols, is_df=is_df)

    def _append_columns(
        self, X: Any, extra_cols: list[NDArray[np.float64]], *, is_df: bool
    ) -> Any:
        if not extra_cols:
            return X
        if is_df:
            import pandas as pd

            extra_df = pd.DataFrame(
                {name: col for name, col in zip(self.out_names_, extra_cols, strict=True)},
                index=X.index,
            )
            return pd.concat([X, extra_df], axis=1)
        arr = np.asarray(X)
        extra_mat = np.column_stack(extra_cols)
        return np.column_stack([arr, extra_mat])
