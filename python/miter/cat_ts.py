"""CatBoost-style categorical CTR and frequency encoding helpers for oblique splits."""

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
_CAT_PERM_OFFSET = 20260401
_MAX_PAIR_CANDIDATE_COLS = 16


@dataclass(frozen=True)
class _SingleTargetSpec:
    out_name: str
    mapping: dict[str, float]
    missing_value: float
    prior: float


@dataclass(frozen=True)
class _CounterSpec:
    out_name: str
    mapping: dict[str, float]
    missing_value: float


@dataclass(frozen=True)
class _SingleColBundle:
    orig_index: int
    stats: list[_SingleTargetSpec]
    counter: _CounterSpec | None


@dataclass(frozen=True)
class _PairTargetSpec:
    out_name: str
    mapping: dict[tuple[str | None, str | None], float]
    prior: float


@dataclass(frozen=True)
class _PairColBundle:
    idx_a: int
    idx_b: int
    stats: list[_PairTargetSpec]


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


def _ordered_encode_buckets(
    buckets: NDArray[np.int32],
    n_buckets: int,
    Y: NDArray[np.float64],
    w: NDArray[np.float64],
    priors: NDArray[np.float64],
    smooth: float,
    perms: list[NDArray[np.intp]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized CatBoost-style ordered prefix target statistics averaged over permutations."""
    n_rows, n_targets = Y.shape
    n_perms = len(perms)
    wy = w[:, None] * Y
    zero_row = np.zeros((1, n_targets), dtype=np.float64)
    idx_arange = np.arange(n_rows)
    ctr_acc = np.zeros((n_rows, n_targets), dtype=np.float64)

    for perm in perms:
        b_p = buckets[perm]
        order = np.argsort(b_p, kind="mergesort")
        b_sorted = b_p[order]
        orig_pos = perm[order]

        is_start = np.empty(n_rows, dtype=bool)
        is_start[0] = True
        if n_rows > 1:
            is_start[1:] = b_sorted[1:] != b_sorted[:-1]
        grp_starts = np.flatnonzero(is_start)
        row_grp = np.cumsum(is_start) - 1
        grp_first_pos = grp_starts[row_grp]

        cum_w = np.cumsum(w[orig_pos])
        cum_wy = np.cumsum(wy[orig_pos], axis=0)
        cum_w0 = np.concatenate(([0.0], cum_w))
        cum_wy0 = np.vstack([zero_row, cum_wy])

        pref_w_sorted = cum_w0[idx_arange] - cum_w0[grp_first_pos]
        pref_wy_sorted = cum_wy0[idx_arange] - cum_wy0[grp_first_pos]

        pref_w = np.empty(n_rows, dtype=np.float64)
        pref_wy = np.empty((n_rows, n_targets), dtype=np.float64)
        pref_w[orig_pos] = pref_w_sorted
        pref_wy[orig_pos] = pref_wy_sorted

        ctr_acc += (pref_wy + smooth * priors[None, :]) / (pref_w[:, None] + smooth)

    ctr_acc /= float(n_perms)

    tot_w = np.bincount(buckets, weights=w, minlength=n_buckets).astype(np.float64)
    tot_wy = np.empty((n_buckets, n_targets), dtype=np.float64)
    for t in range(n_targets):
        tot_wy[:, t] = np.bincount(buckets, weights=wy[:, t], minlength=n_buckets)
    full_lut = (tot_wy + smooth * priors[None, :]) / (tot_w[:, None] + smooth)
    return ctr_acc, full_lut


class CatObliqueTSEncoder:
    """Fit deterministic CatBoost-style CTRs and append them as numeric columns.

    Original categorical columns stay in place so miter retains native K1 splits alongside
    oblique-eligible Q16r target-statistic, pair, and counter columns.
    """

    def __init__(
        self,
        *,
        smooth: float = 10.0,
        n_folds: int = 5,
        random_state: int = 0,
        mode: str = "oof",
        max_pairs: int = 0,
        include_counter: bool = False,
        n_permutations: int = 4,
        max_multiclass_targets: int = 8,
    ) -> None:
        if mode not in ("oof", "ordered"):
            raise ValueError(f"cat_ctr_mode must be 'oof' or 'ordered', got {mode!r}")
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 0:
            raise ValueError(f"max_pairs must be an int >= 0, got {max_pairs!r}")
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be >= 1, got {n_permutations!r}")
        if max_multiclass_targets < 1:
            raise ValueError(
                f"max_multiclass_targets must be >= 1, got {max_multiclass_targets!r}"
            )
        self.smooth = float(smooth)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.mode = str(mode)
        self.max_pairs = int(max_pairs)
        self.include_counter = bool(include_counter)
        self.n_permutations = int(n_permutations)
        self.max_multiclass_targets = int(max_multiclass_targets)

        self.prior_: float = 0.0
        self.priors_: list[float] = []
        self.n_input_features_: int = 0
        self.fitted_on_dataframe_: bool = False
        self.input_feature_names_: list[str] = []
        self.cat_indices_: list[int] = []
        self.single_bundles_: list[_SingleColBundle] = []
        self.pair_bundles_: list[_PairColBundle] = []
        self.out_names_: list[str] = []

    def _build_target_matrix(
        self,
        y: NDArray[np.float64],
        w: NDArray[np.float64],
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
        del w
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

        w = np.ones(n_rows, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        Y, selected_cls = self._build_target_matrix(y, w, n_classes=n_classes)
        n_targets = Y.shape[1]

        total_w = float(np.sum(w))
        total_wy = np.sum(w[:, None] * Y, axis=0)
        priors = (total_wy / total_w) if total_w > 0.0 else np.zeros(n_targets, dtype=np.float64)
        self.prior_ = float(priors[0])
        self.priors_ = [float(p) for p in priors]

        # Build CV fold IDs (used when mode=="oof", and for greedy pair OOF scoring).
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

        perms: list[NDArray[np.intp]] = []
        if self.mode == "ordered":
            perm_seed = abs(int(self.random_state) + _CAT_PERM_OFFSET)
            perm_rng = np.random.default_rng(perm_seed)
            perms = [perm_rng.permutation(n_rows) for _ in range(self.n_permutations)]

        # Factorize every categorical column once.
        fact_cols: dict[int, _FactorizedCol] = {}
        for j in self.cat_indices_:
            raw_col = X.iloc[:, j] if is_df else arr[:, j]  # type: ignore[index]
            vocab, codes = _factorize_categorical_values(raw_col, is_pandas=is_df)
            n_vocab = len(vocab)
            n_buckets = n_vocab + 1
            buckets = np.where(codes < 0, n_vocab, codes).astype(np.int32)
            fact_cols[j] = _FactorizedCol(
                orig_index=j,
                vocab=vocab,
                codes=codes,
                buckets=buckets,
                n_vocab=n_vocab,
                n_buckets=n_buckets,
            )

        used_names = set(orig_names)
        single_bundles: list[_SingleColBundle] = []
        extra_cols: list[NDArray[np.float64]] = []
        out_names: list[str] = []
        single_oof_mse: dict[int, float] = {}
        base_mse = float(np.average(np.mean((Y - priors[None, :]) ** 2, axis=1), weights=w)) if total_w > 0.0 else 0.0

        for j in self.cat_indices_:
            fc = fact_cols[j]
            need_oof_mse = self.max_pairs > 0 and len(self.cat_indices_) >= 2

            if self.mode == "ordered":
                train_mat, full_lut = _ordered_encode_buckets(
                    fc.buckets, fc.n_buckets, Y, w, priors, self.smooth, perms
                )
                if need_oof_mse:
                    oof_for_score, _ = _oof_encode_buckets(
                        fc.buckets, fc.n_buckets, Y, w, fold_ids, k, priors, self.smooth
                    )
                    single_oof_mse[j] = (
                        float(np.average(np.mean((Y - oof_for_score) ** 2, axis=1), weights=w))
                        if total_w > 0.0
                        else base_mse
                    )
            else:
                train_mat, full_lut = _oof_encode_buckets(
                    fc.buckets, fc.n_buckets, Y, w, fold_ids, k, priors, self.smooth
                )
                if need_oof_mse:
                    single_oof_mse[j] = (
                        float(np.average(np.mean((Y - train_mat) ** 2, axis=1), weights=w))
                        if total_w > 0.0
                        else base_mse
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

            counter_spec: _CounterSpec | None = None
            if self.include_counter:
                cnt_name = _unique_name(f"__miter_cat_cnt_{j}_{orig_names[j]}", used_names)
                raw_cnts = np.bincount(fc.buckets, minlength=fc.n_buckets)
                train_cnt = np.log1p(np.maximum(raw_cnts[fc.buckets] - 1, 0)).astype(np.float64)
                full_cnt_lut = np.log1p(raw_cnts.astype(np.float64))
                cnt_mapping = {fc.vocab[b]: float(full_cnt_lut[b]) for b in range(fc.n_vocab)}
                cnt_missing = float(full_cnt_lut[fc.n_vocab])
                counter_spec = _CounterSpec(
                    out_name=cnt_name,
                    mapping=cnt_mapping,
                    missing_value=cnt_missing,
                )
                extra_cols.append(train_cnt)
                out_names.append(cnt_name)

            single_bundles.append(
                _SingleColBundle(orig_index=j, stats=stats_specs, counter=counter_spec)
            )

        pair_bundles: list[_PairColBundle] = []
        if self.max_pairs > 0 and len(self.cat_indices_) >= 2 and total_w > 0.0:
            if len(self.cat_indices_) > _MAX_PAIR_CANDIDATE_COLS:
                cand_cols = sorted(
                    self.cat_indices_,
                    key=lambda idx: (-(base_mse - single_oof_mse[idx]), idx),
                )[:_MAX_PAIR_CANDIDATE_COLS]
                cand_cols.sort()
            else:
                cand_cols = list(self.cat_indices_)

            scored_pairs: list[
                tuple[float, int, int, NDArray[np.int64], NDArray[np.int32], int]
            ] = []
            for p_a_i in range(len(cand_cols)):
                ia = cand_cols[p_a_i]
                fa = fact_cols[ia]
                for p_b_i in range(p_a_i + 1, len(cand_cols)):
                    ib = cand_cols[p_b_i]
                    fb = fact_cols[ib]
                    packed = fa.buckets.astype(np.int64) * np.int64(fb.n_buckets) + fb.buckets.astype(
                        np.int64
                    )
                    uniq_packed, pair_buckets_i64 = np.unique(packed, return_inverse=True)
                    n_pair_buckets = int(len(uniq_packed))
                    if n_pair_buckets >= n_rows:
                        continue
                    pair_buckets = pair_buckets_i64.astype(np.int32)
                    oof_p, _ = _oof_encode_buckets(
                        pair_buckets, n_pair_buckets, Y, w, fold_ids, k, priors, self.smooth
                    )
                    mse_p = float(np.average(np.mean((Y - oof_p) ** 2, axis=1), weights=w))
                    parent_best = min(single_oof_mse[ia], single_oof_mse[ib], base_mse)
                    gain = parent_best - mse_p
                    if gain > 1e-12:
                        scored_pairs.append(
                            (gain, ia, ib, uniq_packed, pair_buckets, n_pair_buckets)
                        )

            scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, ia, ib, uniq_packed, pair_buckets, n_pair_buckets in scored_pairs[
                : self.max_pairs
            ]:
                fa = fact_cols[ia]
                fb = fact_cols[ib]
                if self.mode == "ordered":
                    p_mat, p_lut = _ordered_encode_buckets(
                        pair_buckets, n_pair_buckets, Y, w, priors, self.smooth, perms
                    )
                else:
                    p_mat, p_lut = _oof_encode_buckets(
                        pair_buckets, n_pair_buckets, Y, w, fold_ids, k, priors, self.smooth
                    )

                va_tokens: list[str | None] = [*fa.vocab, None]
                vb_tokens: list[str | None] = [*fb.vocab, None]
                nb = np.int64(fb.n_buckets)
                pair_keys = [
                    (va_tokens[int(pv // nb)], vb_tokens[int(pv % nb)]) for pv in uniq_packed
                ]

                p_specs: list[_PairTargetSpec] = []
                for t in range(n_targets):
                    base = (
                        f"__miter_cat_pair_{ia}_{ib}_{orig_names[ia]}__{orig_names[ib]}"
                        if n_targets == 1
                        else f"__miter_cat_pair_{ia}_{ib}_{orig_names[ia]}__{orig_names[ib]}_c{selected_cls[t]}"
                    )
                    p_name = _unique_name(base, used_names)
                    p_map = {
                        pk: float(p_lut[u_idx, t]) for u_idx, pk in enumerate(pair_keys)
                    }
                    p_specs.append(
                        _PairTargetSpec(out_name=p_name, mapping=p_map, prior=float(priors[t]))
                    )
                    extra_cols.append(np.ascontiguousarray(p_mat[:, t], dtype=np.float64))
                    out_names.append(p_name)

                pair_bundles.append(_PairColBundle(idx_a=ia, idx_b=ib, stats=p_specs))

        self.single_bundles_ = single_bundles
        self.pair_bundles_ = pair_bundles
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
        factorized: dict[int, tuple[list[str], NDArray[np.int32]]] = {}

        def _get_fact(orig_idx: int) -> tuple[list[str], NDArray[np.int32]]:
            if orig_idx not in factorized:
                raw_col = X.iloc[:, orig_idx] if is_df else arr[:, orig_idx]  # type: ignore[index]
                factorized[orig_idx] = _factorize_categorical_values(raw_col, is_pandas=is_df)
            return factorized[orig_idx]

        out_cols: list[NDArray[np.float64]] = []
        for bundle in self.single_bundles_:
            vocab, codes = _get_fact(bundle.orig_index)
            n_vocab = len(vocab)
            lut_idx = np.where(codes < 0, n_vocab, codes)

            for spec in bundle.stats:
                lut = np.empty(n_vocab + 1, dtype=np.float64)
                for idx, key in enumerate(vocab):
                    lut[idx] = spec.mapping.get(key, spec.prior)
                lut[n_vocab] = spec.missing_value
                out_cols.append(lut[lut_idx])

            if bundle.counter is not None:
                cspec = bundle.counter
                c_lut = np.empty(n_vocab + 1, dtype=np.float64)
                for idx, key in enumerate(vocab):
                    c_lut[idx] = cspec.mapping.get(key, 0.0)
                c_lut[n_vocab] = cspec.missing_value
                out_cols.append(c_lut[lut_idx])

        for pbundle in self.pair_bundles_:
            va, ca = _get_fact(pbundle.idx_a)
            vb, cb = _get_fact(pbundle.idx_b)
            n_va = len(va)
            n_vb = len(vb)
            ba = np.where(ca < 0, n_va, ca).astype(np.int64)
            bb = np.where(cb < 0, n_vb, cb).astype(np.int64)
            nb_str = np.int64(n_vb + 1)
            packed = ba * nb_str + bb
            uniq_packed, inv = np.unique(packed, return_inverse=True)

            va_tokens: list[str | None] = [*va, None]
            vb_tokens: list[str | None] = [*vb, None]
            u_keys = [(va_tokens[int(pv // nb_str)], vb_tokens[int(pv % nb_str)]) for pv in uniq_packed]

            for pspec in pbundle.stats:
                p_lut = np.array([pspec.mapping.get(pk, pspec.prior) for pk in u_keys], dtype=np.float64)
                out_cols.append(p_lut[inv])

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
