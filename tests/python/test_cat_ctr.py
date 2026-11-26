"""Tests for CatBoost-style ordered CTRs, pair combinations, counter, and multiclass support."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from miter import MiterClassifier, MiterRegressor, _core
from miter.cat_ts import CatObliqueTSEncoder


def test_ordered_ctr_matches_bruteforce_and_has_zero_self_leakage() -> None:
    df = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "b", "a", "a", None, "b", "singleton", "a", "b", None]),
            "x": np.linspace(0.0, 1.0, 10),
        }
    )
    y = np.array([1.0, 4.0, 2.0, 8.0, -3.0, 6.0, 99.0, 5.0, 0.0, 1.0], dtype=np.float64)
    w = np.array([1.0, 2.0, 1.0, 1.5, 1.0, 0.5, 2.0, 1.0, 1.0, 1.5], dtype=np.float64)
    smooth = 5.0
    n_perm = 4

    enc = CatObliqueTSEncoder(
        smooth=smooth, n_folds=5, random_state=3, mode="ordered", n_permutations=n_perm
    )
    out = enc.fit_transform(df, y, w, [0], is_binary=False)
    got = out["__miter_cat_ts_0_c"].to_numpy()

    # Exact brute-force verification against the same 4 permutations.
    rng = np.random.default_rng(3 + 20260401)
    cats_key = [None if pd.isna(v) else str(v) for v in df["c"]]
    prior = float(np.sum(w * y) / np.sum(w))
    expected = np.zeros(len(df), dtype=np.float64)
    for _ in range(n_perm):
        perm = rng.permutation(len(df))
        seen_w: dict[str | None, float] = {}
        seen_wy: dict[str | None, float] = {}
        for idx in perm:
            key = cats_key[idx]
            pw = seen_w.get(key, 0.0)
            pwy = seen_wy.get(key, 0.0)
            expected[idx] += (pwy + smooth * prior) / (pw + smooth)
            seen_w[key] = pw + w[idx]
            seen_wy[key] = pwy + w[idx] * y[idx]
    expected /= float(n_perm)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)

    # Regression check: singletons with different y must receive identical CTR (= prior), not -y/(N-1),
    # and perturbing the last row in the single permutation cannot leak into earlier rows.
    df_s = pd.DataFrame(
        {"c": pd.Categorical(["s0", "s1", "a", "a"]), "x": np.arange(4, dtype=np.float64)}
    )
    y_s = np.array([0.0, 1.0, 0.25, 0.75], dtype=np.float64)
    enc_s = CatObliqueTSEncoder(smooth=smooth, n_folds=4, random_state=0, mode="ordered", n_permutations=1)
    out_s = enc_s.fit_transform(df_s, y_s, None, [0], is_binary=False)
    ctr_s = out_s["__miter_cat_ts_0_c"].to_numpy()
    assert ctr_s[0] == pytest.approx(float(np.mean(y_s)), abs=1e-14)
    assert ctr_s[1] == pytest.approx(float(np.mean(y_s)), abs=1e-14)

    # Find the earlier 'a' row in the single permutation and verify its prefix (zero predecessors) == prior.
    perm_1 = np.random.default_rng(0 + 20260401).permutation(len(df_s))
    a_positions = [int(i) for i in perm_1 if i in (2, 3)]
    first_a = a_positions[0]
    assert ctr_s[first_a] == pytest.approx(float(np.mean(y_s)), abs=1e-14)


def test_counter_leave_one_out_and_transform_unseen_missing() -> None:
    df = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "a", "a", "b", "singleton", None, None]),
            "x": np.arange(7, dtype=np.float64),
        }
    )
    y = np.linspace(0.0, 1.0, 7)
    enc = CatObliqueTSEncoder(smooth=10.0, n_folds=3, random_state=0, include_counter=True)
    out = enc.fit_transform(df, y, None, [0], is_binary=False)

    cnt = out["__miter_cat_cnt_0_c"].to_numpy()
    expected_tr = np.log1p(np.array([2, 2, 2, 0, 0, 1, 1], dtype=np.float64))
    np.testing.assert_allclose(cnt, expected_tr, atol=1e-14)

    df_te = pd.DataFrame(
        {"c": pd.Categorical(["a", "singleton", "unseen_xyz", None]), "x": np.zeros(4)}
    )
    cnt_te = enc.transform(df_te)["__miter_cat_cnt_0_c"].to_numpy()
    expected_te = np.log1p(np.array([3, 1, 0, 2], dtype=np.float64))
    np.testing.assert_allclose(cnt_te, expected_te, atol=1e-14)


def test_pair_combinations_recover_xor_interaction() -> None:
    rng = np.random.default_rng(101)
    n = 1500
    a = rng.integers(0, 6, size=n)
    b = rng.integers(0, 6, size=n)
    noise = rng.integers(0, 8, size=n)
    # Pure parity interaction between a and b; marginals carry no signal.
    y_bin = ((a ^ b) & 1).astype(np.int64)

    df = pd.DataFrame(
        {
            "a": pd.Categorical([f"a{v}" for v in a]),
            "b": pd.Categorical([f"b{v}" for v in b]),
            "n": pd.Categorical([f"n{v}" for v in noise]),
            "x": rng.normal(scale=0.1, size=n),
        }
    )
    tr, te = slice(0, 1200), slice(1200, 1500)

    clf_no_pairs = MiterClassifier(
        n_estimators=15,
        learning_rate=0.2,
        num_leaves=2,
        max_depth=1,
        m_mult=0,
        cat_oblique_ts=True,
        cat_ctr_mode="ordered",
        cat_ctr_pairs=0,
        min_data_per_group=10,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(df.iloc[tr], y_bin[tr])

    clf_pairs = MiterClassifier(
        n_estimators=15,
        learning_rate=0.2,
        num_leaves=2,
        max_depth=1,
        m_mult=0,
        cat_oblique_ts=True,
        cat_ctr_mode="ordered",
        cat_ctr_pairs=4,
        cat_ctr_counter=True,
        min_data_per_group=10,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(df.iloc[tr], y_bin[tr])

    assert clf_pairs._cat_ts_encoder is not None
    assert len(clf_pairs._cat_ts_encoder.pair_bundles_) == 1
    first_pair = clf_pairs._cat_ts_encoder.pair_bundles_[0]
    assert (first_pair.idx_a, first_pair.idx_b) == (0, 1)

    acc_no_pairs = float(np.mean(clf_no_pairs.predict(df.iloc[te]) == y_bin[te]))
    acc_pairs = float(np.mean(clf_pairs.predict(df.iloc[te]) == y_bin[te]))
    assert acc_pairs > 0.95
    assert acc_pairs > acc_no_pairs + 0.30


@pytest.mark.parametrize("n_classes,expected_targets", [(4, 4), (10, 8)])
def test_multiclass_ctr_cap_and_numpy_end_to_end(n_classes: int, expected_targets: int) -> None:
    rng = np.random.default_rng(77 + n_classes)
    n = 700
    c0 = rng.choice([f"g{k}" for k in range(n_classes)], size=n)
    c1 = rng.choice(["m0", "m1", "m2"], size=n)
    y = ((np.array([int(s[1:]) for s in c0]) + np.array([int(s[1:]) for s in c1])) % n_classes).astype(
        np.int64
    )
    X = np.column_stack([rng.normal(size=n), c0, c1]).astype(object)

    clf = MiterClassifier(
        n_estimators=20,
        learning_rate=0.1,
        cat_oblique_ts=True,
        cat_ctr_mode="ordered",
        cat_ctr_pairs=4,
        cat_ctr_counter=True,
        min_data_per_group=5,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=5,
    ).fit(X[:500], y[:500], categorical_feature=[1, 2])

    enc = clf._cat_ts_encoder
    assert enc is not None
    assert len(enc.single_bundles_[0].stats) == expected_targets

    proba = clf.predict_proba(X[500:])
    assert proba.shape == (200, n_classes)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    unpickled = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(proba, unpickled.predict_proba(X[500:]))


def test_cat_ctr_defaults_bit_identical_and_validation() -> None:
    rng = np.random.default_rng(12)
    df = pd.DataFrame(
        {
            "c": pd.Categorical(rng.choice(["x", "y", "z"], size=120)),
            "x": rng.normal(size=120),
        }
    )
    y = rng.normal(size=120)

    m0 = MiterRegressor(n_estimators=10, early_stopping_rounds=None, n_jobs=1, random_state=2).fit(df, y)
    m1 = MiterRegressor(
        n_estimators=10,
        early_stopping_rounds=None,
        cat_oblique_ts=False,
        cat_ctr_mode="ordered",
        cat_ctr_pairs=8,
        cat_ctr_counter=True,
        n_jobs=1,
        random_state=2,
    ).fit(df, y)
    assert _core.model_to_json(m0._model) == _core.model_to_json(m1._model)
    np.testing.assert_array_equal(m0.predict(df), m1.predict(df))

    with pytest.raises(ValueError, match="cat_ctr_mode"):
        MiterRegressor(cat_ctr_mode="bad").fit(df, y)
    with pytest.raises(ValueError, match="cat_ctr_pairs"):
        MiterRegressor(cat_ctr_pairs=-1).fit(df, y)
    with pytest.raises(ValueError, match="cat_ctr_pairs"):
        MiterRegressor(cat_ctr_pairs=True).fit(df, y)  # type: ignore[arg-type]
