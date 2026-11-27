"""Tests for opt-in OOF target statistics (cat_oblique_ts) for categoricals."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from miter import MiterClassifier, MiterRegressor, _core


def test_cat_oblique_ts_off_and_numeric_only_are_bit_identical() -> None:
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "c0": pd.Categorical(rng.choice(["a", "b", "c", "d"], size=n)),
        }
    )
    y = df["x0"].to_numpy() + (df["c0"] == "a").astype(float).to_numpy() + 0.1 * rng.normal(size=n)

    m_def = MiterRegressor(n_estimators=10, early_stopping_rounds=None, n_jobs=1, random_state=3).fit(df, y)
    m_off = MiterRegressor(
        n_estimators=10, early_stopping_rounds=None, cat_oblique_ts=False, n_jobs=1, random_state=3
    ).fit(df, y)
    assert _core.model_to_json(m_def._model) == _core.model_to_json(m_off._model)
    np.testing.assert_array_equal(m_def.predict(df), m_off.predict(df))

    # Numeric-only input with cat_oblique_ts=True matches cat_oblique_ts=False bit for bit.
    X_num = rng.normal(size=(n, 4))
    y_num = 2.0 * X_num[:, 0] - X_num[:, 1]
    mn_off = MiterRegressor(n_estimators=8, early_stopping_rounds=None, n_jobs=1, random_state=5).fit(
        X_num, y_num
    )
    mn_on = MiterRegressor(
        n_estimators=8, early_stopping_rounds=None, cat_oblique_ts=True, n_jobs=1, random_state=5
    ).fit(X_num, y_num)
    assert _core.model_to_json(mn_off._model) == _core.model_to_json(mn_on._model)
    np.testing.assert_array_equal(mn_off.predict(X_num), mn_on.predict(X_num))


def test_cat_ts_encoder_exact_oof_and_zero_leakage() -> None:
    from miter.cat_ts import CatObliqueTSEncoder

    # 10 rows, smooth=10, 5 folds -> 2 rows per fold; one category appears once, one missing.
    df = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "a", "a", "a", "a", "b", "b", "b", "singleton", None]),
            "x": np.linspace(0.0, 1.0, 10),
        }
    )
    y = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 0.0, 2.0, 4.0, 100.0, -2.0], dtype=np.float64)
    prior = float(np.mean(y))
    enc = CatObliqueTSEncoder(smooth=10.0, n_folds=5, random_state=0)
    out = enc.fit_transform(df, y, None, [0], is_binary=False)
    ts = out["__miter_cat_ts_0_c"].to_numpy()

    # The singleton row (y=100.0) has no other rows outside its fold -> OOF value must equal prior!
    assert ts[8] == pytest.approx(prior, abs=1e-12)
    # Missing row (index 9) only occurred in its own fold -> OOF value must equal prior!
    assert ts[9] == pytest.approx(prior, abs=1e-12)

    # On transform, unseen categories map to prior, known "a" maps to (25 + 10*prior)/15, missing maps to (-2 + 10*prior)/11.
    df_test = pd.DataFrame({"c": pd.Categorical(["unseen_xyz", "a", None]), "x": [0.0, 0.0, 0.0]})
    out_test = enc.transform(df_test)["__miter_cat_ts_0_c"].to_numpy()
    assert out_test[0] == pytest.approx(prior, abs=1e-12)
    assert out_test[1] == pytest.approx((25.0 + 10.0 * prior) / 15.0, abs=1e-12)
    assert out_test[2] == pytest.approx((-2.0 + 10.0 * prior) / 11.0, abs=1e-12)


def test_cat_oblique_ts_recovers_high_card_oblique_interaction_and_bitwise_exactness() -> None:
    rng = np.random.default_rng(42)
    n = 4000
    n_cats = 400  # > max_categories (253)
    cat_effect = rng.normal(scale=2.0, size=n_cats)
    cat_idx = rng.integers(0, n_cats, size=n)
    x_num = rng.normal(size=n)
    cats = pd.Categorical([f"id_{i}" for i in cat_idx])
    cats[np.arange(n) % 40 == 0] = np.nan

    raw_eff = np.where(pd.isna(cats), 0.0, cat_effect[cat_idx])
    y = 1.5 * raw_eff + 1.5 * x_num + 0.1 * rng.normal(size=n)

    df = pd.DataFrame({"cat": cats, "x": x_num})
    df_tr, df_val, df_te = df.iloc[:2800], df.iloc[2800:3200], df.iloc[3200:]
    y_tr, y_val, y_te = y[:2800], y[2800:3200], y[3200:]

    m_base = MiterRegressor(
        n_estimators=150, learning_rate=0.05, early_stopping_rounds=20, n_jobs=1, random_state=0
    ).fit(df_tr, y_tr, eval_set=[(df_val, y_val)])
    m_ts = MiterRegressor(
        n_estimators=150,
        learning_rate=0.05,
        cat_oblique_ts=True,
        early_stopping_rounds=20,
        n_jobs=1,
        random_state=0,
    ).fit(df_tr, y_tr, eval_set=[(df_val, y_val)])

    rmse_base = float(np.sqrt(np.mean((m_base.predict(df_te) - y_te) ** 2)))
    rmse_ts = float(np.sqrt(np.mean((m_ts.predict(df_te) - y_te) ** 2)))
    assert rmse_ts < 0.75 * rmse_base
    assert m_ts.n_features_in_ == 2

    # Bitwise exactness:
    # 1. Without early stopping, train_scores equals Evaluator on deterministic OOF features bit for bit.
    from miter.cat_ts import CatObliqueTSEncoder

    m_ts_full = MiterRegressor(
        n_estimators=25,
        learning_rate=0.05,
        cat_oblique_ts=True,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(df_tr, y_tr)
    X_oof_tr = CatObliqueTSEncoder(smooth=10.0, n_folds=5, random_state=0).fit_transform(
        df_tr, y_tr, None, [0], is_binary=False
    )
    raw_oof_eval = _core.predict_raw_mixed(
        m_ts_full._model,
        len(df_tr),
        [X_oof_tr["x"].to_numpy(dtype=np.float64), X_oof_tr["__miter_cat_ts_0_cat"].to_numpy(dtype=np.float64)],
        [list(df_tr["cat"].cat.categories.astype(str))],
        [np.ascontiguousarray(df_tr["cat"].cat.codes.to_numpy(dtype=np.int32))],
        0,
    )
    np.testing.assert_array_equal(m_ts_full._train_scores_, raw_oof_eval)

    # 2. predict (LignumPredictor on transform(df_te)) equals Evaluator bit for bit.
    assert m_ts._cat_ts_encoder is not None
    X_aug_te = m_ts._cat_ts_encoder.transform(df_te)
    raw_te_eval = _core.predict_raw_mixed(
        m_ts._model,
        len(df_te),
        [X_aug_te["x"].to_numpy(dtype=np.float64), X_aug_te["__miter_cat_ts_0_cat"].to_numpy(dtype=np.float64)],
        [list(df_te["cat"].cat.categories.astype(str))],
        [np.ascontiguousarray(df_te["cat"].cat.codes.to_numpy(dtype=np.int32))],
        0,
    )
    np.testing.assert_array_equal(m_ts.predict(df_te), raw_te_eval)

    unpickled = pickle.loads(pickle.dumps(m_ts))
    np.testing.assert_array_equal(m_ts.predict(df_te), unpickled.predict(df_te))

    m_ts_t4 = MiterRegressor(
        n_estimators=150,
        learning_rate=0.05,
        cat_oblique_ts=True,
        early_stopping_rounds=20,
        n_jobs=4,
        random_state=0,
    ).fit(df_tr, y_tr, eval_set=[(df_val, y_val)])
    np.testing.assert_array_equal(m_ts.predict(df_te), m_ts_t4.predict(df_te))


def test_cat_oblique_ts_classifier_and_validation_errors() -> None:
    rng = np.random.default_rng(19)
    n = 600
    X_np = np.column_stack([rng.normal(size=n), rng.choice(["k0", "k1", "k2", "k3"], size=n)]).astype(object)
    y = ((X_np[:, 0].astype(float) + (X_np[:, 1] == "k0").astype(float)) > 0.3).astype(int)

    clf = MiterClassifier(
        n_estimators=20,
        cat_oblique_ts=True,
        cat_ts_smooth=5.0,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=1,
    ).fit(X_np[:450], y[:450], categorical_feature=[1])
    proba = clf.predict_proba(X_np[450:])
    assert proba.shape == (150, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert clf.n_features_in_ == 2

    with pytest.raises(NotImplementedError, match="save_model"):
        clf.save_model("unused.json")
    with pytest.raises(ValueError, match="cat_ts_smooth"):
        MiterRegressor(cat_oblique_ts=True, cat_ts_smooth=0.0).fit(X_np[:50], y[:50], categorical_feature=[1])


@pytest.mark.parametrize("n_classes,expected_targets", [(4, 4), (10, 8)])
def test_multiclass_cat_oblique_ts_cap_and_numpy_end_to_end(
    n_classes: int, expected_targets: int
) -> None:
    rng = np.random.default_rng(77 + n_classes)
    n = 700
    c0 = rng.choice([f"g{k}" for k in range(n_classes)], size=n)
    c1 = rng.choice(["m0", "m1", "m2"], size=n)
    y = (
        (np.array([int(s[1:]) for s in c0]) + np.array([int(s[1:]) for s in c1])) % n_classes
    ).astype(np.int64)
    X = np.column_stack([rng.normal(size=n), c0, c1]).astype(object)

    clf = MiterClassifier(
        n_estimators=20,
        learning_rate=0.1,
        cat_oblique_ts=True,
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
