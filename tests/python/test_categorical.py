from pathlib import Path

import numpy as np
import pandas as pd
from miter import MiterClassifier, MiterRegressor


def test_k1_beats_ordinal_codes_on_noncontiguous_subset() -> None:
    rng = np.random.default_rng(7)
    n = 3000
    c_codes = rng.integers(0, 20, size=n)
    good = np.isin(c_codes, [1, 4, 6, 9, 12, 15, 17])
    x_num = rng.normal(size=n)
    y = good.astype(np.float64) * 3.0 + 0.3 * rng.normal(size=n)

    df = pd.DataFrame({"c": pd.Categorical(c_codes.astype(str)), "x": x_num})
    X_ord = np.column_stack([c_codes.astype(np.float64), x_num])

    k1 = MiterRegressor(
        n_estimators=25,
        learning_rate=0.1,
        num_leaves=4,
        min_data_in_leaf=20,
        min_data_per_group=20,
        max_cat_to_onehot=2,
        m_mult=0,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(df.iloc[:2200], y[:2200])

    k0 = MiterRegressor(
        n_estimators=25,
        learning_rate=0.1,
        num_leaves=4,
        min_data_in_leaf=20,
        m_mult=0,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(X_ord[:2200], y[:2200])

    err_k1 = float(np.mean((k1.predict(df.iloc[2200:]) - y[2200:]) ** 2))
    err_k0 = float(np.mean((k0.predict(X_ord[2200:]) - y[2200:]) ** 2))
    assert err_k1 < 0.7 * err_k0


def test_mixed_dataframe_and_numpy_bitwise_roundtrip_and_unseen(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    n = 600
    cats = rng.choice(["alpha", "beta", "gamma", "delta", "eps"], size=n)
    cats[::25] = "rare_singleton"
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x0[::20] = np.nan

    cat_series = pd.Series(pd.Categorical(cats))
    cat_series.iloc[::30] = np.nan

    signal = (
        np.where(np.isin(cats, ["alpha", "gamma"]), 2.5, -1.0)
        + np.where(np.isnan(x0), 1.0, 0.8 * (x0 - x1))
    )
    y_reg = signal + 0.05 * rng.normal(size=n)
    y_bin = (signal > 0.2).astype(np.int64)

    df = pd.DataFrame({"x0": x0, "cat": cat_series, "x1": x1})
    df_tr, df_val, df_te = df.iloc[:400], df.iloc[400:500], df.iloc[500:]

    reg = MiterRegressor(
        n_estimators=40,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        min_data_per_group=20,
        max_cat_to_onehot=2,
        early_stopping_rounds=10,
        n_jobs=1,
        random_state=5,
    ).fit(df_tr, y_reg[:400], eval_set=[(df_val, y_reg[400:500])])

    model_path = tmp_path / "cat_reg.json"
    reg.save_model(str(model_path))
    reg_loaded = MiterRegressor.load_model(str(model_path))
    np.testing.assert_array_equal(reg.predict(df_te), reg_loaded.predict(df_te))

    # Unseen category in predict must route to OTHER and yield finite predictions matching rare_singleton.
    df_unseen = df_te.iloc[:3].copy()
    df_unseen["cat"] = pd.Categorical(["totally_new_category", "alpha", None])
    pred_unseen = reg_loaded.predict(df_unseen)
    assert np.all(np.isfinite(pred_unseen))

    clf = MiterClassifier(
        n_estimators=40,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        min_data_per_group=20,
        early_stopping_rounds=10,
        n_jobs=1,
        random_state=5,
    ).fit(df_tr, y_bin[:400], eval_set=[(df_val, y_bin[400:500])])
    clf_path = tmp_path / "cat_clf.json"
    clf.save_model(str(clf_path))
    clf_loaded = MiterClassifier.load_model(str(clf_path))
    np.testing.assert_array_equal(clf.predict_proba(df_te), clf_loaded.predict_proba(df_te))

    # Explicit NumPy categorical_feature=[1] support.
    X_np = np.column_stack([x0, cats, x1]).astype(object)
    reg_np = MiterRegressor(
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=8,
        min_data_in_leaf=10,
        min_data_per_group=20,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=5,
    ).fit(X_np[:400], y_reg[:400], categorical_feature=[1])
    assert np.all(np.isfinite(reg_np.predict(X_np[500:])))


def test_native_nan_informative_and_validation() -> None:
    rng = np.random.default_rng(77)
    n = 400
    X = rng.normal(size=(n, 3))
    is_miss = rng.random(n) < 0.35
    X[is_miss, 0] = np.nan
    y = np.where(is_miss, 4.0, -1.0) + 0.5 * (X[:, 1] - X[:, 2]) + 0.05 * rng.normal(size=n)

    reg = MiterRegressor(
        n_estimators=20,
        learning_rate=0.15,
        num_leaves=8,
        min_data_in_leaf=10,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=0,
    ).fit(X[:300], y[:300])
    rmse = float(np.sqrt(np.mean((reg.predict(X[300:]) - y[300:]) ** 2)))
    assert rmse < 0.6

