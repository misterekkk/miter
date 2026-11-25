"""predict() goes through lignum (M6); results must equal miter's own evaluator bit for bit."""

import numpy as np
import pandas as pd
from miter import MiterClassifier, MiterRegressor, _core


def _data(n: int = 3000, p: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 0] = np.exp(2.0 * X[:, 0])  # heavy tail: saturated codes
    X[rng.random((n, p)) < 0.03] = np.nan
    x1 = np.nan_to_num(X[:, 1])
    y = np.sin(np.nan_to_num(X[:, 0])) + x1 * np.nan_to_num(X[:, 2]) + 0.1 * rng.normal(size=n)
    return X, y


def _test_rows(X: np.ndarray) -> np.ndarray:
    rows = X[:500].copy()
    specials = [np.inf, -np.inf, 1e300, -1e300, 0.0, -0.0, np.nan]
    for i in range(70):
        rows[i, i % X.shape[1]] = specials[i % len(specials)]
    return rows


def _same_bits(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def test_regressor_lignum_equals_evaluator() -> None:
    X, y = _data()
    est = MiterRegressor(n_estimators=40, early_stopping_rounds=None, n_jobs=2, random_state=1).fit(X, y)
    model = est._model
    rows = _test_rows(X)
    ref = _core.predict_raw_f64(model, np.ascontiguousarray(rows), 0)
    _same_bits(est.predict(rows), ref)
    # Layouts and dtypes: F-order and float32 (widened exactly) give the evaluator's results too.
    _same_bits(est.predict(np.asfortranarray(rows)), ref)
    with np.errstate(over="ignore"):  # 1e300 -> inf in float32, on purpose
        rows32 = rows.astype(np.float32)
    _same_bits(est.predict(rows32), _core.predict_raw_f32(model, rows32, 0))
    # num_iteration
    _same_bits(est.predict(rows, num_iteration=7), _core.predict_raw_f64(model, np.ascontiguousarray(rows), 7))
    predictor = est._lignum(0)
    assert predictor is not None and predictor.binned and predictor.n_oblique > 0


def test_classifier_proba_lignum_equals_evaluator() -> None:
    X, y = _data(seed=3)
    yb = (y > np.median(y)).astype(np.int64)
    est = MiterClassifier(n_estimators=30, early_stopping_rounds=None, n_jobs=1, random_state=2).fit(X, yb)
    rows = np.ascontiguousarray(_test_rows(X))
    _same_bits(est.predict_proba(rows), _core.predict_proba_f64(est._model, rows, 0))
    np.testing.assert_array_equal(est.predict(rows), np.argmax(_core.predict_proba_f64(est._model, rows, 0), axis=1))


def test_categorical_dataframe_lignum_equals_evaluator() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    codes = rng.integers(0, 15, size=n)
    x = rng.normal(size=n)
    y = np.isin(codes, [1, 4, 6, 9]).astype(np.float64) * 2.0 + x + 0.2 * rng.normal(size=n)
    cats = pd.Categorical(np.where(rng.random(n) < 0.05, None, codes.astype(str)))
    df = pd.DataFrame({"c": cats, "x": x, "z": rng.normal(size=n)})
    est = MiterRegressor(
        n_estimators=30, min_data_per_group=20, early_stopping_rounds=None, n_jobs=1, random_state=0
    ).fit(df, y)
    # Unseen categories and missing values in the input.
    test = df.iloc[:400].copy()
    test["c"] = test["c"].cat.add_categories(["never_seen"])
    test.loc[test.index[:30], "c"] = "never_seen"
    test.loc[test.index[30:50], "c"] = None
    from miter.booster import _extract_mixed_predict_inputs

    n_rows, num_cols, vocabs, indices = _extract_mixed_predict_inputs(test, est._model, fitted_on_dataframe=True)
    ref = _core.predict_raw_mixed(est._model, n_rows, num_cols, vocabs, indices, 0)
    _same_bits(est.predict(test), ref)


def test_models_lignum_cannot_hold_fall_back_to_the_evaluator() -> None:
    X, y = _data(n=1500, seed=5)
    est = MiterRegressor(n_estimators=15, k_proj=4, early_stopping_rounds=None, n_jobs=1, random_state=0).fit(
        np.nan_to_num(X), y
    )
    rows = np.ascontiguousarray(np.nan_to_num(X[:200]))
    assert est._lignum(0) is None  # oblique splits with k = 4: miter's evaluator predicts
    _same_bits(est.predict(rows), _core.predict_raw_f64(est._model, rows, 0))


def test_refit_rebuilds_the_lignum_model() -> None:
    X, y = _data(n=1500, seed=8)
    est = MiterRegressor(n_estimators=10, early_stopping_rounds=None, n_jobs=1, random_state=0)
    est.fit(X, y)
    first = est.predict(X[:50])
    est.random_state = 1
    est.fit(X, y + 1.0)
    second = est.predict(X[:50])
    _same_bits(second, _core.predict_raw_f64(est._model, np.ascontiguousarray(X[:50]), 0))
    assert not np.array_equal(first, second)
