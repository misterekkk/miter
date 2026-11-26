from pathlib import Path

import miter
import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor


def test_regressor_bitwise_exactness_and_json_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 4))
    X[:, 1] = 3.0  # constant
    X[::10, 2] = np.nan
    X[0, 0] = -6.0
    X[1, 0] = 6.0
    y = 2.5 * X[:, 0] - np.nan_to_num(X[:, 2], nan=1.0) + 0.1 * rng.normal(size=200)

    reg = MiterRegressor(
        n_estimators=40, learning_rate=0.15, num_leaves=8, min_data_in_leaf=5
    ).fit(X, y)

    pred_c = reg.predict(np.ascontiguousarray(X))
    pred_f = reg.predict(np.asfortranarray(X))
    np.testing.assert_array_equal(pred_c, reg._train_scores_)
    np.testing.assert_array_equal(pred_c, pred_f)
    assert np.sqrt(np.mean((pred_c - y) ** 2)) < 0.4 * np.std(y)

    X_probe = X.copy()
    X_probe[0, 0] = -100.0
    X_probe[1, 0] = 100.0
    pred_probe = reg.predict(X_probe)

    model_path = tmp_path / "reg.json"
    reg.save_model(str(model_path))
    loaded = MiterRegressor.load_model(str(model_path))
    np.testing.assert_array_equal(loaded.predict(X_probe), pred_probe)


def test_classifier_binary_early_stopping_and_errors(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    X = rng.normal(size=(240, 3))
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0.0).astype(np.int64)
    X_tr, y_tr = X[:180], y[:180]
    X_va, y_va = X[180:], y[180:]

    clf = MiterClassifier(
        n_estimators=200,
        learning_rate=0.2,
        num_leaves=7,
        min_data_in_leaf=5,
        early_stopping_rounds=10,
    ).fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

    assert 1 <= clf.best_iteration_ <= 200
    proba = clf.predict_proba(X_va)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-15)
    preds = clf.predict(X_va)
    assert np.mean(preds == y_va) > 0.85

    path = tmp_path / "clf.json"
    clf.save_model(str(path))
    loaded = MiterClassifier.load_model(str(path))
    np.testing.assert_array_equal(loaded.predict_proba(X_va), proba)

    with pytest.raises(ValueError, match="1 class"):
        MiterClassifier(n_estimators=5).fit(X_tr, np.zeros(len(X_tr), dtype=np.int64))

    clf3 = MiterClassifier(n_estimators=5).fit(X_tr, np.array([0, 1, 2] * 60))
    assert clf3.predict_proba(X_va).shape == (60, 3)


def test_pandas_feature_name_validation_and_sample_weight() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 0.0, 0.0]})
    y = np.array([0.0, 0.0, 10.0, 10.0])
    w = np.array([1.0, 2.0, 2.0, 1.0])

    reg = miter.MiterRegressor(n_estimators=5, learning_rate=0.5, num_leaves=2, min_data_in_leaf=1)
    reg.fit(df, y, sample_weight=w)
    np.testing.assert_array_equal(reg.predict(df), reg._train_scores_)

    with pytest.raises(ValueError, match="Feature names mismatch"):
        reg.predict(df.rename(columns={"b": "z"}))
