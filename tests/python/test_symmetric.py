"""Tests for GrowPolicy::Symmetric (oblivious + oblique trees)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from miter import MiterClassifier, MiterRegressor


def _reg_data(n: int = 800, p: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[::17, 1] = np.nan
    y = 2.0 * X[:, 0] - np.nan_to_num(X[:, 1]) + 0.8 * X[:, 0] * X[:, 2] + 0.1 * rng.normal(size=n)
    return X, y


def test_symmetric_regressor_and_classifier_roundtrip(tmp_path: Path) -> None:
    X, y = _reg_data()
    reg = MiterRegressor(
        grow_policy="symmetric",
        max_depth=6,
        min_data_in_leaf=1,
        lambda_l2=3.0,
        n_estimators=12,
        early_stopping_rounds=None,
        n_jobs=2,
        random_state=7,
    ).fit(X, y)
    p1 = reg.predict(X)
    assert p1.shape == (len(X),)

    # Thread invariance and JSON/pickle round trip.
    reg_t1 = MiterRegressor(
        grow_policy="symmetric",
        max_depth=6,
        min_data_in_leaf=1,
        lambda_l2=3.0,
        n_estimators=12,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=7,
    ).fit(X, y)
    np.testing.assert_array_equal(p1, reg_t1.predict(X))

    model_path = tmp_path / "sym.json"
    reg.save_model(str(model_path))
    reloaded = MiterRegressor.load_model(str(model_path))
    assert reloaded.grow_policy == "symmetric"
    np.testing.assert_array_equal(p1, reloaded.predict(X))

    unpickled = pickle.loads(pickle.dumps(reg))
    assert unpickled.grow_policy == "symmetric"
    np.testing.assert_array_equal(p1, unpickled.predict(X))

    # Mixed DataFrame with categoricals + binary objective.
    rng = np.random.default_rng(11)
    n_clf = 600
    df = pd.DataFrame(
        {
            "x0": np.where(rng.random(n_clf) < 0.08, np.nan, rng.normal(size=n_clf)),
            "x1": rng.normal(size=n_clf),
            "x2": rng.normal(size=n_clf),
            "c0": pd.Categorical(rng.choice(["a", "b", "c", "d", None], size=n_clf)),
        }
    )
    yb = ((np.nan_to_num(df["x0"]) - df["x1"] + (df["c0"] == "a").astype(float)) > 0.2).astype(int)
    clf = MiterClassifier(
        grow_policy="symmetric",
        max_depth=5,
        min_data_in_leaf=1,
        lambda_l2=3.0,
        n_estimators=10,
        early_stopping_rounds=None,
        n_jobs=2,
        random_state=5,
    ).fit(df, yb)
    proba = clf.predict_proba(df)
    assert proba.shape == (n_clf, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)


def test_symmetric_invalid_params_rejected() -> None:
    X, y = _reg_data(n=100, p=4)
    with pytest.raises(ValueError, match="Unknown grow_policy"):
        MiterRegressor(grow_policy="bad", n_estimators=5, early_stopping_rounds=None).fit(X, y)
    with pytest.raises(ValueError, match="max_depth"):
        MiterRegressor(grow_policy="symmetric", max_depth=9, n_estimators=5, early_stopping_rounds=None).fit(
            X, y
        )


def test_symmetric_early_stopping_and_sklearn_clone() -> None:
    from sklearn.base import clone

    X, y = _reg_data(n=1000, p=6, seed=9)
    est = MiterRegressor(
        grow_policy="symmetric",
        max_depth=5,
        min_data_in_leaf=1,
        lambda_l2=3.0,
        n_estimators=40,
        early_stopping_rounds=10,
        n_jobs=1,
        random_state=3,
    )
    cloned = clone(est)
    assert cloned.get_params()["grow_policy"] == "symmetric"
    cloned.fit(X[:800], y[:800], eval_set=[(X[800:], y[800:])])
    assert 1 <= cloned.best_iteration_ <= 40


def test_symmetric_large_n_thread_invariance() -> None:
    from miter import _core

    rng = np.random.default_rng(123)
    n = 40000
    df = pd.DataFrame(
        {
            "x0": np.where(rng.random(n) < 0.05, np.nan, rng.normal(size=n)),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "c0": pd.Categorical(rng.choice(["a", "b", "c", "d", "e", None], size=n)),
        }
    )
    y = (
        1.5 * np.nan_to_num(df["x0"])
        - 0.7 * df["x1"] * df["x2"]
        + 0.4 * (df["c0"] == "b").astype(float)
        + 0.1 * rng.normal(size=n)
    )
    m1 = MiterRegressor(
        grow_policy="symmetric",
        max_depth=6,
        n_estimators=8,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=17,
    ).fit(df, y)
    m4 = MiterRegressor(
        grow_policy="symmetric",
        max_depth=6,
        n_estimators=8,
        early_stopping_rounds=None,
        n_jobs=4,
        random_state=17,
    ).fit(df, y)
    assert json.loads(_core.model_to_json(m1._model)) == {
        **json.loads(_core.model_to_json(m4._model)),
        "params": {**json.loads(_core.model_to_json(m4._model))["params"], "n_jobs": 1},
    }
    np.testing.assert_array_equal(m1.predict(df), m4.predict(df))
    np.testing.assert_array_equal(m1._train_scores_, m1.predict(df))


def test_min_sum_hessian_in_leaf_applies_with_l2() -> None:
    """The hessian bound holds in symmetric trees whatever lambda_l2 is (it was skipped when
    lambda_l2 != 0); a side with only zero-weight rows still counts as empty, not as too light."""
    X, y = _reg_data()
    kw = dict(grow_policy="symmetric", n_estimators=5, early_stopping_rounds=None, n_jobs=1, lambda_l2=1.0)
    free = MiterRegressor(**kw).fit(X, y)
    bound = MiterRegressor(min_sum_hessian_in_leaf=1e9, **kw).fit(X, y)
    assert np.unique(free.predict(X)).size > 1
    assert np.unique(bound.predict(X)).size == 1  # no split can give each side a hessian of 1e9
    w = np.ones(len(y))
    w[::3] = 0.0
    weighted = MiterRegressor(**kw).fit(X, y, sample_weight=w)
    removed = MiterRegressor(**kw).fit(X[w > 0], y[w > 0])
    assert np.isfinite(weighted.predict(X)).all() and np.isfinite(removed.predict(X)).all()
