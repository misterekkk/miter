"""Tests for the optional feature_fraction / bagging_fraction / extra_trees / robust switches."""

from __future__ import annotations

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor


def _data(n: int = 1500, p: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[::13, 2] = np.nan
    y = X[:, 0] * X[:, 1] + np.nan_to_num(X[:, 2]) + 0.1 * rng.normal(size=n)
    return X, y


def _fit(X: np.ndarray, y: np.ndarray, **kw: object) -> MiterRegressor:
    params: dict[str, object] = dict(
        n_estimators=15, early_stopping_rounds=None, n_jobs=1, random_state=3
    )
    params.update(kw)
    return MiterRegressor(**params).fit(X, y)  # type: ignore[arg-type]


OPTIONS = [
    {"feature_fraction": 0.4},
    {"bagging_fraction": 0.6},
    {"extra_trees": True},
    {"feature_fraction": 0.5, "bagging_fraction": 0.7, "extra_trees": True},
]


def test_explicit_defaults_match_unset() -> None:
    X, y = _data()
    ref = _fit(X, y).predict(X)
    got = _fit(
        X, y, feature_fraction=1.0, bagging_fraction=1.0, extra_trees=False
    ).predict(X)
    np.testing.assert_array_equal(ref, got)


@pytest.mark.parametrize("opts", OPTIONS)
def test_option_changes_predictions(opts: dict[str, object]) -> None:
    X, y = _data()
    ref = _fit(X, y).predict(X)
    got = _fit(X, y, **opts).predict(X)
    assert not np.array_equal(ref, got)
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize("opts", OPTIONS)
def test_same_seed_identical_different_seed_differs(opts: dict[str, object]) -> None:
    X, y = _data()
    a = _fit(X, y, random_state=1, **opts).predict(X)
    b = _fit(X, y, random_state=1, **opts).predict(X)
    c = _fit(X, y, random_state=2, **opts).predict(X)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("opts", OPTIONS)
def test_threads_do_not_change_model(opts: dict[str, object]) -> None:
    X, y = _data(n=6000)
    ref = _fit(X, y, n_jobs=1, **opts).predict(X)
    for n_jobs in (2, 8):
        np.testing.assert_array_equal(ref, _fit(X, y, n_jobs=n_jobs, **opts).predict(X))


def test_classifier_accepts_options() -> None:
    X, y = _data()
    yb = (y > np.median(y)).astype(int)
    clf = MiterClassifier(
        n_estimators=10,
        early_stopping_rounds=None,
        n_jobs=1,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        extra_trees=True,
    ).fit(X, yb)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (clf.predict(X) == yb).mean() > 0.6


def test_bagging_still_learns() -> None:
    X, y = _data()
    pred = _fit(X, y, n_estimators=60, bagging_fraction=0.5).predict(X)
    assert np.corrcoef(pred, y)[0, 1] > 0.7


@pytest.mark.parametrize("name", ["feature_fraction", "bagging_fraction"])
@pytest.mark.parametrize("bad", [0.0, -0.5, 1.01, float("nan"), float("inf")])
def test_fraction_ranges_validated(name: str, bad: float) -> None:
    X, y = _data(n=200)
    with pytest.raises(ValueError, match=name):
        _fit(X, y, **{name: bad})


def test_robust_not_available() -> None:
    X, y = _data(n=200)
    with pytest.raises(ValueError, match="robust=True is not available yet"):
        _fit(X, y, robust=True)
    _fit(X, y, robust=False)


def test_options_survive_pickle() -> None:
    import pickle

    X, y = _data()
    m = _fit(X, y, feature_fraction=0.5, bagging_fraction=0.7, extra_trees=True)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))
    assert m2.get_params()["feature_fraction"] == 0.5
    assert m2.get_params()["extra_trees"] is True
