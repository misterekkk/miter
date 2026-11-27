"""efb: exclusive feature bundling of sparse numeric features in symmetric axis histograms."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor


def _onehot_data(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g1 = np.eye(8)[rng.integers(0, 8, n)]  # one-hot group 1 (mutually exclusive)
    g2 = np.eye(5)[rng.integers(0, 5, n)]  # one-hot group 2
    sparse = np.where(rng.random((n, 3)) < 0.05, rng.integers(1, 4, (n, 3)), 0).astype(float)
    dense = rng.standard_normal((n, 4))
    dense[rng.random((n, 4)) < 0.03] = np.nan
    X = np.hstack([g1, g2, sparse, dense])
    y = g1[:, 2] - g2[:, 1] + sparse[:, 0] + np.nan_to_num(dense[:, 0] - dense[:, 1]) + 0.1 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_efb_gives_the_same_model(n_jobs) -> None:
    X, y = _onehot_data()
    a = MiterRegressor(n_estimators=40, n_jobs=n_jobs).fit(X, y).predict(X)
    b = MiterRegressor(n_estimators=40, n_jobs=n_jobs, efb=True).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_efb_deterministic_across_threads() -> None:
    X, y = _onehot_data(seed=1)
    ref = MiterRegressor(n_estimators=30, efb=True, n_jobs=1).fit(X, y).predict(X)
    for nj in (2, 4):
        np.testing.assert_array_equal(ref, MiterRegressor(n_estimators=30, efb=True, n_jobs=nj).fit(X, y).predict(X))


def test_efb_with_feature_fraction_and_bagging() -> None:
    X, y = _onehot_data(seed=2)
    m = MiterRegressor(n_estimators=40, efb=True, feature_fraction=0.6, bagging_fraction=0.8).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.8


def test_efb_noop_without_sparse_features() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3000, 6))
    y = X[:, 0] - X[:, 1]
    a = MiterRegressor(n_estimators=20).fit(X, y).predict(X)
    b = MiterRegressor(n_estimators=20, efb=True).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_efb_round_trip(tmp_path) -> None:
    X, y = _onehot_data(seed=4)
    clf = MiterClassifier(n_estimators=20, efb=True).fit(X, (y > 0).astype(int))
    path = tmp_path / "c.json"
    clf.save_model(str(path))
    loaded = MiterClassifier.load_model(str(path))
    assert loaded.get_params()["efb"] is True
    np.testing.assert_array_equal(loaded.predict_proba(X), clf.predict_proba(X))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(clf)).predict_proba(X), clf.predict_proba(X))
    with pytest.raises(ValueError, match="efb"):
        MiterRegressor(efb=1).fit(X, y)
