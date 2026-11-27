"""oblique_sample_rows: oblique directions chosen on a strided sample, thresholds exact on all rows."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor


def _data(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    X[rng.random((n, 8)) < 0.02] = np.nan
    y = np.sin(np.nan_to_num(X[:, 0] + 0.7 * X[:, 1] - X[:, 2])) + 0.1 * rng.standard_normal(n)
    return X, y


def test_threshold_above_n_is_identical_to_off() -> None:
    X, y = _data()
    a = MiterRegressor(n_estimators=30).fit(X, y).predict(X)
    b = MiterRegressor(n_estimators=30, oblique_sample_rows=len(X)).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_jobs", [1, 2, 4])
def test_deterministic_across_threads(n_jobs) -> None:
    X, y = _data()
    ref = MiterRegressor(n_estimators=30, oblique_sample_rows=1000, n_jobs=1).fit(X, y).predict(X)
    out = MiterRegressor(n_estimators=30, oblique_sample_rows=1000, n_jobs=n_jobs).fit(X, y).predict(X)
    np.testing.assert_array_equal(ref, out)


def test_quality_close_to_full_search() -> None:
    X, y = _data(8000, 1)
    Xt, yt = _data(4000, 2)
    full = MiterRegressor(n_estimators=150).fit(X, y)
    samp = MiterRegressor(n_estimators=150, oblique_sample_rows=1000).fit(X, y)
    e_full = np.sqrt(np.mean((full.predict(Xt) - yt) ** 2))
    e_samp = np.sqrt(np.mean((samp.predict(Xt) - yt) ** 2))
    assert e_samp < 1.1 * e_full


def test_round_trip_and_validation(tmp_path) -> None:
    X, y = _data()
    clf = MiterClassifier(n_estimators=20, oblique_sample_rows=1000).fit(X, (y > 0).astype(int))
    path = tmp_path / "c.json"
    clf.save_model(str(path))
    loaded = MiterClassifier.load_model(str(path))
    assert loaded.get_params()["oblique_sample_rows"] == 1000
    np.testing.assert_array_equal(loaded.predict_proba(X), clf.predict_proba(X))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(clf)).predict_proba(X), clf.predict_proba(X))
    with pytest.raises(ValueError, match="oblique_sample_rows"):
        MiterRegressor(oblique_sample_rows=-5).fit(X, y)
