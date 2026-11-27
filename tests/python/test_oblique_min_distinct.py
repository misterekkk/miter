"""oblique_min_distinct: numeric features with too few distinct values stay out of oblique projections."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    binary = rng.integers(0, 2, size=(2000, 6)).astype(np.float64)
    cont = rng.standard_normal((2000, 2))
    X = np.hstack([binary, cont])
    y = binary[:, 0] + binary[:, 1] - binary[:, 2] + cont[:, 0] - 0.5 * cont[:, 1] + 0.1 * rng.standard_normal(2000)
    return X, y


def _oblique_features(model: MiterRegressor, tmp_path) -> set[int]:
    path = tmp_path / "m.json"
    model.save_model(str(path))
    feats: set[int] = set()

    def walk(o) -> None:
        if isinstance(o, dict):
            if o.get("kind") == "oblique":
                for key in ("features", "oblique_features"):
                    if key in o:
                        feats.update(int(f) for f in o[key])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(path.read_text()))
    return feats


@pytest.mark.parametrize("grow_policy", ["symmetric", "leafwise"])
def test_binary_features_leave_the_oblique_pool(grow_policy, tmp_path) -> None:
    X, y = _data()
    off = MiterRegressor(n_estimators=40, grow_policy=grow_policy).fit(X, y)
    on = MiterRegressor(n_estimators=40, grow_policy=grow_policy, oblique_min_distinct=3).fit(X, y)
    assert _oblique_features(off, tmp_path) & set(range(6))  # binary features are used by default
    assert not (_oblique_features(on, tmp_path) & set(range(6)))  # and never with oblique_min_distinct=3


def test_zero_is_identical_to_default() -> None:
    X, y = _data(1)
    a = MiterRegressor(n_estimators=30).fit(X, y).predict(X)
    b = MiterRegressor(n_estimators=30, oblique_min_distinct=0).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_round_trips_and_validation(tmp_path) -> None:
    X, y = _data(2)
    clf = MiterClassifier(n_estimators=20, oblique_min_distinct=3).fit(X, (y > 0).astype(int))
    path = tmp_path / "c.json"
    clf.save_model(str(path))
    loaded = MiterClassifier.load_model(str(path))
    assert loaded.get_params()["oblique_min_distinct"] == 3
    np.testing.assert_array_equal(loaded.predict_proba(X), clf.predict_proba(X))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(clf)).predict_proba(X), clf.predict_proba(X))
    with pytest.raises(ValueError, match="oblique_min_distinct"):
        MiterRegressor(oblique_min_distinct=-1).fit(X, y)
