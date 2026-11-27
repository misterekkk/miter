"""oblique_binary_penalty and oblique_gain_margin: gain multipliers for oblique candidates."""

from __future__ import annotations

import json

import numpy as np
import pytest
from miter import MiterRegressor


def _data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    b = rng.integers(0, 2, (n, 4)).astype(float)
    c = rng.standard_normal((n, 4))
    y = c[:, 0] - c[:, 1] + b[:, 0] * c[:, 2] + 0.1 * rng.standard_normal(n)
    return np.hstack([b, c]), y


def _oblique_nodes(model: MiterRegressor, tmp_path) -> list[dict]:
    path = tmp_path / "m.json"
    model.save_model(str(path))
    return [nd for t in json.loads(path.read_text())["trees"] for nd in t["nodes"] if nd.get("kind") == "oblique"]


def test_zero_is_the_default() -> None:
    X, y = _data()
    a = MiterRegressor(n_estimators=30).fit(X, y).predict(X)
    b = MiterRegressor(n_estimators=30, oblique_binary_penalty=0.0, oblique_gain_margin=0.0).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_full_binary_penalty_removes_binary_terms(tmp_path) -> None:
    X, y = _data()
    nodes = _oblique_nodes(MiterRegressor(n_estimators=30, oblique_binary_penalty=1.0).fit(X, y), tmp_path)
    assert nodes and not any(set(nd["features"]) & {0, 1, 2, 3} for nd in nodes)


def test_huge_margin_disables_oblique_splits(tmp_path) -> None:
    X, y = _data()
    assert not _oblique_nodes(MiterRegressor(n_estimators=30, oblique_gain_margin=1e9).fit(X, y), tmp_path)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_deterministic(n_jobs) -> None:
    X, y = _data(seed=1)
    kw = {"n_estimators": 30, "oblique_binary_penalty": 0.25, "oblique_gain_margin": 0.05}
    ref = MiterRegressor(n_jobs=1, **kw).fit(X, y).predict(X)
    np.testing.assert_array_equal(ref, MiterRegressor(n_jobs=n_jobs, **kw).fit(X, y).predict(X))


def test_round_trip_and_validation(tmp_path) -> None:
    X, y = _data(seed=2)
    m = MiterRegressor(n_estimators=20, oblique_binary_penalty=0.25, oblique_gain_margin=0.05).fit(X, y)
    path = tmp_path / "m.json"
    m.save_model(str(path))
    loaded = MiterRegressor.load_model(str(path))
    assert loaded.get_params()["oblique_binary_penalty"] == 0.25
    assert loaded.get_params()["oblique_gain_margin"] == 0.05
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    for bad in ({"oblique_binary_penalty": -0.1}, {"oblique_gain_margin": float("nan")}):
        with pytest.raises(ValueError):
            MiterRegressor(**bad).fit(X, y)
