"""Tests for the optional feature_fraction / bagging_fraction / extra_trees switches and k_proj."""

from __future__ import annotations

from typing import Any

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


@pytest.mark.parametrize("bad", [1, 4, 5, 8, -1, True])
def test_k_proj_is_0_2_or_3(bad: object) -> None:
    X, y = _data(n=200)
    with pytest.raises(ValueError, match="k_proj"):
        _fit(X, y, k_proj=bad)
    for ok in (0, 2, 3):
        _fit(X, y, k_proj=ok)


def test_models_saved_with_removed_options_still_load(tmp_path: Any) -> None:
    """0.1.0 wrote oblique_min_distinct / oblique_binary_penalty / oblique_gain_margin when set."""
    X, y = _data(n=300)
    m = _fit(X, y)
    path = tmp_path / "m.json"
    m.save_model(str(path))
    text = path.read_text()
    old = text.replace('"efb": 1', '"oblique_min_distinct": 3, "oblique_binary_penalty": 0.25, '
                       '"oblique_gain_margin": 0.05, "efb": 1', 1)
    assert old != text
    path.write_text(old)
    loaded = type(m).load_model(str(path))
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))


def test_options_survive_pickle() -> None:
    import pickle

    X, y = _data()
    m = _fit(X, y, feature_fraction=0.5, bagging_fraction=0.7, extra_trees=True)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(m.predict(X), m2.predict(X))
    assert m2.get_params()["feature_fraction"] == 0.5
    assert m2.get_params()["extra_trees"] is True


def test_ctr_min_cardinality_leaves_low_cardinality_features_to_k1(tmp_path: Any) -> None:
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 3000
    lo = rng.integers(0, 10, n)  # 10 categories
    hi = rng.integers(0, 200, n)  # 200 categories
    x = rng.normal(size=n)
    y = 0.3 * lo + np.sin(hi) + x + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"lo": pd.Categorical(lo), "hi": pd.Categorical(hi), "x": x})
    kw = dict(n_estimators=30, random_state=0, max_ctr_complexity=1, early_stopping_rounds=None)
    off = MiterRegressor(n_estimators=30, random_state=0, early_stopping_rounds=None).fit(X, y)
    # A threshold above every cardinality leaves no CTR candidate: same model as CTR off.
    none = MiterRegressor(ctr_min_cardinality=1000, **kw).fit(X, y)
    np.testing.assert_array_equal(none.predict(X), off.predict(X))
    some = MiterRegressor(ctr_min_cardinality=32, **kw).fit(X, y)
    path = tmp_path / "m.json"
    some.save_model(str(path))
    assert '"ctr_min_cardinality": 32' in path.read_text()
    np.testing.assert_array_equal(type(some).load_model(str(path)).predict(X), some.predict(X))
    with pytest.raises(ValueError, match="ctr_min_cardinality"):
        MiterRegressor(ctr_min_cardinality=-1).fit(X, y)


def test_enforce_min_hessian_applies_the_bound_in_symmetric_trees(tmp_path: Any) -> None:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=2000) > 1.8).astype(int)  # about 5% positives
    kw = dict(n_estimators=40, random_state=0, early_stopping_rounds=None, min_sum_hessian_in_leaf=2.0)
    loose = MiterClassifier(**kw).fit(X, y)
    strict = MiterClassifier(enforce_min_hessian=True, **kw).fit(X, y)
    # With lambda_l2 = 3 the bound was not applied in symmetric trees; enforcing it changes the model.
    assert not np.array_equal(loose.predict_proba(X), strict.predict_proba(X))
    # min_sum_hessian_in_leaf at its default without the switch: unchanged models.
    base = MiterClassifier(n_estimators=40, random_state=0, early_stopping_rounds=None).fit(X, y)
    np.testing.assert_array_equal(base.predict_proba(X),
                                  MiterClassifier(n_estimators=40, random_state=0, early_stopping_rounds=None,
                                                  enforce_min_hessian=False).fit(X, y).predict_proba(X))
    path = tmp_path / "m.json"
    strict.save_model(str(path))
    assert '"enforce_min_hessian": true' in path.read_text()
    np.testing.assert_array_equal(type(strict).load_model(str(path)).predict_proba(X), strict.predict_proba(X))
