"""Python API tests for C++20 native online ordered CTRs (`max_ctr_complexity`)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from miter import MiterClassifier, MiterRegressor, _core


def test_native_ctr_xor_discovery_json_and_pickle(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    n = 1500
    a = rng.integers(0, 6, size=n)
    b = rng.integers(0, 6, size=n)
    noise = rng.integers(0, 8, size=n)
    y = ((a ^ b) & 1).astype(np.int64)

    df = pd.DataFrame(
        {
            "a": pd.Categorical([f"a{v}" for v in a]),
            "b": pd.Categorical([f"b{v}" for v in b]),
            "n": pd.Categorical([f"n{v}" for v in noise]),
            "x": rng.normal(scale=0.1, size=n),
        }
    )
    tr, te = slice(0, 1100), slice(1100, 1500)

    clf_off = MiterClassifier(
        n_estimators=15,
        learning_rate=0.2,
        grow_policy="symmetric",
        max_depth=2,
        min_data_in_leaf=10,
        min_data_per_group=10,
        m_mult=0,
        max_ctr_complexity=0,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=7,
    ).fit(df.iloc[tr], y[tr])

    clf_on = MiterClassifier(
        n_estimators=15,
        learning_rate=0.2,
        grow_policy="symmetric",
        max_depth=2,
        min_data_in_leaf=10,
        min_data_per_group=10,
        m_mult=0,
        max_ctr_complexity=2,
        ctr_prior_weight=2.0,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=7,
    ).fit(df.iloc[tr], y[tr])

    assert clf_off._model.n_ctr_tables == 0
    assert clf_on._model.n_ctr_tables >= 1
    assert clf_on._model.n_ctr_projections >= 1

    p_on = clf_on.predict_proba(df.iloc[te])
    acc_off = float(np.mean(clf_off.predict(df.iloc[te]) == y[te]))
    acc_on = float(np.mean(np.argmax(p_on, axis=1) == y[te]))
    assert acc_on > 0.90
    assert acc_on > acc_off + 0.15

    # JSON save/load & pickle preserve native CTRs bit-for-bit.
    json_path = tmp_path / "ctr_model.json"
    clf_on.save_model(str(json_path))
    loaded = MiterClassifier.load_model(str(json_path))
    np.testing.assert_array_equal(p_on, loaded.predict_proba(df.iloc[te]))
    unpickled = pickle.loads(pickle.dumps(clf_on))
    np.testing.assert_array_equal(p_on, unpickled.predict_proba(df.iloc[te]))


def test_native_ctr_default_bit_identical_and_validation() -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"c": pd.Categorical(rng.choice(["p", "q", "r"], size=120)), "x": rng.normal(size=120)}
    )
    y = rng.normal(size=120)

    r_def = MiterRegressor(
        n_estimators=8, grow_policy="symmetric", max_depth=3, early_stopping_rounds=None, n_jobs=1
    ).fit(df, y)
    r_zero = MiterRegressor(
        n_estimators=8,
        grow_policy="symmetric",
        max_depth=3,
        max_ctr_complexity=0,
        early_stopping_rounds=None,
        n_jobs=1,
    ).fit(df, y)
    assert _core.model_to_json(r_def._model) == _core.model_to_json(r_zero._model)

    with pytest.raises(ValueError, match="max_ctr_complexity"):
        MiterRegressor(max_ctr_complexity=4).fit(df, y)
    with pytest.raises(ValueError, match="ctr_prior_weight"):
        MiterRegressor(max_ctr_complexity=1, ctr_prior_weight=0.0).fit(df, y)
