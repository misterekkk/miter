"""The defaults: symmetric trees of depth 6 with the S6_M4 setting; leaf-wise keeps its own."""

from __future__ import annotations

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor, _core


@pytest.mark.parametrize("cls", [MiterRegressor, MiterClassifier])
def test_default_is_symmetric_s6_m4(cls) -> None:
    p = cls()._build_params()
    assert p.grow_policy == _core.GrowPolicy.Symmetric
    assert p.max_depth == -1  # resolved to depth 6 by the engine
    assert (p.min_data_in_leaf, p.lambda_l2, p.m_mult) == (1, 3.0, 4)
    assert p.quantizer_mode == _core.QuantizerMode.Linear
    assert p.max_ctr_complexity == 0
    assert p.oblique_sample_rows == 65536  # E2, exp_speedups_e2_prereg.md


def test_leafwise_keeps_its_defaults() -> None:
    p = MiterRegressor(grow_policy="leafwise")._build_params()
    assert p.grow_policy == _core.GrowPolicy.Leafwise
    assert (p.num_leaves, p.min_data_in_leaf, p.lambda_l2, p.m_mult) == (31, 20, 1.0, 2)


def test_explicit_values_win() -> None:
    p = MiterRegressor(min_data_in_leaf=7, lambda_l2=0.5, m_mult=3)._build_params()
    assert (p.min_data_in_leaf, p.lambda_l2, p.m_mult) == (7, 0.5, 3)


def test_presets_scale_from_the_shape_default() -> None:
    assert MiterRegressor(preset="accurate")._build_params().m_mult == 8
    assert MiterRegressor(grow_policy="leafwise", preset="accurate")._build_params().m_mult == 4
    assert MiterRegressor(preset="accurate", m_mult=5)._build_params().m_mult == 5
    fast = MiterRegressor(preset="fast")._build_params()
    assert (fast.m_mult, fast.oblique_max_depth) == (4, 1)


def test_default_model_trains_symmetric_trees() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    y = X[:, 0] - 2 * X[:, 1] + 0.1 * rng.standard_normal(400)
    reg = MiterRegressor(n_estimators=20, random_state=1).fit(X, y)
    assert reg.get_params()["grow_policy"] == "symmetric"
    assert np.corrcoef(reg.predict(X), y)[0, 1] > 0.8
