from pathlib import Path

import miter
import numpy as np
from miter import MiterClassifier, MiterRegressor


def test_oblique_beats_axis_on_diagonal_boundary() -> None:
    rng = np.random.default_rng(17)
    X = rng.uniform(-1.0, 1.0, size=(600, 2))
    y = ((X[:, 0] + X[:, 1]) > 0.0).astype(np.int64)
    X_tr, y_tr = X[:450], y[:450]
    X_te, y_te = X[450:], y[450:]

    ax = MiterClassifier(
        n_estimators=1, num_leaves=4, min_data_in_leaf=5, m_mult=0, random_state=0
    ).fit(X_tr, y_tr)
    obl = MiterClassifier(
        n_estimators=1, num_leaves=4, min_data_in_leaf=5, m_mult=8, random_state=0
    ).fit(X_tr, y_tr)

    acc_ax = float(np.mean(ax.predict(X_te) == y_te))
    acc_obl = float(np.mean(obl.predict(X_te) == y_te))
    assert acc_obl > acc_ax + 0.05
    assert acc_obl > 0.90


def test_oblique_m_mult_kproj_nan_determinism_and_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2028)
    X = rng.normal(size=(240, 5))
    X[::12, 3] = np.nan
    y = 2.0 * (X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]) + 0.1 * rng.normal(size=240)

    for m_mult, k_proj in [(None, 0), (2, 0), (8, 3)]:
        reg1 = MiterRegressor(
            n_estimators=15,
            learning_rate=0.1,
            num_leaves=8,
            min_data_in_leaf=5,
            m_mult=m_mult,
            k_proj=k_proj,
            random_state=11,
        ).fit(X, y)
        reg2 = MiterRegressor(
            n_estimators=15,
            learning_rate=0.1,
            num_leaves=8,
            min_data_in_leaf=5,
            m_mult=m_mult,
            k_proj=k_proj,
            random_state=11,
        ).fit(X, y)

        p1 = reg1.predict(np.ascontiguousarray(X))
        p2 = reg2.predict(np.asfortranarray(X))
        np.testing.assert_array_equal(p1, reg1._train_scores_)
        np.testing.assert_array_equal(p1, p2)

        path = tmp_path / f"m_{m_mult}_{k_proj}.json"
        reg1.save_model(str(path))
        loaded = MiterRegressor.load_model(str(path))
        np.testing.assert_array_equal(loaded.predict(X), p1)
        assert loaded.predict(X).shape == (240,)
        assert miter._core.model_to_json(reg1._model) == miter._core.model_to_json(reg2._model)
