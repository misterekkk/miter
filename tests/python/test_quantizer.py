"""Python API tests for adaptive piecewise-linear quantile Q16 quantization."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from miter import MiterClassifier, MiterRegressor, _core


def test_quantizer_linear_default_auto_json_and_pickle(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    n = 1200
    body = rng.uniform(0.001, 0.1, size=n)
    outliers = rng.choice(n, size=12, replace=False)
    body[outliers] = 1e9
    clean = rng.normal(size=n)
    X = np.column_stack([body, clean])
    y = ((body < 0.05).astype(np.int64) ^ (clean > 0.0).astype(np.int64))

    tr, te = slice(0, 900), slice(900, n)

    clf_def = MiterClassifier(
        n_estimators=20,
        grow_policy="symmetric",
        max_depth=4,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=3,
    ).fit(X[tr], y[tr])
    clf_lin = MiterClassifier(
        n_estimators=20,
        grow_policy="symmetric",
        max_depth=4,
        quantizer="linear",
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=3,
    ).fit(X[tr], y[tr])
    assert clf_def._model.n_quantile_features == 0
    assert _core.model_to_json(clf_def._model) == _core.model_to_json(clf_lin._model)

    clf_auto = MiterClassifier(
        n_estimators=20,
        grow_policy="symmetric",
        max_depth=4,
        quantizer="auto",
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=3,
    ).fit(X[tr], y[tr])
    assert clf_auto._model.n_quantile_features == 1

    p_lin = clf_lin.predict_proba(X[te])
    p_auto = clf_auto.predict_proba(X[te])
    assert float(np.mean(np.argmax(p_auto, axis=1) == y[te])) > 0.85
    assert float(np.mean(np.argmax(p_auto, axis=1) == y[te])) > float(
        np.mean(np.argmax(p_lin, axis=1) == y[te])
    )

    path = tmp_path / "auto_quant_model.json"
    clf_auto.save_model(str(path))
    loaded = MiterClassifier.load_model(str(path))
    assert loaded.quantizer == "auto"
    np.testing.assert_array_equal(p_auto, loaded.predict_proba(X[te]))
    unpickled = pickle.loads(pickle.dumps(clf_auto))
    np.testing.assert_array_equal(p_auto, unpickled.predict_proba(X[te]))

    with pytest.raises(ValueError, match="quantizer"):
        MiterRegressor(quantizer="bogus").fit(X, y.astype(np.float64))
