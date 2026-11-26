from __future__ import annotations

import pickle
from pathlib import Path

import miter
import numpy as np
import pandas as pd
import pytest


def _multiclass_data(
    n: int = 450, n_classes: int = 4, seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=n)
    X1 = rng.normal(size=n)
    cats = np.array(["a", "b", "c", "d"])[rng.integers(0, 4, size=n)]
    score = X0 + 0.7 * X1 + (cats == "b") * 0.8 - (cats == "d") * 0.8
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y_idx = np.searchsorted(edges, score)
    X0[::17] = np.nan
    df = pd.DataFrame(
        {
            "x0": X0,
            "x1": X1,
            "x2": rng.normal(size=n),
            "c0": pd.Categorical(cats),
        }
    )
    labels = np.array([f"cls_{k}" for k in y_idx])
    return df, labels


@pytest.mark.parametrize("policy", ["leafwise", "symmetric"])
def test_multiclass_classifier_end_to_end(policy: str, tmp_path: Path) -> None:
    df, y = _multiclass_data(n=500, n_classes=4, seed=11)
    tr_df, va_df = df.iloc[:350], df.iloc[350:]
    tr_y, va_y = y[:350], y[350:]

    clf = miter.MiterClassifier(
        n_estimators=40,
        learning_rate=0.1,
        grow_policy=policy,
        num_leaves=15,
        max_depth=4,
        min_data_in_leaf=1 if policy == "symmetric" else 10,
        min_data_per_group=10,
        early_stopping_rounds=10,
        n_jobs=1,
        random_state=7,
    ).fit(tr_df, tr_y, eval_set=[(va_df, va_y)])

    assert clf.feature_importances_[3] > 0
    proba = clf.predict_proba(va_df)
    assert proba.shape == (len(va_df), 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    preds = clf.predict(va_df)
    assert set(preds) <= set(clf.classes_)
    acc = float(np.mean(preds == va_y))
    assert acc > 0.70

    # Thread determinism n_jobs=1 vs n_jobs=4 on pure numeric view
    X_num = tr_df[["x0", "x1", "x2"]].to_numpy(dtype=np.float64)
    c1 = miter.MiterClassifier(
        n_estimators=10, grow_policy=policy, early_stopping_rounds=None, n_jobs=1, random_state=3
    ).fit(X_num, tr_y)
    c4 = miter.MiterClassifier(
        n_estimators=10, grow_policy=policy, early_stopping_rounds=None, n_jobs=4, random_state=3
    ).fit(X_num, tr_y)
    np.testing.assert_array_equal(c1.predict_proba(X_num), c4.predict_proba(X_num))

    # Pickle and JSON roundtrip
    unpickled = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(unpickled.predict_proba(va_df), proba)
    np.testing.assert_array_equal(unpickled.predict(va_df), preds)

    p_json = tmp_path / f"mc_{policy}.json"
    c1.save_model(str(p_json))
    loaded = miter.MiterClassifier.load_model(str(p_json))
    np.testing.assert_array_equal(loaded.predict_proba(X_num), c1.predict_proba(X_num))


def test_multiclass_unseen_eval_label_rejected() -> None:
    X = np.random.default_rng(0).normal(size=(60, 3))
    y = np.tile([0, 1, 2], 20)
    clf = miter.MiterClassifier(n_estimators=5, early_stopping_rounds=3)
    with pytest.raises(ValueError, match="unseen"):
        clf.fit(X[:40], y[:40], eval_set=[(X[40:], np.full(20, 3))])
