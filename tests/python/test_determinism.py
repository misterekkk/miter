from pathlib import Path

import numpy as np
import pandas as pd
from miter import MiterClassifier, MiterRegressor


def test_python_bitwise_determinism_across_threads(tmp_path: Path) -> None:
    rng = np.random.default_rng(314)
    n = 10000
    X = rng.normal(size=(n, 10))
    X[::19, 2] = np.nan
    y = X[:, 0] - 0.8 * X[:, 1] + 0.2 * rng.normal(size=n)

    ref_bytes: bytes | None = None
    ref_pred: np.ndarray | None = None
    for threads in (1, 2, 4, 8):
        reg = MiterRegressor(
            n_estimators=8,
            num_leaves=15,
            min_data_in_leaf=20,
            early_stopping_rounds=None,
            n_jobs=threads,
            random_state=7,
        ).fit(X, y)
        p = tmp_path / f"reg_{threads}.json"
        reg.save_model(str(p))
        raw_bytes = p.read_bytes()
        # Normalize only n_jobs inside params metadata so structure and splits compare byte-for-byte.
        norm = raw_bytes.decode("utf-8").replace(f'"n_jobs": {threads}', '"n_jobs": 0')
        pred = reg.predict(X)
        if ref_bytes is None:
            ref_bytes = norm.encode("utf-8")
            ref_pred = pred
        else:
            assert norm.encode("utf-8") == ref_bytes
            assert ref_pred is not None
            np.testing.assert_array_equal(pred, ref_pred)


def test_python_mixed_binary_determinism_1_vs_8(tmp_path: Path) -> None:
    rng = np.random.default_rng(999)
    n = 9000
    cats = rng.choice(["a", "b", "c", "d", "e"], size=n)
    cat_s = pd.Series(pd.Categorical(cats))
    cat_s.iloc[::29] = np.nan
    x0 = rng.normal(size=n)
    x0[::17] = np.nan
    df = pd.DataFrame({
        "cat": cat_s,
        "x0": x0,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    yb = ((df["cat"] == "b").to_numpy(dtype=float) + df["x1"].to_numpy() > 0.5).astype(int)

    m1 = MiterClassifier(
        n_estimators=6,
        num_leaves=12,
        min_data_per_group=20,
        early_stopping_rounds=None,
        n_jobs=1,
        random_state=11,
    ).fit(df, yb)
    m8 = MiterClassifier(
        n_estimators=6,
        num_leaves=12,
        min_data_per_group=20,
        early_stopping_rounds=None,
        n_jobs=8,
        random_state=11,
    ).fit(df, yb)

    p1 = tmp_path / "m1.json"
    p8 = tmp_path / "m8.json"
    m1.save_model(str(p1))
    m8.save_model(str(p8))
    assert (
        p1.read_text(encoding="utf-8").replace('"n_jobs": 1', '"n_jobs": 0')
        == p8.read_text(encoding="utf-8").replace('"n_jobs": 8', '"n_jobs": 0')
    )
    np.testing.assert_array_equal(m1.predict_proba(df), m8.predict_proba(df))
