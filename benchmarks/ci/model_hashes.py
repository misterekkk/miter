"""Hash saved models and predictions for many configurations (bit-identity check between builds).

    python benchmarks/ci/model_hashes.py out.txt

Run it once per build (e.g. with PYTHONPATH pointing at each build) and diff the outputs: a change
that claims to be bit-identical must leave every line unchanged. Data are synthetic and seeded.
"""
import hashlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from miter import MiterClassifier, MiterRegressor

rng = np.random.default_rng(0)

def make(n, p, task, cats=0, sparse=0, missing=0.0):
    X = rng.normal(size=(n, p))
    if missing:
        X[rng.random(X.shape) < missing] = np.nan
    cols = {f"x{i}": X[:, i] for i in range(p)}
    for j in range(sparse):  # one-hot-like exclusive sparse block
        cols[f"s{j}"] = np.zeros(n)
    if sparse:
        idx = rng.integers(0, sparse * 4, size=n)
        for j in range(sparse):
            m = idx == j
            cols[f"s{j}"][m] = rng.integers(1, 5, size=m.sum())
    for j in range(cats):
        cols[f"c{j}"] = pd.Categorical(rng.integers(0, 12 + 5 * j, size=n).astype(str))
    df = pd.DataFrame(cols)
    def c(i: int) -> np.ndarray:
        return np.nan_to_num(X[:, i % p])

    z = c(0) * 1.5 + np.sin(c(1) * 2) + c(2) * c(3)
    if cats:
        z = z + df["c0"].cat.codes.to_numpy() % 3 * 0.7
    if sparse:
        z = z + 0.4 * df["s0"].to_numpy() - 0.3 * df["s1"].to_numpy()
    z = z + rng.normal(scale=0.5, size=n)
    if task == "reg":
        y = z
    elif task == "bin":
        y = (z > np.median(z)).astype(int)
    else:
        y = np.digitize(z, np.quantile(z, [0.33, 0.66]))
    return df, y

DATA = {
    "reg": make(3000, 8, "reg", missing=0.05),
    "bin": make(3000, 8, "bin", cats=2),
    "multi": make(2500, 6, "multi", cats=1, missing=0.02),
    "sparse": make(6000, 6, "reg", sparse=12),
    "big": make(80000, 10, "bin", sparse=6),
    "one_num": make(3000, 1, "multi", cats=3),
    "two_num": make(4000, 2, "reg", cats=1, missing=0.03),
    "wide": make(40000, 60, "reg", missing=0.01),
}

CONFIGS = [
    ("reg", {}), ("bin", {}), ("multi", {}), ("sparse", {}), ("sparse", {"efb": False}),
    ("reg", {"m_mult": 2}), ("reg", {"max_depth": 8}), ("reg", {"max_depth": 4, "k_proj": 2}),
    ("bin", {"max_ctr_complexity": 2}),
    ("multi", {"extra_trees": True}),
    ("reg", {"bagging_fraction": 0.7, "feature_fraction": 0.8}),
    ("reg", {"quantizer": "auto"}),
    ("big", {}), ("big", {"oblique_sample_rows": 4096}), ("big", {"oblique_sample_rows": 0}),
    ("big", {"efb": False, "oblique_sample_rows": 4096}),
    ("reg", {"grow_policy": "leafwise"}), ("bin", {"grow_policy": "leafwise"}),
    ("one_num", {}), ("one_num", {"max_ctr_complexity": 2, "bagging_fraction": 0.8}),
    ("two_num", {}), ("multi", {"bagging_fraction": 0.7, "max_ctr_complexity": 2}),
    ("wide", {}), ("wide", {"bagging_fraction": 0.6}), ("wide", {"k_proj": 3}),
    ("wide", {"oblique_sample_rows": 0, "max_depth": 8}), ("big", {"bagging_fraction": 0.5}),
]

out = []
for name, kw in CONFIGS:
    X, y = DATA[name]
    for nj in (1, 4):
        Est = MiterRegressor if name in ("reg", "sparse", "two_num", "wide") else MiterClassifier
        n_est = 30 if name in ("big", "wide") else 60
        try:
            m = Est(n_estimators=n_est, learning_rate=0.1, early_stopping_rounds=None, n_jobs=nj, random_state=1, **kw)
            m.fit(X, y)
        except Exception as e:  # noqa: BLE001
            out.append(f"{name} {kw} nj={nj} ERROR {type(e).__name__}: {e}")
            continue
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.bin")
            try:
                m.save_model(path)
                hm = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
            except NotImplementedError:
                hm = "n/a"
        pred = m.predict_proba(X) if hasattr(m, "predict_proba") else m.predict(X)
        hp = hashlib.sha256(np.ascontiguousarray(pred, dtype=np.float64).tobytes()).hexdigest()[:16]
        out.append(f"{name} {kw} nj={nj} model={hm} pred={hp}")
        print(out[-1], flush=True)
open(sys.argv[1], "w").write("\n".join(out) + "\n")
