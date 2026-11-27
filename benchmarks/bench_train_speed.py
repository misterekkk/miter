"""Training speed of the current build: ms per tree for LightGBM, CatBoost and miter configurations.

Every model trains a fixed number of trees (no early stopping) on the same data; the time of `fit`
divided by the number of trees is reported (median of `--runs` runs). Data: synthetic regression
with an oblique signal at several shapes, and covertype (binary, 581k x 54, via scikit-learn).

Usage:
    .venv/bin/python benchmarks/bench_train_speed.py            # writes results/train_speed_summary.md
    .venv/bin/python benchmarks/bench_train_speed.py --quick    # small shapes only, 1 run
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import miter
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "train_speed_summary.md"
N_TREES = 100

MITER_CONFIGS: dict[str, dict[str, Any]] = {
    "miter default": {},  # symmetric depth 6, m_mult 4 (S6_M4)
    "miter fast": {"preset": "fast"},
    "miter leaf-wise": {"grow_policy": "leafwise"},  # the default before v1.0
    "miter + auto quantizer + CTR": {"quantizer": "auto", "max_ctr_complexity": "auto", "ctr_n_permutations": 4},
}


def _synthetic(n: int, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    w = rng.standard_normal(min(p, 8))
    y = np.sin(X[:, : len(w)] @ w) + 0.5 * X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


def _covertype() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_covtype

    X, y = fetch_covtype(return_X_y=True)
    return X.astype(np.float64), (y == 2).astype(np.float64)


def _median_time(fn: Callable[[], None], runs: int) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _fitters(X: np.ndarray, y: np.ndarray, classification: bool, threads: int) -> dict[str, Callable[[], None]]:
    import catboost
    import lightgbm

    out: dict[str, Callable[[], None]] = {}

    def lgb() -> None:
        params = {
            "objective": "binary" if classification else "regression",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 20,
            "num_threads": threads,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
        }
        lightgbm.train(params, lightgbm.Dataset(X, y), num_boost_round=N_TREES)

    def cb() -> None:
        cls = catboost.CatBoostClassifier if classification else catboost.CatBoostRegressor
        cls(iterations=N_TREES, depth=6, learning_rate=0.05, thread_count=threads, verbose=False,
            allow_writing_files=False).fit(X, y)

    out["LightGBM (31 leaves)"] = lgb
    out["CatBoost (depth 6)"] = cb
    for name, cfg in MITER_CONFIGS.items():
        cls = miter.MiterClassifier if classification else miter.MiterRegressor

        def fit_miter(cls: Any = cls, cfg: dict[str, Any] = cfg) -> None:
            cls(n_estimators=N_TREES, early_stopping_rounds=None, n_jobs=threads, **cfg).fit(X, y)

        out[name] = fit_miter
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    runs = 1 if a.quick else a.runs
    all_threads = os.cpu_count() or 1
    shapes: list[tuple[str, Callable[[], tuple[np.ndarray, np.ndarray]], bool]] = [
        ("20k x 16", lambda: _synthetic(20_000, 16), False),
        ("20k x 100", lambda: _synthetic(20_000, 100), False),
    ]
    if not a.quick:
        shapes += [
            ("100k x 50", lambda: _synthetic(100_000, 50), False),
            ("500k x 50", lambda: _synthetic(500_000, 50), False),
            ("covertype 581k x 54", _covertype, True),
        ]
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    lines = [
        "# Training speed: seconds for 100 trees",
        "",
        f"- Machine: {cpu or platform.processor()}, {all_threads} threads; {platform.system()} {platform.release()}",
        f"- miter {miter.__version__} at commit `{git}`; {N_TREES} trees, no early stopping; median of {runs} run(s).",
        "- Data: synthetic regression (Gaussian features, oblique signal) and covertype (binary).",
        "- Time of `fit` for 100 trees, including dataset construction. Runs above 200k rows: one run.",
        "",
    ]
    for threads in (1, all_threads):
        lines += [f"## {threads} thread{'s' if threads > 1 else ''}", ""]
        header_done = False
        for label, load, classification in shapes:
            X, y = load()
            fitters = _fitters(X, y, classification, threads)
            if not header_done:
                lines += ["| data | " + " | ".join(fitters) + " |", "|---|" + "---:|" * len(fitters)]
                header_done = True
            secs = {}
            for name, fn in fitters.items():
                secs[name] = _median_time(fn, 1 if X.shape[0] > 200_000 else runs)
                print(f"threads={threads} {label} {name}: {secs[name]:.2f} s", flush=True)
            lgb, cb = secs["LightGBM (31 leaves)"], secs["CatBoost (depth 6)"]
            lines.append(
                f"| {label} | "
                + " | ".join(
                    f"{v:.2f}" if k.startswith(("LightGBM", "CatBoost")) else f"{v:.2f} ({v / lgb:.1f}× / {v / cb:.2f}×)"
                    for k, v in secs.items()
                )
                + " |"
            )
        lines += ["", "miter cells: seconds (× LightGBM / × CatBoost).", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
