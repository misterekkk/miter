"""Interleaved A/B training-speed comparison of two miter builds.

    python benchmarks/ci/speed_ab.py --build base=DIR_A --build head=DIR_B --reps 5 --out ab.md
    python benchmarks/ci/speed_ab.py --build def=DIR --build m2=DIR --params m2='{"m_mult": 2}' ...

Each build is an installed copy of miter (`pip install --target DIR .`). Every measurement runs in
a fresh process, and the builds alternate run by run, so drifts in machine speed (other load,
thermal state) hit both equally. Reports the median seconds for 100 trees and the ratio to the
first build.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ONE_RUN = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import miter
assert miter.__file__.startswith(sys.argv[1]), miter.__file__
import json
which, threads, params = sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4])
if which == "covertype":
    from sklearn.datasets import fetch_covtype
    X, y = fetch_covtype(return_X_y=True)
    X, y, cls = X.astype(np.float64), (y == 2).astype(np.float64), miter.MiterClassifier
else:
    n, p = (int(v) for v in which.split("x"))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, p))
    w = rng.standard_normal(min(p, 8))
    y = np.sin(X[:, : len(w)] @ w) + 0.5 * X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(n)
    cls = miter.MiterRegressor
t = time.perf_counter()
cls(n_estimators=100, early_stopping_rounds=None, n_jobs=threads, **params).fit(X, y)
print(f"{time.perf_counter() - t:.4f}")
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="append", required=True, help="label=DIR (first one is the reference)")
    ap.add_argument("--params", action="append", default=[], help="label=JSON constructor arguments")
    ap.add_argument("--data", nargs="+", default=["covertype", "500000x50", "100000x20"])
    ap.add_argument("--threads", nargs="+", type=int, default=[1, os.cpu_count() or 1])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    builds = [tuple(b.split("=", 1)) for b in a.build]
    params = {lbl: json.loads(js) for lbl, js in (p.split("=", 1) for p in a.params)}
    times: dict[tuple[str, int, str], list[float]] = {}
    for data in a.data:
        for threads in a.threads:
            for rep in range(a.reps):
                order = builds if rep % 2 == 0 else builds[::-1]
                for label, path in order:
                    r = subprocess.run([sys.executable, "-I", "-c", ONE_RUN, str(Path(path).resolve()), data, str(threads),
                                        json.dumps(params.get(label, {}))],
                                       capture_output=True, text=True, check=True)
                    t = float(r.stdout.strip().splitlines()[-1])
                    times.setdefault((data, threads, label), []).append(t)
                    print(f"{data} T{threads} {label} rep{rep}: {t:.3f}s", flush=True)
    ref = builds[0][0]
    lines = [f"Median of {a.reps} interleaved runs, seconds for 100 trees (fresh process each run).", "",
             "| data | threads | " + " | ".join(lbl for lbl, _ in builds) + " |",
             "|---|---:|" + "---:|" * len(builds)]
    for data in a.data:
        for threads in a.threads:
            base = statistics.median(times[(data, threads, ref)])
            cells = []
            for lbl, _ in builds:
                ts = times[(data, threads, lbl)]
                m = statistics.median(ts)
                spread = (max(ts) - min(ts)) / m
                cells.append(f"{m:.2f}" + ("" if lbl == ref else f" ({m / base - 1:+.1%})") + f" ±{spread / 2:.0%}")
            lines.append(f"| {data} | {threads} | " + " | ".join(cells) + " |")
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
