"""Training speed-ups (E1 oblique_min_distinct, E2 oblique row sampling, E3 feature bundling) checked for
quality on the 32 held-out OpenML datasets of exp_scaling_defaults part 3 (no TabArena overlap).

Every variant is the v1.0 default (`BASE`: symmetric depth 6, m_mult 4) plus one change; each is
compared with BASE fitted in the same run. Pre-registrations: results/exp_speedups_*_prereg.md.

Usage:
    .venv/bin/python benchmarks/bench_exp_speedups.py run --variants BASE E1 --shard 0 --nshards 8
    .venv/bin/python benchmarks/bench_exp_speedups.py summarize --variants E1 --out results/exp_speedups_e1_summary.md
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_exp_external_holdout import _split_cv, eval_metrics, load_dataset
from bench_exp_scaling_defaults import PART3_DATASETS, assert_zero_overlap
from scipy.stats import wilcoxon

RAW_DIR = HERE / "results" / "raw" / "exp_speedups"
FIELDS = ["kind", "dataset", "n_eval", "p_num", "p_cat", "fold", "variant", "metric", "trees", "secs"]

from speedup_variants import BASE, VARIANTS  # noqa: F401


def _fit(kind: str, cfg: dict[str, Any], X_tr: Any, y_tr: Any, X_val: Any, y_val: Any, X_te: Any, seed: int):
    import miter

    t0 = time.perf_counter()
    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(n_jobs=1, random_state=seed, **cfg).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    if kind == "reg":
        pred = m.predict(X_te)
    elif kind == "clf":
        pred = m.predict_proba(X_te)[:, 1]
    else:
        pred = m.predict_proba(X_te)
    return pred, int(m.best_iteration_), time.perf_counter() - t0


def _csv(variant: str, name: str, fold: int) -> Path:
    return RAW_DIR / f"{variant}_{name}_f{fold}.csv"


def cmd_run(variants: list[str], shard: int, nshards: int) -> None:
    assert_zero_overlap()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(kind, name, did, fold) for (kind, name, did) in PART3_DATASETS for fold in range(5)]
    for i, (kind, name, did, fold) in enumerate(jobs):
        if i % nshards != shard:
            continue
        todo = [v for v in variants if not _csv(v, name, fold).exists()]
        if not todo:
            continue
        X, y, cat_cols, _, _ = load_dataset(kind, did)
        a, b, te = _split_cv(kind, X, y)[fold]
        for v in todo:
            pred, trees, secs = _fit(kind, VARIANTS[v], X.iloc[a], y[a], X.iloc[b], y[b], X.iloc[te], fold)
            met, _ = eval_metrics(kind, y[te], pred)
            with _csv(v, name, fold).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerow({"kind": kind, "dataset": name, "n_eval": len(X), "p_num": X.shape[1] - len(cat_cols),
                            "p_cat": len(cat_cols), "fold": fold, "variant": v, "metric": f"{met:.12g}",
                            "trees": trees, "secs": f"{secs:.6f}"})
        print(f"DONE {name} f{fold} {todo}", flush=True)


def _load() -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in RAW_DIR.glob("*.csv")], ignore_index=True)


def paired(df: pd.DataFrame, a: str, b: str = "BASE") -> dict[str, float]:
    """Per-dataset mean over folds, then relative error of a vs b (negative = a better)."""
    g = df.groupby(["dataset", "variant"]).agg(metric=("metric", "mean"), secs=("secs", "sum")).reset_index()
    m = g.pivot(index="dataset", columns="variant", values="metric")[[a, b]].dropna()
    s = g.pivot(index="dataset", columns="variant", values="secs").loc[m.index, [a, b]]
    rel = 100.0 * (m[a] - m[b]) / m[b]
    nz = rel[rel != 0]
    p_worse = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) else 1.0
    return {
        "n": len(rel), "wins": int((rel < 0).sum()), "ties": int((rel == 0).sum()),
        "median": float(rel.median()), "mean": float(rel.mean()), "worst": float(rel.max()),
        "p_worse": p_worse, "time_ratio_total": float(s[a].sum() / s[b].sum()),
        "time_ratio_median": float((s[a] / s[b]).median()),
    }


def cmd_summarize(variants: list[str], out: Path) -> None:
    df = _load()
    lines = ["# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds", "",
             "Relative error per dataset (mean over folds), negative = better; time = total fit seconds "
             "(one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.", "",
             "| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in variants:
        r = paired(df, v)
        lines.append(f"| {v} | {r['n']} | {r['wins']} | {r['ties']} | {r['median']:+.3f}% | {r['mean']:+.3f}% | "
                     f"{r['worst']:+.2f}% | {r['p_worse']:.4f} | {r['time_ratio_total']:.2f}x | {r['time_ratio_median']:.2f}x |")
    g = df.groupby(["dataset", "variant"])["metric"].mean().unstack()
    lines += ["", "## Per dataset (mean metric over folds)", "", "| dataset | " + " | ".join(["BASE", *variants]) + " |",
              "|---|" + "---:|" * (1 + len(variants))]
    for d, row in g.iterrows():
        lines.append(f"| {d} | " + " | ".join(f"{row.get(v, math.nan):.5g}" for v in ["BASE", *variants]) + " |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def cmd_speed(variants: list[str], out: Path) -> None:
    """Seconds for 100 trees (no early stopping) on covertype and on synthetic 500k x 50, 1 and all threads."""
    import os

    import miter
    from bench_train_speed import _covertype, _synthetic

    data = {"covertype 581k x 54": (*_covertype(), True), "synthetic 500k x 50": (*_synthetic(500_000, 50), False)}
    lines = ["# Speed: seconds for 100 trees (median of 3)", "", "| data | threads | " + " | ".join(variants) + " |",
             "|---|---:|" + "---:|" * len(variants)]
    for label, (X, y, clf) in data.items():
        for threads in (1, os.cpu_count() or 1):
            row = []
            for v in variants:
                cls = miter.MiterClassifier if clf else miter.MiterRegressor
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    cls(n_estimators=100, early_stopping_rounds=None, n_jobs=threads, **VARIANTS[v]).fit(X, y)
                    ts.append(time.perf_counter() - t0)
                row.append(float(np.median(ts)))
                print(f"{label} T{threads} {v}: {row[-1]:.2f}s", flush=True)
            base = row[0]
            lines.append(f"| {label} | {threads} | " + " | ".join(
                f"{t:.2f}" if i == 0 else f"{t:.2f} ({t / base:.2f}x)" for i, t in enumerate(row)) + " |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--variants", nargs="+", required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    s = sp.add_parser("summarize")
    s.add_argument("--variants", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    sp_speed = sp.add_parser("speed")
    sp_speed.add_argument("--variants", nargs="+", required=True)
    sp_speed.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.variants, a.shard, a.nshards)
    elif a.cmd == "speed":
        cmd_speed(a.variants, HERE / a.out if not a.out.is_absolute() else a.out)
    else:
        cmd_summarize(a.variants, HERE / a.out if not a.out.is_absolute() else a.out)
