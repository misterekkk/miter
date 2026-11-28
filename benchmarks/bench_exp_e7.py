"""E7: CTR splits in the default (results/exp_e7_ctr_default_prereg.md and its amendments).

Same protocol as bench_exp_speedups.py (5 folds, early stopping 50, one thread), on two dataset
sets: `dev` (the phase-A categorical development pool, amendment 1) and `control` (the phase-B
control set C).

Usage:
    .venv/bin/python benchmarks/bench_exp_e7.py run --set dev --variants D0 E7_C1 --shard 0 --nshards 4
    .venv/bin/python benchmarks/bench_exp_e7.py summarize --set dev --variants E7_C1 E7_C1P4 E7_AUTO
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_exp_external_holdout import _split_cv, eval_metrics, load_dataset  # noqa: E402
from bench_exp_scaling_defaults import fit_predict_any  # noqa: E402
from bench_exp_speedups import FIELDS, _fit  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RAW_DIR = HERE / "results" / "raw" / "exp_e7"

VARIANTS: dict[str, dict[str, Any]] = {
    "D0": {},
    "E7_C1": {"max_ctr_complexity": 1},
    "E7_C1P4": {"max_ctr_complexity": 1, "ctr_n_permutations": 4},
    "E7_AUTO": {"max_ctr_complexity": "auto", "ctr_n_permutations": 4},
    # Frozen by amendment 2: the phase-B candidate.
    "D1": {"max_ctr_complexity": 1, "ctr_n_permutations": 4},
}
REFERENCE = "CAT"  # CatBoost, fixed learning rate 0.05, depth 6 (bench_exp_scaling_defaults.fit_predict_any)

# (kind, name, OpenML dataset id)
DEV: list[tuple[str, str, int]] = [
    ("reg", "employee_salaries", 42125),
    ("clf", "rl", 41160),
    ("reg", "wine-reviews", 41275),
    ("clf", "kdd_internet_usage", 981),
    ("clf", "kdd_ipums_la_97-small", 993),
    ("mc", "cjs", 473),
    ("mc", "BachChoralHarmony", 4552),
    ("clf", "KDD98", 23513),
]

CONTROL: list[tuple[str, str, int]] = [
    ("clf", "kr-vs-kp", 3), ("mc", "cmc", 23), ("clf", "tic-tac-toe", 50), ("clf", "electricity", 151),
    ("reg", "boston", 531), ("reg", "sensory", 546), ("clf", "ilpd", 1480), ("clf", "PhishingWebsites", 4534),
    ("mc", "MiceProtein", 40966), ("clf", "Australian", 40981), ("clf", "jasmine", 41143),
    ("reg", "black_friday", 41540), ("reg", "SAT11-HAND-runtime-regression", 41980),
    ("reg", "house_prices_nominal", 42563), ("reg", "Allstate_Claims_Severity", 42571),
    ("mc", "okcupid-stem", 42734), ("reg", "MIP-2016-regression", 43071), ("reg", "analcatdata_supreme", 44055),
    ("reg", "visualizing_soil", 44056), ("reg", "auction_verification", 44958), ("reg", "video_transcoding", 44974),
    ("reg", "fifa", 45012), ("clf", "albert", 45035), ("clf", "road-safety", 45038),
    ("clf", "compas-two-years", 45039), ("reg", "topo_2_1", 45041), ("reg", "seattlecrime6", 45043),
]

SETS = {"dev": DEV, "control": CONTROL}


def _csv(set_name: str, variant: str, name: str, fold: int) -> Path:
    return RAW_DIR / set_name / f"{variant}_{name}_f{fold}.csv"


def cmd_run(set_name: str, variants: list[str], shard: int, nshards: int) -> None:
    (RAW_DIR / set_name).mkdir(parents=True, exist_ok=True)
    jobs = [(kind, name, did, fold) for (kind, name, did) in SETS[set_name] for fold in range(5)]
    for i, (kind, name, did, fold) in enumerate(jobs):
        if i % nshards != shard:
            continue
        assert all(v in VARIANTS or v == REFERENCE for v in variants), variants
        todo = [v for v in variants if not _csv(set_name, v, name, fold).exists()]
        if not todo:
            continue
        try:
            X, y, cat_cols, _, _ = load_dataset(kind, did)
            a, b, te = _split_cv(kind, X, y)[fold]
        except Exception as e:  # reported and left out (prereg)
            print(f"FAIL load {name} f{fold}: {type(e).__name__}: {e}", flush=True)
            continue
        for v in todo:
            try:
                if v == REFERENCE:
                    pred, trees, secs = fit_predict_any(kind, v, X.iloc[a], y[a], X.iloc[b], y[b], X.iloc[te],
                                                        cat_cols, fold)
                else:
                    pred, trees, secs = _fit(kind, VARIANTS[v], X.iloc[a], y[a], X.iloc[b], y[b], X.iloc[te], fold)
                met, _ = eval_metrics(kind, y[te], pred)
            except Exception as e:
                print(f"FAIL fit {name} f{fold} {v}: {type(e).__name__}: {e}", flush=True)
                continue
            with _csv(set_name, v, name, fold).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerow({"kind": kind, "dataset": name, "n_eval": len(X), "p_num": X.shape[1] - len(cat_cols),
                            "p_cat": len(cat_cols), "fold": fold, "variant": v, "metric": f"{met:.12g}",
                            "trees": trees, "secs": f"{secs:.6f}"})
        print(f"DONE {name} f{fold} {todo}", flush=True)


def cmd_summarize(set_name: str, variants: list[str], base: str = "D0") -> None:
    df = pd.concat([pd.read_csv(p) for p in (RAW_DIR / set_name).glob("*.csv")], ignore_index=True)
    # Only folds present for every arm, so that means compare like with like.
    arms = [base, *variants]
    df = df[df.variant.isin(arms)]
    complete = df.groupby(["dataset", "fold"]).variant.nunique() == len(arms)
    df = df.set_index(["dataset", "fold"]).loc[complete[complete].index].reset_index()
    g = df.groupby(["dataset", "variant"]).agg(metric=("metric", "mean"), secs=("secs", "sum")).reset_index()
    m = g.pivot(index="dataset", columns="variant", values="metric")
    s = g.pivot(index="dataset", columns="variant", values="secs")
    print(f"{set_name}: {len(m)} datasets with all arms; negative = better than {base}")
    for v in variants:
        rel = 100.0 * (m[v] - m[base]) / m[base]
        nz = rel[rel != 0]
        pb = float(wilcoxon(nz, alternative="less").pvalue) if len(nz) > 1 else 1.0
        pw = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 1 else 1.0
        print(f"  {v:8s} wins {int((rel < 0).sum())}, ties {int((rel == 0).sum())}, losses {int((rel > 0).sum())}; "
              f"median {rel.median():+.3f}%, mean {rel.mean():+.2f}%, worst {rel.max():+.2f}%; "
              f"p_better {pb:.3f}, p_worse {pw:.3f}; time {s[v].sum() / s[base].sum():.2f}x")
    table = pd.DataFrame({v: 100.0 * (m[v] - m[base]) / m[base] for v in variants})
    print(table.round(2).sort_values(variants[0]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--set", choices=list(SETS), required=True)
    r.add_argument("--variants", nargs="+", required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    s = sp.add_parser("summarize")
    s.add_argument("--set", choices=list(SETS), required=True)
    s.add_argument("--variants", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.set, a.variants, a.shard, a.nshards)
    else:
        cmd_summarize(a.set, a.variants)
