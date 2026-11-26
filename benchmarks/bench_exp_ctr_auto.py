"""Exploratory ablation benchmark: Adaptive Native CTR v3 (`exp-ctr-auto`).

Evaluates the 3 adaptive CTR mechanisms across the 16 datasets x 5 folds from
`exp-native-ctr-v2`, reusing saved baseline rows (`S6`, `S6_CTR1`, `S6_CTR2`, `CAT`).

Usage:
    .venv/bin/python benchmarks/bench_exp_ctr_auto.py pilot
    .venv/bin/python benchmarks/bench_exp_ctr_auto.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_ctr_auto.py summarize
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
from bench_exp_native_ctr_v2 import (
    DATASETS,
    FIELDS,
    _paired_summary_row,
    _split_cv,
    eval_metrics,
    load_dataset,
)

HERE = Path(__file__).resolve().parent
BASELINE_RAW_DIR = HERE / "results" / "raw" / "exp_native_ctr_v2"
RAW_DIR = HERE / "results" / "raw" / "exp_ctr_auto"
SUMMARY_PATH = HERE / "results" / "exp_ctr_auto_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

NEW_VARIANTS: list[str] = [
    "S6_CTR2_PEN",
    "S6_AUTO",
    "S6_AUTO_NC",
    "S6_AUTO_NC_P4",
]
ALL_VARIANTS: list[str] = ["S6", "S6_CTR1", "S6_CTR2", "CAT", *NEW_VARIANTS]

VARIANT_PARAMS: dict[str, dict[str, Any]] = {
    "S6_CTR2_PEN": {
        "max_ctr_complexity": 2,
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": False,
        "ctr_n_permutations": 1,
    },
    "S6_AUTO": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": False,
        "ctr_n_permutations": 1,
    },
    "S6_AUTO_NC": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 1,
    },
    "S6_AUTO_NC_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 4,
    },
}


def fit_predict_adaptive(
    kind: str,
    variant: str,
    X: Any,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, int, int, int, float]:
    import miter

    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]
    vprm = VARIANT_PARAMS[variant]
    kw: dict[str, Any] = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "grow_policy": "symmetric",
        "max_depth": 6,
        "min_data_in_leaf": 1,
        "lambda_l2": 3.0,
        "m_mult": 2,
        "k_proj": 0,
        "max_bin": 255,
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "random_state": fold,
        **vprm,
    }
    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    pred = m.predict(X_te) if kind == "reg" else m.predict_proba(X_te)[:, 1]
    trees = int(m.best_iteration_)
    n_tbl = int(m._model.n_ctr_tables)
    n_proj = int(m._model.n_ctr_projections)
    return pred, trees, n_tbl, n_proj, time.perf_counter() - t0


def _load_csv_dir(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(raw_dir.glob("shard_*.csv")):
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "kind": row["kind"],
                        "dataset": row["dataset"],
                        "did": int(row["did"]),
                        "n_eval": int(row["n_eval"]),
                        "p_num": int(row["p_num"]),
                        "p_cat": int(row["p_cat"]),
                        "max_card": int(row["max_card"]),
                        "fold": int(row["fold"]),
                        "variant": row["variant"],
                        "metric": float(row["metric"]),
                        "auc": float("nan") if row["auc"] == "" else float(row["auc"]),
                        "trees": int(row["trees"]),
                        "n_ctr_tables": int(row["n_ctr_tables"]),
                        "n_ctr_projections": int(row["n_ctr_projections"]),
                        "secs": float(row["secs"]),
                    }
                )
    return rows


def cmd_pilot() -> None:
    base_rows = _load_csv_dir(BASELINE_RAW_DIR) if BASELINE_RAW_DIR.exists() else []
    for kind, name, ik, oid in [
        ("clf", "credit-g", "did", 31),
        ("clf", "Amazon_employee_access", "did", 4135),
    ]:
        X, y, cat_cols, did, mcard = load_dataset(kind, ik, oid)
        a, b, te = _split_cv(kind, X, y)[0]
        print(
            f"=== PILOT {name} did={did} n={len(X)} p_cat={len(cat_cols)} max_card={mcard} ===",
            flush=True,
        )
        for br in base_rows:
            if br["dataset"] == name and br["fold"] == 0 and br["variant"] in ("S6", "S6_CTR2", "CAT"):
                print(
                    f"  [base] {br['variant']:13s}: metric={br['metric']:.5f} auc={br['auc']:.5f} "
                    f"trees={br['trees']:4d} tbl={br['n_ctr_tables']:3d} secs={br['secs']:.2f}s",
                    flush=True,
                )
        for v in NEW_VARIANTS:
            pred, trees, n_tbl, n_proj, secs = fit_predict_adaptive(kind, v, X, y, a, b, te, 0)
            met, auc = eval_metrics(kind, y[te], pred)
            print(
                f"  [new]  {v:13s}: metric={met:.5f} auc={auc:.5f} trees={trees:4d} "
                f"tbl={n_tbl:3d} proj={n_proj:3d} secs={secs:.2f}s",
                flush=True,
            )


def cmd_run(shard: int, nshards: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RAW_DIR / f"shard_{shard}_of_{nshards}.csv"
    done: set[tuple[str, int, str]] = set()
    if raw_csv.exists():
        with raw_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["dataset"], int(row["fold"]), row["variant"]))

    write_header = not raw_csv.exists() or raw_csv.stat().st_size == 0
    selected = [d for idx, d in enumerate(DATASETS) if idx % nshards == shard]

    with raw_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            fh.flush()

        for kind, name, ik, oid in selected:
            X, y, cat_cols, did, mcard = load_dataset(kind, ik, oid)
            p_cat = len(cat_cols)
            p_num = X.shape[1] - p_cat
            splits = _split_cv(kind, X, y)
            for fold, (a, b, te) in enumerate(splits):
                for v in NEW_VARIANTS:
                    if (name, fold, v) in done:
                        continue
                    pred, trees, n_tbl, n_proj, secs = fit_predict_adaptive(
                        kind, v, X, y, a, b, te, fold
                    )
                    met, auc = eval_metrics(kind, y[te], pred)
                    writer.writerow(
                        {
                            "kind": kind,
                            "dataset": name,
                            "did": did,
                            "n_eval": len(X),
                            "p_num": p_num,
                            "p_cat": p_cat,
                            "max_card": mcard,
                            "fold": fold,
                            "variant": v,
                            "metric": repr(met),
                            "auc": "" if math.isnan(auc) else repr(auc),
                            "trees": trees,
                            "n_ctr_tables": n_tbl,
                            "n_ctr_projections": n_proj,
                            "secs": f"{secs:.3f}",
                        }
                    )
                    fh.flush()
                    print(
                        f"[{shard}/{nshards}] {kind:3s} {name:30s} f={fold} {v:13s}: "
                        f"{met:.5f} trees={trees:4d} tbl={n_tbl:3d} {secs:5.2f}s",
                        flush=True,
                    )


def cmd_summarize() -> None:
    rows = _load_csv_dir(BASELINE_RAW_DIR) + _load_csv_dir(RAW_DIR)
    if not rows:
        raise SystemExit("No rows found for baseline or exp-ctr-auto")

    agg: dict[tuple[str, str], float] = {}
    agg_secs: dict[tuple[str, str], float] = {}
    agg_one_m_auc: dict[tuple[str, str], float] = {}
    agg_tbl: dict[tuple[str, str], float] = {}
    ds_meta: dict[str, dict[str, Any]] = {}
    all_names = [n for _, n, _, _ in DATASETS]
    clf_names = [n for k, n, _, _ in DATASETS if k == "clf"]

    for _, name, _, _ in DATASETS:
        sub_all = [r for r in rows if r["dataset"] == name]
        if sub_all:
            ds_meta[name] = sub_all[0]
        for v in ALL_VARIANTS:
            sub = [r for r in sub_all if r["variant"] == v]
            if len(sub) != 5:
                raise SystemExit(f"Incomplete runs for {name}/{v}: expected 5 folds, got {len(sub)}")
            agg[(name, v)] = statistics.fmean(r["metric"] for r in sub)
            agg_secs[(name, v)] = statistics.fmean(r["secs"] for r in sub)
            agg_tbl[(name, v)] = statistics.fmean(r["n_ctr_tables"] for r in sub)
            if sub[0]["kind"] == "clf":
                agg_one_m_auc[(name, v)] = 1.0 - statistics.fmean(r["auc"] for r in sub)

    lines = [
        "# Exploratory Ablation Summary: Adaptive Native CTR v3 (`exp-ctr-auto`)",
        "",
        "**Status: Exploratory** (no pre-registration; 16 datasets x 5 folds = 80 paired runs/variant, baselines reused from `benchmarks/results/raw/exp_native_ctr_v2/`).",
        "",
        "## 1. Exploratory Scorecard vs Baselines (Primary metric; negative = better error)",
        "",
        "| Variant | Baseline | Mean % | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | Worst % | Time vs S6 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for cand in NEW_VARIANTS:
        t_ratio = float(
            statistics.median(
                agg_secs[(n, cand)] / max(agg_secs[(n, "S6")], 1e-6) for n in all_names
            )
        )
        for base in ["S6", "S6_CTR1", "S6_CTR2", "CAT"]:
            deltas, p_val, (ci_lo, ci_hi) = _paired_summary_row(
                f"{cand} vs {base}", all_names, agg, cand, base
            )
            wins = sum(1 for d in deltas if d < 0.0)
            lines.append(
                f"| `{cand}` | `{base}` | {statistics.fmean(deltas):+.2f}% | "
                f"{statistics.median(deltas):+.2f}% | [{ci_lo:+.2f}%, {ci_hi:+.2f}%] | "
                f"{wins}/16 | {p_val:.4g} | {max(deltas):+.2f}% | {t_ratio:.2f}x |"
            )

    lines += [
        "",
        "## 2. Subgroup Breakdown (Exploratory)",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    hi_names = [n for n in all_names if ds_meta[n]["max_card"] > 253]
    for cand in NEW_VARIANTS:
        for label, sub_names, src_agg, base in [
            (f"{cand} vs CAT (1-AUC)", clf_names, agg_one_m_auc, "CAT"),
            (f"{cand} vs S6 (1-AUC)", clf_names, agg_one_m_auc, "S6"),
            (f"{cand} vs S6_CTR2 (max_card>253)", hi_names, agg, "S6_CTR2"),
        ]:
            deltas, p_val, (lo, hi) = _paired_summary_row(label, sub_names, src_agg, cand, base)
            lines.append(
                f"| `{label}` | {len(sub_names)} | {statistics.fmean(deltas):+.2f}% | "
                f"{statistics.median(deltas):+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] | "
                f"{sum(1 for d in deltas if d < 0)}/{len(sub_names)} | {p_val:.4g} | {max(deltas):+.2f}% |"
            )

    lines += [
        "",
        "## 3. Per-dataset Primary Metric (5-fold mean; lower is better)",
        "",
        "| kind | dataset | n | p_cat | S6 | S6_CTR2 | CAT | CTR2_PEN | AUTO | AUTO_NC | AUTO_NC_P4 | AUTO_NC vs CTR2 | AUTO_NC vs CAT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _, _ in DATASETS:
        m = ds_meta[name]
        v_ctr2 = agg[(name, "S6_CTR2")]
        v_cat = agg[(name, "CAT")]
        v_nc = agg[(name, "S6_AUTO_NC")]
        lines.append(
            f"| {kind} | `{name}` | {m['n_eval']} | {m['p_cat']} | "
            f"{agg[(name, 'S6')]:.5f} | {v_ctr2:.5f} | {v_cat:.5f} | "
            f"{agg[(name, 'S6_CTR2_PEN')]:.5f} | {agg[(name, 'S6_AUTO')]:.5f} | "
            f"{v_nc:.5f} | {agg[(name, 'S6_AUTO_NC_P4')]:.5f} | "
            f"{(v_nc - v_ctr2) / v_ctr2 * 100.0:+.2f}% | {(v_nc - v_cat) / v_cat * 100.0:+.2f}% |"
        )

    lines += [
        "",
        "## 4. Wall Time & Mean CTR Tables per Variant",
        "",
        "| dataset | S6 t | CTR2 t | CAT t | CTR2_PEN t | AUTO t | AUTO_NC t | AUTO_NC_P4 t | CTR2 tbl | CTR2_PEN tbl | AUTO tbl | AUTO_NC tbl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, name, _, _ in DATASETS:
        lines.append(
            f"| `{name}` | {agg_secs[(name, 'S6')]:.2f}s | {agg_secs[(name, 'S6_CTR2')]:.2f}s | "
            f"{agg_secs[(name, 'CAT')]:.2f}s | {agg_secs[(name, 'S6_CTR2_PEN')]:.2f}s | "
            f"{agg_secs[(name, 'S6_AUTO')]:.2f}s | {agg_secs[(name, 'S6_AUTO_NC')]:.2f}s | "
            f"{agg_secs[(name, 'S6_AUTO_NC_P4')]:.2f}s | {agg_tbl[(name, 'S6_CTR2')]:.1f} | "
            f"{agg_tbl[(name, 'S6_CTR2_PEN')]:.1f} | {agg_tbl[(name, 'S6_AUTO')]:.1f} | "
            f"{agg_tbl[(name, 'S6_AUTO_NC')]:.1f} |"
        )

    lines += [
        "",
        "## 5. Exploratory Binary Classification ROC-AUC (higher is better)",
        "",
        "| dataset | S6 | S6_CTR2 | CAT | CTR2_PEN | AUTO | AUTO_NC | AUTO_NC_P4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in clf_names:
        lines.append(
            f"| `{name}` | {1.0 - agg_one_m_auc[(name, 'S6')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_CTR2')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'CAT')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_CTR2_PEN')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO_NC')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO_NC_P4')]:.5f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sp = sub.add_parser("run")
    sp.add_argument("--shard", type=int, default=0)
    sp.add_argument("--nshards", type=int, default=1)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "run":
        cmd_run(args.shard, args.nshards)
    elif args.cmd == "summarize":
        cmd_summarize()


if __name__ == "__main__":
    main()
