"""Exploratory ablation benchmark v2: Gated num x cat CTR (`exp-ctr-auto-v2`).

Reuses baseline rows (`S6`, `S6_CTR1`, `S6_CTR2`, `CAT` from `exp_native_ctr_v2`) and
previous ablation rows (`S6_CTR2_PEN`, `S6_AUTO`, `S6_AUTO_NC`, `S6_AUTO_NC_P4` from
`exp_ctr_auto`), running only the 5 new variants across the 16 datasets x 5 folds.

Usage:
    .venv/bin/python benchmarks/bench_exp_ctr_auto_v2.py pilot
    .venv/bin/python benchmarks/bench_exp_ctr_auto_v2.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_ctr_auto_v2.py summarize
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
from bench_exp_ctr_auto import _load_csv_dir
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
PREV_RAW_DIR = HERE / "results" / "raw" / "exp_ctr_auto"
RAW_DIR = HERE / "results" / "raw" / "exp_ctr_auto_v2"
SUMMARY_PATH = HERE / "results" / "exp_ctr_auto_v2_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

PREV_VARIANTS: list[str] = [
    "S6_CTR1",
    "S6_CTR2",
    "S6_CTR2_PEN",
    "S6_AUTO",
    "S6_AUTO_NC",
    "S6_AUTO_NC_P4",
]
NEW_VARIANTS: list[str] = [
    "S6_AUTO_P4",
    "S6_NC_M25_P4",
    "S6_NC_M50_P4",
    "S6_NC_NEWTBL_P4",
    "S6_NC_SMART_P4",
]
ALL_VARIANTS: list[str] = ["S6", "CAT", *PREV_VARIANTS, *NEW_VARIANTS]

VARIANT_PARAMS: dict[str, dict[str, Any]] = {
    "S6_AUTO_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": False,
        "ctr_n_permutations": 4,
    },
    "S6_NC_M25_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 4,
        "ctr_num_gain_margin": 1.25,
    },
    "S6_NC_M50_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 4,
        "ctr_num_gain_margin": 1.50,
    },
    "S6_NC_NEWTBL_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 4,
        "ctr_new_table_penalty": 0.35,
    },
    "S6_NC_SMART_P4": {
        "max_ctr_complexity": "auto",
        "ctr_model_size_reg": 0.5,
        "ctr_num_combinations": True,
        "ctr_n_permutations": 4,
        "ctr_num_max_depth": 1,
        "ctr_num_skip_onehot": True,
        "ctr_num_gain_margin": 1.25,
        "ctr_new_table_penalty": 0.25,
    },
}


def fit_predict_v2(
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


def cmd_pilot() -> None:
    prev_rows = (
        (_load_csv_dir(BASELINE_RAW_DIR) if BASELINE_RAW_DIR.exists() else [])
        + (_load_csv_dir(PREV_RAW_DIR) if PREV_RAW_DIR.exists() else [])
    )
    for kind, name, ik, oid in [
        ("reg", "avocado-sales", "did", 41210),
        ("clf", "Amazon_employee_access", "did", 4135),
    ]:
        X, y, cat_cols, did, mcard = load_dataset(kind, ik, oid)
        a, b, te = _split_cv(kind, X, y)[0]
        print(
            f"=== PILOT {name} did={did} n={len(X)} p_cat={len(cat_cols)} max_card={mcard} ===",
            flush=True,
        )
        for pr in prev_rows:
            if (
                pr["dataset"] == name
                and pr["fold"] == 0
                and pr["variant"] in ("S6", "S6_AUTO", "S6_AUTO_NC_P4", "CAT")
            ):
                auc_s = f"{pr['auc']:.5f}" if not math.isnan(pr["auc"]) else "  nan  "
                print(
                    f"  [prev] {pr['variant']:15s}: metric={pr['metric']:.5f} auc={auc_s} "
                    f"trees={pr['trees']:4d} tbl={pr['n_ctr_tables']:4d} secs={pr['secs']:.2f}s",
                    flush=True,
                )
        for v in NEW_VARIANTS:
            pred, trees, n_tbl, n_proj, secs = fit_predict_v2(kind, v, X, y, a, b, te, 0)
            met, auc = eval_metrics(kind, y[te], pred)
            auc_s = f"{auc:.5f}" if not math.isnan(auc) else "  nan  "
            print(
                f"  [new]  {v:15s}: metric={met:.5f} auc={auc_s} trees={trees:4d} "
                f"tbl={n_tbl:4d} proj={n_proj:3d} secs={secs:.2f}s",
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
                    pred, trees, n_tbl, n_proj, secs = fit_predict_v2(kind, v, X, y, a, b, te, fold)
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
                        f"[{shard}/{nshards}] {kind:3s} {name:30s} f={fold} {v:15s}: "
                        f"{met:.5f} trees={trees:4d} tbl={n_tbl:4d} {secs:5.2f}s",
                        flush=True,
                    )


def cmd_summarize() -> None:
    rows = (
        _load_csv_dir(BASELINE_RAW_DIR)
        + _load_csv_dir(PREV_RAW_DIR)
        + _load_csv_dir(RAW_DIR)
    )
    if not rows:
        raise SystemExit("No rows found for baseline / prev / exp-ctr-auto-v2")

    agg: dict[tuple[str, str], float] = {}
    agg_secs: dict[tuple[str, str], float] = {}
    agg_one_m_auc: dict[tuple[str, str], float] = {}
    agg_tbl: dict[tuple[str, str], float] = {}
    ds_meta: dict[str, dict[str, Any]] = {}
    all_names = [n for _, n, _, _ in DATASETS]
    reg_names = [n for k, n, _, _ in DATASETS if k == "reg"]
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
        "# Exploratory Ablation Summary v2: Gated num x cat CTR (`exp-ctr-auto-v2`)",
        "",
        "**Status: Exploratory** (no pre-registration; 16 datasets x 5 folds = 80 paired runs/variant. "
        "Previous results reused verbatim from `exp_native_ctr_v2` and `exp_ctr_auto`).",
        "",
        "## 1. Unified Exploratory Scorecard (Previous on Top, 5 New Variants Below)",
        "",
        "Negative `%` = lower error than baseline.",
        "",
        "| Stage | Variant | vs Baseline | Mean % | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | Worst % | Time vs S6 | Med Tbls |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    ordered = [(v, "prev") for v in PREV_VARIANTS] + [(v, "new") for v in NEW_VARIANTS]
    for cand, stage in ordered:
        t_ratio = float(
            statistics.median(
                agg_secs[(n, cand)] / max(agg_secs[(n, "S6")], 1e-6) for n in all_names
            )
        )
        med_tbl = float(statistics.median(agg_tbl[(n, cand)] for n in all_names))
        for base in ["S6", "CAT"]:
            deltas, p_val, (ci_lo, ci_hi) = _paired_summary_row(
                f"{cand} vs {base}", all_names, agg, cand, base
            )
            wins = sum(1 for d in deltas if d < 0.0)
            lines.append(
                f"| {stage} | `{cand}` | `{base}` | {statistics.fmean(deltas):+.2f}% | "
                f"{statistics.median(deltas):+.2f}% | [{ci_lo:+.2f}%, {ci_hi:+.2f}%] | "
                f"{wins}/16 | {p_val:.4g} | {max(deltas):+.2f}% | {t_ratio:.2f}x | {med_tbl:.1f} |"
            )

    lines += [
        "",
        "## 2. Subgroup Breakdown (7 Regression vs 9 Classification, Exploratory)",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    focus_vars = ["S6_AUTO", "S6_AUTO_NC_P4", *NEW_VARIANTS]
    for cand in focus_vars:
        for label, sub_names, src_agg, base in [
            (f"{cand} vs S6 (reg)", reg_names, agg, "S6"),
            (f"{cand} vs S6 (clf logloss)", clf_names, agg, "S6"),
            (f"{cand} vs S6 (clf 1-AUC)", clf_names, agg_one_m_auc, "S6"),
            (f"{cand} vs CAT (reg)", reg_names, agg, "CAT"),
            (f"{cand} vs CAT (clf logloss)", clf_names, agg, "CAT"),
            (f"{cand} vs CAT (clf 1-AUC)", clf_names, agg_one_m_auc, "CAT"),
        ]:
            deltas, p_val, (lo, hi) = _paired_summary_row(label, sub_names, src_agg, cand, base)
            lines.append(
                f"| `{label}` | {len(sub_names)} | {statistics.fmean(deltas):+.2f}% | "
                f"{statistics.median(deltas):+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] | "
                f"{sum(1 for d in deltas if d < 0)}/{len(sub_names)} | {p_val:.4g} | {max(deltas):+.2f}% |"
            )

    lines += [
        "",
        "## 3. Per-dataset Primary Metric & Mean CTR Tables (5-fold mean; lower is better)",
        "",
        "| kind | dataset | S6 | CAT | S6_AUTO | NC_P4 | AUTO_P4 | M25_P4 | M50_P4 | NEWTBL_P4 | SMART_P4 | tbl:NC_P4 | tbl:SMART_P4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _, _ in DATASETS:
        lines.append(
            f"| {kind} | `{name}` | {agg[(name, 'S6')]:.5f} | {agg[(name, 'CAT')]:.5f} | "
            f"{agg[(name, 'S6_AUTO')]:.5f} | {agg[(name, 'S6_AUTO_NC_P4')]:.5f} | "
            f"{agg[(name, 'S6_AUTO_P4')]:.5f} | {agg[(name, 'S6_NC_M25_P4')]:.5f} | "
            f"{agg[(name, 'S6_NC_M50_P4')]:.5f} | {agg[(name, 'S6_NC_NEWTBL_P4')]:.5f} | "
            f"{agg[(name, 'S6_NC_SMART_P4')]:.5f} | {agg_tbl[(name, 'S6_AUTO_NC_P4')]:.1f} | "
            f"{agg_tbl[(name, 'S6_NC_SMART_P4')]:.1f} |"
        )

    lines += [
        "",
        "## 4. Mean CTR Tables & Wall Time Comparison (New Variants)",
        "",
        "| dataset | AUTO t | NC_P4 t | AUTO_P4 t | M25_P4 t | M50_P4 t | NEWTBL t | SMART t | tbl:AUTO | tbl:AUTO_P4 | tbl:M25 | tbl:M50 | tbl:NEWTBL | tbl:SMART |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, name, _, _ in DATASETS:
        lines.append(
            f"| `{name}` | {agg_secs[(name, 'S6_AUTO')]:.2f}s | {agg_secs[(name, 'S6_AUTO_NC_P4')]:.2f}s | "
            f"{agg_secs[(name, 'S6_AUTO_P4')]:.2f}s | {agg_secs[(name, 'S6_NC_M25_P4')]:.2f}s | "
            f"{agg_secs[(name, 'S6_NC_M50_P4')]:.2f}s | {agg_secs[(name, 'S6_NC_NEWTBL_P4')]:.2f}s | "
            f"{agg_secs[(name, 'S6_NC_SMART_P4')]:.2f}s | {agg_tbl[(name, 'S6_AUTO')]:.1f} | "
            f"{agg_tbl[(name, 'S6_AUTO_P4')]:.1f} | {agg_tbl[(name, 'S6_NC_M25_P4')]:.1f} | "
            f"{agg_tbl[(name, 'S6_NC_M50_P4')]:.1f} | {agg_tbl[(name, 'S6_NC_NEWTBL_P4')]:.1f} | "
            f"{agg_tbl[(name, 'S6_NC_SMART_P4')]:.1f} |"
        )

    lines += [
        "",
        "## 5. Exploratory Binary Classification ROC-AUC (higher is better)",
        "",
        "| dataset | CAT | S6_AUTO | NC_P4 | AUTO_P4 | M25_P4 | M50_P4 | NEWTBL_P4 | SMART_P4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in clf_names:
        lines.append(
            f"| `{name}` | {1.0 - agg_one_m_auc[(name, 'CAT')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO_NC_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_NC_M25_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_NC_M50_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_NC_NEWTBL_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_NC_SMART_P4')]:.5f} |"
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
