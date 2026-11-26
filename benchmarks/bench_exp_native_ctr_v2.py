"""Confirmatory benchmark: Native C++20 Online Ordered CTR v2 (`exp-native-ctr-v2`).

Pre-registered in `benchmarks/results/exp_native_ctr_v2_prereg.md`.

Usage:
    .venv/bin/python benchmarks/bench_exp_native_ctr_v2.py pilot
    .venv/bin/python benchmarks/bench_exp_native_ctr_v2.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_native_ctr_v2.py summarize
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
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "results" / "raw" / "exp_native_ctr_v2"
SUMMARY_PATH = HERE / "results" / "exp_native_ctr_v2_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# (kind, name, id_kind, openml_id)
DATASETS: list[tuple[str, str, str, int]] = [
    ("reg", "abalone", "did", 44956),
    ("reg", "diamonds", "did", 42225),
    ("reg", "Mercedes_Benz", "did", 42570),
    ("reg", "avocado-sales", "did", 41210),
    ("reg", "nyc-taxi-green-dec-2016", "did", 42729),
    ("reg", "colleges", "did", 42727),
    ("reg", "Food_Delivery_Time", "tid", 363672),
    ("clf", "credit-g", "did", 31),
    ("clf", "bank-marketing", "did", 1461),
    ("clf", "adult", "did", 1590),
    ("clf", "kick", "did", 41162),
    ("clf", "Amazon_employee_access", "did", 4135),
    ("clf", "Click_prediction_small", "did", 42733),
    ("clf", "Bank_Customer_Churn", "tid", 363619),
    ("clf", "in_vehicle_coupon_recommendation", "tid", 363681),
    ("clf", "online_shoppers_intention", "tid", 363691),
]

VARIANTS = ["S6", "S6_CTR1", "S6_CTR2", "CAT"]

FIELDS = [
    "kind",
    "dataset",
    "did",
    "n_eval",
    "p_num",
    "p_cat",
    "max_card",
    "fold",
    "variant",
    "metric",
    "auc",
    "trees",
    "n_ctr_tables",
    "n_ctr_projections",
    "secs",
]


def _resolve_did(id_kind: str, oid: int) -> int:
    if id_kind == "did":
        return oid
    task = openml.tasks.get_task(oid, download_splits=False, download_qualities=False)
    return int(task.dataset_id)


def load_dataset(
    kind: str, id_kind: str, oid: int
) -> tuple[pd.DataFrame, np.ndarray, list[str], int, int]:
    did = _resolve_did(id_kind, oid)
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    t = d.default_target_attribute
    X_df, y_s, cat_ind, _ = d.get_data(target=t, dataset_format="dataframe")
    X = X_df.copy()
    cat_cols: list[str] = []
    for col, is_cat in zip(X.columns, cat_ind, strict=True):
        if (
            is_cat
            or isinstance(X[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[col])
            or pd.api.types.is_object_dtype(X[col])
        ):
            X[col] = X[col].astype("category")
            cat_cols.append(str(col))
        else:
            X[col] = X[col].astype(np.float64)
    X.columns = [str(c) for c in X.columns]

    if kind == "reg":
        y = y_s.astype("float64").to_numpy()
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.int64)

    n = len(X)
    if n > 25000:
        rng = np.random.default_rng(42)
        if kind == "clf":
            idx0 = np.flatnonzero(y == 0)
            idx1 = np.flatnonzero(y == 1)
            n0 = int(round(25000 * len(idx0) / n))
            n1 = 25000 - n0
            pick = np.sort(
                np.concatenate(
                    [
                        rng.choice(idx0, size=n0, replace=False),
                        rng.choice(idx1, size=n1, replace=False),
                    ]
                )
            )
        else:
            pick = np.sort(rng.choice(n, size=25000, replace=False))
        X = X.iloc[pick].reset_index(drop=True)
        y = y[pick]
    else:
        X = X.reset_index(drop=True)

    max_card = max((int(X[c].nunique(dropna=True)) for c in cat_cols), default=0)
    return X, y, cat_cols, did, max_card


def eval_metrics(kind: str, y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if kind == "reg":
        nrmse = float(np.sqrt(np.mean((pred - y_true) ** 2)) / np.std(y_true))
        return nrmse, float("nan")
    p = np.clip(pred, 1e-15, 1.0 - 1e-15)
    logloss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
    auc = float(roc_auc_score(y_true, pred))
    return logloss, auc


def _prep_catboost_df(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        s = Xc[c].astype(object)
        Xc[c] = s.where(pd.notna(s), "__MITER_MISSING__").astype(str)
    return Xc


def fit_predict_variant(
    kind: str,
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, int, int, int, float]:
    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]

    if variant in ("S6", "S6_CTR1", "S6_CTR2"):
        import miter

        ctr_cplx = 0 if variant == "S6" else (1 if variant == "S6_CTR1" else 2)
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
            "max_ctr_complexity": ctr_cplx,
            "early_stopping_rounds": 50,
            "n_jobs": 1,
            "random_state": fold,
        }
        cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
        m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        pred = m.predict(X_te) if kind == "reg" else m.predict_proba(X_te)[:, 1]
        trees = int(m.best_iteration_)
        n_tbl = int(m._model.n_ctr_tables)
        n_proj = int(m._model.n_ctr_projections)
    elif variant == "CAT":
        from catboost import CatBoostClassifier, CatBoostRegressor

        Xc_tr = _prep_catboost_df(X_tr, cat_cols)
        Xc_val = _prep_catboost_df(X_val, cat_cols)
        Xc_te = _prep_catboost_df(X_te, cat_cols)
        common = {
            "iterations": 5000,
            "learning_rate": 0.05,
            "early_stopping_rounds": 50,
            "depth": 6,
            "thread_count": 1,
            "random_seed": fold,
            "cat_features": cat_cols,
            "verbose": 0,
        }
        if kind == "reg":
            mc = CatBoostRegressor(**common)
            mc.fit(Xc_tr, y_tr, eval_set=(Xc_val, y_val))
            pred = np.asarray(mc.predict(Xc_te), dtype=np.float64)
        else:
            mc = CatBoostClassifier(**common)
            mc.fit(Xc_tr, y_tr, eval_set=(Xc_val, y_val))
            pred = np.asarray(mc.predict_proba(Xc_te)[:, 1], dtype=np.float64)
        trees = int(mc.get_best_iteration() + 1)
        n_tbl = 0
        n_proj = 0
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    return pred, trees, n_tbl, n_proj, time.perf_counter() - t0


def _split_cv(
    kind: str, X: pd.DataFrame, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if kind == "clf"
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splitter.split(X, y if kind == "clf" else None):
        strat = y[tr_idx] if kind == "clf" else None
        a, b = train_test_split(tr_idx, test_size=0.2, random_state=0, stratify=strat)
        out.append((a, b, te_idx))
    return out


def _bootstrap_median_ci(
    deltas: list[float], *, n_boot: int = 10000, seed: int = 2026
) -> tuple[float, float]:
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    meds = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def _holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        prev_max = max(prev_max, val)
        adj[idx] = prev_max
    return adj


def cmd_pilot() -> None:
    for kind, name, ik, oid in [
        ("reg", "abalone", "did", 44956),
        ("clf", "credit-g", "did", 31),
    ]:
        X, y, cat_cols, did, mcard = load_dataset(kind, ik, oid)
        a, b, te = _split_cv(kind, X, y)[0]
        print(
            f"=== PILOT {name} did={did} n={len(X)} p_cat={len(cat_cols)} max_card={mcard} ===",
            flush=True,
        )
        for v in VARIANTS:
            pred, trees, n_tbl, n_proj, secs = fit_predict_variant(
                kind, v, X, y, cat_cols, a, b, te, 0
            )
            met, auc = eval_metrics(kind, y[te], pred)
            print(
                f"  {v:8s}: metric={met:.5f} auc={auc:.5f} trees={trees:4d} "
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
                for v in VARIANTS:
                    if (name, fold, v) in done:
                        continue
                    pred, trees, n_tbl, n_proj, secs = fit_predict_variant(
                        kind, v, X, y, cat_cols, a, b, te, fold
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
                        f"[{shard}/{nshards}] {kind:3s} {name:30s} f={fold} {v:8s}: "
                        f"{met:.5f} trees={trees:4d} tbl={n_tbl:3d} {secs:5.2f}s",
                        flush=True,
                    )


def _paired_summary_row(
    label: str,
    names: list[str],
    agg: dict[tuple[str, str], float],
    cand: str,
    base: str,
) -> tuple[list[float], float, tuple[float, float]]:
    deltas = [(agg[(n, cand)] - agg[(n, base)]) / agg[(n, base)] * 100.0 for n in names]
    nz = [d for d in deltas if d != 0.0]
    p_val = 1.0 if not nz else float(wilcoxon(nz, alternative="two-sided").pvalue)
    ci_lo, ci_hi = _bootstrap_median_ci(deltas, n_boot=10000, seed=2026)
    return deltas, p_val, (ci_lo, ci_hi)


def cmd_summarize() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(RAW_DIR.glob("shard_*.csv")):
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
    if not rows:
        raise SystemExit(f"No rows found in {RAW_DIR}")

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
        for v in VARIANTS:
            sub = [r for r in sub_all if r["variant"] == v]
            if len(sub) != 5:
                raise SystemExit(f"Incomplete runs for {name}/{v}: expected 5 folds, got {len(sub)}")
            agg[(name, v)] = statistics.fmean(r["metric"] for r in sub)
            agg_secs[(name, v)] = statistics.fmean(r["secs"] for r in sub)
            agg_tbl[(name, v)] = statistics.fmean(r["n_ctr_tables"] for r in sub)
            if sub[0]["kind"] == "clf":
                agg_one_m_auc[(name, v)] = 1.0 - statistics.fmean(r["auc"] for r in sub)

    confirmatory = [
        ("Primary: S6_CTR2 vs S6", "S6_CTR2", "S6"),
        ("Secondary: S6_CTR1 vs S6", "S6_CTR1", "S6"),
    ]
    raw_p: list[float] = []
    comp_data: list[dict[str, Any]] = []
    for label, cand, base in confirmatory:
        deltas, p_val, (ci_lo, ci_hi) = _paired_summary_row(label, all_names, agg, cand, base)
        raw_p.append(p_val)
        t_ratio = float(
            statistics.median(
                agg_secs[(n, cand)] / max(agg_secs[(n, base)], 1e-6) for n in all_names
            )
        )
        comp_data.append(
            {
                "label": label,
                "med": float(statistics.median(deltas)),
                "mean": float(statistics.fmean(deltas)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "wins": sum(1 for d in deltas if d < 0.0),
                "worst": float(max(deltas)),
                "p_raw": p_val,
                "t_ratio": t_ratio,
            }
        )

    holm_p = _holm_bonferroni(raw_p)
    lines = [
        "# Confirmatory Summary: Native C++20 Online Ordered CTR v2 (`exp-native-ctr-v2`)",
        "",
        "Pre-registered in `benchmarks/results/exp_native_ctr_v2_prereg.md` (16 datasets x 5 folds = 80 paired runs/variant).",
        "",
        "## 1. Anti-Fluke Confirmatory Scorecard (Holm-Bonferroni m=2, 10k bootstrap CI)",
        "",
        "| Comparison | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | p_Holm | Worst % | Time vs S6 | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for cd, ph in zip(comp_data, holm_p, strict=True):
        passed = (
            cd["med"] < -0.5
            and ph < 0.05
            and cd["ci_hi"] < 0.0
            and cd["wins"] >= 11
            and cd["worst"] <= 1.5
            and cd["t_ratio"] <= 3.0
        )
        verdict = "**PASS**" if passed else "FAIL"
        lines.append(
            f"| {cd['label']} | {cd['med']:+.2f}% | [{cd['ci_lo']:+.2f}%, {cd['ci_hi']:+.2f}%] | "
            f"{cd['wins']}/16 | {cd['p_raw']:.4g} | {ph:.4g} | {cd['worst']:+.2f}% | "
            f"{cd['t_ratio']:.2f}x | {verdict} |"
        )

    hi_names = [n for n in all_names if ds_meta[n]["max_card"] > 253]
    lines += [
        "",
        "## 2. Exploratory Comparisons (Primary metric unless noted)",
        "",
        "| Comparison | Subgroup | N | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    exp_specs = [
        ("S6 vs CAT", all_names, agg, "S6", "CAT"),
        ("S6_CTR1 vs CAT", all_names, agg, "S6_CTR1", "CAT"),
        ("S6_CTR2 vs CAT", all_names, agg, "S6_CTR2", "CAT"),
        ("S6_CTR2 vs S6_CTR1", all_names, agg, "S6_CTR2", "S6_CTR1"),
        ("S6_CTR2 vs S6 (1-AUC)", clf_names, agg_one_m_auc, "S6_CTR2", "S6"),
        ("S6_CTR2 vs CAT (1-AUC)", clf_names, agg_one_m_auc, "S6_CTR2", "CAT"),
    ]
    if hi_names:
        exp_specs.extend(
            [
                ("S6_CTR2 vs S6 (max_card>253)", hi_names, agg, "S6_CTR2", "S6"),
                ("S6_CTR2 vs CAT (max_card>253)", hi_names, agg, "S6_CTR2", "CAT"),
            ]
        )
    for label, sub_names, src_agg, cand, base in exp_specs:
        deltas, p_val, (lo, hi) = _paired_summary_row(label, sub_names, src_agg, cand, base)
        lines.append(
            f"| {label} | {len(sub_names)} | {len(sub_names)} | {statistics.median(deltas):+.2f}% | "
            f"[{lo:+.2f}%, {hi:+.2f}%] | {sum(1 for d in deltas if d < 0)}/{len(sub_names)} | "
            f"{p_val:.4g} | {max(deltas):+.2f}% |"
        )

    lines += [
        "",
        "## 3. Per-dataset Primary Metric & Wall Time (5-fold mean; lower is better)",
        "",
        "| kind | dataset | n | p_cat | max_card | S6 | S6_CTR1 | S6_CTR2 | CAT | CTR2 vs S6 | S6 t | CTR2 t | CAT t | mean tbls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _, _ in DATASETS:
        m = ds_meta[name]
        s6_m = agg[(name, "S6")]
        ctr2_m = agg[(name, "S6_CTR2")]
        d_pct = (ctr2_m - s6_m) / s6_m * 100.0
        lines.append(
            f"| {kind} | `{name}` | {m['n_eval']} | {m['p_cat']} | {m['max_card']} | "
            f"{s6_m:.5f} | {agg[(name, 'S6_CTR1')]:.5f} | {ctr2_m:.5f} | "
            f"{agg[(name, 'CAT')]:.5f} | {d_pct:+.2f}% | {agg_secs[(name, 'S6')]:.2f}s | "
            f"{agg_secs[(name, 'S6_CTR2')]:.2f}s | {agg_secs[(name, 'CAT')]:.2f}s | "
            f"{agg_tbl[(name, 'S6_CTR2')]:.1f} |"
        )

    lines += [
        "",
        "## 4. Exploratory Binary Classification ROC-AUC (higher is better)",
        "",
        "| dataset | S6 | S6_CTR1 | S6_CTR2 | CAT |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in clf_names:
        lines.append(
            f"| `{name}` | {1.0 - agg_one_m_auc[(name, 'S6')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_CTR1')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_CTR2')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'CAT')]:.5f} |"
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
