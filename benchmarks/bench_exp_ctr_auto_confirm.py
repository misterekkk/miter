"""Confirmatory benchmark: Adaptive Native CTR (`exp-ctr-auto-confirm`) on 16 unseen datasets.

Pre-registered in `benchmarks/results/exp_ctr_auto_confirm_prereg.md`.

Usage:
    .venv/bin/python benchmarks/bench_exp_ctr_auto_confirm.py pilot
    .venv/bin/python benchmarks/bench_exp_ctr_auto_confirm.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_ctr_auto_confirm.py summarize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from bench_exp_native_ctr_v2 import _bootstrap_median_ci, _holm_bonferroni
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "results" / "exp_ctr_auto_confirm_prereg.md"
RAW_DIR = HERE / "results" / "raw" / "exp_ctr_auto_confirm"
SUMMARY_PATH = HERE / "results" / "exp_ctr_auto_confirm_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# (kind, name, did) — ZERO overlap with exp_native_ctr_v2 / exp_ctr_auto / exp_ctr_auto_v2
DATASETS: list[tuple[str, str, int]] = [
    ("clf", "Diabetes130US", 46922),
    ("clf", "E-CommereShippingData", 46924),
    ("clf", "Fitness_Club", 46927),
    ("clf", "HR_Analytics_Job_Change", 46935),
    ("clf", "Is-this-a-good-customer", 46938),
    ("clf", "Marketing_Campaign", 46940),
    ("clf", "churn", 46915),
    ("clf", "coil2000", 46916),
    ("clf", "credit_card_clients_default", 46919),
    ("clf", "customer_satisfaction_in_airline", 46920),
    ("clf", "kddcup09_appetency", 46939),
    ("clf", "qsar-biodeg", 46952),
    ("mc", "anneal", 46906),
    ("mc", "splice", 46958),
    ("mc", "students_dropout", 46960),
    ("reg", "healthcare_insurance_expenses", 46931),
]

VARIANTS = ["S6", "S6_AUTO_P4", "S6_NC_SMART_P4", "CAT"]
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


def load_dataset(kind: str, did: int) -> tuple[pd.DataFrame, np.ndarray, list[str], int]:
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
        if kind == "reg":
            rng = np.random.default_rng(42)
            pick = np.sort(rng.choice(n, size=25000, replace=False))
        else:
            _, pick = train_test_split(np.arange(n), test_size=25000, random_state=42, stratify=y)
            pick = np.sort(pick)
        X = X.iloc[pick].reset_index(drop=True)
        y = y[pick]
    else:
        X = X.reset_index(drop=True)

    max_card = max((int(X[c].nunique(dropna=True)) for c in cat_cols), default=0)
    return X, y, cat_cols, max_card


def eval_metrics(kind: str, y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if kind == "reg":
        return float(np.sqrt(np.mean((pred - y_true) ** 2)) / np.std(y_true)), float("nan")
    if kind == "clf":
        p = np.clip(pred, 1e-15, 1.0 - 1e-15)
        logloss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
        auc = float(roc_auc_score(y_true, pred))
        return logloss, auc
    probs = np.clip(pred[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(probs))), float("nan")


def _prep_catboost_df(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        s = Xc[c].astype(object)
        Xc[c] = s.where(pd.notna(s), "__MITER_MISSING__").astype(str)
    return Xc


def _split_cv(
    kind: str, X: pd.DataFrame, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splitter = (
        KFold(n_splits=5, shuffle=True, random_state=42)
        if kind == "reg"
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splitter.split(X, None if kind == "reg" else y):
        strat = None if kind == "reg" else y[tr_idx]
        a, b = train_test_split(tr_idx, test_size=0.2, random_state=0, stratify=strat)
        out.append((a, b, te_idx))
    return out


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

    if variant in ("S6", "S6_AUTO_P4", "S6_NC_SMART_P4"):
        import miter

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
        }
        if variant == "S6_AUTO_P4":
            kw.update(
                {
                    "max_ctr_complexity": "auto",
                    "ctr_model_size_reg": 0.5,
                    "ctr_num_combinations": False,
                    "ctr_n_permutations": 4,
                }
            )
        elif variant == "S6_NC_SMART_P4":
            kw.update(
                {
                    "max_ctr_complexity": "auto",
                    "ctr_model_size_reg": 0.5,
                    "ctr_num_combinations": True,
                    "ctr_n_permutations": 4,
                    "ctr_num_max_depth": 1,
                    "ctr_num_skip_onehot": True,
                    "ctr_num_gain_margin": 1.25,
                    "ctr_new_table_penalty": 0.25,
                }
            )
        cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
        m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        if kind == "reg":
            pred = m.predict(X_te)
        elif kind == "clf":
            pred = m.predict_proba(X_te)[:, 1]
        else:
            pred = m.predict_proba(X_te)
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
            loss_fn = "Logloss" if kind == "clf" else "MultiClass"
            mc = CatBoostClassifier(loss_function=loss_fn, **common)
            mc.fit(Xc_tr, y_tr, eval_set=(Xc_val, y_val))
            proba = np.asarray(mc.predict_proba(Xc_te), dtype=np.float64)
            pred = proba[:, 1] if kind == "clf" else proba
        trees = int(mc.get_best_iteration() + 1)
        n_tbl = 0
        n_proj = 0
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    return pred, trees, n_tbl, n_proj, time.perf_counter() - t0


def cmd_pilot() -> None:
    for kind, name, did in [
        ("clf", "Fitness_Club", 46927),
        ("clf", "HR_Analytics_Job_Change", 46935),
        ("mc", "anneal", 46906),
    ]:
        X, y, cat_cols, mcard = load_dataset(kind, did)
        a, b, te = _split_cv(kind, X, y)[0]
        print(
            f"=== PILOT {kind} {name} did={did} n={len(X)} p_cat={len(cat_cols)} max_card={mcard} ===",
            flush=True,
        )
        for v in VARIANTS:
            pred, trees, n_tbl, n_proj, secs = fit_predict_variant(
                kind, v, X, y, cat_cols, a, b, te, 0
            )
            met, auc = eval_metrics(kind, y[te], pred)
            auc_s = f"{auc:.5f}" if not math.isnan(auc) else "  nan  "
            print(
                f"  {v:15s}: metric={met:.5f} auc={auc_s} trees={trees:4d} "
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

        for kind, name, did in selected:
            X, y, cat_cols, mcard = load_dataset(kind, did)
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
                        f"[{shard}/{nshards}] {kind:3s} {name:30s} f={fold} {v:15s}: "
                        f"{met:.5f} trees={trees:4d} tbl={n_tbl:3d} {secs:5.2f}s",
                        flush=True,
                    )


def _paired_stats(
    names: list[str], agg: dict[tuple[str, str], float], cand: str, base: str
) -> dict[str, Any]:
    deltas = [(agg[(n, cand)] - agg[(n, base)]) / agg[(n, base)] * 100.0 for n in names]
    nz = [d for d in deltas if abs(d) > 1e-12]
    p_val = 1.0 if not nz else float(wilcoxon(nz, alternative="two-sided").pvalue)
    ci_lo, ci_hi = _bootstrap_median_ci(deltas, n_boot=10000, seed=2026)
    return {
        "mean": float(statistics.fmean(deltas)),
        "med": float(statistics.median(deltas)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wins": sum(1 for d in deltas if d < -1e-12),
        "worst": float(max(deltas)),
        "p_raw": p_val,
    }


def cmd_summarize() -> None:
    sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
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
    all_names = [n for _, n, _ in DATASETS]
    clf_names = [n for k, n, _ in DATASETS if k == "clf"]

    for _, name, _ in DATASETS:
        sub_all = [r for r in rows if r["dataset"] == name]
        if not sub_all:
            raise SystemExit(f"Missing dataset {name}")
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

    conf_specs = [
        ("Primary 1: S6_AUTO_P4 vs S6", "S6_AUTO_P4", "S6", "primary"),
        ("Primary 2: S6_NC_SMART_P4 vs S6", "S6_NC_SMART_P4", "S6", "primary"),
        ("Head-to-head: S6_NC_SMART_P4 vs S6_AUTO_P4", "S6_NC_SMART_P4", "S6_AUTO_P4", "h2h"),
    ]
    raw_ps: list[float] = []
    stats_list: list[dict[str, Any]] = []
    for _, cand, base, _ in conf_specs:
        st = _paired_stats(all_names, agg, cand, base)
        raw_ps.append(st["p_raw"])
        st["t_ratio"] = float(
            statistics.median(
                agg_secs[(n, cand)] / max(agg_secs[(n, "S6")], 1e-6) for n in all_names
            )
        )
        stats_list.append(st)

    holm = _holm_bonferroni(raw_ps)
    lines = [
        "# Confirmatory Summary: Adaptive Native CTR (`exp-ctr-auto-confirm`)",
        "",
        f"- Pre-registration: `benchmarks/results/exp_ctr_auto_confirm_prereg.md` (SHA-256: `{sha}`)",
        "- Scope: 16 completely unseen OpenML datasets × 5 folds = 80 paired runs/variant (`n_jobs=1`).",
        "",
        "## 1. Anti-Fluke Confirmatory Scorecard (Holm-Bonferroni `m=3`, 10k bootstrap CI)",
        "",
        "| Comparison | Mean % | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | p_Holm | Worst % | Time vs S6 | Verdict |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for (label, _, _, rule), st, ph in zip(conf_specs, stats_list, holm, strict=True):
        if rule == "primary":
            passed = (
                st["med"] < -0.25
                and ph < 0.05
                and st["ci_hi"] < 0.0
                and st["wins"] >= 11
                and st["worst"] <= 1.5
                and st["t_ratio"] <= 3.0
            )
        else:
            passed = ph < 0.05 and st["med"] < -0.10 and st["wins"] >= 10
        verdict = "**PASS**" if passed else ("FAIL (PARSIMONY -> S6_AUTO_P4)" if rule == "h2h" else "FAIL")
        lines.append(
            f"| {label} | {st['mean']:+.2f}% | {st['med']:+.2f}% | "
            f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/16 | "
            f"{st['p_raw']:.4g} | {ph:.4g} | {st['worst']:+.2f}% | {st['t_ratio']:.2f}x | {verdict} |"
        )

    hi_names = [n for n in all_names if ds_meta[n]["max_card"] > 10]
    lines += [
        "",
        "## 2. Exploratory Comparisons (Primary metric unless noted)",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    exp_specs = [
        ("S6 vs CAT", all_names, agg, "S6", "CAT"),
        ("S6_AUTO_P4 vs CAT", all_names, agg, "S6_AUTO_P4", "CAT"),
        ("S6_NC_SMART_P4 vs CAT", all_names, agg, "S6_NC_SMART_P4", "CAT"),
        ("S6_AUTO_P4 vs S6 (1-AUC)", clf_names, agg_one_m_auc, "S6_AUTO_P4", "S6"),
        ("S6_NC_SMART_P4 vs S6 (1-AUC)", clf_names, agg_one_m_auc, "S6_NC_SMART_P4", "S6"),
        ("S6_AUTO_P4 vs CAT (1-AUC)", clf_names, agg_one_m_auc, "S6_AUTO_P4", "CAT"),
        ("S6_NC_SMART_P4 vs CAT (1-AUC)", clf_names, agg_one_m_auc, "S6_NC_SMART_P4", "CAT"),
    ]
    if hi_names:
        exp_specs.extend(
            [
                ("S6_AUTO_P4 vs S6 (max_card>10)", hi_names, agg, "S6_AUTO_P4", "S6"),
                ("S6_NC_SMART_P4 vs S6 (max_card>10)", hi_names, agg, "S6_NC_SMART_P4", "S6"),
            ]
        )
    for label, sub_names, src_agg, cand, base in exp_specs:
        st = _paired_stats(sub_names, src_agg, cand, base)
        lines.append(
            f"| {label} | {len(sub_names)} | {st['mean']:+.2f}% | {st['med']:+.2f}% | "
            f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/{len(sub_names)} | "
            f"{st['p_raw']:.4g} | {st['worst']:+.2f}% |"
        )

    lines += [
        "",
        "## 3. Per-dataset Primary Metric, Wall Time & CTR Tables (5-fold mean; lower is better)",
        "",
        "| kind | dataset | n | p_cat | max_card | S6 | S6_AUTO_P4 | SMART_P4 | CAT | AUTO_P4 vs S6 | SMART vs S6 | S6 t | AUTO_P4 t | SMART t | CAT t | tbl:AUTO_P4 | tbl:SMART |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _ in DATASETS:
        m = ds_meta[name]
        s6 = agg[(name, "S6")]
        ap = agg[(name, "S6_AUTO_P4")]
        sm = agg[(name, "S6_NC_SMART_P4")]
        ct = agg[(name, "CAT")]
        lines.append(
            f"| {kind} | `{name}` | {m['n_eval']} | {m['p_cat']} | {m['max_card']} | "
            f"{s6:.5f} | {ap:.5f} | {sm:.5f} | {ct:.5f} | "
            f"{(ap - s6) / s6 * 100.0:+.2f}% | {(sm - s6) / s6 * 100.0:+.2f}% | "
            f"{agg_secs[(name, 'S6')]:.2f}s | {agg_secs[(name, 'S6_AUTO_P4')]:.2f}s | "
            f"{agg_secs[(name, 'S6_NC_SMART_P4')]:.2f}s | {agg_secs[(name, 'CAT')]:.2f}s | "
            f"{agg_tbl[(name, 'S6_AUTO_P4')]:.1f} | {agg_tbl[(name, 'S6_NC_SMART_P4')]:.1f} |"
        )

    lines += [
        "",
        "## 4. Exploratory Binary Classification ROC-AUC (higher is better)",
        "",
        "| dataset | S6 | S6_AUTO_P4 | SMART_P4 | CAT |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in clf_names:
        lines.append(
            f"| `{name}` | {1.0 - agg_one_m_auc[(name, 'S6')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_AUTO_P4')]:.5f} | "
            f"{1.0 - agg_one_m_auc[(name, 'S6_NC_SMART_P4')]:.5f} | "
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
