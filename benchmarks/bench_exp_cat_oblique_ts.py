"""Pre-registered benchmark for OOF Target Statistics in oblique splits (exp-cat-oblique-ts).

Frozen pre-registration: benchmarks/results/exp_cat_oblique_ts_prereg.md
Usage:
    python benchmarks/bench_exp_cat_oblique_ts.py pilot
    python benchmarks/bench_exp_cat_oblique_ts.py run --shard 0 --nshards 4
    python benchmarks/bench_exp_cat_oblique_ts.py summarize
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
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "results" / "exp_cat_oblique_ts_prereg.md"
SUMMARY_PATH = HERE / "results" / "exp_cat_oblique_ts_summary.md"
RAW_DIR = HERE / "results" / "raw" / "exp_cat_oblique_ts"

# (kind, name, openml_did)
DATASETS: list[tuple[str, str, int]] = [
    ("reg", "abalone", 44956),
    ("reg", "diamonds", 42225),
    ("reg", "Mercedes_Benz", 42570),
    ("reg", "avocado-sales", 41210),
    ("reg", "nyc-taxi-green-dec-2016", 42729),
    ("reg", "colleges", 42727),
    ("clf", "credit-g", 31),
    ("clf", "bank-marketing", 1461),
    ("clf", "adult", 1590),
    ("clf", "kick", 41162),
    ("clf", "Amazon_employee_access", 4135),
    ("clf", "Click_prediction_small", 42733),
]

VARIANTS = ["B", "TS", "TS_S6", "CAT", "LGB"]
FIELDS = [
    "kind",
    "dataset",
    "did",
    "n_eval",
    "p_num",
    "p_cat",
    "max_card_full",
    "max_card_eval",
    "fold",
    "variant",
    "metric",
    "auc",
    "trees",
    "secs",
]


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def load_dataset(kind: str, did: int) -> tuple[pd.DataFrame, np.ndarray, list[str], int, int]:
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    t = d.default_target_attribute
    X_df, y_s, cat_ind, _ = d.get_data(target=t, dataset_format="dataframe")
    X = X_df.copy()
    cat_cols: list[str] = []
    for col, is_cat in zip(X.columns, cat_ind, strict=True):
        if is_cat or str(X[col].dtype) in ("category", "object", "string"):
            X[col] = X[col].astype("category")
            cat_cols.append(str(col))
        else:
            X[col] = X[col].astype(np.float64)
    X.columns = [str(c) for c in X.columns]
    max_card_full = max((int(X[c].nunique(dropna=True)) for c in cat_cols), default=0)

    if kind == "reg":
        y = y_s.astype("float64").to_numpy()
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.int64)

    n = len(X)
    if n > 30000:
        rng = np.random.default_rng(42)
        if kind == "clf":
            idx0 = np.flatnonzero(y == 0)
            idx1 = np.flatnonzero(y == 1)
            n0 = int(round(25000 * len(idx0) / n))
            n1 = 25000 - n0
            pick = np.sort(
                np.concatenate(
                    [rng.choice(idx0, size=n0, replace=False), rng.choice(idx1, size=n1, replace=False)]
                )
            )
        else:
            pick = np.sort(rng.choice(n, size=25000, replace=False))
        X = X.iloc[pick].reset_index(drop=True)
        y = y[pick]
    else:
        X = X.reset_index(drop=True)

    max_card_eval = max((int(X[c].nunique(dropna=True)) for c in cat_cols), default=0)
    return X, y, cat_cols, max_card_full, max_card_eval


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
) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]

    if variant in ("B", "TS", "TS_S6"):
        import miter

        kw: dict[str, Any] = {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "m_mult": 2,
            "k_proj": 0,
            "max_bin": 255,
            "early_stopping_rounds": 50,
            "n_jobs": 1,
            "random_state": fold,
        }
        if variant == "TS_S6":
            kw.update(
                {
                    "grow_policy": "symmetric",
                    "max_depth": 6,
                    "min_data_in_leaf": 1,
                    "lambda_l2": 3.0,
                    "cat_oblique_ts": True,
                    "cat_ts_smooth": 10.0,
                }
            )
        else:
            kw.update(
                {
                    "grow_policy": "leafwise",
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "lambda_l2": 1.0,
                    "cat_oblique_ts": (variant == "TS"),
                    "cat_ts_smooth": 10.0,
                }
            )
        cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
        m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        pred = m.predict(X_te) if kind == "reg" else m.predict_proba(X_te)[:, 1]
        trees = int(m.best_iteration_)
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
    elif variant == "LGB":
        import lightgbm as lgb

        common_lgb = {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "max_bin": 255,
            "n_jobs": 1,
            "random_state": fold,
            "deterministic": True,
            "verbose": -1,
        }
        if kind == "reg":
            ml = lgb.LGBMRegressor(**common_lgb)
            ml.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = np.asarray(ml.predict(X_te), dtype=np.float64)
        else:
            ml = lgb.LGBMClassifier(**common_lgb)
            ml.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = np.asarray(ml.predict_proba(X_te)[:, 1], dtype=np.float64)
        trees = int(ml.best_iteration_)
    else:
        raise ValueError(f"Unknown variant {variant}")
    return pred, trees, time.perf_counter() - t0


def _split_cv(kind: str, X: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splitter = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if kind == "clf"
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold, (tr_idx, te_idx) in enumerate(splitter.split(X, y if kind == "clf" else None)):
        strat = y[tr_idx] if kind == "clf" else None
        a, b = train_test_split(tr_idx, test_size=0.15, random_state=fold, stratify=strat)
        out.append((a, b, te_idx))
    return out


def cmd_pilot() -> None:
    for kind, name, did in [("clf", "credit-g", 31), ("clf", "Amazon_employee_access", 4135)]:
        X, y, cat_cols, mcf, mce = load_dataset(kind, did)
        a, b, te = _split_cv(kind, X, y)[0]
        print(f"=== PILOT {name} n={len(X)} p_cat={len(cat_cols)} max_card={mcf}/{mce} ===", flush=True)
        for v in VARIANTS:
            pred, trees, secs = fit_predict_variant(kind, v, X, y, cat_cols, a, b, te, 0)
            met, auc = eval_metrics(kind, y[te], pred)
            print(f"  {v:5s}: metric={met:.5f} auc={auc:.5f} trees={trees:4d} secs={secs:.2f}s", flush=True)


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
            X, y, cat_cols, mcf, mce = load_dataset(kind, did)
            p_cat = len(cat_cols)
            p_num = X.shape[1] - p_cat
            splits = _split_cv(kind, X, y)
            for fold, (a, b, te) in enumerate(splits):
                for v in VARIANTS:
                    if (name, fold, v) in done:
                        continue
                    pred, trees, secs = fit_predict_variant(kind, v, X, y, cat_cols, a, b, te, fold)
                    met, auc = eval_metrics(kind, y[te], pred)
                    writer.writerow(
                        {
                            "kind": kind,
                            "dataset": name,
                            "did": did,
                            "n_eval": len(X),
                            "p_num": p_num,
                            "p_cat": p_cat,
                            "max_card_full": mcf,
                            "max_card_eval": mce,
                            "fold": fold,
                            "variant": v,
                            "metric": repr(met),
                            "auc": "" if math.isnan(auc) else repr(auc),
                            "trees": trees,
                            "secs": f"{secs:.3f}",
                        }
                    )
                    fh.flush()
                    print(
                        f"[{shard}/{nshards}] {kind:3s} {name:24s} f={fold} {v:5s}: "
                        f"{met:.5f} trees={trees:4d} {secs:5.2f}s",
                        flush=True,
                    )


def _group_stats(deltas: list[float]) -> tuple[float, float, int, float, str]:
    med = float(statistics.median(deltas))
    mean = float(statistics.fmean(deltas))
    wins = sum(1 for x in deltas if x < 0.0)
    worst = float(max(deltas))
    nz = [x for x in deltas if x != 0.0]
    p_str = "1.0" if not nz else f"{float(wilcoxon(nz, alternative='two-sided').pvalue):.4g}"
    return med, mean, wins, worst, p_str


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
                        "max_card_full": int(row["max_card_full"]),
                        "max_card_eval": int(row["max_card_eval"]),
                        "fold": int(row["fold"]),
                        "variant": row["variant"],
                        "metric": float(row["metric"]),
                        "auc": float("nan") if row["auc"] == "" else float(row["auc"]),
                        "trees": int(row["trees"]),
                        "secs": float(row["secs"]),
                    }
                )

    expected = len(DATASETS) * 5 * len(VARIANTS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows in {RAW_DIR}, got {len(rows)}")

    # Aggregate 5-fold mean per (dataset, variant)
    ds_meta: dict[str, dict[str, Any]] = {}
    agg: dict[tuple[str, str], float] = {}
    agg_auc: dict[tuple[str, str], float] = {}
    agg_secs: dict[tuple[str, str], float] = {}
    for _, name, _ in DATASETS:
        sub = [r for r in rows if r["dataset"] == name]
        ds_meta[name] = {
            "kind": sub[0]["kind"],
            "did": sub[0]["did"],
            "n_eval": sub[0]["n_eval"],
            "p_num": sub[0]["p_num"],
            "p_cat": sub[0]["p_cat"],
            "max_card_full": sub[0]["max_card_full"],
            "max_card_eval": sub[0]["max_card_eval"],
        }
        for v in VARIANTS:
            vs = [r for r in sub if r["variant"] == v]
            assert len(vs) == 5, f"Missing folds for {name} {v}"
            agg[(name, v)] = float(statistics.fmean(r["metric"] for r in vs))
            agg_secs[(name, v)] = float(statistics.fmean(r["secs"] for r in vs))
            if sub[0]["kind"] == "clf":
                agg_auc[(name, v)] = float(statistics.fmean(r["auc"] for r in vs))

    deltas_all = [(agg[(n, "TS")] - agg[(n, "B")]) / agg[(n, "B")] * 100.0 for _, n, _ in DATASETS]
    med_all, mean_all, wins_all, worst_all, p_all = _group_stats(deltas_all)
    verdict_pass = (
        med_all < -0.5
        and float(p_all) < 0.05
        and wins_all >= 9
        and worst_all <= 1.5
    )

    lines: list[str] = [
        "# Experimental Report: Categoricals in Oblique Splits via OOF Target Statistics (`exp-cat-oblique-ts`)",
        "",
        "- Pre-registration: [`exp_cat_oblique_ts_prereg.md`](exp_cat_oblique_ts_prereg.md)",
        f"- Pre-registration SHA-256: `{prereg_sha256()}`",
        f"- Rows collected: `{len(rows)} / {expected}` (12 datasets × 5 folds × 5 variants, `n_jobs=1`)",
        "",
        "## 1. Pre-registered Confirmatory Verdict (`TS` vs `B`, 12 datasets)",
        "",
        "| Comparison | Scope | median % vs B | mean % vs B | wins | Wilcoxon p | worst dataset % | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| **TS vs B** | All (12) | {med_all:+.2f}% | {mean_all:+.2f}% | {wins_all}/12 | {p_all} | {worst_all:+.2f}% | **{'PASS' if verdict_pass else 'FAIL'}** |",
    ]

    subgroups = [
        ("High-card (`max_card > 253`)", [n for _, n, _ in DATASETS if ds_meta[n]["max_card_full"] > 253]),
        ("Med/High-card (`max_card > 50`)", [n for _, n, _ in DATASETS if ds_meta[n]["max_card_full"] > 50]),
        ("Low/Med-card (`max_card <= 50`)", [n for _, n, _ in DATASETS if ds_meta[n]["max_card_full"] <= 50]),
        ("Regression (6)", [n for k, n, _ in DATASETS if k == "reg"]),
        ("Classification (6)", [n for k, n, _ in DATASETS if k == "clf"]),
    ]

    lines += [
        "",
        "## 2. Exploratory Subgroup Breakdowns & Reference Comparisons",
        "",
        "| Variant / Slice | Scope | median % | mean % | wins | Wilcoxon p | worst % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, names in subgroups:
        ds_d = [(agg[(n, "TS")] - agg[(n, "B")]) / agg[(n, "B")] * 100.0 for n in names]
        m, mn, w, wr, pv = _group_stats(ds_d)
        lines.append(f"| TS vs B | {label} ({len(names)}) | {m:+.2f}% | {mn:+.2f}% | {w}/{len(names)} | {pv} | {wr:+.2f}% |")

    for ref in ("CAT", "LGB"):
        d_ts = [(agg[(n, "TS")] - agg[(n, ref)]) / agg[(n, ref)] * 100.0 for _, n, _ in DATASETS]
        m, mn, w, wr, pv = _group_stats(d_ts)
        lines.append(f"| TS vs {ref} | All (12) | {m:+.2f}% | {mn:+.2f}% | {w}/12 | {pv} | {wr:+.2f}% |")
        d_s6 = [(agg[(n, "TS_S6")] - agg[(n, ref)]) / agg[(n, ref)] * 100.0 for _, n, _ in DATASETS]
        m, mn, w, wr, pv = _group_stats(d_s6)
        lines.append(f"| TS_S6 vs {ref} | All (12) | {m:+.2f}% | {mn:+.2f}% | {w}/12 | {pv} | {wr:+.2f}% |")

    d_s6_b = [(agg[(n, "TS_S6")] - agg[(n, "B")]) / agg[(n, "B")] * 100.0 for _, n, _ in DATASETS]
    m, mn, w, wr, pv = _group_stats(d_s6_b)
    lines.append(f"| TS_S6 vs B | All (12) | {m:+.2f}% | {mn:+.2f}% | {w}/12 | {pv} | {wr:+.2f}% |")

    lines += [
        "",
        "## 3. Per-dataset Results (5-fold mean; lower metric is better)",
        "",
        "| kind | dataset | n | p_cat | max_card | B | TS | TS vs B % | TS_S6 | CAT | LGB | B time | TS time |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _ in DATASETS:
        meta = ds_meta[name]
        b_m = agg[(name, "B")]
        ts_m = agg[(name, "TS")]
        d_pct = (ts_m - b_m) / b_m * 100.0
        lines.append(
            f"| {kind} | `{name}` | {meta['n_eval']} | {meta['p_cat']} | {meta['max_card_full']} ({meta['max_card_eval']}) | "
            f"{b_m:.5f} | {ts_m:.5f} | {d_pct:+.2f}% | {agg[(name, 'TS_S6')]:.5f} | "
            f"{agg[(name, 'CAT')]:.5f} | {agg[(name, 'LGB')]:.5f} | "
            f"{agg_secs[(name, 'B')]:.2f}s | {agg_secs[(name, 'TS')]:.2f}s |"
        )

    lines += [
        "",
        "## 4. Exploratory Classification ROC-AUC (5-fold mean, higher is better)",
        "",
        "| dataset | B | TS | TS_S6 | CAT | LGB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind, name, _ in DATASETS:
        if kind != "clf":
            continue
        lines.append(
            f"| `{name}` | {agg_auc[(name, 'B')]:.5f} | {agg_auc[(name, 'TS')]:.5f} | "
            f"{agg_auc[(name, 'TS_S6')]:.5f} | {agg_auc[(name, 'CAT')]:.5f} | {agg_auc[(name, 'LGB')]:.5f} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
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
