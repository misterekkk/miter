"""Exploratory Out-of-TabArena External Holdout Validation (`exp-external-holdout`).

Strictly verifies our adaptive CTR guard (<1500 rows or <300 rows/class -> require card > 16),
sample-size complexity penalty (`ctr_model_size_reg=0.5`), and `quantizer="auto"` on OpenML-CC18 /
OpenML-CTR23 datasets with ZERO overlap against TabArena 51 and prior exploratory benchmarks.

Usage:
    .venv/bin/python benchmarks/bench_exp_external_holdout.py pilot
    .venv/bin/python benchmarks/bench_exp_external_holdout.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_external_holdout.py summarize
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from bench_exp_native_ctr_v2 import _bootstrap_median_ci
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "results" / "raw" / "exp_external_holdout"
SUMMARY_PATH = HERE / "results" / "exp_external_holdout_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# Zero-overlap Suite A: Categorical & Mixed datasets (small/multiclass guard + medium/large CTR)
SUITE_A_DATASETS: list[tuple[str, str, int]] = [
    ("clf", "credit-approval", 29),
    ("mc", "analcatdata_dmft", 469),
    ("mc", "vowel", 307),
    ("reg", "student_performance_por", 44967),
    ("reg", "solar_flare", 44966),
    ("clf", "cylinder-bands", 6332),
    ("clf", "dresses-sales", 23381),
    ("mc", "eucalyptus", 188),
    ("reg", "Moneyball", 41021),
    ("reg", "socmob", 44987),
    ("mc", "car", 40975),
    ("mc", "dna", 40670),
    ("clf", "sick", 38),
    ("reg", "kings_county", 44989),
    ("reg", "fps_benchmark", 44992),
    ("clf", "nomao", 1486),
]
SUITE_A_VARIANTS = ["S6", "S6_OLD_AUTO_P4", "S6_AUTO_P4", "S6_AUTO_NOREG", "CAT"]

# Zero-overlap Suite B: Numeric-heavy datasets (`quantizer="auto"` vs linear / quantile)
SUITE_B_DATASETS: list[tuple[str, str, int]] = [
    ("reg", "forest_fires", 44962),
    ("reg", "energy_efficiency", 44960),
    ("reg", "cars", 44994),
    ("reg", "geographical_origin_of_music", 44965),
    ("reg", "space_ga", 45402),
    ("reg", "cpu_activity", 44978),
    ("reg", "kin8nm", 44980),
    ("reg", "grid_stability", 44973),
    ("reg", "naval_propulsion_plant", 44969),
    ("reg", "wave_energy", 44975),
    ("reg", "sarcos", 44976),
    ("clf", "spambase", 44),
    ("clf", "ozone-level-8hr", 1487),
    ("mc", "segment", 40984),
    ("mc", "first-order-theorem-proving", 1475),
    ("mc", "GesturePhaseSegmentationProcessed", 4538),
]
SUITE_B_VARIANTS = ["LIN", "AUTO", "QUANT", "S6_LIN", "S6_AUTO"]

FIELDS = [
    "suite",
    "kind",
    "dataset",
    "did",
    "n_eval",
    "n_classes",
    "p_num",
    "p_cat",
    "max_card",
    "fold",
    "variant",
    "metric",
    "auc",
    "trees",
    "aux_count",
    "old_auto",
    "new_auto",
    "guard_flipped",
    "secs",
]

EXCLUDED_NAMES = [
    "airfoil_self_noise", "Amazon_employee_access", "anneal",
    "Another-Dataset-on-used-Fiat-500", "APSFailure", "Bank_Customer_Churn",
    "bank-marketing", "Bioresponse", "blood-transfusion-service-center",
    "churn", "coil2000_insurance_policies", "coil2000", "concrete_compressive_strength",
    "credit-g", "credit_card_clients_default", "customer_satisfaction_in_airline",
    "diabetes", "Diabetes130US", "diamonds", "E-CommereShippingData", "Fitness_Club",
    "Food_Delivery_Time", "GiveMeSomeCredit", "hazelnut-spread-contaminant-detection",
    "healthcare_insurance_expenses", "health_insurance", "heloc", "hiva_agnostic",
    "houses", "california_housing", "HR_Analytics_Job_Change_of_Data_Scientists",
    "HR_Analytics_Job_Change", "in_vehicle_coupon_recommendation",
    "Is-this-a-good-customer", "jm1", "kddcup09_appetency", "Marketing_Campaign",
    "maternal_health_risk", "miami_housing", "MIC", "NATICUSdroid",
    "online_shoppers_intention", "physiochemical_protein", "polish_companies_bankruptcy",
    "QSAR-TID-11", "qsar-biodeg", "QSAR_fish_toxicity", "SDSS17", "seismic-bumps",
    "splice", "students_dropout_and_academic_success", "students_dropout",
    "superconductivity", "taiwanese_bankruptcy_prediction", "website_phishing",
    "wine_quality", "white_wine", "red_wine", "abalone", "Mercedes_Benz",
    "avocado-sales", "nyc-taxi-green-dec-2016", "colleges", "adult", "kick",
    "Click_prediction_small", "brazilian_houses",
]


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def assert_zero_overlap() -> None:
    excl = {_norm_name(x) for x in EXCLUDED_NAMES}
    seen_dids: set[int] = set()
    for suite_name, ds_list in (("A", SUITE_A_DATASETS), ("B", SUITE_B_DATASETS)):
        for _, name, did in ds_list:
            if _norm_name(name) in excl:
                raise AssertionError(f"Overlap detected in Suite {suite_name}: {name} (did={did})")
            if did in seen_dids:
                raise AssertionError(f"Duplicate dataset id {did} across Suite A/B")
            seen_dids.add(did)


def load_dataset(kind: str, did: int) -> tuple[pd.DataFrame, np.ndarray, list[str], int, int]:
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
        n_classes = 1
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.int64)
        n_classes = int(len(np.unique(y)))

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
    return X, y, cat_cols, max_card, n_classes


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


def _auto_complexities_on_train(
    X_tr: pd.DataFrame, cat_cols: list[str], n_classes: int, onehot_max: int = 4
) -> tuple[int, int]:
    p_cat = len(cat_cols)
    if p_cat == 0:
        return 0, 0
    n = len(X_tr)
    cards = [int(X_tr[c].nunique(dropna=False)) for c in cat_cols]
    n_high = sum(1 for c in cards if c > onehot_max)
    n_very_high = sum(1 for c in cards if c > max(onehot_max, 16))

    # Pre-guard rule (before small-sample / multiclass density fix)
    if p_cat == 1 or n < 1500:
        old_auto = 1 if n_high >= 1 else 0
    elif n < 8000:
        old_auto = 2
    elif p_cat >= 3:
        old_auto = 3
    else:
        old_auto = 2

    # Post-guard rule matching src/core/ctr.cpp resolve_effective_ctr_complexity
    if n < 1500 or n < 300 * max(n_classes, 1):
        new_auto = 1 if n_very_high >= 1 else 0
    elif p_cat == 1:
        new_auto = 1 if n_high >= 1 else 0
    elif n < 8000:
        new_auto = 2
    elif p_cat >= 3:
        new_auto = 3
    else:
        new_auto = 2

    return old_auto, new_auto


def _prep_catboost_df(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        s = Xc[c].astype(object)
        Xc[c] = s.where(pd.notna(s), "__MITER_MISSING__").astype(str)
    return Xc


def fit_predict_suite_a(
    kind: str,
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
    old_auto: int,
) -> tuple[np.ndarray, int, int, float]:
    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]

    if variant == "CAT":
        from catboost import CatBoostClassifier, CatBoostRegressor

        Xc_tr = _prep_catboost_df(X_tr, cat_cols)
        Xc_val = _prep_catboost_df(X_val, cat_cols)
        Xc_te = _prep_catboost_df(X_te, cat_cols)
        common: dict[str, Any] = {
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
            mc = CatBoostRegressor(**common).fit(Xc_tr, y_tr, eval_set=(Xc_val, y_val))
            pred = np.asarray(mc.predict(Xc_te), dtype=np.float64)
        else:
            loss_fn = "Logloss" if kind == "clf" else "MultiClass"
            mc = CatBoostClassifier(loss_function=loss_fn, **common).fit(
                Xc_tr, y_tr, eval_set=(Xc_val, y_val)
            )
            proba = np.asarray(mc.predict_proba(Xc_te), dtype=np.float64)
            pred = proba[:, 1] if kind == "clf" else proba
        return pred, int(mc.get_best_iteration() + 1), 0, time.perf_counter() - t0

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
        kw.update({"max_ctr_complexity": "auto", "ctr_model_size_reg": 0.5, "ctr_n_permutations": 4})
    elif variant == "S6_OLD_AUTO_P4":
        kw.update({"max_ctr_complexity": old_auto, "ctr_model_size_reg": 0.5, "ctr_n_permutations": 4})
    elif variant == "S6_AUTO_NOREG":
        kw.update({"max_ctr_complexity": "auto", "ctr_model_size_reg": 0.0, "ctr_n_permutations": 4})

    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    pred = m.predict(X_te) if kind == "reg" else (m.predict_proba(X_te)[:, 1] if kind == "clf" else m.predict_proba(X_te))
    return pred, int(m.best_iteration_), int(m._model.n_ctr_tables), time.perf_counter() - t0


def fit_predict_suite_b(
    kind: str,
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, int, int, float]:
    import miter

    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]
    qmode = "linear" if variant in ("LIN", "S6_LIN") else ("quantile" if variant == "QUANT" else "auto")
    is_sym = variant.startswith("S6")
    kw: dict[str, Any] = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "grow_policy": "symmetric" if is_sym else "leafwise",
        "num_leaves": 31,
        "max_depth": 6 if is_sym else -1,
        "min_data_in_leaf": 1 if is_sym else 20,
        "lambda_l2": 3.0 if is_sym else 1.0,
        "m_mult": 2,
        "k_proj": 0,
        "max_bin": 255,
        "quantizer": qmode,
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "random_state": fold,
    }
    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    pred = m.predict(X_te) if kind == "reg" else (m.predict_proba(X_te)[:, 1] if kind == "clf" else m.predict_proba(X_te))
    return pred, int(m.best_iteration_), int(m._model.n_quantile_features), time.perf_counter() - t0


def _csv_path(suite: str, name: str, fold: int) -> Path:
    return RAW_DIR / f"suite_{suite}_{name}_f{fold}.csv"


def run_job(suite: str, kind: str, name: str, did: int, fold: int) -> None:
    p = _csv_path(suite, name, fold)
    variants = SUITE_A_VARIANTS if suite == "A" else SUITE_B_VARIANTS
    if p.exists():
        with p.open() as f:
            done = {r["variant"] for r in csv.DictReader(f)}
        if set(variants).issubset(done):
            return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    X, y, cat_cols, max_card, n_classes = load_dataset(kind, did)
    a, b, te = _split_cv(kind, X, y)[fold]
    old_auto, new_auto = _auto_complexities_on_train(X.iloc[a], cat_cols, n_classes)
    flipped = int(old_auto != new_auto)

    rows = []
    for v in variants:
        if suite == "A":
            pred, trees, aux, secs = fit_predict_suite_a(
                kind, v, X, y, cat_cols, a, b, te, fold, old_auto
            )
        else:
            pred, trees, aux, secs = fit_predict_suite_b(kind, v, X, y, a, b, te, fold)
        met, auc = eval_metrics(kind, y[te], pred)
        rows.append(
            {
                "suite": suite,
                "kind": kind,
                "dataset": name,
                "did": did,
                "n_eval": len(X),
                "n_classes": n_classes,
                "p_num": X.shape[1] - len(cat_cols),
                "p_cat": len(cat_cols),
                "max_card": max_card,
                "fold": fold,
                "variant": v,
                "metric": f"{met:.12g}",
                "auc": f"{auc:.12g}" if not math.isnan(auc) else "nan",
                "trees": trees,
                "aux_count": aux,
                "old_auto": old_auto,
                "new_auto": new_auto,
                "guard_flipped": flipped,
                "secs": f"{secs:.6f}",
            }
        )
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"DONE suite={suite} {name} f{fold} old_auto={old_auto} new_auto={new_auto} n={len(X)}", flush=True)


def cmd_pilot() -> None:
    assert_zero_overlap()
    for suite, kind, name, did in [
        ("A", "clf", "credit-approval", 29),
        ("B", "reg", "forest_fires", 44962),
    ]:
        X, y, cat_cols, max_card, n_classes = load_dataset(kind, did)
        a, b, te = _split_cv(kind, X, y)[0]
        old_auto, new_auto = _auto_complexities_on_train(X.iloc[a], cat_cols, n_classes)
        print(
            f"=== PILOT Suite {suite} {name} did={did} n={len(X)} K={n_classes} "
            f"p_cat={len(cat_cols)} max_card={max_card} old_auto={old_auto} new_auto={new_auto} ===",
            flush=True,
        )
        variants = SUITE_A_VARIANTS if suite == "A" else SUITE_B_VARIANTS
        for v in variants:
            if suite == "A":
                pred, trees, aux, secs = fit_predict_suite_a(
                    kind, v, X, y, cat_cols, a, b, te, 0, old_auto
                )
            else:
                pred, trees, aux, secs = fit_predict_suite_b(kind, v, X, y, a, b, te, 0)
            met, auc = eval_metrics(kind, y[te], pred)
            print(f"  {v:16s} met={met:.5f} auc={auc:.5f} trees={trees:4d} aux={aux:3d} {secs:.2f}s", flush=True)


def cmd_run(shard: int, nshards: int) -> None:
    assert_zero_overlap()
    jobs = (
        [("A", k, n, d, f) for k, n, d in SUITE_A_DATASETS for f in range(5)]
        + [("B", k, n, d, f) for k, n, d in SUITE_B_DATASETS for f in range(5)]
    )
    mine = [j for idx, j in enumerate(jobs) if idx % nshards == shard]
    print(f"Shard {shard}/{nshards}: {len(mine)}/{len(jobs)} (ds,fold) jobs", flush=True)
    for suite, kind, name, did, fold in mine:
        run_job(suite, kind, name, did, fold)


def _paired_summary(df: pd.DataFrame, datasets: list[str], a: str, b: str) -> dict[str, Any]:
    piv = df.pivot_table(index="dataset", columns="variant", values="metric", aggfunc="mean")
    tim = df.pivot_table(index="dataset", columns="variant", values="secs", aggfunc="sum")
    ds_ok = [d for d in datasets if d in piv.index and a in piv.columns and b in piv.columns]
    piv = piv.loc[ds_ok, [a, b]].dropna()
    rel = (100.0 * (piv[a] - piv[b]) / piv[b]).to_numpy(dtype=np.float64)
    wins = int((rel < 0.0).sum())
    ties = int((rel == 0.0).sum())
    p = float(wilcoxon(rel).pvalue) if np.any(rel != 0.0) else 1.0
    ci_lo, ci_hi = _bootstrap_median_ci(rel, n_boot=10000, seed=2026)
    t_ratio = float(tim.loc[piv.index, a].sum() / max(tim.loc[piv.index, b].sum(), 1e-9))
    return {
        "n": len(rel),
        "wins": wins,
        "ties": ties,
        "mean": float(np.mean(rel)),
        "median": float(np.median(rel)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p": p,
        "worst": float(np.max(rel)),
        "t_ratio": t_ratio,
    }


def _stat_row(label: str, st: dict[str, Any]) -> str:
    return (
        f"| {label} | {st['n']} | {st['mean']:+.2f}% | {st['median']:+.2f}% | "
        f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/{st['n']} ({st['ties']}t) | "
        f"{st['p']:.4g} | {st['worst']:+.2f}% | {st['t_ratio']:.2f}x |"
    )


def cmd_summarize() -> None:
    assert_zero_overlap()
    all_rows = []
    for fpath in sorted(RAW_DIR.glob("suite_*.csv")):
        with fpath.open() as f:
            for r in csv.DictReader(f):
                r["metric"] = float(r["metric"])
                r["secs"] = float(r["secs"])
                r["trees"] = int(r["trees"])
                r["aux_count"] = int(r["aux_count"])
                r["old_auto"] = int(r["old_auto"])
                r["new_auto"] = int(r["new_auto"])
                r["guard_flipped"] = int(r["guard_flipped"])
                r["n_eval"] = int(r["n_eval"])
                r["n_classes"] = int(r["n_classes"])
                r["p_num"] = int(r["p_num"])
                r["p_cat"] = int(r["p_cat"])
                r["max_card"] = int(r["max_card"])
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df_a = df[df["suite"] == "A"].copy()
    df_b = df[df["suite"] == "B"].copy()

    ds_a = [n for _, n, _ in SUITE_A_DATASETS]
    ds_b = [n for _, n, _ in SUITE_B_DATASETS]

    # Guard subgroups in Suite A
    a_meta = df_a[df_a["variant"] == "S6_AUTO_P4"].groupby("dataset").agg(
        old_auto=("old_auto", "max"),
        new_auto=("new_auto", "max"),
        guard_flipped=("guard_flipped", "max"),
        max_card=("max_card", "max"),
        n_eval=("n_eval", "max"),
        n_classes=("n_classes", "max"),
    )
    ds_a_flipped = [d for d in ds_a if d in a_meta.index and a_meta.loc[d, "guard_flipped"] > 0]
    ds_a_small_hi = [
        d for d in ds_a
        if d in a_meta.index
        and a_meta.loc[d, "guard_flipped"] == 0
        and (a_meta.loc[d, "n_eval"] < 1500 or a_meta.loc[d, "n_eval"] < 300 * max(a_meta.loc[d, "n_classes"], 1))
        and a_meta.loc[d, "new_auto"] >= 1
    ]
    ds_a_med_hi = [
        d for d in ds_a
        if d in a_meta.index and a_meta.loc[d, "new_auto"] >= 2
    ]

    # Triggered subgroup in Suite B (nq >= 1)
    b_meta = df_b[df_b["variant"] == "S6_AUTO"].groupby("dataset").agg(nq=("aux_count", "mean"))
    ds_b_trig = [d for d in ds_b if d in b_meta.index and b_meta.loc[d, "nq"] >= 1.0]

    lines = [
        "# Out-of-TabArena External Holdout Validation (`exp-external-holdout`)",
        "",
        "- **Purpose:** Exploratory out-of-sample check of our heuristic thresholds (`max_ctr_complexity='auto'` guard `< 1500` / `< 300*K` / `card > 16`, `ctr_model_size_reg=0.5`, and `quantizer='auto'`) on OpenML-CC18 / OpenML-CTR23 datasets with **ZERO overlap** against TabArena 51 and all prior `exp_*` benchmarks.",
        "- Protocol: 5-fold stratified/K-fold CV, `n_jobs=1`, `random_state=fold`, early stopping `50` rounds.",
        "",
        "## Suite A: Categorical & Mixed Guard + Regularization Check (`N=16` unseen non-TabArena datasets)",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % | Time |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
        _stat_row("S6_AUTO_P4 vs S6 (all)", _paired_summary(df_a, ds_a, "S6_AUTO_P4", "S6")),
        _stat_row("S6_OLD_AUTO_P4 vs S6 (all)", _paired_summary(df_a, ds_a, "S6_OLD_AUTO_P4", "S6")),
        _stat_row("S6_AUTO_P4 vs S6_OLD_AUTO_P4 (all)", _paired_summary(df_a, ds_a, "S6_AUTO_P4", "S6_OLD_AUTO_P4")),
        _stat_row("S6_AUTO_P4 vs S6_OLD_AUTO_P4 (guard flipped 1->0)", _paired_summary(df_a, ds_a_flipped, "S6_AUTO_P4", "S6_OLD_AUTO_P4")),
        _stat_row("S6_AUTO_P4 vs S6 (small allowed card>16)", _paired_summary(df_a, ds_a_small_hi, "S6_AUTO_P4", "S6")),
        _stat_row("S6_AUTO_P4 vs S6_AUTO_NOREG (all)", _paired_summary(df_a, ds_a, "S6_AUTO_P4", "S6_AUTO_NOREG")),
        _stat_row("S6_AUTO_P4 vs S6_AUTO_NOREG (cplx>=2)", _paired_summary(df_a, ds_a_med_hi, "S6_AUTO_P4", "S6_AUTO_NOREG")),
        _stat_row("S6_AUTO_P4 vs CAT (all)", _paired_summary(df_a, ds_a, "S6_AUTO_P4", "CAT")),
        _stat_row("S6 vs CAT (all)", _paired_summary(df_a, ds_a, "S6", "CAT")),
        "",
        "### Suite A Per-dataset (5-fold mean)",
        "",
        "| kind | dataset | n | K | p_cat | max_card | old/new | S6 | OLD_AUTO | AUTO_P4 | NOREG | CAT | AUTO vs OLD | NOREG vs AUTO | tbl |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    piv_a = df_a.pivot_table(index="dataset", columns="variant", values="metric", aggfunc="mean")
    tbl_a = df_a[df_a["variant"] == "S6_AUTO_P4"].groupby("dataset")["aux_count"].mean()
    info_a = df_a[df_a["variant"] == "S6_AUTO_P4"].groupby("dataset").first()
    for _, name, _ in SUITE_A_DATASETS:
        if name not in piv_a.index:
            continue
        r = piv_a.loc[name]
        inf = info_a.loc[name]
        oa = int(a_meta.loc[name, "old_auto"])
        na = int(a_meta.loc[name, "new_auto"])
        d_old = 100.0 * (r["S6_AUTO_P4"] - r["S6_OLD_AUTO_P4"]) / r["S6_OLD_AUTO_P4"]
        d_reg = 100.0 * (r["S6_AUTO_P4"] - r["S6_AUTO_NOREG"]) / r["S6_AUTO_NOREG"]
        lines.append(
            f"| {inf['kind']} | `{name}` | {int(inf['n_eval'])} | {int(inf['n_classes'])} | "
            f"{int(inf['p_cat'])} | {int(inf['max_card'])} | {oa}->{na} | "
            f"{r['S6']:.5f} | {r['S6_OLD_AUTO_P4']:.5f} | {r['S6_AUTO_P4']:.5f} | "
            f"{r['S6_AUTO_NOREG']:.5f} | {r['CAT']:.5f} | {d_old:+.2f}% | {d_reg:+.2f}% | {tbl_a.loc[name]:.1f} |"
        )

    lines += [
        "",
        "## Suite B: Numeric `quantizer='auto'` Out-of-TabArena Check (`N=16` unseen datasets)",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % | Time |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
        _stat_row("AUTO vs LIN (all)", _paired_summary(df_b, ds_b, "AUTO", "LIN")),
        _stat_row("QUANT vs LIN (all)", _paired_summary(df_b, ds_b, "QUANT", "LIN")),
        _stat_row("S6_AUTO vs S6_LIN (all)", _paired_summary(df_b, ds_b, "S6_AUTO", "S6_LIN")),
        _stat_row("S6_AUTO vs S6_LIN (trig nq>=1)", _paired_summary(df_b, ds_b_trig, "S6_AUTO", "S6_LIN")),
        _stat_row("AUTO vs LIN (trig nq>=1)", _paired_summary(df_b, ds_b_trig, "AUTO", "LIN")),
        "",
        "### Suite B Per-dataset (5-fold mean)",
        "",
        "| kind | dataset | n | p_num | nq | LIN | AUTO | QUANT | S6_LIN | S6_AUTO | AUTO vs LIN | S6_AUTO vs S6_LIN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    piv_b = df_b.pivot_table(index="dataset", columns="variant", values="metric", aggfunc="mean")
    nq_b = df_b[df_b["variant"] == "S6_AUTO"].groupby("dataset")["aux_count"].mean()
    info_b = df_b[df_b["variant"] == "S6_AUTO"].groupby("dataset").first()
    for _, name, _ in SUITE_B_DATASETS:
        if name not in piv_b.index:
            continue
        r = piv_b.loc[name]
        inf = info_b.loc[name]
        da = 100.0 * (r["AUTO"] - r["LIN"]) / r["LIN"]
        ds6 = 100.0 * (r["S6_AUTO"] - r["S6_LIN"]) / r["S6_LIN"]
        lines.append(
            f"| {inf['kind']} | `{name}` | {int(inf['n_eval'])} | {int(inf['p_num'])} | "
            f"{nq_b.loc[name]:.1f} | {r['LIN']:.5f} | {r['AUTO']:.5f} | {r['QUANT']:.5f} | "
            f"{r['S6_LIN']:.5f} | {r['S6_AUTO']:.5f} | {da:+.2f}% | {ds6:+.2f}% |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
    print("\n".join(lines[:35]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("pilot")
    r = sp.add_parser("run")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    sp.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "run":
        cmd_run(args.shard, args.nshards)
    else:
        cmd_summarize()


if __name__ == "__main__":
    main()
