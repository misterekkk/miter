"""Exploratory Scaling Laws & Default Settings Study (0 overlap with TabArena 51).

Three sub-studies strictly outside TabArena 51 (asserted programmatically):
  1. Part 1 (Real OpenML Learning Curves): 8 large zero-overlap OpenML datasets x 3 folds
     across n_train in {300, 1000, 3000, 10000, 16000} (fixed n_test=4000), comparing 12
     miter capacity/regularization/oblique/CTR configs + CAT + LGBM.
  2. Part 2 (Controlled Synthetic Scaling Laws):
     - 2a oblique_n: rotated nonlinear manifold up to n_train=150,000
     - 2b oblique_p: fixed n_train=10,000 across p in {10, 50, 150}
     - 2c mixed_cat_n: numeric + pairwise-XOR categorical (card=50) up to n_train=50,000
  3. Part 3 (Single-Model Default Showdown): 7 default candidates (DEF_LW, S6, S6_AUTO_P4,
     S6_AUTO_P4_Q, S6_M4, CAT, LGBM) across all 32 external OpenML datasets (Suite A + B),
     broken down by task type, sample size regime, and categorical vs pure-numeric.

Usage:
    .venv/bin/python benchmarks/bench_exp_scaling_defaults.py pilot
    .venv/bin/python benchmarks/bench_exp_scaling_defaults.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_scaling_defaults.py summarize
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from bench_exp_external_holdout import (
    EXCLUDED_NAMES,
    SUITE_A_DATASETS,
    SUITE_B_DATASETS,
    _norm_name,
    _prep_catboost_df,
    _split_cv,
    eval_metrics,
    load_dataset,
)
from bench_exp_native_ctr_v2 import _bootstrap_median_ci
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "results" / "raw" / "exp_scaling_defaults"
SUMMARY_PATH = HERE / "results" / "exp_scaling_defaults_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# Part 1: 8 large zero-overlap OpenML-CC18 / OpenML-CTR23 datasets (all N >= 21,613)
PART1_DATASETS: list[tuple[str, str, int]] = [
    ("clf", "nomao", 1486),
    ("clf", "numerai28.6", 23517),
    ("mc", "connect-4", 40668),
    ("mc", "jungle_chess_2pcs_raw_endgame_complete", 41027),
    ("reg", "kings_county", 44989),
    ("reg", "cps88wages", 44984),
    ("reg", "wave_energy", 44975),
    ("reg", "sarcos", 44976),
]
PART1_NS = [300, 1000, 3000, 10000, 16000]
PART1_N_FOLDS = 3
PART1_N_TEST = 4000
PART1_VARIANTS = [
    "LW_15",
    "LW_31",
    "LW_63",
    "LW_L2_10",
    "S4",
    "S6",
    "S7",
    "S6_L2_10",
    "S6_M4",
    "S6_AUTO_P4",
    "CAT",
    "LGBM",
]

# Part 2: Synthetic scaling laws
SYN_2A_NS = [500, 2000, 10000, 50000, 150000]
SYN_2A_VARS = ["LW_31", "S6", "S6_M4", "LGBM", "CAT"]
SYN_2B_PS = [10, 50, 150]
SYN_2B_VARS = ["LW_31", "S6", "S6_M4", "LGBM", "CAT"]
SYN_2C_NS = [500, 2000, 10000, 50000]
SYN_2C_VARS = ["LW_31", "S6", "S6_AUTO_P4", "LGBM", "CAT"]
SYN_N_SEEDS = 3

# Part 3: Single-model default showdown across 32 zero-overlap external OpenML datasets
PART3_DATASETS: list[tuple[str, str, int]] = SUITE_A_DATASETS + SUITE_B_DATASETS
PART3_VARIANTS = ["DEF_LW", "S6", "S6_AUTO_P4", "S6_AUTO_P4_Q", "S6_M4", "CAT", "LGBM"]

MITER_CONFIGS: dict[str, dict[str, Any]] = {
    "DEF_LW": {"grow_policy": "leafwise", "num_leaves": 31, "min_data_in_leaf": 20, "lambda_l2": 1.0},
    "LW_15": {"grow_policy": "leafwise", "num_leaves": 15, "min_data_in_leaf": 20, "lambda_l2": 1.0},
    "LW_31": {"grow_policy": "leafwise", "num_leaves": 31, "min_data_in_leaf": 20, "lambda_l2": 1.0},
    "LW_63": {"grow_policy": "leafwise", "num_leaves": 63, "min_data_in_leaf": 20, "lambda_l2": 1.0},
    "LW_L2_10": {"grow_policy": "leafwise", "num_leaves": 31, "min_data_in_leaf": 20, "lambda_l2": 10.0},
    "S4": {"grow_policy": "symmetric", "max_depth": 4, "min_data_in_leaf": 1, "lambda_l2": 3.0},
    "S6": {"grow_policy": "symmetric", "max_depth": 6, "min_data_in_leaf": 1, "lambda_l2": 3.0},
    "S7": {"grow_policy": "symmetric", "max_depth": 7, "min_data_in_leaf": 1, "lambda_l2": 3.0},
    "S6_L2_10": {"grow_policy": "symmetric", "max_depth": 6, "min_data_in_leaf": 1, "lambda_l2": 10.0},
    "S6_M4": {"grow_policy": "symmetric", "max_depth": 6, "min_data_in_leaf": 1, "lambda_l2": 3.0, "m_mult": 4},
    "S6_AUTO_P4": {
        "grow_policy": "symmetric",
        "max_depth": 6,
        "min_data_in_leaf": 1,
        "lambda_l2": 3.0,
        "max_ctr_complexity": "auto",
        "ctr_n_permutations": 4,
        "ctr_model_size_reg": 0.5,
    },
    "S6_AUTO_P4_Q": {
        "grow_policy": "symmetric",
        "max_depth": 6,
        "min_data_in_leaf": 1,
        "lambda_l2": 3.0,
        "max_ctr_complexity": "auto",
        "ctr_n_permutations": 4,
        "ctr_model_size_reg": 0.5,
        "quantizer": "auto",
    },
}


def assert_zero_overlap() -> None:
    excl = {_norm_name(x) for x in EXCLUDED_NAMES}
    for scope, ds_list in (("Part1", PART1_DATASETS), ("Part3", PART3_DATASETS)):
        for _, name, did in ds_list:
            if _norm_name(name) in excl:
                raise AssertionError(f"Overlap detected in {scope}: {name} (did={did})")


def _safe_lgbm_df(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out.columns = [f"f_{i}" for i in range(out.shape[1])]
    return out


def _stratified_val_split(
    kind: str, X_n: Any, y_n: np.ndarray, seed: int
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    idx = np.arange(len(y_n))
    strat = None
    if kind != "reg":
        _, counts = np.unique(y_n, return_counts=True)
        if np.all(counts >= 5):
            strat = y_n
    a, b = train_test_split(idx, test_size=0.2, random_state=seed, stratify=strat)
    if isinstance(X_n, pd.DataFrame):
        return X_n.iloc[a], X_n.iloc[b], y_n[a], y_n[b]
    return X_n[a], X_n[b], y_n[a], y_n[b]


def fit_predict_any(
    kind: str,
    variant: str,
    X_tr: Any,
    y_tr: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    X_te: Any,
    cat_cols: list[str],
    seed: int,
) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    if variant == "CAT":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if isinstance(X_tr, pd.DataFrame) and cat_cols:
            Xt = _prep_catboost_df(X_tr, cat_cols)
            Xv = _prep_catboost_df(X_val, cat_cols)
            Xe = _prep_catboost_df(X_te, cat_cols)
        else:
            Xt, Xv, Xe = X_tr, X_val, X_te
        common: dict[str, Any] = {
            "iterations": 5000,
            "learning_rate": 0.05,
            "early_stopping_rounds": 50,
            "depth": 6,
            "thread_count": 1,
            "random_seed": seed,
            "verbose": 0,
        }
        if cat_cols:
            common["cat_features"] = cat_cols
        if kind == "reg":
            mc = CatBoostRegressor(**common).fit(Xt, y_tr, eval_set=(Xv, y_val))
            pred = np.asarray(mc.predict(Xe), dtype=np.float64)
        else:
            loss_fn = "Logloss" if kind == "clf" else "MultiClass"
            mc = CatBoostClassifier(loss_function=loss_fn, **common).fit(Xt, y_tr, eval_set=(Xv, y_val))
            proba = np.asarray(mc.predict_proba(Xe), dtype=np.float64)
            pred = proba[:, 1] if kind == "clf" else proba
        return pred, int(mc.get_best_iteration() + 1), time.perf_counter() - t0

    if variant == "LGBM":
        import lightgbm as lgb

        if isinstance(X_tr, pd.DataFrame):
            Xt, Xv, Xe = _safe_lgbm_df(X_tr), _safe_lgbm_df(X_val), _safe_lgbm_df(X_te)
        else:
            Xt, Xv, Xe = X_tr, X_val, X_te
        common = {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_jobs": 1,
            "random_state": seed,
            "verbosity": -1,
        }
        cb = [lgb.early_stopping(50, verbose=False)]
        if kind == "reg":
            ml = lgb.LGBMRegressor(**common).fit(Xt, y_tr, eval_set=[(Xv, y_val)], callbacks=cb)
            pred = np.asarray(ml.predict(Xe), dtype=np.float64)
        else:
            ml = lgb.LGBMClassifier(**common).fit(Xt, y_tr, eval_set=[(Xv, y_val)], callbacks=cb)
            proba = np.asarray(ml.predict_proba(Xe), dtype=np.float64)
            pred = proba[:, 1] if kind == "clf" else proba
        trees = int(ml.best_iteration_ or common["n_estimators"])
        return pred, trees, time.perf_counter() - t0

    import miter

    cfg = MITER_CONFIGS[variant]
    kw: dict[str, Any] = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "m_mult": 2,
        "k_proj": 0,
        "max_bin": 255,
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "random_state": seed,
    }
    kw.update(cfg)
    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    if kind == "reg":
        pred = m.predict(X_te)
    elif kind == "clf":
        pred = m.predict_proba(X_te)[:, 1]
    else:
        pred = m.predict_proba(X_te)
    return pred, int(m.best_iteration_), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Part 1: Real OpenML learning curves
# ---------------------------------------------------------------------------

P1_FIELDS = ["kind", "dataset", "did", "fold", "n_train", "p_num", "p_cat", "variant", "metric", "auc", "trees", "secs"]


def _p1_csv(name: str, fold: int, n_train: int) -> Path:
    return RAW_DIR / f"p1_{name}_f{fold}_n{n_train}.csv"


def _load_p1_full(kind: str, did: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, cat_ind, _ = d.get_data(target=d.default_target_attribute, dataset_format="dataframe")
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
    y = y_s.astype("float64").to_numpy() if kind == "reg" else y_s.astype("category").cat.codes.to_numpy(dtype=np.int64)
    return X.reset_index(drop=True), y, cat_cols


def run_p1_job(kind: str, name: str, did: int, fold: int, n_train: int) -> None:
    p = _p1_csv(name, fold, n_train)
    if p.exists():
        with p.open() as f:
            if set(PART1_VARIANTS).issubset({r["variant"] for r in csv.DictReader(f)}):
                return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    X, y, cat_cols = _load_p1_full(kind, did)
    idx_all = np.arange(len(X))
    strat = None if kind == "reg" else y
    pool_idx, te_idx = train_test_split(idx_all, test_size=PART1_N_TEST, random_state=42 + fold, stratify=strat)
    rng = np.random.default_rng(100 + fold)
    # Shuffle pool with stratification-friendly permutation
    perm = rng.permutation(len(pool_idx))
    pool_idx = pool_idx[perm]
    sub_idx = pool_idx[:n_train]
    X_sub, y_sub = X.iloc[sub_idx].reset_index(drop=True), y[sub_idx]
    X_te, y_te = X.iloc[te_idx].reset_index(drop=True), y[te_idx]
    X_tr, X_val, y_tr, y_val = _stratified_val_split(kind, X_sub, y_sub, seed=fold)

    rows = []
    for v in PART1_VARIANTS:
        pred, trees, secs = fit_predict_any(kind, v, X_tr, y_tr, X_val, y_val, X_te, cat_cols, fold)
        met, auc = eval_metrics(kind, y_te, pred)
        rows.append(
            {
                "kind": kind,
                "dataset": name,
                "did": did,
                "fold": fold,
                "n_train": n_train,
                "p_num": X.shape[1] - len(cat_cols),
                "p_cat": len(cat_cols),
                "variant": v,
                "metric": f"{met:.12g}",
                "auc": f"{auc:.12g}" if not math.isnan(auc) else "nan",
                "trees": trees,
                "secs": f"{secs:.6f}",
            }
        )
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=P1_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"DONE P1 {name} f{fold} n={n_train}", flush=True)


# ---------------------------------------------------------------------------
# Part 2: Controlled synthetic scaling laws
# ---------------------------------------------------------------------------

P2_FIELDS = ["family", "n_train", "p_dim", "fold", "variant", "metric", "auc", "trees", "secs"]


def _gen_oblique(n_tr: int, n_te: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    w_rng = np.random.default_rng(9999)
    n_tot = n_tr + n_te
    X = rng.normal(size=(n_tot, p)).astype(np.float64)
    p_rel = min(p, 20)
    sig = np.zeros(n_tot, dtype=np.float64)
    for t in range(6):
        cols = w_rng.choice(p_rel, size=3, replace=False)
        w = w_rng.normal(size=3)
        w /= np.linalg.norm(w)
        z = X[:, cols] @ w
        if t % 3 == 0:
            sig += np.sin(1.6 * z)
        elif t % 3 == 1:
            sig += 1.2 * np.tanh(z)
        else:
            sig += 0.8 * np.abs(z) - 0.6 * z * z * 0.2
    sig = (sig - np.mean(sig)) / max(float(np.std(sig)), 1e-9)
    y = sig + 0.2 * rng.normal(size=n_tot)
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]


def _gen_mixed_cat(n_tr: int, n_te: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed + 2000)
    n_tot = n_tr + n_te
    n_num = 10
    n_cat = 6
    card = 50
    X_num = rng.normal(size=(n_tot, n_num)).astype(np.float64)
    cats = [rng.integers(0, card, size=n_tot) for _ in range(n_cat)]

    # Pairwise categorical XOR + numeric oblique interaction + additive terms
    pair0 = np.where(((cats[0] % 5) == (cats[1] % 5)), 1.5, -0.6)
    pair1 = np.where((((cats[2] + cats[3]) % 4) == 0), 1.3, -0.5)
    num_obl = np.sin(X_num[:, 0] + 0.8 * X_num[:, 1] - 0.5 * X_num[:, 2])
    c_num = np.where((cats[4] % 2 == 0) & (X_num[:, 3] + X_num[:, 4] > 0.2), 1.1, -0.4)
    logit = 0.8 * pair0 + 0.7 * pair1 + 0.9 * num_obl + 0.6 * c_num + 0.3 * X_num[:, 5]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n_tot) < prob).astype(np.int64)

    data: dict[str, Any] = {f"x{j}": X_num[:, j] for j in range(n_num)}
    cat_cols = [f"c{k}" for k in range(n_cat)]
    for k, col_name in enumerate(cat_cols):
        data[col_name] = pd.Categorical([f"v_{v}" for v in cats[k]])
    df = pd.DataFrame(data)
    return df.iloc[:n_tr].reset_index(drop=True), y[:n_tr], df.iloc[n_tr:].reset_index(drop=True), y[n_tr:], cat_cols


def _p2_csv(family: str, n_tr: int, p_dim: int, fold: int) -> Path:
    return RAW_DIR / f"p2_{family}_n{n_tr}_p{p_dim}_f{fold}.csv"


def run_p2_job(family: str, n_tr: int, p_dim: int, fold: int) -> None:
    p = _p2_csv(family, n_tr, p_dim, fold)
    variants = SYN_2C_VARS if family == "mixed_cat_n" else (SYN_2A_VARS if family == "oblique_n" else SYN_2B_VARS)
    if p.exists():
        with p.open() as f:
            if set(variants).issubset({r["variant"] for r in csv.DictReader(f)}):
                return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if family in ("oblique_n", "oblique_p"):
        kind = "reg"
        X_sub, y_sub, X_te, y_te = _gen_oblique(n_tr, 20000, p_dim, fold)
        cat_cols: list[str] = []
    else:
        kind = "clf"
        X_sub, y_sub, X_te, y_te, cat_cols = _gen_mixed_cat(n_tr, 10000, fold)
    X_tr, X_val, y_tr, y_val = _stratified_val_split(kind, X_sub, y_sub, seed=fold)

    rows = []
    for v in variants:
        pred, trees, secs = fit_predict_any(kind, v, X_tr, y_tr, X_val, y_val, X_te, cat_cols, fold)
        met, auc = eval_metrics(kind, y_te, pred)
        rows.append(
            {
                "family": family,
                "n_train": n_tr,
                "p_dim": p_dim,
                "fold": fold,
                "variant": v,
                "metric": f"{met:.12g}",
                "auc": f"{auc:.12g}" if not math.isnan(auc) else "nan",
                "trees": trees,
                "secs": f"{secs:.6f}",
            }
        )
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=P2_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"DONE P2 {family} n={n_tr} p={p_dim} f{fold}", flush=True)


# ---------------------------------------------------------------------------
# Part 3: 32-dataset External Default Showdown
# ---------------------------------------------------------------------------

P3_FIELDS = ["kind", "dataset", "did", "n_eval", "n_classes", "p_num", "p_cat", "fold", "variant", "metric", "auc", "trees", "secs"]


def _p3_csv(name: str, fold: int) -> Path:
    return RAW_DIR / f"p3_{name}_f{fold}.csv"


def run_p3_job(kind: str, name: str, did: int, fold: int) -> None:
    p = _p3_csv(name, fold)
    if p.exists():
        with p.open() as f:
            if set(PART3_VARIANTS).issubset({r["variant"] for r in csv.DictReader(f)}):
                return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    X, y, cat_cols, _, n_classes = load_dataset(kind, did)
    a, b, te = _split_cv(kind, X, y)[fold]
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val, y_te = y[a], y[b], y[te]

    rows = []
    for v in PART3_VARIANTS:
        pred, trees, secs = fit_predict_any(kind, v, X_tr, y_tr, X_val, y_val, X_te, cat_cols, fold)
        met, auc = eval_metrics(kind, y_te, pred)
        rows.append(
            {
                "kind": kind,
                "dataset": name,
                "did": did,
                "n_eval": len(X),
                "n_classes": n_classes,
                "p_num": X.shape[1] - len(cat_cols),
                "p_cat": len(cat_cols),
                "fold": fold,
                "variant": v,
                "metric": f"{met:.12g}",
                "auc": f"{auc:.12g}" if not math.isnan(auc) else "nan",
                "trees": trees,
                "secs": f"{secs:.6f}",
            }
        )
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=P3_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"DONE P3 {name} f{fold} n={len(X)}", flush=True)


# ---------------------------------------------------------------------------
# CLI & Summarization
# ---------------------------------------------------------------------------

def cmd_pilot() -> None:
    assert_zero_overlap()
    t0 = time.perf_counter()
    X, y, cat_cols = _load_p1_full("reg", 44984)  # cps88wages (already cached)
    pool_idx, te_idx = train_test_split(np.arange(len(X)), test_size=PART1_N_TEST, random_state=42)
    X_sub, y_sub = X.iloc[pool_idx[:1000]].reset_index(drop=True), y[pool_idx[:1000]]
    X_te, y_te = X.iloc[te_idx].reset_index(drop=True), y[te_idx]
    X_tr, X_val, y_tr, y_val = _stratified_val_split("reg", X_sub, y_sub, seed=0)
    print("=== PILOT P1 cps88wages n=1000 ===", flush=True)
    for v in ["LW_31", "S6", "S6_AUTO_P4", "CAT", "LGBM"]:
        pred, trees, secs = fit_predict_any("reg", v, X_tr, y_tr, X_val, y_val, X_te, cat_cols, 0)
        met, _ = eval_metrics("reg", y_te, pred)
        print(f"  {v:12s} nRMSE={met:.5f} trees={trees:4d} {secs:.2f}s", flush=True)

    print("=== PILOT P2 oblique_n n=2000 & mixed_cat_n n=2000 ===", flush=True)
    Xo_s, yo_s, Xo_te, yo_te = _gen_oblique(2000, 20000, 20, 0)
    Xo_tr, Xo_val, yo_tr, yo_val = _stratified_val_split("reg", Xo_s, yo_s, seed=0)
    for v in SYN_2A_VARS:
        pred, trees, secs = fit_predict_any("reg", v, Xo_tr, yo_tr, Xo_val, yo_val, Xo_te, [], 0)
        met, _ = eval_metrics("reg", yo_te, pred)
        print(f"  oblique {v:10s} nRMSE={met:.5f} trees={trees:4d} {secs:.2f}s", flush=True)

    Xm_s, ym_s, Xm_te, ym_te, mcats = _gen_mixed_cat(2000, 10000, 0)
    Xm_tr, Xm_val, ym_tr, ym_val = _stratified_val_split("clf", Xm_s, ym_s, seed=0)
    for v in SYN_2C_VARS:
        pred, trees, secs = fit_predict_any("clf", v, Xm_tr, ym_tr, Xm_val, ym_val, Xm_te, mcats, 0)
        met, auc = eval_metrics("clf", ym_te, pred)
        print(f"  mixed   {v:10s} logloss={met:.5f} auc={auc:.5f} trees={trees:4d} {secs:.2f}s", flush=True)
    print(f"Pilot finished in {time.perf_counter() - t0:.1f}s", flush=True)


def _all_jobs() -> list[tuple[str, Any]]:
    jobs: list[tuple[str, Any]] = []
    # Heavier jobs interspersed / first for balanced 3-shard wall time
    for k, n, d in PART1_DATASETS:
        for n_tr in sorted(PART1_NS, reverse=True):
            for f in range(PART1_N_FOLDS):
                jobs.append(("p1", (k, n, d, f, n_tr)))
    for n_tr in sorted(SYN_2A_NS, reverse=True):
        for f in range(SYN_N_SEEDS):
            jobs.append(("p2", ("oblique_n", n_tr, 20, f)))
    for p_dim in sorted(SYN_2B_PS, reverse=True):
        for f in range(SYN_N_SEEDS):
            jobs.append(("p2", ("oblique_p", 10000, p_dim, f)))
    for n_tr in sorted(SYN_2C_NS, reverse=True):
        for f in range(SYN_N_SEEDS):
            jobs.append(("p2", ("mixed_cat_n", n_tr, 16, f)))
    for k, n, d in PART3_DATASETS:
        for f in range(5):
            jobs.append(("p3", (k, n, d, f)))
    return jobs


def cmd_run(shard: int, nshards: int) -> None:
    assert_zero_overlap()
    jobs = _all_jobs()
    mine = [j for idx, j in enumerate(jobs) if idx % nshards == shard]
    print(f"Shard {shard}/{nshards}: {len(mine)}/{len(jobs)} jobs", flush=True)
    for kind_tag, payload in mine:
        if kind_tag == "p1":
            run_p1_job(*payload)
        elif kind_tag == "p2":
            run_p2_job(*payload)
        else:
            run_p3_job(*payload)


def _paired_stats(df: pd.DataFrame, datasets: list[str], a: str, b: str) -> dict[str, Any]:
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


def _stat_line(label: str, st: dict[str, Any]) -> str:
    return (
        f"| {label} | {st['n']} | {st['mean']:+.2f}% | {st['median']:+.2f}% | "
        f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/{st['n']} ({st['ties']}t) | "
        f"{st['p']:.4g} | {st['worst']:+.2f}% | {st['t_ratio']:.2f}x |"
    )


def _fit_power_alpha(ns: np.ndarray, errs: np.ndarray) -> float:
    mask = (ns > 0) & (errs > 0)
    if np.sum(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(ns[mask]), np.log(errs[mask]), 1)
    return float(-slope)


def cmd_summarize() -> None:
    assert_zero_overlap()
    p1_rows, p2_rows, p3_rows = [], [], []
    for fpath in sorted(RAW_DIR.glob("p1_*.csv")):
        with fpath.open() as f:
            for r in csv.DictReader(f):
                r["n_train"] = int(r["n_train"])
                r["metric"] = float(r["metric"])
                r["secs"] = float(r["secs"])
                r["trees"] = int(r["trees"])
                p1_rows.append(r)
    for fpath in sorted(RAW_DIR.glob("p2_*.csv")):
        with fpath.open() as f:
            for r in csv.DictReader(f):
                r["n_train"] = int(r["n_train"])
                r["p_dim"] = int(r["p_dim"])
                r["metric"] = float(r["metric"])
                r["auc"] = float(r["auc"]) if r["auc"] != "nan" else float("nan")
                r["secs"] = float(r["secs"])
                r["trees"] = int(r["trees"])
                p2_rows.append(r)
    for fpath in sorted(RAW_DIR.glob("p3_*.csv")):
        with fpath.open() as f:
            for r in csv.DictReader(f):
                r["n_eval"] = int(r["n_eval"])
                r["n_classes"] = int(r["n_classes"])
                r["p_cat"] = int(r["p_cat"])
                r["metric"] = float(r["metric"])
                r["secs"] = float(r["secs"])
                r["trees"] = int(r["trees"])
                p3_rows.append(r)

    df1 = pd.DataFrame(p1_rows)
    df2 = pd.DataFrame(p2_rows)
    df3 = pd.DataFrame(p3_rows)

    lines = [
        "# Exploratory Scaling Laws & Default Settings Study (`exp-scaling-defaults`)",
        "",
        "- Strictly **ZERO overlap with TabArena 51** (verified via `assert_zero_overlap()`).",
        "- Single-thread (`n_jobs=1`), early stopping `50` rounds.",
        "",
        "## Part 1: Real OpenML Learning Curves (8 datasets x 3 folds, fixed n_test=4000)",
        "",
        "### Relative Error Median (%) vs `LW_31` (frozen default) across `n_train` (negative = better)",
        "",
        "| Variant | n=300 | n=1000 | n=3000 | n=10000 | n=16000 | alpha (med) | alpha (mean) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    ds1 = [n for _, n, _ in PART1_DATASETS]
    piv1 = df1.pivot_table(index=["dataset", "n_train"], columns="variant", values="metric", aggfunc="mean")
    for v in PART1_VARIANTS:
        meds = []
        for n_tr in PART1_NS:
            sub = piv1.xs(n_tr, level="n_train")
            rel = 100.0 * (sub[v] - sub["LW_31"]) / sub["LW_31"]
            meds.append(f"{float(np.median(rel)):+.2f}%")
        alphas = []
        for ds in ds1:
            sub_ds = piv1.loc[ds]
            alphas.append(_fit_power_alpha(sub_ds.index.to_numpy(dtype=np.float64), sub_ds[v].to_numpy(dtype=np.float64)))
        a_arr = np.asarray([a for a in alphas if not math.isnan(a)], dtype=np.float64)
        lines.append(f"| `{v}` | " + " | ".join(meds) + f" | {np.median(a_arr):.3f} | {np.mean(a_arr):.3f} |")

    lines += [
        "",
        "### Part 1 Median Relative Error (%) vs `CAT` (negative = beats CAT) and vs `LGBM`",
        "",
        "| Comparison | n=300 | n=1000 | n=3000 | n=10000 | n=16000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for a, b in [
        ("LW_31", "CAT"), ("S6", "CAT"), ("S6_AUTO_P4", "CAT"), ("S6_M4", "CAT"), ("LGBM", "CAT"),
        ("LW_31", "LGBM"), ("S6", "LGBM"), ("S6_AUTO_P4", "LGBM"), ("S6_M4", "LGBM"),
    ]:
        cells = []
        for n_tr in PART1_NS:
            sub = piv1.xs(n_tr, level="n_train")
            rel = 100.0 * (sub[a] - sub[b]) / sub[b]
            w = int((rel < 0.0).sum())
            cells.append(f"{float(np.median(rel)):+.2f}% ({w}/8)")
        lines.append(f"| `{a} vs {b}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Part 1 Capacity & Regularization Breakdown by `n_train` (wins/8 vs `LW_31` or baseline)",
        "",
        "| Regime Check | n=300 | n=1000 | n=3000 | n=10000 | n=16000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, a, b in [
        ("LW_15 vs LW_31", "LW_15", "LW_31"),
        ("LW_63 vs LW_31", "LW_63", "LW_31"),
        ("LW_L2_10 vs LW_31", "LW_L2_10", "LW_31"),
        ("S4 vs S6", "S4", "S6"),
        ("S7 vs S6", "S7", "S6"),
        ("S6_L2_10 vs S6", "S6_L2_10", "S6"),
        ("S6_M4 vs S6", "S6_M4", "S6"),
        ("S6_AUTO_P4 vs S6", "S6_AUTO_P4", "S6"),
    ]:
        cells = []
        for n_tr in PART1_NS:
            sub = piv1.xs(n_tr, level="n_train")
            rel = 100.0 * (sub[a] - sub[b]) / sub[b]
            w = int((rel < 0.0).sum())
            cells.append(f"{float(np.median(rel)):+.2f}% ({w}/8)")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Part 2: Controlled Synthetic Scaling Laws (3 seeds)",
        "",
        "### 2a. Rotated Oblique Manifold (`oblique_n`, `p=20`, `n_test=20000`): Test nRMSE & Power-Law `alpha`",
        "",
        "| Variant | n=500 | n=2000 | n=10000 | n=50000 | n=150000 | alpha | t(150k) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    df2a = df2[df2["family"] == "oblique_n"]
    piv2a = df2a.pivot_table(index="variant", columns="n_train", values="metric", aggfunc="mean")
    tim2a = df2a.pivot_table(index="variant", columns="n_train", values="secs", aggfunc="mean")
    ns2a = np.asarray(SYN_2A_NS, dtype=np.float64)
    for v in SYN_2A_VARS:
        vals = [piv2a.loc[v, n_tr] for n_tr in SYN_2A_NS]
        alpha = _fit_power_alpha(ns2a, np.asarray(vals, dtype=np.float64))
        lines.append(
            f"| `{v}` | " + " | ".join(f"{x:.4f}" for x in vals) +
            f" | {alpha:.3f} | {tim2a.loc[v, 150000]:.2f}s |"
        )

    lines += [
        "",
        "### 2b. Dimensionality Scaling (`oblique_p`, `n_train=10000`, `n_test=20000`)",
        "",
        "| Variant | p=10 nRMSE (t) | p=50 nRMSE (t) | p=150 nRMSE (t) |",
        "|---|---:|---:|---:|",
    ]
    df2b = df2[df2["family"] == "oblique_p"]
    piv2b = df2b.pivot_table(index="variant", columns="p_dim", values="metric", aggfunc="mean")
    tim2b = df2b.pivot_table(index="variant", columns="p_dim", values="secs", aggfunc="mean")
    for v in SYN_2B_VARS:
        cells = [f"{piv2b.loc[v, p]:.4f} ({tim2b.loc[v, p]:.2f}s)" for p in SYN_2B_PS]
        lines.append(f"| `{v}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### 2c. Mixed Pairwise-Categorical XOR + Oblique (`mixed_cat_n`, `p_num=10, p_cat=6, card=50`)",
        "",
        "| Variant | n=500 | n=2000 | n=10000 | n=50000 | alpha | t(50k) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    df2c = df2[df2["family"] == "mixed_cat_n"]
    piv2c = df2c.pivot_table(index="variant", columns="n_train", values="metric", aggfunc="mean")
    tim2c = df2c.pivot_table(index="variant", columns="n_train", values="secs", aggfunc="mean")
    ns2c = np.asarray(SYN_2C_NS, dtype=np.float64)
    for v in SYN_2C_VARS:
        vals = [piv2c.loc[v, n_tr] for n_tr in SYN_2C_NS]
        alpha = _fit_power_alpha(ns2c, np.asarray(vals, dtype=np.float64))
        lines.append(
            f"| `{v}` | " + " | ".join(f"{x:.4f}" for x in vals) +
            f" | {alpha:.3f} | {tim2c.loc[v, 50000]:.2f}s |"
        )

    # Part 3 breakdown
    ds3_all = [n for _, n, _ in PART3_DATASETS]
    meta3 = df3.groupby("dataset").agg(
        kind=("kind", "first"),
        n_eval=("n_eval", "first"),
        p_cat=("p_cat", "first"),
    )
    ds3_reg = [d for d in ds3_all if meta3.loc[d, "kind"] == "reg"]
    ds3_clf = [d for d in ds3_all if meta3.loc[d, "kind"] == "clf"]
    ds3_mc = [d for d in ds3_all if meta3.loc[d, "kind"] == "mc"]
    ds3_small = [d for d in ds3_all if meta3.loc[d, "n_eval"] < 1500]
    ds3_mid = [d for d in ds3_all if 1500 <= meta3.loc[d, "n_eval"] < 8000]
    ds3_large = [d for d in ds3_all if meta3.loc[d, "n_eval"] >= 8000]
    ds3_cat = [d for d in ds3_all if meta3.loc[d, "p_cat"] >= 1]
    ds3_num = [d for d in ds3_all if meta3.loc[d, "p_cat"] == 0]

    lines += [
        "",
        "## Part 3: Single-Model Default Showdown (32 zero-overlap OpenML datasets x 5 folds)",
        "",
        "### Overall (`N=32`) vs Frozen Default (`DEF_LW`), vs `CAT`, and vs `LGBM`",
        "",
        "| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % | Time |",
        "|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    for a in ["S6", "S6_AUTO_P4", "S6_AUTO_P4_Q", "S6_M4", "CAT", "LGBM"]:
        lines.append(_stat_line(f"{a} vs DEF_LW", _paired_stats(df3, ds3_all, a, "DEF_LW")))
    for a in ["DEF_LW", "S6", "S6_AUTO_P4", "S6_AUTO_P4_Q", "S6_M4", "LGBM"]:
        lines.append(_stat_line(f"{a} vs CAT", _paired_stats(df3, ds3_all, a, "CAT")))
    for a in ["DEF_LW", "S6", "S6_AUTO_P4", "S6_AUTO_P4_Q", "S6_M4", "CAT"]:
        lines.append(_stat_line(f"{a} vs LGBM", _paired_stats(df3, ds3_all, a, "LGBM")))

    lines += [
        "",
        "### Stratified Breakdown vs `CAT` & `DEF_LW` (by task type, sample size `n`, and categorical presence)",
        "",
        "| Slice | Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % | Time |",
        "|---|---|---:|---:|---:|---|---|---|---|---:|",
    ]
    slices = [
        ("reg", ds3_reg),
        ("clf", ds3_clf),
        ("mc", ds3_mc),
        ("n < 1500", ds3_small),
        ("1500 <= n < 8k", ds3_mid),
        ("n >= 8000", ds3_large),
        ("has_cat", ds3_cat),
        ("pure_num", ds3_num),
    ]
    for s_name, s_ds in slices:
        for a, b in [("DEF_LW", "CAT"), ("S6", "CAT"), ("S6_AUTO_P4", "CAT"), ("S6_AUTO_P4_Q", "CAT"), ("S6_AUTO_P4", "DEF_LW")]:
            st = _paired_stats(df3, s_ds, a, b)
            lines.append(
                f"| {s_name} | {a} vs {b} | {st['n']} | {st['mean']:+.2f}% | {st['median']:+.2f}% | "
                f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/{st['n']} ({st['ties']}t) | "
                f"{st['p']:.4g} | {st['worst']:+.2f}% | {st['t_ratio']:.2f}x |"
            )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
    print("\n".join(lines[:45]))


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
