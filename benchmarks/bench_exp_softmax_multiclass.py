"""Evening confirmatory benchmark: Native Softmax vs One-vs-Rest (`exp-softmax-multiclass`).

Pre-registered in `benchmarks/results/exp_softmax_tabarena51_prereg.md` (Part A).

Usage:
    .venv_tabarena/bin/python benchmarks/bench_exp_softmax_multiclass.py pilot
    .venv_tabarena/bin/python benchmarks/bench_exp_softmax_multiclass.py run --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/bench_exp_softmax_multiclass.py summarize
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "results" / "raw" / "exp_softmax_multiclass"
SUMMARY_PATH = HERE / "results" / "exp_softmax_multiclass_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas2"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# 8 TabArena multiclass tasks (`tid`) + 4 OpenML CC18 multiclass datasets (`did`)
DATASETS: list[tuple[str, str, int]] = [
    ("tid", "anneal", 363614),
    ("tid", "hiva_agnostic", 363677),
    ("tid", "maternal_health_risk", 363685),
    ("tid", "MIC", 363711),
    ("tid", "SDSS17", 363699),
    ("tid", "splice", 363702),
    ("tid", "students_dropout_and_academic_success", 363704),
    ("tid", "website_phishing", 363707),
    ("did", "satimage", 182),
    ("did", "segment", 40984),
    ("did", "pendigits", 32),
    ("did", "letter", 6),
]

VARIANTS = ["OVR", "SM_B", "SM_S6", "SM_B+S6", "CAT", "LGB"]

FIELDS = [
    "dataset",
    "did",
    "n_rows",
    "n_classes",
    "p_num",
    "p_cat",
    "fold",
    "variant",
    "logloss",
    "accuracy",
    "trees",
    "secs",
]


def _resolve_did(id_kind: str, oid: int) -> int:
    if id_kind == "did":
        return oid
    task = openml.tasks.get_task(oid, download_splits=False, download_qualities=False)
    return int(task.dataset_id)


def load_dataset(id_kind: str, oid: int) -> tuple[pd.DataFrame, np.ndarray, list[str], int, int]:
    did = _resolve_did(id_kind, oid)
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
    X = X.reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(np.asarray(y_s.astype(str))).astype(np.int64)
    n_classes = int(len(le.classes_))
    return X, y, cat_cols, did, n_classes


def eval_multiclass(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    row_sums = proba.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    norm_p = np.where(row_sums > 0.0, proba / safe_sums, 1.0 / proba.shape[1])
    clipped = np.clip(norm_p, 1e-15, 1.0 - 1e-15)
    clipped /= clipped.sum(axis=1, keepdims=True)
    ll = float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))
    acc = float(accuracy_score(y_true, np.argmax(norm_p, axis=1)))
    return ll, acc


def _prep_catboost_df(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    Xc = X.copy()
    for c in cat_cols:
        s = Xc[c].astype(object)
        Xc[c] = s.where(pd.notna(s), "__MITER_MISSING__").astype(str)
    return Xc


def _split_cv(X: pd.DataFrame, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in skf.split(X, y):
        counts = np.bincount(y[tr_idx])
        strat = y[tr_idx] if int(counts.min()) >= 5 else None
        a, b = train_test_split(tr_idx, test_size=0.2, random_state=0, stratify=strat)
        out.append((a, b, te_idx))
    return out


def _fit_miter_softmax(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    n_classes: int,
    *,
    symmetric: bool,
    fold: int,
) -> tuple[np.ndarray, int, float]:
    import miter

    t0 = time.perf_counter()
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
    if symmetric:
        kw.update({"grow_policy": "symmetric", "max_depth": 6, "min_data_in_leaf": 1, "lambda_l2": 3.0})
    else:
        kw.update({"grow_policy": "leafwise", "num_leaves": 31, "min_data_in_leaf": 20, "lambda_l2": 1.0})

    present = np.unique(y_tr)
    eval_mask = np.isin(y_val, present)
    fit_kw: dict[str, Any] = {}
    if np.any(eval_mask):
        fit_kw["eval_set"] = [(X_val.iloc[eval_mask], y_val[eval_mask])]

    clf = miter.MiterClassifier(**kw).fit(X_tr, y_tr, **fit_kw)
    p_sub = clf.predict_proba(X_te)
    if len(present) == n_classes and np.array_equal(present, np.arange(n_classes)):
        proba = p_sub
    else:
        proba = np.zeros((len(X_te), n_classes), dtype=np.float64)
        proba[:, present] = p_sub
    return proba, int(clf.best_iteration_), time.perf_counter() - t0


def _fit_miter_ovr(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    n_classes: int,
    *,
    fold: int,
) -> tuple[np.ndarray, int, float]:
    import miter

    t0 = time.perf_counter()
    cols: list[np.ndarray] = []
    trees_sum = 0
    for k in range(n_classes):
        yk_tr = (y_tr == k).astype(np.float64)
        if np.min(yk_tr) == np.max(yk_tr):
            cols.append(np.full(len(X_te), float(yk_tr[0]), dtype=np.float64))
            continue
        yk_val = (y_val == k).astype(np.float64)
        clf = miter.MiterClassifier(
            n_estimators=5000,
            learning_rate=0.05,
            grow_policy="leafwise",
            num_leaves=31,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            m_mult=2,
            k_proj=0,
            max_bin=255,
            early_stopping_rounds=50,
            n_jobs=1,
            random_state=fold,
        ).fit(X_tr, yk_tr, eval_set=[(X_val, yk_val)])
        cols.append(clf.predict_proba(X_te)[:, 1])
        trees_sum += int(clf.best_iteration_)

    proba = np.column_stack(cols)
    totals = proba.sum(axis=1, keepdims=True)
    safe_tot = np.where(totals > 0.0, totals, 1.0)
    proba = np.where(totals > 0.0, proba / safe_tot, 1.0 / n_classes)
    return proba, max(trees_sum, 1), time.perf_counter() - t0


def _fit_lgb_subprocess(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_te: pd.DataFrame,
    n_classes: int,
    *,
    fold: int,
) -> tuple[np.ndarray, int, float]:
    """Run LightGBM in an isolated subprocess to prevent macOS libomp SIGSEGV with miter."""
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        X_tr.to_pickle(tdir / "X_tr.pkl")
        X_val.to_pickle(tdir / "X_val.pkl")
        X_te.to_pickle(tdir / "X_te.pkl")
        np.savez(tdir / "targets.npz", y_tr=y_tr, y_val=y_val)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_lgb-worker",
            "--workdir",
            str(tdir),
            "--n-classes",
            str(n_classes),
            "--fold",
            str(fold),
        ]
        subprocess.run(cmd, check=True)
        meta = json.loads((tdir / "meta.json").read_text())
        proba = np.load(tdir / "proba.npy")
        return proba, int(meta["trees"]), float(meta["secs"])


def _cmd_lgb_worker(workdir: Path, n_classes: int, fold: int) -> None:
    import lightgbm as lgb

    X_tr = pd.read_pickle(workdir / "X_tr.pkl")
    X_val = pd.read_pickle(workdir / "X_val.pkl")
    X_te = pd.read_pickle(workdir / "X_te.pkl")
    data = np.load(workdir / "targets.npz")
    y_tr, y_val = data["y_tr"], data["y_val"]

    t0 = time.perf_counter()
    present = np.unique(y_tr)
    eval_mask = np.isin(y_val, present)
    fit_kw: dict[str, Any] = {}
    if np.any(eval_mask):
        fit_kw["eval_set"] = [(X_val.iloc[eval_mask], y_val[eval_mask])]
        fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_lambda=1.0,
        max_bin=255,
        n_jobs=1,
        random_state=fold,
        deterministic=True,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr, **fit_kw)
    secs = time.perf_counter() - t0
    p_sub = np.asarray(clf.predict_proba(X_te), dtype=np.float64)
    if len(clf.classes_) == n_classes and np.array_equal(clf.classes_, np.arange(n_classes)):
        proba = p_sub
    else:
        proba = np.zeros((len(X_te), n_classes), dtype=np.float64)
        proba[:, np.asarray(clf.classes_, dtype=np.int64)] = p_sub
    trees = int(clf.best_iteration_ or clf.n_estimators)
    np.save(workdir / "proba.npy", proba)
    (workdir / "meta.json").write_text(json.dumps({"trees": trees, "secs": secs}))


def _prob_cache_path(name: str, fold: int, variant: str) -> Path:
    return RAW_DIR / "prob_cache" / f"{name}_f{fold}_{variant}.npy"


def fit_predict_variant(
    name: str,
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    cat_cols: list[str],
    n_classes: int,
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
    *,
    cache_probs: bool = True,
) -> tuple[np.ndarray, int, float]:
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]

    if variant == "OVR":
        return _fit_miter_ovr(X_tr, y_tr, X_val, y_val, X_te, n_classes, fold=fold)
    if variant in ("SM_B", "SM_S6"):
        proba, trees, secs = _fit_miter_softmax(
            X_tr, y_tr, X_val, y_val, X_te, n_classes, symmetric=(variant == "SM_S6"), fold=fold
        )
        if cache_probs:
            cp = _prob_cache_path(name, fold, variant)
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(cp, proba)
        return proba, trees, secs
    if variant == "SM_B+S6":
        pb_path = _prob_cache_path(name, fold, "SM_B")
        ps_path = _prob_cache_path(name, fold, "SM_S6")
        t_extra = 0.0
        if pb_path.exists():
            pb = np.load(pb_path)
            t_b = 0.0
        else:
            pb, tb_trees, t_b = _fit_miter_softmax(
                X_tr, y_tr, X_val, y_val, X_te, n_classes, symmetric=False, fold=fold
            )
            t_extra += tb_trees * 0.0
        if ps_path.exists():
            ps = np.load(ps_path)
            t_s = 0.0
        else:
            ps, _, t_s = _fit_miter_softmax(
                X_tr, y_tr, X_val, y_val, X_te, n_classes, symmetric=True, fold=fold
            )
        return 0.5 * (pb + ps), 0, t_b + t_s + t_extra
    if variant == "CAT":
        from catboost import CatBoostClassifier

        t0 = time.perf_counter()
        Xc_tr = _prep_catboost_df(X_tr, cat_cols)
        Xc_val = _prep_catboost_df(X_val, cat_cols)
        Xc_te = _prep_catboost_df(X_te, cat_cols)
        present = np.unique(y_tr)
        eval_mask = np.isin(y_val, present)
        mc = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=5000,
            early_stopping_rounds=50,
            depth=6,
            thread_count=1,
            random_seed=fold,
            cat_features=cat_cols,
            verbose=0,
        )
        if np.any(eval_mask):
            mc.fit(Xc_tr, y_tr, eval_set=(Xc_val.iloc[eval_mask], y_val[eval_mask]))
        else:
            mc.fit(Xc_tr, y_tr)
        p_sub = np.asarray(mc.predict_proba(Xc_te), dtype=np.float64)
        cls_arr = np.asarray(mc.classes_, dtype=np.int64)
        if len(cls_arr) == n_classes and np.array_equal(cls_arr, np.arange(n_classes)):
            proba = p_sub
        else:
            proba = np.zeros((len(X_te), n_classes), dtype=np.float64)
            proba[:, cls_arr] = p_sub
        return proba, int(mc.get_best_iteration() + 1), time.perf_counter() - t0
    if variant == "LGB":
        return _fit_lgb_subprocess(X_tr, y_tr, X_val, y_val, X_te, n_classes, fold=fold)
    raise ValueError(f"Unknown variant {variant!r}")


def _bootstrap_median_ci(deltas: list[float], *, n_boot: int = 10000, seed: int = 2026) -> tuple[float, float]:
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    meds = np.median(arr[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_pilot() -> None:
    for ik, name, oid in [("tid", "maternal_health_risk", 363685), ("did", "segment", 40984)]:
        X, y, cat_cols, did, n_classes = load_dataset(ik, oid)
        a, b, te = _split_cv(X, y)[0]
        print(f"=== PILOT {name} did={did} K={n_classes} n={len(X)} p_cat={len(cat_cols)} ===", flush=True)
        fold_times: dict[str, float] = {}
        for v in VARIANTS:
            proba, trees, secs = fit_predict_variant(
                name, v, X, y, cat_cols, n_classes, a, b, te, 0, cache_probs=True
            )
            if v == "SM_B+S6":
                secs = fold_times.get("SM_B", 0.0) + fold_times.get("SM_S6", 0.0)
            fold_times[v] = secs
            ll, acc = eval_multiclass(y[te], proba)
            print(f"  {v:7s}: logloss={ll:.5f} acc={acc:.5f} trees={trees:4d} secs={secs:.2f}s", flush=True)


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

        for ik, name, oid in selected:
            X, y, cat_cols, did, n_classes = load_dataset(ik, oid)
            p_cat = len(cat_cols)
            p_num = X.shape[1] - p_cat
            splits = _split_cv(X, y)
            for fold, (a, b, te) in enumerate(splits):
                fold_times: dict[str, float] = {}
                fold_trees: dict[str, int] = {}
                for v in VARIANTS:
                    if (name, fold, v) in done:
                        continue
                    proba, trees, secs = fit_predict_variant(
                        name, v, X, y, cat_cols, n_classes, a, b, te, fold, cache_probs=True
                    )
                    if v == "SM_B+S6" and secs == 0.0:
                        secs = fold_times.get("SM_B", 0.0) + fold_times.get("SM_S6", 0.0)
                        trees = fold_trees.get("SM_B", 0) + fold_trees.get("SM_S6", 0)
                    fold_times[v] = secs
                    fold_trees[v] = trees
                    ll, acc = eval_multiclass(y[te], proba)
                    writer.writerow(
                        {
                            "dataset": name,
                            "did": did,
                            "n_rows": len(X),
                            "n_classes": n_classes,
                            "p_num": p_num,
                            "p_cat": p_cat,
                            "fold": fold,
                            "variant": v,
                            "logloss": repr(ll),
                            "accuracy": repr(acc),
                            "trees": trees,
                            "secs": f"{secs:.3f}",
                        }
                    )
                    fh.flush()
                    print(
                        f"[{shard}/{nshards}] {name:34s} K={n_classes:2d} f={fold} {v:7s}: "
                        f"ll={ll:.5f} acc={acc:.4f} trees={trees:5d} {secs:6.2f}s",
                        flush=True,
                    )


def cmd_summarize() -> None:
    rows: list[dict[str, Any]] = []
    for p in sorted(RAW_DIR.glob("shard_*.csv")):
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "did": int(row["did"]),
                        "n_rows": int(row["n_rows"]),
                        "n_classes": int(row["n_classes"]),
                        "fold": int(row["fold"]),
                        "variant": row["variant"],
                        "logloss": float(row["logloss"]),
                        "accuracy": float(row["accuracy"]),
                        "secs": float(row["secs"]),
                    }
                )
    if not rows:
        raise SystemExit(f"No rows found in {RAW_DIR}")

    agg_ll: dict[tuple[str, str], float] = {}
    agg_acc: dict[tuple[str, str], float] = {}
    agg_secs: dict[tuple[str, str], float] = {}
    ds_meta: dict[str, dict[str, Any]] = {}
    for _, name, _ in DATASETS:
        sub_all = [r for r in rows if r["dataset"] == name]
        if sub_all:
            ds_meta[name] = sub_all[0]
        for v in VARIANTS:
            sub = [r for r in sub_all if r["variant"] == v]
            if len(sub) != 5:
                raise SystemExit(f"Incomplete runs for {name}/{v}: expected 5 folds, got {len(sub)}")
            agg_ll[(name, v)] = statistics.fmean(r["logloss"] for r in sub)
            agg_acc[(name, v)] = statistics.fmean(r["accuracy"] for r in sub)
            agg_secs[(name, v)] = statistics.fmean(r["secs"] for r in sub)

    # Normalize SM_B+S6 runtime = SM_B + SM_S6 if recorded as 0 across resumes.
    for _, name, _ in DATASETS:
        if agg_secs[(name, "SM_B+S6")] == 0.0:
            agg_secs[(name, "SM_B+S6")] = agg_secs[(name, "SM_B")] + agg_secs[(name, "SM_S6")]

    def _stats(cand: str, base: str) -> tuple[float, float, float, float, int, float, float]:
        deltas = [(agg_ll[(n, cand)] - agg_ll[(n, base)]) / agg_ll[(n, base)] * 100.0 for _, n, _ in DATASETS]
        nz = [d for d in deltas if d != 0.0]
        p_val = 1.0 if not nz else float(wilcoxon(nz, alternative="two-sided").pvalue)
        lo, hi = _bootstrap_median_ci(deltas, n_boot=10000, seed=2026)
        return (
            float(statistics.median(deltas)),
            float(statistics.fmean(deltas)),
            lo,
            hi,
            sum(1 for d in deltas if d < 0.0),
            p_val,
            float(max(deltas)),
        )

    med, _mn, lo, hi, wins, p_val, worst = _stats("SM_B", "OVR")
    t_med_smb = float(statistics.median(agg_secs[(n, "SM_B")] for _, n, _ in DATASETS))
    t_med_ovr = float(statistics.median(agg_secs[(n, "OVR")] for _, n, _ in DATASETS))
    primary_pass = (
        med < -2.0
        and wins >= 10
        and p_val < 0.01
        and hi < 0.0
        and worst <= 1.0
        and t_med_smb <= t_med_ovr
    )

    cat_med, _cmn, cat_lo, cat_hi, cat_wins, cat_p, cat_worst = _stats("SM_B+S6", "CAT")
    ahead_cat = cat_wins >= 9 and cat_p < 0.05 and cat_hi < 0.0

    lines = [
        "# Confirmatory Summary: Native Softmax vs One-vs-Rest (`exp-softmax-multiclass`)",
        "",
        "Pre-registered in `benchmarks/results/exp_softmax_tabarena51_prereg.md` (Part A, 12 datasets x 5 folds).",
        "",
        "## 1. Anti-Fluke Confirmatory Verdict",
        "",
        "| Comparison | Median Δ | 95% Boot CI | Wins | Wilcoxon p | Worst Δ | Median Time (Cand vs Base) | Status |",
        "|---|---:|---|---:|---:|---:|---|---|",
        f"| **Primary: SM_B vs OVR** | {med:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] | {wins}/12 | {p_val:.4g} | {worst:+.2f}% | {t_med_smb:.2f}s vs {t_med_ovr:.2f}s | {'**PASS**' if primary_pass else 'FAIL'} |",
        f"| Secondary: SM_B+S6 vs CAT | {cat_med:+.2f}% | [{cat_lo:+.2f}%, {cat_hi:+.2f}%] | {cat_wins}/12 | {cat_p:.4g} | {cat_worst:+.2f}% | - | {'**AHEAD OF CAT**' if ahead_cat else 'NOT AHEAD (level/behind)'} |",
        "",
        "## 2. Reference Comparisons (Exploratory, N=12)",
        "",
        "| Comparison | Median Δ | 95% Boot CI | Wins | Wilcoxon p | Worst Δ |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for cand, base in [("SM_S6", "OVR"), ("SM_B", "CAT"), ("SM_B", "LGB"), ("SM_S6", "CAT"), ("SM_B+S6", "LGB")]:
        m, _mn, clo, chi, w, pv, wr = _stats(cand, base)
        lines.append(f"| {cand} vs {base} | {m:+.2f}% | [{clo:+.2f}%, {chi:+.2f}%] | {w}/12 | {pv:.4g} | {wr:+.2f}% |")

    lines += [
        "",
        "## 3. Per-dataset LogLoss & Wall Time (5-fold mean; lower is better)",
        "",
        "| dataset | n | K | OVR | SM_B | SM_S6 | SM_B+S6 | CAT | LGB | SM_B vs OVR | OVR time | SM_B time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, name, _ in DATASETS:
        m = ds_meta[name]
        ovr_ll = agg_ll[(name, "OVR")]
        smb_ll = agg_ll[(name, "SM_B")]
        d_pct = (smb_ll - ovr_ll) / ovr_ll * 100.0
        lines.append(
            f"| `{name}` | {m['n_rows']} | {m['n_classes']} | "
            f"{ovr_ll:.5f} | {smb_ll:.5f} | {agg_ll[(name, 'SM_S6')]:.5f} | {agg_ll[(name, 'SM_B+S6')]:.5f} | "
            f"{agg_ll[(name, 'CAT')]:.5f} | {agg_ll[(name, 'LGB')]:.5f} | {d_pct:+.2f}% | "
            f"{agg_secs[(name, 'OVR')]:.2f}s | {agg_secs[(name, 'SM_B')]:.2f}s |"
        )

    lines += [
        "",
        "## 4. Exploratory Accuracy (5-fold mean; higher is better)",
        "",
        "| dataset | OVR | SM_B | SM_S6 | SM_B+S6 | CAT | LGB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, name, _ in DATASETS:
        lines.append(
            f"| `{name}` | {agg_acc[(name, 'OVR')]:.4f} | {agg_acc[(name, 'SM_B')]:.4f} | "
            f"{agg_acc[(name, 'SM_S6')]:.4f} | {agg_acc[(name, 'SM_B+S6')]:.4f} | "
            f"{agg_acc[(name, 'CAT')]:.4f} | {agg_acc[(name, 'LGB')]:.4f} |"
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
    wk = sub.add_parser("_lgb-worker")
    wk.add_argument("--workdir", type=Path, required=True)
    wk.add_argument("--n-classes", type=int, required=True)
    wk.add_argument("--fold", type=int, required=True)

    args = ap.parse_args()
    if args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "run":
        cmd_run(args.shard, args.nshards)
    elif args.cmd == "summarize":
        cmd_summarize()
    elif args.cmd == "_lgb-worker":
        _cmd_lgb_worker(args.workdir, args.n_classes, args.fold)


if __name__ == "__main__":
    main()
