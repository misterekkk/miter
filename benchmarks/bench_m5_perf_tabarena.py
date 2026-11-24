"""M5 performance, OpenMP scaling, and 12-dataset TabArena leaderboard benchmark."""

from __future__ import annotations

import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import KFold, train_test_split

TABARENA_REG_12 = [
    ("airfoil_self_noise", 46904),
    ("Another-Dataset-on-used-Fiat-500", 46907),
    ("concrete_compressive_strength", 46917),
    ("diamonds", 46923),
    ("Food_Delivery_Time", 46928),
    ("healthcare_insurance_expenses", 46931),
    ("houses", 46934),
    ("miami_housing", 46942),
    ("physiochemical_protein", 46949),
    ("QSAR-TID-11", 46953),
    ("QSAR_fish_toxicity", 46954),
    ("superconductivity", 46961),
]


def _median_runs(fn: Any, runs: int = 5) -> float:
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _prep_tabarena_frames(
    X_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df_cat = X_df.copy()
    cat_cols: list[str] = []
    for c in df_cat.columns:
        if str(df_cat[c].dtype) in ("category", "object", "string", "str"):
            df_cat[c] = df_cat[c].astype("category")
            cat_cols.append(str(c))
        else:
            df_cat[c] = df_cat[c].astype(np.float64)

    df_cb = df_cat.copy()
    for c in cat_cols:
        df_cb[c] = df_cb[c].astype(object).where(df_cb[c].notna(), "__MISSING__").astype(str)
    return df_cat, df_cb, cat_cols


def _measure_peak_rss_mb(snippet: str) -> float:
    code = (
        "import resource, sys\n"
        + snippet
        + "\nru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
        + "mb = ru / (1024.0 * 1024.0) if sys.platform == 'darwin' else ru / 1024.0\n"
        + "print(f'{mb:.1f}')\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return float(out.splitlines()[-1])


def _compute_tabarena_leaderboard(
    methods: list[str],
    split_errors: dict[str, list[float]],
    dataset_means: dict[str, list[float]],
    train_times: dict[str, list[float]],
) -> list[dict[str, Any]]:
    k_m = len(methods)
    n_splits = len(split_errors[methods[0]])
    n_datasets = len(dataset_means[methods[0]])

    # Build pairwise win-counts across all (dataset, fold) splits
    wins = np.zeros((k_m, k_m), dtype=np.float64)
    totals = np.zeros((k_m, k_m), dtype=np.float64)
    ranks_accum = np.zeros(k_m, dtype=np.float64)

    for s in range(n_splits):
        errs = np.array([split_errors[m][s] for m in methods], dtype=np.float64)
        ranks_accum += rankdata(errs, method="average")
        for i in range(k_m):
            for j in range(i + 1, k_m):
                totals[i, j] += 1.0
                totals[j, i] += 1.0
                if abs(errs[i] - errs[j]) <= 1e-12:
                    wins[i, j] += 0.5
                    wins[j, i] += 0.5
                elif errs[i] < errs[j]:
                    wins[i, j] += 1.0
                else:
                    wins[j, i] += 1.0

    # Empirical win-rate across all opponent comparisons
    winrates = np.zeros(k_m, dtype=np.float64)
    for i in range(k_m):
        opp_tot = float(np.sum(totals[i]))
        winrates[i] = float(np.sum(wins[i])) / opp_tot if opp_tot > 0.0 else 0.5

    # Bradley-Terry MLE -> Elo (base 1000, scale 400 / ln(10))
    def bt_nll(r: np.ndarray) -> float:
        r_c = r - np.mean(r)
        loss = 1e-5 * float(np.sum(r_c * r_c))
        for i in range(k_m):
            for j in range(i + 1, k_m):
                if totals[i, j] <= 0.0:
                    continue
                diff = r_c[i] - r_c[j]
                p_ij = 1.0 / (1.0 + math.exp(-np.clip(diff, -30.0, 30.0)))
                loss -= wins[i, j] * math.log(max(p_ij, 1e-15)) + wins[j, i] * math.log(
                    max(1.0 - p_ij, 1e-15)
                )
        return loss

    opt = minimize(bt_nll, np.zeros(k_m, dtype=np.float64), method="BFGS")
    r_hat = opt.x - np.mean(opt.x)
    elo = 1000.0 + (400.0 / math.log(10.0)) * r_hat

    # Per-dataset Improvability (%) and Normalized Error
    improv = {m: [] for m in methods}
    norm_err = {m: [] for m in methods}
    for d in range(n_datasets):
        d_errs = np.array([dataset_means[m][d] for m in methods], dtype=np.float64)
        best_e = float(np.min(d_errs))
        med_e = float(np.median(d_errs))
        for m, val in zip(methods, d_errs, strict=True):
            improv[m].append(100.0 * (val - best_e) / max(val, 1e-15))
            if med_e > best_e:
                norm_err[m].append(float(np.clip((val - best_e) / (med_e - best_e), 0.0, 1.0)))
            else:
                norm_err[m].append(0.0)

    rows = []
    for i, m in enumerate(methods):
        rows.append(
            {
                "method": m,
                "elo": float(elo[i]),
                "winrate_pct": 100.0 * float(winrates[i]),
                "avg_rank": float(ranks_accum[i] / n_splits),
                "improv_pct": float(np.mean(improv[m])),
                "norm_err": float(np.mean(norm_err[m])),
                "med_train_s": float(np.median(train_times[m])),
            }
        )
    rows.sort(key=lambda r: r["elo"], reverse=True)
    return rows


def run_part_a_single_thread(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    from catboost import CatBoostRegressor

    shapes = [(20_000, 16), (20_000, 50), (20_000, 100), (50_000, 50), (10_000, 1024)]
    n_trees = 100

    lines.append(
        "## Part A: Single-thread training speed (100 trees, median of 5 runs, `n_jobs=1`)\n"
    )
    lines.append(
        "| shape | LightGBM ms/tree | CatBoost ms/tree | miter AX | miter default | miter fast | miter acc | miter/LGBM | miter/CB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for n, p in shapes:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, p)).astype(np.float64)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.1 * rng.normal(size=n)).astype(
            np.float64
        )

        def run_lgb(X_in: np.ndarray = X, y_in: np.ndarray = y) -> None:
            ds = lgb.Dataset(X_in, label=y_in, params={"max_bin": 255, "verbose": -1})
            lgb.train(
                {
                    "objective": "regression",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "lambda_l2": 1.0,
                    "max_bin": 255,
                    "num_threads": 1,
                    "deterministic": True,
                    "seed": 0,
                    "verbose": -1,
                },
                ds,
                num_boost_round=n_trees,
            )

        def run_cb(X_in: np.ndarray = X, y_in: np.ndarray = y) -> None:
            CatBoostRegressor(
                iterations=n_trees,
                depth=6,
                learning_rate=0.05,
                thread_count=1,
                random_seed=0,
                allow_writing_files=False,
                verbose=0,
            ).fit(X_in, y_in)

        def run_miter(
            m_mult: int, oblique_max_depth: int, X_in: np.ndarray = X, y_in: np.ndarray = y
        ) -> None:
            ds = miter.Dataset(X_in, max_bin=255)
            prm = miter._core.Params()
            prm.n_estimators = n_trees
            prm.learning_rate = 0.05
            prm.num_leaves = 31
            prm.min_data_in_leaf = 20
            prm.lambda_l2 = 1.0
            prm.max_bin = 255
            prm.m_mult = m_mult
            prm.oblique_max_depth = oblique_max_depth
            prm.early_stopping_rounds = 0
            prm.n_jobs = 1
            miter._core.fit_booster(
                ds._inner,
                miter._core.Objective.Regression,
                y_in,
                None,
                None,
                None,
                None,
                prm,
            )

        lgb_ms = _median_runs(run_lgb) * 1000.0 / n_trees
        cb_ms = _median_runs(run_cb) * 1000.0 / n_trees
        ax_ms = _median_runs(lambda: run_miter(0, -1)) * 1000.0 / n_trees
        def_ms = _median_runs(lambda: run_miter(2, -1)) * 1000.0 / n_trees
        fast_ms = _median_runs(lambda: run_miter(2, 1)) * 1000.0 / n_trees
        acc_ms = _median_runs(lambda: run_miter(4, -1)) * 1000.0 / n_trees

        row = (
            f"| {n} x {p} | {lgb_ms:.3f} | {cb_ms:.3f} | {ax_ms:.3f} | {def_ms:.3f} | "
            f"{fast_ms:.3f} | {acc_ms:.3f} | {def_ms / lgb_ms:.2f}x | {def_ms / cb_ms:.2f}x |"
        )
        print(row, flush=True)
        lines.append(row)

    # Clean subprocess peak RSS on (50000, 50), 100 trees
    rss_lgb = _measure_peak_rss_mb(
        "import numpy as np, lightgbm as lgb; rng=np.random.default_rng(0); "
        "X=rng.normal(size=(50000,50)); y=X[:,0]; "
        "lgb.train({'objective':'regression','num_threads':1,'verbose':-1}, lgb.Dataset(X,y), 100)"
    )
    rss_cb = _measure_peak_rss_mb(
        "import numpy as np; from catboost import CatBoostRegressor; rng=np.random.default_rng(0); "
        "X=rng.normal(size=(50000,50)); y=X[:,0]; "
        "CatBoostRegressor(iterations=100,depth=6,thread_count=1,allow_writing_files=False,verbose=0).fit(X,y)"
    )
    rss_mit = _measure_peak_rss_mb(
        "import numpy as np, miter; rng=np.random.default_rng(0); "
        "X=rng.normal(size=(50000,50)); y=X[:,0]; "
        "miter.MiterRegressor(n_estimators=100,early_stopping_rounds=None,n_jobs=1).fit(X,y)"
    )
    lines.append(
        f"\n- **Peak RSS on `50000 x 50` (100 trees):** LightGBM `{rss_lgb:.1f} MB`, "
        f"CatBoost `{rss_cb:.1f} MB`, miter default `{rss_mit:.1f} MB`"
    )


def run_part_b_multithread(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml
    from catboost import CatBoostRegressor

    n_trees = 100
    thread_list = [1, 2, 4, 8]

    rng = np.random.default_rng(123)
    X_500k = rng.normal(size=(500_000, 50)).astype(np.float64)
    y_500k = (X_500k[:, 0] + 0.5 * X_500k[:, 1] - 0.3 * X_500k[:, 2]).astype(np.float64)

    ds_cov = openml.datasets.get_dataset(44121, download_data=True, download_qualities=False)
    X_cov_df, y_cov_s, _, _ = ds_cov.get_data(
        target=ds_cov.default_target_attribute, dataset_format="dataframe"
    )
    X_cov = X_cov_df.to_numpy(dtype=np.float64)
    y_cov = y_cov_s.astype("category").cat.codes.to_numpy(dtype=np.float64)

    lines.append("\n## Part B: Multithreaded scaling (`1, 2, 4, 8` threads, 100 trees, median of 5 runs)\n")
    lines.append("| dataset | model | T=1 (ms/tree) | T=2 (ms/tree) | T=4 (ms/tree) | T=8 (ms/tree) | Speedup T1/T8 |")
    lines.append("|---|---|---|---|---|---|---|")

    # Verify bitwise determinism 1 vs 8 on 500k x 50
    ds_sub = miter.Dataset(X_500k, max_bin=255)
    p1 = miter._core.Params()
    p1.n_estimators = 5
    p1.early_stopping_rounds = 0
    p1.n_jobs = 1
    p8 = miter._core.Params()
    p8.n_estimators = 5
    p8.early_stopping_rounds = 0
    p8.n_jobs = 8
    r1 = miter._core.fit_booster(ds_sub._inner, miter._core.Objective.Regression, y_500k, None, None, None, None, p1)
    r8 = miter._core.fit_booster(ds_sub._inner, miter._core.Objective.Regression, y_500k, None, None, None, None, p8)
    j1 = miter._core.model_to_json(r1.model).replace('"n_jobs": 1', '"n_jobs": 0')
    j8 = miter._core.model_to_json(r8.model).replace('"n_jobs": 8', '"n_jobs": 0')
    assert j1 == j8, "500k x 50 1-thread vs 8-thread model JSON mismatch!"

    datasets = [
        ("synth 500k x 50 (reg)", X_500k, y_500k, "regression"),
        (f"covertype {X_cov.shape[0]} x {X_cov.shape[1]} (clf)", X_cov, y_cov, "binary"),
    ]

    for d_label, X_d, y_d, obj_str in datasets:
        obj_enum = (
            miter._core.Objective.Regression if obj_str == "regression" else miter._core.Objective.Binary
        )
        ds_mit = miter.Dataset(X_d, max_bin=255)
        ds_lgb = lgb.Dataset(X_d, label=y_d, params={"max_bin": 255, "verbose": -1})
        ds_lgb.construct()

        for model_name in ("LightGBM", "CatBoost", "miter AX", "miter default"):
            per_t: list[float] = []
            for th in thread_list:
                if model_name == "LightGBM":
                    def fn_l(
                        t_cnt: int = th,
                        obj_s: str = obj_str,
                        ds_l: Any = ds_lgb,
                    ) -> None:
                        lgb.train(
                            {
                                "objective": obj_s,
                                "learning_rate": 0.05,
                                "num_leaves": 31,
                                "min_data_in_leaf": 20,
                                "lambda_l2": 1.0,
                                "max_bin": 255,
                                "num_threads": t_cnt,
                                "deterministic": True,
                                "seed": 0,
                                "verbose": -1,
                            },
                            ds_l,
                            num_boost_round=n_trees,
                        )
                    ms = _median_runs(fn_l) * 1000.0 / n_trees
                elif model_name == "CatBoost":
                    def fn_c(
                        t_cnt: int = th,
                        obj_s: str = obj_str,
                        X_ref: Any = X_d,
                        y_ref: Any = y_d,
                    ) -> None:
                        if obj_s == "regression":
                            CatBoostRegressor(
                                iterations=n_trees,
                                depth=6,
                                learning_rate=0.05,
                                thread_count=t_cnt,
                                random_seed=0,
                                allow_writing_files=False,
                                verbose=0,
                            ).fit(X_ref, y_ref)
                        else:
                            from catboost import CatBoostClassifier

                            CatBoostClassifier(
                                iterations=n_trees,
                                depth=6,
                                learning_rate=0.05,
                                thread_count=t_cnt,
                                random_seed=0,
                                allow_writing_files=False,
                                verbose=0,
                            ).fit(X_ref, y_ref)
                    ms = _median_runs(fn_c) * 1000.0 / n_trees
                else:
                    mult = 0 if model_name == "miter AX" else 2
                    def fn_m(
                        t_cnt: int = th,
                        m_val: int = mult,
                        ds_m: Any = ds_mit,
                        obj_e: Any = obj_enum,
                        y_ref: Any = y_d,
                    ) -> None:
                        prm = miter._core.Params()
                        prm.n_estimators = n_trees
                        prm.learning_rate = 0.05
                        prm.num_leaves = 31
                        prm.min_data_in_leaf = 20
                        prm.lambda_l2 = 1.0
                        prm.max_bin = 255
                        prm.m_mult = m_val
                        prm.early_stopping_rounds = 0
                        prm.n_jobs = t_cnt
                        miter._core.fit_booster(
                            ds_m._inner, obj_e, y_ref, None, None, None, None, prm
                        )
                    ms = _median_runs(fn_m) * 1000.0 / n_trees
                per_t.append(ms)

            row = (
                f"| {d_label} | {model_name} | {per_t[0]:.3f} | {per_t[1]:.3f} | "
                f"{per_t[2]:.3f} | {per_t[3]:.3f} | {per_t[0] / per_t[3]:.2f}x |"
            )
            print(row, flush=True)
            lines.append(row)

    lines.append("\n- Verified byte-for-byte identical JSON model for `500k x 50` (5 trees, `m_mult=2`) across `n_jobs=1` vs `n_jobs=8`.")


def run_part_c_tabarena(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml
    from catboost import CatBoostRegressor

    methods = [
        "miter (accurate)",
        "miter (default)",
        "miter (fast)",
        "miter (AX)",
        "CatBoost (default)",
        "LightGBM (default)",
    ]
    split_errors: dict[str, list[float]] = {m: [] for m in methods}
    dataset_means: dict[str, list[float]] = {m: [] for m in methods}
    train_times: dict[str, list[float]] = {m: [] for m in methods}
    ds_rows: list[str] = []

    n_threads = 4
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for name, did in TABARENA_REG_12:
        ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
        X_raw, y_s, _, _ = ds_oml.get_data(
            target=ds_oml.default_target_attribute, dataset_format="dataframe"
        )
        df_cat, df_cb, cat_cols = _prep_tabarena_frames(X_raw)
        yv = y_s.astype("float64").to_numpy()

        fold_errs: dict[str, list[float]] = {m: [] for m in methods}
        for fold, (tr, te) in enumerate(kf.split(df_cat)):
            a, b = train_test_split(tr, test_size=0.2, random_state=0)
            sd = float(np.std(yv[te]))

            # 1. LightGBM
            t0 = time.perf_counter()
            m_lgb = lgb.train(
                {
                    "objective": "regression",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "lambda_l2": 1.0,
                    "max_bin": 255,
                    "num_threads": n_threads,
                    "deterministic": True,
                    "seed": fold,
                    "verbose": -1,
                },
                lgb.Dataset(df_cat.iloc[a], yv[a]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(df_cat.iloc[b], yv[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            dt = time.perf_counter() - t0
            e_lgb = float(np.sqrt(np.mean((m_lgb.predict(df_cat.iloc[te], num_iteration=m_lgb.best_iteration) - yv[te]) ** 2)) / sd)
            fold_errs["LightGBM (default)"].append(e_lgb)
            split_errors["LightGBM (default)"].append(e_lgb)
            train_times["LightGBM (default)"].append(dt)

            # 2. CatBoost
            t0 = time.perf_counter()
            m_cb = CatBoostRegressor(
                iterations=5000,
                early_stopping_rounds=50,
                thread_count=n_threads,
                random_seed=fold,
                allow_writing_files=False,
                verbose=0,
            )
            if cat_cols:
                m_cb.fit(df_cb.iloc[a], yv[a], cat_features=cat_cols, eval_set=(df_cb.iloc[b], yv[b]))
            else:
                m_cb.fit(df_cb.iloc[a], yv[a], eval_set=(df_cb.iloc[b], yv[b]))
            dt = time.perf_counter() - t0
            e_cb = float(np.sqrt(np.mean((m_cb.predict(df_cb.iloc[te]) - yv[te]) ** 2)) / sd)
            fold_errs["CatBoost (default)"].append(e_cb)
            split_errors["CatBoost (default)"].append(e_cb)
            train_times["CatBoost (default)"].append(dt)

            # 3-6. miter variants (shares deterministic Dataset preprocessing across presets)
            for label, preset, m_mult in [
                ("miter (AX)", "default", 0),
                ("miter (fast)", "fast", 2),
                ("miter (default)", "default", 2),
                ("miter (accurate)", "accurate", 4),
            ]:
                t0 = time.perf_counter()
                m_mit = miter.MiterRegressor(
                    n_estimators=5000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_data_in_leaf=20,
                    lambda_l2=1.0,
                    max_bin=255,
                    m_mult=m_mult,
                    preset=preset,
                    early_stopping_rounds=50,
                    n_jobs=n_threads,
                    random_state=fold,
                ).fit(df_cat.iloc[a], yv[a], eval_set=[(df_cat.iloc[b], yv[b])])
                dt = time.perf_counter() - t0
                e_mit = float(np.sqrt(np.mean((m_mit.predict(df_cat.iloc[te]) - yv[te]) ** 2)) / sd)
                fold_errs[label].append(e_mit)
                split_errors[label].append(e_mit)
                train_times[label].append(dt)

        for m in methods:
            dataset_means[m].append(float(np.mean(fold_errs[m])))

        d_ax = dataset_means["miter (AX)"][-1]
        d_fast = dataset_means["miter (fast)"][-1]
        d_def = dataset_means["miter (default)"][-1]
        d_acc = dataset_means["miter (accurate)"][-1]
        d_lgb = dataset_means["LightGBM (default)"][-1]
        d_cb = dataset_means["CatBoost (default)"][-1]
        row = (
            f"| {name} ({X_raw.shape[0]}x{X_raw.shape[1]}) | {d_lgb:.4f} | {d_cb:.4f} | "
            f"{d_ax:.4f} | {d_fast:.4f} | {d_def:.4f} | {d_acc:.4f} | "
            f"{100.0 * (d_def - d_lgb) / d_lgb:+.2f}% | {100.0 * (d_def - d_cb) / d_cb:+.2f}% |"
        )
        print(row, flush=True)
        ds_rows.append(row)

    lb = _compute_tabarena_leaderboard(methods, split_errors, dataset_means, train_times)
    lines.append(
        "\n## Part C: Hard TabArena 12-Dataset Regression Suite (3 folds, ES=50, `n_jobs=4`)\n"
    )
    lines.append("### TabArena-Style Leaderboard Score\n")
    lines.append("| Method | Elo | Win-rate (%) | Avg Rank | Improvability (%) | Normalized Error | Median Train (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in lb:
        lines.append(
            f"| {r['method']} | {r['elo']:.1f} | {r['winrate_pct']:.1f}% | "
            f"{r['avg_rank']:.2f} | {r['improv_pct']:.2f}% | {r['norm_err']:.3f} | {r['med_train_s']:.2f} |"
        )

    lines.append("\n### Per-Dataset Mean Test nRMSE (3 folds)\n")
    lines.append(
        "| dataset | LGBM | CatBoost | miter AX | miter fast | miter default | miter acc | def vs LGBM | def vs CB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.extend(ds_rows)

    for label in ("miter (default)", "miter (accurate)", "miter (fast)"):
        ma = np.array(dataset_means[label])
        for ref_name in ("LightGBM (default)", "CatBoost (default)"):
            ra = np.array(dataset_means[ref_name])
            rel = 100.0 * (ma - ra) / ra
            _, p_w = wilcoxon(ma, ra)
            lines.append(
                f"- **{label} vs {ref_name} (12 datasets):** mean {rel.mean():+.2f}%, "
                f"**median {np.median(rel):+.2f}%**, wins {int(np.sum(ma < ra))}/12, Wilcoxon p = {p_w:.4g}"
            )


def main() -> None:
    import catboost
    import lightgbm as lgb
    import miter

    lines = [
        "# M5 Performance, OpenMP Scaling, and TabArena Leaderboard Summary",
        "",
        f"- Hardware: `{platform.machine()}` (`{platform.platform()}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Versions: `miter {miter.__version__}`, `lightgbm {lgb.__version__}`, `catboost {catboost.__version__}`, `numpy {np.__version__}`",
        "- Protocol: Part A uses 1 thread; Part B tests 1/2/4/8 threads; Part C uses `n_jobs=4` across 12 TabArena datasets (3 folds, ES=50).",
        "",
    ]
    run_part_a_single_thread(lines)
    run_part_b_multithread(lines)
    run_part_c_tabarena(lines)

    out_path = Path(__file__).resolve().parent / "results" / "m5_perf_tabarena_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
