"""M4 categorical (K1) and native NaN benchmark vs histree, LightGBM, and CatBoost."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold, train_test_split

HISTREE_SRC = Path((str(Path(os.environ.get("HISTREE_DIR", Path(__file__).resolve().parents[2] / "histree"))) + "/src"))

TABARENA_CAT_DATASETS = [
    ("airfoil_self_noise", 46904),
    ("Another-Dataset-on-used-Fiat-500", 46907),
    ("diamonds", 46923),
    ("Food_Delivery_Time", 46928),
    ("healthcare_insurance_expenses", 46931),
]

NAN_REG_DATASETS = [
    ("abalone", 45033),
    ("wine_quality", 44136),
    ("cpu_act", 44132),
    ("sulfur", 44145),
]


def _median_runs(fn: Any, runs: int = 5) -> float:
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _prep_frames(
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


def run_part_a_categorical(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml
    from catboost import CatBoostRegressor

    if str(HISTREE_SRC) not in sys.path:
        sys.path.insert(0, str(HISTREE_SRC))
    from histree.gb.oblique_cat import ObliqueGBCat  # type: ignore

    # Warmup numba JIT on mixed categorical + numeric
    xw = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "b", "c", "a"] * 25),
            "x0": np.linspace(0.0, 1.0, 100),
            "x1": np.linspace(1.0, 0.0, 100),
        }
    )
    ObliqueGBCat(
        selector="R",
        max_trees=2,
        early_stopping_rounds=5,
        min_data_per_group=5,
        learning_rate=0.05,
        random_state=0,
    ).fit(xw, xw["x0"].to_numpy())

    lines.append(
        "## Part A: 5 TabArena categorical regression datasets x 5 folds (1 thread, ES=50, max_trees=5000)\n"
    )
    lines.append(
        "| dataset | LightGBM | CatBoost | histree K1 | miter K1 | miter vs LGBM | miter vs CB | miter vs histree | ms/tree (L/CB/H/M) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    mit_means: list[float] = []
    lgb_means: list[float] = []
    cb_means: list[float] = []
    hist_means: list[float] = []

    for name, did in TABARENA_CAT_DATASETS:
        ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
        X_raw, y_s, _, _ = ds_oml.get_data(
            target=ds_oml.default_target_attribute, dataset_format="dataframe"
        )
        df_cat, df_cb, cat_cols = _prep_frames(X_raw)
        yv = y_s.astype("float64").to_numpy()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        f_lgb, f_cb, f_hist, f_mit = [], [], [], []
        ms_l_list, ms_cb_list, ms_h_list, ms_m_list = [], [], [], []

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
                    "min_data_per_group": 100,
                    "max_cat_to_onehot": 4,
                    "cat_smooth": 10.0,
                    "cat_l2": 10.0,
                    "max_cat_threshold": 32,
                    "num_threads": 1,
                    "deterministic": True,
                    "seed": fold,
                    "verbose": -1,
                },
                lgb.Dataset(df_cat.iloc[a], yv[a]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(df_cat.iloc[b], yv[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            t_l = time.perf_counter() - t0
            p_lgb = m_lgb.predict(df_cat.iloc[te], num_iteration=m_lgb.best_iteration)
            f_lgb.append(float(np.sqrt(np.mean((p_lgb - yv[te]) ** 2)) / sd))
            ms_l_list.append(t_l * 1000.0 / max(len(m_lgb.dump_model()["tree_info"]), 1))

            # 2. CatBoost (native categoricals, 1 thread)
            t0 = time.perf_counter()
            m_cb = CatBoostRegressor(
                iterations=5000,
                early_stopping_rounds=50,
                thread_count=1,
                random_seed=fold,
                allow_writing_files=False,
                verbose=0,
            )
            m_cb.fit(df_cb.iloc[a], yv[a], cat_features=cat_cols, eval_set=(df_cb.iloc[b], yv[b]))
            t_cb = time.perf_counter() - t0
            p_cb = m_cb.predict(df_cb.iloc[te])
            f_cb.append(float(np.sqrt(np.mean((p_cb - yv[te]) ** 2)) / sd))
            ms_cb_list.append(t_cb * 1000.0 / max(int(m_cb.tree_count_), 1))

            # 3. histree ObliqueGBCat K1
            t0 = time.perf_counter()
            m_hist = ObliqueGBCat(
                selector="R",
                max_trees=5000,
                early_stopping_rounds=50,
                max_leaves=31,
                min_data_in_leaf=20,
                reg_lambda=1.0,
                learning_rate=0.05,
                n_bins=255,
                random_state=fold,
            ).fit(df_cat.iloc[a], yv[a], df_cat.iloc[b], yv[b])
            t_h = time.perf_counter() - t0
            p_hist = m_hist.predict(df_cat.iloc[te])
            f_hist.append(float(np.sqrt(np.mean((p_hist - yv[te]) ** 2)) / sd))
            ms_h_list.append(t_h * 1000.0 / max(len(m_hist.trees_), 1))

            # 4. miter R_uni + K1
            t0 = time.perf_counter()
            m_mit = miter.MiterRegressor(
                n_estimators=5000,
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=20,
                lambda_l2=1.0,
                max_bin=255,
                early_stopping_rounds=50,
                n_jobs=1,
                random_state=fold,
            ).fit(df_cat.iloc[a], yv[a], eval_set=[(df_cat.iloc[b], yv[b])])
            t_m = time.perf_counter() - t0
            p_mit = m_mit.predict(df_cat.iloc[te])
            f_mit.append(float(np.sqrt(np.mean((p_mit - yv[te]) ** 2)) / sd))
            ms_m_list.append(t_m * 1000.0 / max(len(m_mit.train_loss_), 1))

        ml = float(np.mean(f_lgb))
        mcb = float(np.mean(f_cb))
        mh = float(np.mean(f_hist))
        mm = float(np.mean(f_mit))
        lgb_means.append(ml)
        cb_means.append(mcb)
        hist_means.append(mh)
        mit_means.append(mm)

        row = (
            f"| {name} | {ml:.4f} | {mcb:.4f} | {mh:.4f} | {mm:.4f} | "
            f"{100.0 * (mm - ml) / ml:+.2f}% | {100.0 * (mm - mcb) / mcb:+.2f}% | "
            f"{100.0 * (mm - mh) / mh:+.2f}% | "
            f"{np.mean(ms_l_list):.2f} / {np.mean(ms_cb_list):.2f} / "
            f"{np.mean(ms_h_list):.2f} / {np.mean(ms_m_list):.2f} |"
        )
        print(row, flush=True)
        lines.append(row)

    mit_a = np.array(mit_means)
    for label, ref in [
        ("LightGBM", np.array(lgb_means)),
        ("CatBoost", np.array(cb_means)),
        ("histree K1", np.array(hist_means)),
    ]:
        rel = 100.0 * (mit_a - ref) / ref
        _, p_w = wilcoxon(mit_a, ref)
        lines.append(
            f"- **miter K1 vs {label} (5 cat):** mean {rel.mean():+.2f}%, "
            f"**median {np.median(rel):+.2f}%**, wins {int(np.sum(mit_a < ref))}/5, Wilcoxon p = {p_w:.4g}"
        )


def _inject_nans(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    seed: int,
    rate: float = 0.15,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    n, p = X_out.shape
    if mode == "mcar":
        mask = rng.random((n, p)) < rate
        X_out[mask] = np.nan
        return X_out
    # Target-dependent (MNAR) on top-variance numeric feature + lighter MCAR elsewhere.
    top_col = int(np.argmax(np.var(X_out, axis=0)))
    y_norm = (y - np.mean(y)) / max(float(np.std(y)), 1e-12)
    prob = np.clip(0.05 + 0.35 / (1.0 + np.exp(-2.0 * y_norm)), 0.0, 0.9)
    X_out[rng.random(n) < prob, top_col] = np.nan
    for j in range(p):
        if j != top_col:
            X_out[rng.random(n) < 0.05, j] = np.nan
    return X_out


def _impute_with_indicators(
    X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    outs = []
    for X in (X_tr, X_val, X_te):
        miss = np.isnan(X)
        X_imp = np.where(miss, med, X)
        outs.append(np.hstack([X_imp, miss.astype(np.float64)]))
    return outs[0], outs[1], outs[2]


def run_part_b_nans(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml
    from catboost import CatBoostRegressor

    suite = openml.study.get_suite(336)
    task_map: dict[int, object] = {}
    for tid in suite.tasks:
        t_meta = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        task_map[t_meta.dataset_id] = tid

    lines.append(
        "\n## Part B: Native NaN handling (4 OpenML reg x 3 folds, 1 thread, ES=50, MCAR 15% & MNAR)\n"
    )
    lines.append(
        "| regime | dataset | LightGBM | CatBoost | miter native NaN | miter median+ind | miter vs LGBM | miter vs CB | miter vs imputed |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for regime in ("mcar", "mnar"):
        mit_vals, lgb_vals, cb_vals, imp_vals = [], [], [], []
        for name, did in NAN_REG_DATASETS:
            task = openml.tasks.get_task(
                task_map[did], download_data=False, download_qualities=False, download_splits=True
            )
            ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
            X_df, y_s, _, _ = ds_oml.get_data(target=task.target_name, dataset_format="dataframe")
            X = X_df.to_numpy(dtype=np.float64)
            yv = y_s.astype("float64").to_numpy()

            f_l, f_cb, f_m, f_imp = [], [], [], []
            for fold in range(3):
                tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
                X_nan = _inject_nans(X, yv, regime, seed=100 + fold)
                a, b = train_test_split(tr, test_size=0.2, random_state=0)
                sd = float(np.std(yv[te]))

                m_lgb = lgb.train(
                    {
                        "objective": "regression",
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "min_data_in_leaf": 20,
                        "lambda_l2": 1.0,
                        "max_bin": 255,
                        "num_threads": 1,
                        "deterministic": True,
                        "seed": fold,
                        "verbose": -1,
                    },
                    lgb.Dataset(X_nan[a], yv[a]),
                    num_boost_round=5000,
                    valid_sets=[lgb.Dataset(X_nan[b], yv[b])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                p_lgb = m_lgb.predict(X_nan[te], num_iteration=m_lgb.best_iteration)
                f_l.append(float(np.sqrt(np.mean((p_lgb - yv[te]) ** 2)) / sd))

                m_cb = CatBoostRegressor(
                    iterations=5000,
                    early_stopping_rounds=50,
                    thread_count=1,
                    random_seed=fold,
                    allow_writing_files=False,
                    verbose=0,
                )
                m_cb.fit(X_nan[a], yv[a], eval_set=(X_nan[b], yv[b]))
                p_cb = m_cb.predict(X_nan[te])
                f_cb.append(float(np.sqrt(np.mean((p_cb - yv[te]) ** 2)) / sd))

                m_mit = miter.MiterRegressor(
                    n_estimators=5000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_data_in_leaf=20,
                    lambda_l2=1.0,
                    early_stopping_rounds=50,
                    n_jobs=1,
                    random_state=fold,
                ).fit(X_nan[a], yv[a], eval_set=[(X_nan[b], yv[b])])
                p_mit = m_mit.predict(X_nan[te])
                f_m.append(float(np.sqrt(np.mean((p_mit - yv[te]) ** 2)) / sd))

                Xa_i, Xb_i, Xte_i = _impute_with_indicators(X_nan[a], X_nan[b], X_nan[te])
                m_imp = miter.MiterRegressor(
                    n_estimators=5000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_data_in_leaf=20,
                    lambda_l2=1.0,
                    early_stopping_rounds=50,
                    n_jobs=1,
                    random_state=fold,
                ).fit(Xa_i, yv[a], eval_set=[(Xb_i, yv[b])])
                p_imp = m_imp.predict(Xte_i)
                f_imp.append(float(np.sqrt(np.mean((p_imp - yv[te]) ** 2)) / sd))

            ml, mcb, mm, mimp = (
                float(np.mean(f_l)),
                float(np.mean(f_cb)),
                float(np.mean(f_m)),
                float(np.mean(f_imp)),
            )
            lgb_vals.append(ml)
            cb_vals.append(mcb)
            mit_vals.append(mm)
            imp_vals.append(mimp)
            row = (
                f"| {regime} | {name} | {ml:.4f} | {mcb:.4f} | {mm:.4f} | {mimp:.4f} | "
                f"{100.0 * (mm - ml) / ml:+.2f}% | {100.0 * (mm - mcb) / mcb:+.2f}% | "
                f"{100.0 * (mm - mimp) / mimp:+.2f}% |"
            )
            print(row, flush=True)
            lines.append(row)

        ma = np.array(mit_vals)
        for label, ref in [
            ("LightGBM", np.array(lgb_vals)),
            ("CatBoost", np.array(cb_vals)),
            ("imputed+ind", np.array(imp_vals)),
        ]:
            rel = 100.0 * (ma - ref) / ref
            lines.append(
                f"- **{regime}: miter native NaN vs {label}:** mean {rel.mean():+.2f}%, "
                f"median {np.median(rel):+.2f}%, wins {int(np.sum(ma < ref))}/4"
            )


def run_part_c_speed(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    from catboost import CatBoostRegressor

    if str(HISTREE_SRC) not in sys.path:
        sys.path.insert(0, str(HISTREE_SRC))
    from histree.gb.oblique_cat import ObliqueGBCat  # type: ignore

    rng = np.random.default_rng(7)
    n, n_num, n_cat = 20_000, 30, 10
    cols: dict[str, Any] = {}
    for j in range(n_num):
        cols[f"x{j}"] = rng.normal(size=n)
    for c in range(n_cat):
        cols[f"c{c}"] = pd.Categorical([f"k{v}" for v in rng.integers(0, 20, size=n)])
    df_cat = pd.DataFrame(cols)
    df_cb = df_cat.copy()
    cat_names = [f"c{c}" for c in range(n_cat)]
    for cn in cat_names:
        df_cb[cn] = df_cb[cn].astype(str)
    y = cols["x0"] + 0.5 * cols["x1"] + (df_cat["c0"].cat.codes % 3 == 0).astype(float)
    n_trees = 100

    # Warmup histree
    ObliqueGBCat(selector="R", max_trees=2, early_stopping_rounds=5, min_data_per_group=50).fit(
        df_cat.iloc[:200], y[:200]
    )

    def run_lgb() -> None:
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
            lgb.Dataset(df_cat, label=y),
            num_boost_round=n_trees,
        )

    def run_cb() -> None:
        CatBoostRegressor(
            iterations=n_trees,
            depth=6,
            learning_rate=0.05,
            thread_count=1,
            random_seed=0,
            allow_writing_files=False,
            verbose=0,
        ).fit(df_cb, y, cat_features=cat_names)

    def run_miter_ax() -> None:
        miter.MiterRegressor(
            n_estimators=n_trees,
            m_mult=0,
            early_stopping_rounds=None,
            n_jobs=1,
            random_state=0,
        ).fit(df_cat, y)

    def run_miter_r() -> None:
        miter.MiterRegressor(
            n_estimators=n_trees,
            m_mult=2,
            early_stopping_rounds=None,
            n_jobs=1,
            random_state=0,
        ).fit(df_cat, y)

    def run_hist() -> None:
        ObliqueGBCat(
            selector="R",
            max_trees=n_trees,
            early_stopping_rounds=n_trees + 10,
            random_state=0,
        ).fit(df_cat, y)

    ms_lgb = _median_runs(run_lgb) * 1000.0 / n_trees
    ms_cb = _median_runs(run_cb) * 1000.0 / n_trees
    ms_ax = _median_runs(run_miter_ax) * 1000.0 / n_trees
    ms_r = _median_runs(run_miter_r) * 1000.0 / n_trees
    ms_hist = _median_runs(run_hist) * 1000.0 / n_trees

    lines.append("\n## Part C: Single-thread mixed speed (20000 x [30 num + 10 cat], 100 trees, median of 5 runs)\n")
    lines.append("| LightGBM ms/tree | CatBoost ms/tree | miter AX+K1 | miter R_uni+K1 | miter/LGBM | histree K1 |")
    lines.append("|---|---|---|---|---|---|")
    row = f"| {ms_lgb:.3f} | {ms_cb:.3f} | {ms_ax:.3f} | {ms_r:.3f} | {ms_r / ms_lgb:.2f}x | {ms_hist:.3f} |"
    print(row, flush=True)
    lines.append(row)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import catboost
    import lightgbm as lgb
    import miter

    lines = [
        "# M4 Categorical K1 and Native NaN Benchmark Summary",
        "",
        f"- Hardware: `{platform.machine()}` (`{platform.platform()}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Versions: `miter {miter.__version__}`, `lightgbm {lgb.__version__}`, `catboost {catboost.__version__}`, `numpy {np.__version__}`",
        "- Protocol: single-threaded (`n_jobs=1` / `num_threads=1` / `thread_count=1`). Negative relative difference = lower error = better.",
        "",
    ]
    run_part_a_categorical(lines)
    run_part_b_nans(lines)
    run_part_c_speed(lines)

    out_path = Path(__file__).resolve().parent / "results" / "m4_cat_nan_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
