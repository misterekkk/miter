"""M3 oblique booster speed and quality parity benchmark vs LightGBM and histree."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

HISTREE_SRC = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/src")
HISTREE_T7_DIR = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/research/results/t7")

REG_DATASETS = [
    ("abalone", 45033),
    ("wine_quality", 44136),
    ("cpu_act", 44132),
    ("yprop_4_1", 45032),
    ("sulfur", 44145),
    ("Brazilian_houses", 44141),
    ("Ailerons", 44137),
    ("MiamiHousing2016", 44147),
    ("pol", 44133),
    ("elevators", 44134),
    ("Bike_Sharing_Demand", 44142),
    ("houses", 44138),
    ("superconduct", 44148),
    ("house_sales", 44144),
    ("house_16H", 44139),
    ("diamonds", 44140),
]

CLF_DATASETS = [
    ("eye_movements", 44130),
    ("heloc", 45026),
    ("pol", 44122),
    ("bank-marketing", 44126),
    ("default-of-credit-card-clients", 45020),
    ("MagicTelescope", 44125),
    ("house_16H", 44123),
    ("credit", 44089),
    ("california", 45028),
    ("electricity", 44120),
    ("jannis", 45021),
    ("Diabetes130US", 45022),
    ("MiniBooNE", 44128),
]


def _median_runs(fn, runs: int = 5) -> float:
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def run_speed(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter

    if str(HISTREE_SRC) not in sys.path:
        sys.path.insert(0, str(HISTREE_SRC))
    from histree.gb.oblique_int import ObliqueGBInt  # type: ignore

    # Warmup numba JIT
    xw = np.arange(400, dtype=np.float64).reshape(100, 4)
    ObliqueGBInt(
        selector="R",
        m=4,
        max_trees=2,
        early_stopping_rounds=5,
        max_leaves=31,
        min_data_in_leaf=20,
        reg_lambda=1.0,
        learning_rate=0.05,
        n_bins=255,
    ).fit(xw, xw[:, 0])

    shapes = [(20_000, 16), (20_000, 50), (20_000, 100), (50_000, 50)]
    n_trees = 100

    lines.append("## Part A: Single-thread training speed (100 trees, median of 5 runs)\n")
    lines.append(
        "| shape | LightGBM ms/tree | miter AX | miter default (2m0) | miter/LGBM | miter fast | miter acc (4m0) | histree ObliqueGBInt |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for n, p in shapes:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, p)).astype(np.float64)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.1 * rng.normal(size=n)).astype(np.float64)

        def run_lgb(X_in: np.ndarray = X, y_in: np.ndarray = y) -> None:
            ds = lgb.Dataset(X_in, label=y_in, params={"max_bin": 255, "verbose": -1})
            lgb.train(
                {
                    "objective": "regression",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "min_sum_hessian_in_leaf": 1e-3,
                    "lambda_l2": 1.0,
                    "min_gain_to_split": 0.0,
                    "max_bin": 255,
                    "num_threads": 1,
                    "deterministic": True,
                    "seed": 0,
                    "verbose": -1,
                },
                ds,
                num_boost_round=n_trees,
            )

        def run_miter_cfg(
            m_mult: int, oblique_max_depth: int, X_in: np.ndarray = X, y_in: np.ndarray = y
        ) -> None:
            ds = miter.Dataset(X_in, max_bin=255)
            prm = miter._core.Params()
            prm.n_estimators = n_trees
            prm.learning_rate = 0.05
            prm.num_leaves = 31
            prm.min_data_in_leaf = 20
            prm.min_sum_hessian_in_leaf = 1e-3
            prm.lambda_l2 = 1.0
            prm.min_gain_to_split = 0.0
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

        lo = np.quantile(X, 0.0001, axis=0)
        hi = np.quantile(X, 0.9999, axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        Xq = np.clip(np.rint((X - lo) / span * 65535.0), 0.0, 65535.0)
        m0 = int(math.ceil(math.sqrt(p)))

        def run_hist(Xq_in: np.ndarray = Xq, y_in: np.ndarray = y, m_val: int = 2 * m0) -> None:
            ObliqueGBInt(
                selector="R",
                m=m_val,
                max_trees=n_trees,
                early_stopping_rounds=n_trees + 10,
                max_leaves=31,
                min_data_in_leaf=20,
                reg_lambda=1.0,
                min_gain_to_split=0.0,
                learning_rate=0.05,
                n_bins=255,
                random_state=0,
            ).fit(Xq_in, y_in)

        lgb_ms = _median_runs(run_lgb) * 1000.0 / n_trees
        ax_ms = _median_runs(lambda: run_miter_cfg(0, -1)) * 1000.0 / n_trees
        def_ms = _median_runs(lambda: run_miter_cfg(2, -1)) * 1000.0 / n_trees
        fast_ms = _median_runs(lambda: run_miter_cfg(2, 1)) * 1000.0 / n_trees
        acc_ms = _median_runs(lambda: run_miter_cfg(4, -1)) * 1000.0 / n_trees
        hist_ms = _median_runs(run_hist) * 1000.0 / n_trees

        row = (
            f"| {n} x {p} | {lgb_ms:.3f} | {ax_ms:.3f} | {def_ms:.3f} | "
            f"{def_ms / lgb_ms:.2f}x | {fast_ms:.3f} | {acc_ms:.3f} | {hist_ms:.3f} |"
        )
        print(row, flush=True)
        lines.append(row)


def run_openml_regression_parity(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml

    suite = openml.study.get_suite(336)
    task_map: dict[int, object] = {}
    for tid in suite.tasks:
        t_meta = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        task_map[t_meta.dataset_id] = tid

    lines.append("\n## Part B: 16 OpenML regression datasets x 5 folds (1 thread, ES=50, max_trees=5000)\n")
    lines.append("| dataset | LightGBM nRMSE | miter R_uni nRMSE | histree T7 Q16r_int | miter vs LGBM | miter vs histree T7 | ms/tree (L/M) |")
    lines.append("|---|---|---|---|---|---|---|")

    mit_means: list[float] = []
    lgb_means: list[float] = []
    t7_means: list[float] = []

    for name, did in REG_DATASETS:
        task = openml.tasks.get_task(
            task_map[did], download_data=False, download_qualities=False, download_splits=True
        )
        ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
        X_df, y_s, _, _ = ds_oml.get_data(target=task.target_name, dataset_format="dataframe")
        X = X_df.to_numpy(dtype=np.float64)
        yv = y_s.astype("float64").to_numpy()

        f_mit: list[float] = []
        f_lgb: list[float] = []
        f_t7: list[float] = []
        ms_l_list: list[float] = []
        ms_m_list: list[float] = []

        for fold in range(5):
            tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
            a, b = train_test_split(tr, test_size=0.2, random_state=0)
            sd = float(np.std(yv[te]))

            t0 = time.perf_counter()
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
                lgb.Dataset(X[a], yv[a]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X[b], yv[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            t_l = time.perf_counter() - t0
            p_lgb = m_lgb.predict(X[te], num_iteration=m_lgb.best_iteration)
            f_lgb.append(float(np.sqrt(np.mean((p_lgb - yv[te]) ** 2)) / sd))
            ms_l_list.append(t_l * 1000.0 / max(len(m_lgb.dump_model()["tree_info"]), 1))

            t0 = time.perf_counter()
            m_mit = miter.MiterRegressor(
                n_estimators=5000,
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=20,
                lambda_l2=1.0,
                min_gain_to_split=0.0,
                max_bin=255,
                m_mult=2,
                k_proj=0,
                early_stopping_rounds=50,
                n_jobs=1,
                random_state=fold,
            ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
            t_m = time.perf_counter() - t0
            p_mit = m_mit.predict(X[te])
            f_mit.append(float(np.sqrt(np.mean((p_mit - yv[te]) ** 2)) / sd))
            ms_m_list.append(t_m * 1000.0 / max(len(m_mit.train_loss_), 1))

            t7_path = HISTREE_T7_DIR / f"{name}_{fold}.json"
            if t7_path.exists():
                t7_data = json.loads(t7_path.read_text(encoding="utf-8"))
                f_t7.append(float(t7_data["clean|Q16r_int"]["nrmse"]))

        mm = float(np.mean(f_mit))
        ml = float(np.mean(f_lgb))
        mt7 = float(np.mean(f_t7)) if len(f_t7) == 5 else float("nan")
        mit_means.append(mm)
        lgb_means.append(ml)
        t7_means.append(mt7)
        diff_l = 100.0 * (mm - ml) / ml
        diff_t7 = 100.0 * (mm - mt7) / mt7 if not math.isnan(mt7) else float("nan")
        row = (
            f"| {name} | {ml:.4f} | {mm:.4f} | {mt7:.4f} | "
            f"{diff_l:+.2f}% | {diff_t7:+.2f}% | {np.mean(ms_l_list):.2f} / {np.mean(ms_m_list):.2f} |"
        )
        print(row, flush=True)
        lines.append(row)

    mit_a = np.array(mit_means)
    lgb_a = np.array(lgb_means)
    t7_a = np.array(t7_means)

    rel_lgb = 100.0 * (mit_a - lgb_a) / lgb_a
    _, p_lgb_w = wilcoxon(mit_a, lgb_a)
    rel_t7 = 100.0 * (mit_a - t7_a) / t7_a
    _, p_t7_w = wilcoxon(mit_a, t7_a)

    lines.append("")
    lines.append(
        f"- **miter vs LightGBM (16 reg):** mean {rel_lgb.mean():+.2f}%, **median {np.median(rel_lgb):+.2f}%**, "
        f"wins {int(np.sum(mit_a < lgb_a))}/16, Wilcoxon p = {p_lgb_w:.4g}"
    )
    lines.append(
        f"- **miter vs histree T7 Q16r_int (16 reg):** mean {rel_t7.mean():+.2f}%, **median {np.median(rel_t7):+.2f}%**, "
        f"wins {int(np.sum(mit_a < t7_a))}/16, Wilcoxon p = {p_t7_w:.4g}"
    )


def run_openml_clf_parity(lines: list[str]) -> None:
    import lightgbm as lgb
    import miter
    import openml

    suite = openml.study.get_suite(337)
    task_map: dict[int, object] = {}
    for tid in suite.tasks:
        t_meta = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        task_map[t_meta.dataset_id] = tid

    lines.append("\n## Part C: 13 OpenML binary classification datasets (fold 0, 1 thread, ES=50, max_trees=5000)\n")
    lines.append("| dataset | LightGBM log-loss | miter R_uni log-loss | rel diff | ms/tree (L/M) |")
    lines.append("|---|---|---|---|---|")

    mit_vals: list[float] = []
    lgb_vals: list[float] = []

    for name, did in CLF_DATASETS:
        task = openml.tasks.get_task(
            task_map[did], download_data=False, download_qualities=False, download_splits=True
        )
        ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
        X_df, y_s, _, _ = ds_oml.get_data(target=task.target_name, dataset_format="dataframe")
        X = X_df.to_numpy(dtype=np.float64)
        yv = y_s.astype("category").cat.codes.to_numpy(dtype=np.float64)

        tr, te = task.get_train_test_split_indices(repeat=0, fold=0)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)

        t0 = time.perf_counter()
        m_lgb = lgb.train(
            {
                "objective": "binary",
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
            lgb.Dataset(X[a], yv[a]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X[b], yv[b])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        t_l = time.perf_counter() - t0
        p_lgb = np.clip(m_lgb.predict(X[te], num_iteration=m_lgb.best_iteration), 1e-15, 1.0 - 1e-15)
        ll_l = float(-np.mean(yv[te] * np.log(p_lgb) + (1.0 - yv[te]) * np.log(1.0 - p_lgb)))

        t0 = time.perf_counter()
        m_mit = miter.MiterClassifier(
            n_estimators=5000,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            min_gain_to_split=0.0,
            max_bin=255,
            m_mult=2,
            k_proj=0,
            early_stopping_rounds=50,
            n_jobs=1,
            random_state=0,
        ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
        t_m = time.perf_counter() - t0
        p_mit = np.clip(m_mit.predict_proba(X[te])[:, 1], 1e-15, 1.0 - 1e-15)
        ll_m = float(-np.mean(yv[te] * np.log(p_mit) + (1.0 - yv[te]) * np.log(1.0 - p_mit)))

        mit_vals.append(ll_m)
        lgb_vals.append(ll_l)
        diff = 100.0 * (ll_m - ll_l) / ll_l
        ms_l = t_l * 1000.0 / max(len(m_lgb.dump_model()["tree_info"]), 1)
        ms_m = t_m * 1000.0 / max(len(m_mit.train_loss_), 1)
        row = f"| {name} | {ll_l:.4f} | {ll_m:.4f} | {diff:+.2f}% | {ms_l:.2f} / {ms_m:.2f} |"
        print(row, flush=True)
        lines.append(row)

    mit_a = np.array(mit_vals)
    lgb_a = np.array(lgb_vals)
    rel = 100.0 * (mit_a - lgb_a) / lgb_a
    _, p_w = wilcoxon(mit_a, lgb_a)
    lines.append("")
    lines.append(
        f"- **miter vs LightGBM (13 clf, fold 0):** mean {rel.mean():+.2f}%, **median {np.median(rel):+.2f}%**, "
        f"wins {int(np.sum(mit_a < lgb_a))}/13, Wilcoxon p = {p_w:.4g}"
    )


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import lightgbm as lgb
    import miter

    lines = [
        "# M3 Oblique Booster Benchmark Summary",
        "",
        f"- Hardware: `{platform.machine()}` (`{platform.platform()}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Versions: `miter {miter.__version__}`, `lightgbm {lgb.__version__}`, `numpy {np.__version__}`",
        "- Protocol: single-threaded (`n_jobs=1` / `num_threads=1`). Negative relative difference = lower error = better.",
        "",
    ]
    run_speed(lines)
    run_openml_regression_parity(lines)
    run_openml_clf_parity(lines)
    out_path = Path(__file__).resolve().parent / "results" / "m3_oblique_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
