"""M2 AX booster speed and quality benchmark vs LightGBM and histree (single-threaded)."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HISTREE_SRC = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/src")


def _median_runs(fn, runs: int = 5) -> tuple[float, object]:
    times: list[float] = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), last


def run_synthetic_speed(out_lines: list[str]) -> None:
    import lightgbm as lgb
    import miter

    has_histree = False
    try:
        if HISTREE_SRC.exists() and str(HISTREE_SRC) not in sys.path:
            sys.path.insert(0, str(HISTREE_SRC))
        from histree.gb.oblique import ObliqueGBRegressor  # type: ignore

        has_histree = True
    except Exception:
        ObliqueGBRegressor = None  # type: ignore

    if has_histree:
        # Warm up Numba JIT before timing.
        xw = np.random.default_rng(0).normal(size=(200, 4))
        yw = xw[:, 0]
        ObliqueGBRegressor(
            selector="AX",
            max_trees=2,
            early_stopping_rounds=10,
            max_leaves=31,
            min_data_in_leaf=20,
            reg_lambda=1.0,
            learning_rate=0.05,
            n_bins=255,
            parallel=False,
        ).fit(xw, yw)

    shapes = [(20_000, 16), (20_000, 50), (20_000, 100), (50_000, 50)]
    n_trees = 100

    out_lines.append("## Part A: Single-thread training speed (100 trees, median of 5 runs)\n")
    out_lines.append(
        "| shape | LightGBM ms/tree | miter ms/tree (total) | miter / LGBM | miter prep ms | miter loop ms/tree | histree AX ms/tree |"
    )
    out_lines.append("|---|---|---|---|---|---|---|")

    for n, p in shapes:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, p)).astype(np.float64)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.1 * rng.normal(size=n)).astype(np.float64)

        def run_lgb(X_in: np.ndarray = X, y_in: np.ndarray = y) -> float:
            ds = lgb.Dataset(X_in, label=y_in, params={"max_bin": 255, "verbose": -1})
            prm = {
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
            }
            lgb.train(prm, ds, num_boost_round=n_trees)
            return 0.0

        def run_miter(X_in: np.ndarray = X, y_in: np.ndarray = y) -> tuple[float, float]:
            t_prep0 = time.perf_counter()
            ds = miter.Dataset(X_in, max_bin=255)
            t_prep = time.perf_counter() - t_prep0
            prm = miter._core.Params()
            prm.n_estimators = n_trees
            prm.learning_rate = 0.05
            prm.num_leaves = 31
            prm.min_data_in_leaf = 20
            prm.min_sum_hessian_in_leaf = 1e-3
            prm.lambda_l2 = 1.0
            prm.min_gain_to_split = 0.0
            prm.max_bin = 255
            prm.early_stopping_rounds = 0
            prm.n_jobs = 1
            t_loop0 = time.perf_counter()
            miter._core.fit_axis_booster(
                ds._inner,
                miter._core.Objective.Regression,
                y_in,
                None,
                None,
                None,
                None,
                prm,
            )
            t_loop = time.perf_counter() - t_loop0
            return t_prep, t_loop

        lgb_s, _ = _median_runs(run_lgb, runs=5)
        miter_times: list[tuple[float, float, float]] = []
        for _ in range(5):
            t0 = time.perf_counter()
            tp, tl = run_miter()
            miter_times.append((time.perf_counter() - t0, tp, tl))
        miter_times.sort(key=lambda item: item[0])
        miter_s, prep_s, loop_s = miter_times[2]

        histree_str = "n/a"
        if has_histree and ObliqueGBRegressor is not None:

            def run_histree(X_in: np.ndarray = X, y_in: np.ndarray = y) -> None:
                ObliqueGBRegressor(
                    selector="AX",
                    max_trees=n_trees,
                    early_stopping_rounds=n_trees + 10,
                    max_leaves=31,
                    min_data_in_leaf=20,
                    reg_lambda=1.0,
                    min_gain_to_split=0.0,
                    learning_rate=0.05,
                    n_bins=255,
                    parallel=False,
                    random_state=0,
                ).fit(X_in, y_in)

            hs, _ = _median_runs(run_histree, runs=5)
            histree_str = f"{(hs * 1000.0 / n_trees):.3f}"

        lgb_ms = lgb_s * 1000.0 / n_trees
        miter_ms = miter_s * 1000.0 / n_trees
        ratio = miter_ms / lgb_ms
        row = (
            f"| {n} x {p} | {lgb_ms:.3f} | {miter_ms:.3f} | {ratio:.2f}x "
            f"| {prep_s * 1000.0:.1f} | {loop_s * 1000.0 / n_trees:.3f} | {histree_str} |"
        )
        print(row, flush=True)
        out_lines.append(row)


def run_openml_parity(out_lines: list[str]) -> None:
    try:
        import openml
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        out_lines.append(f"\nOpenML parity skipped (missing dependency: {exc})\n")
        return

    import lightgbm as lgb
    import miter

    has_histree = False
    try:
        if HISTREE_SRC.exists() and str(HISTREE_SRC) not in sys.path:
            sys.path.insert(0, str(HISTREE_SRC))
        from histree.gb.oblique import ObliqueGBRegressor  # type: ignore

        has_histree = True
    except Exception:
        ObliqueGBRegressor = None  # type: ignore

    tasks = [
        (336, "cpu_act", 44132, "reg"),
        (336, "elevators", 44134, "reg"),
        (336, "pol", 44133, "reg"),
        (336, "houses", 44138, "reg"),
        (336, "sulfur", 44145, "reg"),
        (337, "california", 45028, "clf"),
        (337, "electricity", 44120, "clf"),
        (337, "MagicTelescope", 44125, "clf"),
    ]

    out_lines.append("\n## Part B: OpenML fold 0 quality & speed parity (1 thread, ES=50, max_trees=3000)\n")
    out_lines.append(
        "| dataset | kind | metric | LightGBM | miter AX | histree AX | trees (L/M/H) | ms/tree (L/M) |"
    )
    out_lines.append("|---|---|---|---|---|---|---|---|")

    suites = {336: openml.study.get_suite(336), 337: openml.study.get_suite(337)}

    for suite_id, name, did, kind in tasks:
        try:
            task_obj = None
            for tid in suites[suite_id].tasks:
                t_meta = openml.tasks.get_task(
                    tid, download_data=False, download_qualities=False, download_splits=False
                )
                if t_meta.dataset_id == did:
                    task_obj = openml.tasks.get_task(
                        tid, download_data=False, download_qualities=False, download_splits=True
                    )
                    break
            if task_obj is None:
                continue
            ds_oml = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
            X_df, y_s, _, _ = ds_oml.get_data(target=task_obj.target_name, dataset_format="dataframe")
            X = X_df.to_numpy(dtype=np.float64)
            yv = (
                y_s.astype("float64").to_numpy()
                if kind == "reg"
                else y_s.astype("category").cat.codes.to_numpy(dtype=np.float64)
            )
            tr, te = task_obj.get_train_test_split_indices(repeat=0, fold=0)
            a, b = train_test_split(tr, test_size=0.2, random_state=0)

            t0 = time.perf_counter()
            prm_l = {
                "objective": "regression" if kind == "reg" else "binary",
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
            }
            m_lgb = lgb.train(
                prm_l,
                lgb.Dataset(X[a], yv[a]),
                num_boost_round=3000,
                valid_sets=[lgb.Dataset(X[b], yv[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            t_lgb = time.perf_counter() - t0
            p_lgb = m_lgb.predict(X[te], num_iteration=m_lgb.best_iteration)

            t0 = time.perf_counter()
            if kind == "reg":
                m_mit = miter.MiterRegressor(
                    n_estimators=3000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_data_in_leaf=20,
                    min_sum_hessian_in_leaf=1e-3,
                    lambda_l2=1.0,
                    min_gain_to_split=0.0,
                    max_bin=255,
                    early_stopping_rounds=50,
                    n_jobs=1,
                ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
                p_mit = m_mit.predict(X[te])
            else:
                m_mit = miter.MiterClassifier(
                    n_estimators=3000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_data_in_leaf=20,
                    min_sum_hessian_in_leaf=1e-3,
                    lambda_l2=1.0,
                    min_gain_to_split=0.0,
                    max_bin=255,
                    early_stopping_rounds=50,
                    n_jobs=1,
                ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
                p_mit = m_mit.predict_proba(X[te])[:, 1]
            t_mit = time.perf_counter() - t0

            h_metric_str = "n/a"
            h_trees_str = "-"
            if has_histree and ObliqueGBRegressor is not None:
                m_hist = ObliqueGBRegressor(
                    selector="AX",
                    objective="regression" if kind == "reg" else "binary",
                    max_trees=3000,
                    early_stopping_rounds=50,
                    max_leaves=31,
                    min_data_in_leaf=20,
                    reg_lambda=1.0,
                    min_gain_to_split=0.0,
                    learning_rate=0.05,
                    n_bins=255,
                    parallel=False,
                    random_state=0,
                ).fit(X[a], yv[a], X[b], yv[b])
                p_hist = (
                    m_hist.predict(X[te])
                    if kind == "reg"
                    else m_hist.predict_proba(X[te])
                )
                h_trees_str = str(m_hist.best_iteration_)
            else:
                p_hist = None

            if kind == "reg":
                sd = float(np.std(yv[te]))
                val_l = float(np.sqrt(np.mean((p_lgb - yv[te]) ** 2)) / sd)
                val_m = float(np.sqrt(np.mean((p_mit - yv[te]) ** 2)) / sd)
                if p_hist is not None:
                    h_metric_str = f"{float(np.sqrt(np.mean((p_hist - yv[te]) ** 2)) / sd):.4f}"
                metric_name = "nRMSE"
            else:
                eps = 1e-15
                pl = np.clip(p_lgb, eps, 1.0 - eps)
                pm = np.clip(p_mit, eps, 1.0 - eps)
                val_l = float(-np.mean(yv[te] * np.log(pl) + (1.0 - yv[te]) * np.log(1.0 - pl)))
                val_m = float(-np.mean(yv[te] * np.log(pm) + (1.0 - yv[te]) * np.log(1.0 - pm)))
                if p_hist is not None:
                    ph = np.clip(p_hist, eps, 1.0 - eps)
                    h_metric_str = (
                        f"{float(-np.mean(yv[te] * np.log(ph) + (1.0 - yv[te]) * np.log(1.0 - ph))):.4f}"
                    )
                metric_name = "log-loss"

            ms_l = (t_lgb * 1000.0) / max(len(m_lgb.dump_model()["tree_info"]), 1)
            ms_m = (t_mit * 1000.0) / max(len(m_mit.train_loss_), 1)
            row = (
                f"| {name} | {kind} | {metric_name} | {val_l:.4f} | {val_m:.4f} | {h_metric_str} | "
                f"{m_lgb.best_iteration}/{m_mit.best_iteration_}/{h_trees_str} | "
                f"{ms_l:.2f} / {ms_m:.2f} |"
            )
            print(row, flush=True)
            out_lines.append(row)
        except Exception as exc:
            out_lines.append(f"| {name} | {kind} | error | {exc!r} | - | - | - | - |")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lightgbm as lgb
    import miter

    lines = [
        "# M2 AX Booster Benchmark Summary",
        "",
        f"- Hardware: `{platform.machine()}` (`{platform.platform()}`)",
        f"- Python: `{platform.python_version()}`",
        f"- Versions: `miter {miter.__version__}`, `lightgbm {lgb.__version__}`, `numpy {np.__version__}`",
        "- Protocol: single-threaded (`n_jobs=1` / `num_threads=1`), median of 5 runs for timing.",
        "",
    ]
    run_synthetic_speed(lines)
    run_openml_parity(lines)
    out_path = Path(__file__).resolve().parent / "results" / "m2_ax_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
