"""M3 first check (§2.4 / §9): multi-seed sulfur check (5 folds x 5 seeds = 25 runs)."""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

HISTREE_SRC = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/src")


def q16r_proto(X_train: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    lo = np.quantile(X_train, 0.0001, axis=0)
    hi = np.quantile(X_train, 0.9999, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.clip(np.rint((X_all - lo) / span * 65535.0), 0.0, 65535.0)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    import lightgbm as lgb
    import miter
    import openml

    if str(HISTREE_SRC) not in sys.path:
        sys.path.insert(0, str(HISTREE_SRC))
    from histree.gb import oblique as ob  # type: ignore

    suite = openml.study.get_suite(336)
    task = None
    for tid in suite.tasks:
        t_meta = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        if t_meta.dataset_id == 44145:
            task = openml.tasks.get_task(
                tid, download_data=False, download_qualities=False, download_splits=True
            )
            break
    if task is None:
        raise RuntimeError("Could not find OpenML task for sulfur (44145)")

    ds_oml = openml.datasets.get_dataset(44145, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = ds_oml.get_data(target=task.target_name, dataset_format="dataframe")
    X = X_df.to_numpy(dtype=np.float64)
    yv = y_s.astype("float64").to_numpy()
    p = X.shape[1]
    m0 = int(math.ceil(math.sqrt(p)))

    rows: list[dict[str, float]] = []
    # Warmup histree numba JIT
    xw = X[:200]
    yw = yv[:200]
    ob.ObliqueGBRegressor(
        selector="R",
        m=2 * m0,
        bin_sample=-3,
        max_trees=2,
        early_stopping_rounds=5,
        max_leaves=31,
        min_data_in_leaf=20,
        reg_lambda=1.0,
        learning_rate=0.05,
        n_bins=255,
    ).fit(xw, yw)

    t_start = time.perf_counter()
    for fold in range(5):
        tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        sd = float(np.std(yv[te]))
        Q = q16r_proto(X[a], X)

        for s in range(5):
            seed = fold + 100 * s

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
                random_state=seed,
            ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
            e_mit = float(np.sqrt(np.mean((m_mit.predict(X[te]) - yv[te]) ** 2)) / sd)

            m_flt = ob.ObliqueGBRegressor(
                selector="R",
                m=2 * m0,
                bin_sample=-3,
                max_leaves=31,
                min_data_in_leaf=20,
                reg_lambda=1.0,
                learning_rate=0.05,
                n_bins=255,
                max_trees=5000,
                early_stopping_rounds=50,
                random_state=seed,
            ).fit(X[a], yv[a], X[b], yv[b])
            e_flt = float(np.sqrt(np.mean((m_flt.predict(X[te]) - yv[te]) ** 2)) / sd)

            m_qflt = ob.ObliqueGBRegressor(
                selector="R",
                m=2 * m0,
                bin_sample=-3,
                max_leaves=31,
                min_data_in_leaf=20,
                reg_lambda=1.0,
                learning_rate=0.05,
                n_bins=255,
                max_trees=5000,
                early_stopping_rounds=50,
                random_state=seed,
            ).fit(Q[a], yv[a], Q[b], yv[b])
            e_qflt = float(np.sqrt(np.mean((m_qflt.predict(Q[te]) - yv[te]) ** 2)) / sd)

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
                    "seed": seed,
                    "verbose": -1,
                },
                lgb.Dataset(X[a], yv[a]),
                num_boost_round=5000,
                valid_sets=[lgb.Dataset(X[b], yv[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            e_lgb = float(
                np.sqrt(np.mean((m_lgb.predict(X[te], num_iteration=m_lgb.best_iteration) - yv[te]) ** 2))
                / sd
            )

            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "miter_int": e_mit,
                    "histree_float": e_flt,
                    "histree_q16r_float": e_qflt,
                    "lgbm": e_lgb,
                }
            )
            print(
                f"fold={fold} seed={seed:3d} miter={e_mit:.4f} float={e_flt:.4f} "
                f"q16r_flt={e_qflt:.4f} lgbm={e_lgb:.4f}",
                flush=True,
            )

    mit_arr = np.array([r["miter_int"] for r in rows])
    flt_arr = np.array([r["histree_float"] for r in rows])
    qflt_arr = np.array([r["histree_q16r_float"] for r in rows])
    lgb_arr = np.array([r["lgbm"] for r in rows])

    def cmp_stats(a_arr: np.ndarray, b_arr: np.ndarray) -> tuple[float, float, int, float]:
        rel = 100.0 * (a_arr - b_arr) / b_arr  # negative = lower error = better
        wins = int(np.sum(a_arr < b_arr))
        _, pval = wilcoxon(a_arr, b_arr)
        return float(np.mean(rel)), float(np.median(rel)), wins, float(pval)

    m_flt_mean, m_flt_med, m_flt_w, m_flt_p = cmp_stats(mit_arr, flt_arr)
    m_qf_mean, m_qf_med, m_qf_w, m_qf_p = cmp_stats(mit_arr, qflt_arr)
    m_lg_mean, m_lg_med, m_lg_w, m_lg_p = cmp_stats(mit_arr, lgb_arr)

    lines = [
        "# M3 Sulfur Multi-Seed Check (OpenML 44145)",
        "",
        f"- Runs: 5 folds x 5 seeds = 25 paired runs (elapsed: {time.perf_counter() - t_start:.1f}s)",
        f"- `miter` R_uni (C++ Q16r int): mean nRMSE = {mit_arr.mean():.4f}, median = {np.median(mit_arr):.4f}, std = {mit_arr.std():.4f}",
        f"- `histree` R_uni (float): mean nRMSE = {flt_arr.mean():.4f}, median = {np.median(flt_arr):.4f}, std = {flt_arr.std():.4f}",
        f"- `histree` R_uni (Q16r float): mean nRMSE = {qflt_arr.mean():.4f}, median = {np.median(qflt_arr):.4f}, std = {qflt_arr.std():.4f}",
        f"- `LightGBM` (axis): mean nRMSE = {lgb_arr.mean():.4f}, median = {np.median(lgb_arr):.4f}, std = {lgb_arr.std():.4f}",
        "",
        "| comparison | mean rel diff (%) | median rel diff (%) | wins (x/25) | Wilcoxon p |",
        "|---|---|---|---|---|",
        f"| miter int vs histree float | {m_flt_mean:+.2f}% | {m_flt_med:+.2f}% | {m_flt_w}/25 | {m_flt_p:.4g} |",
        f"| miter int vs histree Q16r float | {m_qf_mean:+.2f}% | {m_qf_med:+.2f}% | {m_qf_w}/25 | {m_qf_p:.4g} |",
        f"| miter int vs LightGBM | {m_lg_mean:+.2f}% | {m_lg_med:+.2f}% | {m_lg_w}/25 | {m_lg_p:.4g} |",
        "",
        "(Negative relative diff = lower error than reference.)",
    ]
    out_path = Path(__file__).resolve().parent / "results" / "m3_sulfur_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
