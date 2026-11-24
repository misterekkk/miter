"""M3 parity re-check after the axis-binning fix: miter C++ vs histree T7 Q16r_int on 16 OpenML datasets.

Same protocol as bench_m3_oblique.py Part B (5 OpenML folds, inner train_test_split random_state=0,
random_state=fold, lr 0.05, 31 leaves, min_data_in_leaf 20, l2 1, m_mult 2, ES 50, max 5000 trees,
1 thread). histree T7 values are the cached `clean|Q16r_int` results, which a live re-run reproduced
exactly (benchmarks/results/m3_parity_brazil_pol_summary.md).

Usage: python check_m3_parity_t7_16.py run <dataset>   |   python check_m3_parity_t7_16.py agg
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_m3_oblique import HISTREE_T7_DIR, REG_DATASETS  # noqa: E402

RAW = Path(__file__).resolve().parent / "results" / "raw"
OUT = Path(__file__).resolve().parent / "results" / "m3_parity_t7_16_summary.md"
OLD = {  # Part B of m3_oblique_summary.md (pre-fix binning), miter vs T7
    "abalone": -0.12, "wine_quality": -0.80, "cpu_act": -2.87, "yprop_4_1": 0.09, "sulfur": -3.31,
    "Brazilian_houses": 13.09, "Ailerons": -0.09, "MiamiHousing2016": -0.61, "pol": 3.37,
    "elevators": -0.08, "Bike_Sharing_Demand": 0.10, "houses": 0.09, "superconduct": -0.14,
    "house_sales": -0.41, "house_16H": -1.61, "diamonds": 0.11,
}


def run(name: str) -> None:
    import miter
    import openml

    did = dict(REG_DATASETS)[name]
    suite = openml.study.get_suite(336)
    tid = next(t for t in suite.tasks if openml.tasks.get_task(
        t, download_data=False, download_qualities=False, download_splits=False).dataset_id == did)
    task = openml.tasks.get_task(tid, download_data=False, download_qualities=False, download_splits=True)
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    X = X_df.to_numpy(dtype=np.float64)
    yv = y_s.astype("float64").to_numpy()
    res = []
    for fold in range(5):
        tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        sd = float(np.std(yv[te]))
        t0 = time.perf_counter()
        m = miter.MiterRegressor(
            n_estimators=5000, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, lambda_l2=1.0,
            min_gain_to_split=0.0, max_bin=255, m_mult=2, k_proj=0, early_stopping_rounds=50,
            n_jobs=1, random_state=fold,
        ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
        secs = time.perf_counter() - t0
        p = m.predict(X[te])
        t7 = json.loads((HISTREE_T7_DIR / f"{name}_{fold}.json").read_text(encoding="utf-8"))
        res.append(dict(fold=fold, miter=float(np.sqrt(np.mean((p - yv[te]) ** 2)) / sd),
                        t7=float(t7["clean|Q16r_int"]["nrmse"]), trees=len(m.train_loss_), secs=secs))
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"m3_parity_t7_16_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(name, "done", flush=True)


def agg() -> None:
    import miter

    rows, mm, tt = [], [], []
    for name, _ in REG_DATASETS:
        r = json.loads((RAW / f"m3_parity_t7_16_{name}.json").read_text(encoding="utf-8"))
        m = float(np.mean([x["miter"] for x in r]))
        t = float(np.mean([x["t7"] for x in r]))
        mm.append(m)
        tt.append(t)
        rows.append(f"| {name} | {m:.4f} | {t:.4f} | {100 * (m - t) / t:+.2f}% | {OLD[name]:+.2f}% |")
    mm_a, tt_a = np.array(mm), np.array(tt)
    rel = 100.0 * (mm_a - tt_a) / tt_a
    _, p = wilcoxon(mm_a, tt_a)
    old = np.array(list(OLD.values()))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()
    lines = [
        "# M3 parity re-check after the axis-binning fix (miter C++ vs histree T7 Q16r_int)",
        "",
        f"- Hardware: `{platform.machine()}` (`{platform.platform()}`); miter `{miter.__version__}`; "
        f"built from `{commit}` + working-tree axis-binner fix (research quantile rule)",
        "- Protocol: as `bench_m3_oblique.py` Part B (5 folds, seed = fold, 1 thread). T7 = cached "
        "`clean|Q16r_int` (reproduced exactly by a live run, see `m3_parity_brazil_pol_summary.md`).",
        "- Mean nRMSE over 5 folds; negative = miter lower error. Last column: Part B before the fix.",
        "",
        "| dataset | miter nRMSE | histree T7 | miter vs T7 | before fix |",
        "|---|---|---|---|---|",
        *rows,
        "",
        f"- **After fix (16 reg):** mean {rel.mean():+.2f}%, **median {np.median(rel):+.2f}%**, "
        f"wins {int(np.sum(mm_a < tt_a))}/16, Wilcoxon p = {p:.4g}, max |diff| {np.abs(rel).max():.2f}%",
        f"- Before fix (Part B, same protocol): mean {old.mean():+.2f}%, median {np.median(old):+.2f}%",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg()
    else:
        run(sys.argv[2])
