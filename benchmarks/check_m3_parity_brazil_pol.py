"""M3 parity check: Brazilian_houses and pol, miter C++ vs LIVE histree T7 Q16r_int.

Same outer folds (OpenML task splits), same inner split (train_test_split random_state=0),
same seeds for both methods. 5 folds x 5 seeds (seed = fold + 100*s) per dataset.

Usage:
  run:       python check_m3_parity_brazil_pol.py run <dataset> <miter|histree> [n_seeds]
  aggregate: python check_m3_parity_brazil_pol.py agg
Raw per-run JSON goes to benchmarks/results/raw/m3_parity_<dataset>_<method>.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

HISTREE_SRC = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/src")
T7_DIR = Path("/Users/kacperpiotrowski/Studia/Projekt/histree/research/results/t7")
RAW = Path(__file__).resolve().parent / "results" / "raw"
DATASETS = {"Brazilian_houses": 44141, "pol": 44133}
CFG = dict(max_leaves=31, min_data_in_leaf=20, reg_lambda=1.0, learning_rate=0.05,
           n_bins=255, max_trees=5000, early_stopping_rounds=50)


def load(did: int):
    import openml

    suite = openml.study.get_suite(336)
    tid = None
    for t in suite.tasks:
        tm = openml.tasks.get_task(t, download_data=False, download_qualities=False, download_splits=False)
        if tm.dataset_id == did:
            tid = t
            break
    task = openml.tasks.get_task(tid, download_data=False, download_qualities=False, download_splits=True)
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    return task, X_df.to_numpy(dtype=np.float64), y_s.astype("float64").to_numpy()


def q16r_proto(XA: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Exactly t7_int.py::q16r."""
    lo = np.quantile(XA, 0.0001, axis=0)
    hi = np.quantile(XA, 0.9999, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.clip(np.rint((X_all - lo) / span * 65535), 0, 65535)


def run(name: str, method: str, n_seeds: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    task, X, yv = load(DATASETS[name])
    p = X.shape[1]
    m0 = int(math.ceil(math.sqrt(p)))
    if method == "histree":
        sys.path.insert(0, str(HISTREE_SRC))
        from histree.gb.oblique_int import ObliqueGBInt  # type: ignore
        ObliqueGBInt(selector="R", m=2 * m0, bin_sample=-3, random_state=0,
                     **{**CFG, "max_trees": 2}).fit(
            q16r_proto(X[:300], X[:300]), yv[:300], q16r_proto(X[:300], X[:300]), yv[:300])
    else:
        import miter
    rows = []
    t_start = time.perf_counter()
    for fold in range(5):
        tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        sd = float(np.std(yv[te]))
        Q = q16r_proto(X[a], X) if method == "histree" else None
        for s in range(n_seeds):
            seed = fold + 100 * s
            t0 = time.perf_counter()
            if method == "miter":
                m = miter.MiterRegressor(
                    n_estimators=5000, learning_rate=0.05, num_leaves=31, min_data_in_leaf=20,
                    lambda_l2=1.0, min_gain_to_split=0.0, max_bin=255, m_mult=2, k_proj=0,
                    early_stopping_rounds=50, n_jobs=1, random_state=seed,
                ).fit(X[a], yv[a], eval_set=[(X[b], yv[b])])
                pred = m.predict(X[te])
                trees = len(m.train_loss_)
            else:
                m = ObliqueGBInt(selector="R", m=2 * m0, bin_sample=-3, random_state=seed, **CFG
                                 ).fit(Q[a], yv[a], Q[b], yv[b])
                pred = m.predict(Q[te])
                trees = int(m.best_iteration_)
            e = float(np.sqrt(np.mean((pred - yv[te]) ** 2)) / sd)
            secs = time.perf_counter() - t0
            rows.append(dict(fold=fold, seed=seed, nrmse=e, trees=trees, secs=secs))
            print(f"{name} {method} fold={fold} seed={seed} nrmse={e:.6f} trees={trees} {secs:.1f}s", flush=True)
    out = dict(dataset=name, method=method, elapsed=time.perf_counter() - t_start, rows=rows)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"m3_parity_{name}_{method}.json").write_text(json.dumps(out, indent=1))


def agg() -> None:
    from scipy.stats import wilcoxon
    import platform
    import subprocess

    def ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "?"

    out = ["# M3 parity check: Brazilian_houses and pol, miter C++ vs live histree T7 Q16r_int", ""]
    out.append("## Provenance")
    out.append(f"- hardware/OS: {platform.platform()} / {platform.processor()}, python {platform.python_version()}")
    out.append(f"- numpy {ver('numpy')}, scipy {ver('scipy')}, sklearn {ver('sklearn')}, numba {ver('numba')}, miter {ver('miter')}")
    out.append("- miter source commit: d1b4d12 (branch m5-performance); histree: " +
               subprocess.run(["git", "-C", str(HISTREE_SRC.parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip())
    out.append("- seeds: seed = fold + 100*s, s=0..4; both methods get the same random_state, same outer folds, inner split train_test_split(random_state=0)")
    out.append("- histree gets Q16r codes computed as in t7_int.py (np.quantile 1e-4/0.9999 of inner-train part, rint, clip); miter quantizes internally from raw X")
    out.append("")
    for name in DATASETS:
        pm = RAW / f"m3_parity_{name}_miter.json"
        ph = RAW / f"m3_parity_{name}_histree.json"
        if not (pm.exists() and ph.exists()):
            continue
        dm, dh = json.loads(pm.read_text()), json.loads(ph.read_text())
        em = np.array([r["nrmse"] for r in dm["rows"]])
        eh = np.array([r["nrmse"] for r in dh["rows"]])
        folds = np.array([r["fold"] for r in dm["rows"]])
        seeds = np.array([r["seed"] for r in dm["rows"]])
        assert (folds == np.array([r["fold"] for r in dh["rows"]])).all()
        assert (seeds == np.array([r["seed"] for r in dh["rows"]])).all()
        rel = 100 * (em - eh) / eh
        _, p = wilcoxon(em, eh)
        out.append(f"## {name}")
        out.append(f"- miter elapsed {dm['elapsed']:.0f}s, histree elapsed {dh['elapsed']:.0f}s (under load, approximate)")
        out.append("")
        out.append("| fold | seed | miter nRMSE | histree nRMSE | rel diff % | miter trees | histree trees |")
        out.append("|---|---|---|---|---|---|---|")
        for i, (rm, rh) in enumerate(zip(dm["rows"], dh["rows"])):
            out.append(f"| {rm['fold']} | {rm['seed']} | {rm['nrmse']:.6f} | {rh['nrmse']:.6f} | {rel[i]:+.2f} | {rm['trees']} | {rh['trees']} |")
        out.append("")
        out.append("| stat | miter | histree |")
        out.append("|---|---|---|")
        out.append(f"| mean | {em.mean():.6f} | {eh.mean():.6f} |")
        out.append(f"| median | {np.median(em):.6f} | {np.median(eh):.6f} |")
        # seed std: per fold, std over seeds (ddof=1), averaged over folds
        sm = np.mean([np.std(em[folds == f], ddof=1) for f in range(5)])
        sh = np.mean([np.std(eh[folds == f], ddof=1) for f in range(5)])
        out.append(f"| seed std (ddof=1, within fold, mean over folds) | {sm:.6f} | {sh:.6f} |")
        out.append("")
        out.append("| per-fold mean over seeds | miter | histree | rel diff % |")
        out.append("|---|---|---|---|")
        for f in range(5):
            a_, b_ = em[folds == f].mean(), eh[folds == f].mean()
            out.append(f"| {f} | {a_:.6f} | {b_:.6f} | {100*(a_-b_)/b_:+.2f} |")
        out.append("")
        out.append(f"- paired rel diff: mean {rel.mean():+.2f}%, median {np.median(rel):+.2f}%; mean-of-nRMSE ratio {100*(em.mean()-eh.mean())/eh.mean():+.2f}%")
        out.append(f"- miter wins (lower nRMSE): {int((em < eh).sum())}/{len(em)}; Wilcoxon p = {p:.4g}")
        out.append(f"- seed-to-seed std of rel diff: {rel.std(ddof=1):.2f} pct points")
        # cached T7 reproduction (seed index 0 -> seed == fold)
        out.append("")
        out.append("| fold | cached T7 clean|Q16r_int | live histree (seed=fold) | abs diff |")
        out.append("|---|---|---|---|")
        for f in range(5):
            cp = T7_DIR / f"{name}_{f}.json"
            if cp.exists():
                c = json.loads(cp.read_text())["clean|Q16r_int"]["nrmse"]
                l = eh[(folds == f) & (seeds == f)][0]
                out.append(f"| {f} | {c:.6f} | {l:.6f} | {abs(c-l):.3g} |")
        out.append("")
    (Path(__file__).resolve().parent / "results" / "m3_parity_brazil_pol_summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "agg":
        agg()
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 5)
