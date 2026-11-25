"""Pre-registered benchmark for Oblivious + Oblique (grow_policy="symmetric", S6) vs Leafwise.

Frozen pre-registration: benchmarks/results/exp_symmetric_oblique_prereg.md
Usage:
    python bench_exp_symmetric_oblique.py pilot
    python bench_exp_symmetric_oblique.py speed
    python bench_exp_symmetric_oblique.py run --shard 0 --nshards 3 --folds 5
    python bench_exp_symmetric_oblique.py diag --folds 5
    python bench_exp_symmetric_oblique.py report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_m3_oblique import CLF_DATASETS, REG_DATASETS  # noqa: E402

PREREG_PATH = HERE / "results" / "exp_symmetric_oblique_prereg.md"
SUMMARY_PATH = HERE / "results" / "exp_symmetric_oblique_summary.md"
RAW_DIR = HERE / "results" / "raw" / "exp_sym"

VARIANTS = ["B", "S6", "B+S6", "S6_l1", "CAT", "LGB"]
FIELDS = ["kind", "dataset", "n_fit", "fold", "variant", "metric", "trees", "secs"]

DIAG_TASKS = [
    ("concrete_compressive_strength", 363625),
    ("miami_housing", 363686),
    ("airfoil_self_noise", 363612),
]


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def load_tasks(kind: str) -> dict[int, int]:
    suite = openml.study.get_suite(336 if kind == "reg" else 337)
    task_map: dict[int, int] = {}
    for tid in suite.tasks:
        t = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        task_map[t.dataset_id] = tid
    return task_map


def load_openml_dataset(kind: str, did: int, task_map: dict[int, int]) -> tuple[Any, np.ndarray, np.ndarray]:
    task = openml.tasks.get_task(
        task_map[did], download_data=False, download_qualities=False, download_splits=True
    )
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = d.get_data(target=task.target_name, dataset_format="dataframe")
    X = X_df.to_numpy(dtype=np.float64)
    if kind == "reg":
        y = y_s.astype("float64").to_numpy()
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.float64)
    return task, X, y


def eval_metric(kind: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    if kind == "reg":
        return float(np.sqrt(np.mean((pred - y_true) ** 2)) / np.std(y_true))
    p = np.clip(pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_predict_variant(
    kind: str,
    variant: str,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    if variant in ("B", "S6", "S6_l1"):
        import miter

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
        if variant == "B":
            kw.update(
                {
                    "grow_policy": "leafwise",
                    "num_leaves": 31,
                    "min_data_in_leaf": 20,
                    "lambda_l2": 1.0,
                }
            )
        else:
            kw.update(
                {
                    "grow_policy": "symmetric",
                    "max_depth": 6,
                    "min_data_in_leaf": 1,
                    "lambda_l2": 3.0 if variant == "S6" else 1.0,
                }
            )
        cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
        m = cls(**kw).fit(X[a], y[a], eval_set=[(X[b], y[b])])
        pred = m.predict(X[te]) if kind == "reg" else m.predict_proba(X[te])[:, 1]
        trees = int(m.best_iteration_)
    elif variant == "CAT":
        from catboost import CatBoostClassifier, CatBoostRegressor

        common = {
            "iterations": 5000,
            "early_stopping_rounds": 50,
            "depth": 6,
            "thread_count": 1,
            "random_seed": fold,
            "verbose": 0,
        }
        if kind == "reg":
            m_cat = CatBoostRegressor(**common)
            m_cat.fit(X[a], y[a], eval_set=(X[b], y[b]))
            pred = np.asarray(m_cat.predict(X[te]), dtype=np.float64)
        else:
            m_cat = CatBoostClassifier(**common)
            m_cat.fit(X[a], y[a], eval_set=(X[b], y[b]))
            pred = np.asarray(m_cat.predict_proba(X[te])[:, 1], dtype=np.float64)
        trees = int(m_cat.get_best_iteration() + 1)
    elif variant == "LGB":
        import lightgbm as lgb

        common_lgb = {
            "n_estimators": 5000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "max_bin": 255,
            "n_jobs": 1,
            "random_state": fold,
            "deterministic": True,
            "verbose": -1,
        }
        if kind == "reg":
            m_lgb = lgb.LGBMRegressor(**common_lgb)
            m_lgb.fit(
                X[a],
                y[a],
                eval_set=[(X[b], y[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred = np.asarray(m_lgb.predict(X[te]), dtype=np.float64)
        else:
            m_lgb = lgb.LGBMClassifier(**common_lgb)
            m_lgb.fit(
                X[a],
                y[a],
                eval_set=[(X[b], y[b])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred = np.asarray(m_lgb.predict_proba(X[te])[:, 1], dtype=np.float64)
        trees = int(m_lgb.best_iteration_)
    else:
        raise ValueError(f"Unknown variant {variant}")
    return pred, trees, time.perf_counter() - t0


def run_fold_group(
    kind: str,
    name: str,
    task: Any,
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    done: set[tuple[str, str, int, str]],
    writer: csv.DictWriter,
    fh: Any,
) -> None:
    for fold in range(folds):
        tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        n_fit = int(len(a))

        preds_cache: dict[str, tuple[np.ndarray, int, float]] = {}
        need_blend = (kind, name, fold, "B+S6") not in done
        for v in ("B", "S6", "S6_l1", "CAT", "LGB"):
            need_this = (kind, name, fold, v) not in done or (need_blend and v in ("B", "S6"))
            if not need_this:
                continue
            pred, trees, secs = fit_predict_variant(kind, v, X, y, a, b, te, fold)
            preds_cache[v] = (pred, trees, secs)
            if (kind, name, fold, v) not in done:
                metric = eval_metric(kind, y[te], pred)
                writer.writerow(
                    {
                        "kind": kind,
                        "dataset": name,
                        "n_fit": n_fit,
                        "fold": fold,
                        "variant": v,
                        "metric": repr(metric),
                        "trees": trees,
                        "secs": f"{secs:.3f}",
                    }
                )
                done.add((kind, name, fold, v))
                print(
                    f"{kind:4s} {name:30s} f={fold} {v:5s}: {metric:.5f} trees={trees:4d} {secs:6.2f}s",
                    flush=True,
                )

        if need_blend and "B" in preds_cache and "S6" in preds_cache:
            p_b, tb, sb = preds_cache["B"]
            p_s, ts, ss = preds_cache["S6"]
            p_blend = 0.5 * (p_b + p_s)
            m_blend = eval_metric(kind, y[te], p_blend)
            writer.writerow(
                {
                    "kind": kind,
                    "dataset": name,
                    "n_fit": n_fit,
                    "fold": fold,
                    "variant": "B+S6",
                    "metric": repr(m_blend),
                    "trees": tb + ts,
                    "secs": f"{sb + ss:.3f}",
                }
            )
            done.add((kind, name, fold, "B+S6"))
            print(
                f"{kind:4s} {name:30s} f={fold} B+S6 : {m_blend:.5f} trees={tb + ts:4d} {sb + ss:6.2f}s",
                flush=True,
            )
        fh.flush()


def cmd_pilot() -> None:
    print("=== PILOT ===", flush=True)
    for kind, name, did in [("reg", "abalone", 45033), ("clf", "eye_movements", 44130)]:
        tmap = load_tasks(kind)
        task, X, y = load_openml_dataset(kind, did, tmap)
        tr, te = task.get_train_test_split_indices(repeat=0, fold=0)
        a, b = train_test_split(tr, test_size=0.2, random_state=0)
        p_b, tb, sb = fit_predict_variant(kind, "B", X, y, a, b, te, 0)
        p_s, ts, ss = fit_predict_variant(kind, "S6", X, y, a, b, te, 0)
        _, _, sc = fit_predict_variant(kind, "CAT", X, y, a, b, te, 0)
        mb = eval_metric(kind, y[te], p_b)
        ms = eval_metric(kind, y[te], p_s)
        mbl = eval_metric(kind, y[te], 0.5 * (p_b + p_s))
        print(
            f"{kind} {name} X={X.shape}: B={mb:.5f} ({tb}t, {sb:.2f}s) "
            f"S6={ms:.5f} ({ts}t, {ss:.2f}s) B+S6={mbl:.5f} CAT={sc:.2f}s",
            flush=True,
        )

    import miter

    rng = np.random.default_rng(42)
    X5 = rng.normal(size=(500_000, 50)).astype(np.float64)
    y5 = (X5[:, 0] + 0.5 * X5[:, 1] - 0.3 * X5[:, 2] + 0.1 * rng.normal(size=500_000)).astype(np.float64)
    ds5 = miter.Dataset(X5, max_bin=255, n_jobs=4)
    for nj in (1, 4):
        for label, gp, md, mdl, l2 in [
            ("B", miter._core.GrowPolicy.Leafwise, -1, 20, 1.0),
            ("S6", miter._core.GrowPolicy.Symmetric, 6, 1, 3.0),
        ]:
            prm = miter._core.Params()
            prm.n_estimators = 15
            prm.grow_policy = gp
            prm.max_depth = md
            prm.min_data_in_leaf = mdl
            prm.lambda_l2 = l2
            prm.early_stopping_rounds = 0
            prm.n_jobs = nj
            miter._core.fit_booster(ds5._inner, miter._core.Objective.Regression, y5, None, None, None, None, prm)
            t0 = time.perf_counter()
            miter._core.fit_booster(ds5._inner, miter._core.Objective.Regression, y5, None, None, None, None, prm)
            print(f"500kx50 T={nj} {label}: {(time.perf_counter()-t0)*1000/15:.2f} ms/tree", flush=True)


def _median_runs(fn: Any, runs: int = 5) -> float:
    fn()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(statistics.median(times))


def _measure_peak_rss_mb(code: str) -> float:
    cmd = [
        sys.executable,
        "-c",
        (
            "import resource, sys\n"
            + code
            + "\nru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
            "print(ru / (1024*1024) if sys.platform == 'darwin' else ru / 1024)\n"
        ),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip().splitlines()[-1])


def cmd_speed() -> None:
    import lightgbm as lgb
    import miter
    from catboost import CatBoostRegressor

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    speed_out: dict[str, Any] = {"ms_per_tree": {}, "es_500k": {}, "rss_mb": {}}
    n_trees = 25

    for n in (20_000, 100_000, 500_000):
        p = 50
        key = f"{n}x{p}"
        rng = np.random.default_rng(42)
        X = rng.normal(size=(n, p)).astype(np.float64)
        y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.1 * rng.normal(size=n)).astype(np.float64)

        ds_mit = miter.Dataset(X, max_bin=255, n_jobs=4)
        ds_lgb = lgb.Dataset(X, label=y, params={"max_bin": 255, "verbose": -1})
        ds_lgb.construct()

        speed_out["ms_per_tree"][key] = {}
        for th in (1, 4):
            def run_lgb(tc: int = th, dsl: Any = ds_lgb) -> None:
                lgb.train(
                    {
                        "objective": "regression",
                        "learning_rate": 0.05,
                        "num_leaves": 31,
                        "min_data_in_leaf": 20,
                        "lambda_l2": 1.0,
                        "max_bin": 255,
                        "num_threads": tc,
                        "deterministic": True,
                        "seed": 0,
                        "verbose": -1,
                    },
                    dsl,
                    num_boost_round=n_trees,
                )

            def run_cat(tc: int = th, x_cur: np.ndarray = X, y_cur: np.ndarray = y) -> None:
                CatBoostRegressor(
                    iterations=n_trees,
                    learning_rate=0.05,
                    depth=6,
                    thread_count=tc,
                    random_seed=0,
                    verbose=0,
                ).fit(x_cur, y_cur)

            def run_miter(
                gp: Any,
                md: int,
                mdl: int,
                l2: float,
                tc: int = th,
                dsm: Any = ds_mit,
                y_cur: np.ndarray = y,
            ) -> None:
                prm = miter._core.Params()
                prm.n_estimators = n_trees
                prm.learning_rate = 0.05
                prm.grow_policy = gp
                prm.max_depth = md
                prm.min_data_in_leaf = mdl
                prm.lambda_l2 = l2
                prm.m_mult = 2
                prm.early_stopping_rounds = 0
                prm.n_jobs = tc
                miter._core.fit_booster(
                    dsm._inner, miter._core.Objective.Regression, y_cur, None, None, None, None, prm
                )

            ms_b = _median_runs(lambda tc=th: run_miter(miter._core.GrowPolicy.Leafwise, -1, 20, 1.0, tc), runs=5) * 1000.0 / n_trees
            ms_s6 = _median_runs(lambda tc=th: run_miter(miter._core.GrowPolicy.Symmetric, 6, 1, 3.0, tc), runs=5) * 1000.0 / n_trees
            ms_lgb = _median_runs(run_lgb, runs=5) * 1000.0 / n_trees
            ms_cat = _median_runs(run_cat, runs=5) * 1000.0 / n_trees
            speed_out["ms_per_tree"][key][f"T{th}"] = {
                "B": ms_b,
                "S6": ms_s6,
                "LGB": ms_lgb,
                "CAT": ms_cat,
                "speedup_B_over_S6": ms_b / ms_s6,
            }
            print(
                f"{key:10s} T={th} | B={ms_b:6.2f} S6={ms_s6:6.2f} ({ms_b/ms_s6:.2f}x) LGB={ms_lgb:6.2f} CAT={ms_cat:6.2f}",
                flush=True,
            )

    # Full fit to early stopping on synthetic 500k x 50 regression (T=4 and T=1)
    rng = np.random.default_rng(77)
    X = rng.normal(size=(500_000, 50)).astype(np.float64)
    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4] + X[:, 5])
        + 0.25 * rng.normal(size=500_000)
    ).astype(np.float64)
    X_tr, X_val, y_tr, y_val = X[:400_000], X[400_000:450_000], y[:400_000], y[400_000:450_000]
    X_te, y_te = X[450_000:], y[450_000:]

    for nj in (1, 4):
        es_entry: dict[str, Any] = {}
        for label, gp, md, mdl, l2 in [
            ("B", "leafwise", -1, 20, 1.0),
            ("S6", "symmetric", 6, 1, 3.0),
        ]:
            t0 = time.perf_counter()
            m = miter.MiterRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                grow_policy=gp,
                num_leaves=31,
                max_depth=md,
                min_data_in_leaf=mdl,
                lambda_l2=l2,
                early_stopping_rounds=50,
                n_jobs=nj,
                random_state=0,
            ).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            secs = time.perf_counter() - t0
            nrmse = float(np.sqrt(np.mean((m.predict(X_te) - y_te) ** 2)) / np.std(y_te))
            es_entry[label] = {"secs": secs, "trees": int(m.best_iteration_), "nrmse": nrmse}

        t0 = time.perf_counter()
        ml = lgb.LGBMRegressor(
            n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
            reg_lambda=1.0, n_jobs=nj, random_state=0, deterministic=True, verbose=-1
        ).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        secs_l = time.perf_counter() - t0
        nrmse_l = float(np.sqrt(np.mean((ml.predict(X_te) - y_te) ** 2)) / np.std(y_te))
        es_entry["LGB"] = {"secs": secs_l, "trees": int(ml.best_iteration_), "nrmse": nrmse_l}

        t0 = time.perf_counter()
        mc = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6, thread_count=nj, random_seed=0, verbose=0
        ).fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        secs_c = time.perf_counter() - t0
        nrmse_c = float(np.sqrt(np.mean((mc.predict(X_te) - y_te) ** 2)) / np.std(y_te))
        es_entry["CAT"] = {"secs": secs_c, "trees": int(mc.get_best_iteration() + 1), "nrmse": nrmse_c}

        speed_out["es_500k"][f"T{nj}"] = es_entry
        print(f"ES 500k T={nj}: " + " | ".join(f"{k}={v['secs']:.2f}s({v['trees']}t,{v['nrmse']:.4f})" for k, v in es_entry.items()), flush=True)

    # Peak RSS on 500k x 50
    speed_out["rss_mb"]["B"] = _measure_peak_rss_mb(
        "import numpy as np, miter; rng=np.random.default_rng(42); "
        "X=rng.normal(size=(500000,50)); y=X[:,0]; "
        "miter.MiterRegressor(n_estimators=25,early_stopping_rounds=None,n_jobs=1).fit(X,y)"
    )
    speed_out["rss_mb"]["S6"] = _measure_peak_rss_mb(
        "import numpy as np, miter; rng=np.random.default_rng(42); "
        "X=rng.normal(size=(500000,50)); y=X[:,0]; "
        "miter.MiterRegressor(grow_policy='symmetric',max_depth=6,min_data_in_leaf=1,lambda_l2=3.0,"
        "n_estimators=25,early_stopping_rounds=None,n_jobs=1).fit(X,y)"
    )

    (RAW_DIR / "speed.json").write_text(json.dumps(speed_out, indent=2), encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"raw_shard{args.shard}.csv"
    done: set[tuple[str, str, int, str]] = set()
    for f in RAW_DIR.glob("raw_shard*.csv"):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["kind"], r["dataset"], int(r["fold"]), r["variant"]))

    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        writer.writeheader()

    units = [(kind, name, did) for kind, lst in (("reg", REG_DATASETS), ("clf", CLF_DATASETS)) for name, did in lst]
    units = [u for i, u in enumerate(units) if i % args.nshards == args.shard]
    maps: dict[str, dict[int, int]] = {}
    for kind, name, did in units:
        if kind not in maps:
            maps[kind] = load_tasks(kind)
        task, X, y = load_openml_dataset(kind, did, maps[kind])
        print(f"== {kind} {name} X={X.shape}", flush=True)
        run_fold_group(kind, name, task, X, y, args.folds, done, writer, fh)
    fh.close()


def cmd_diag(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "diag.csv"
    done: set[tuple[str, str, int, str]] = set()
    if out_path.exists():
        with out_path.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["kind"], r["dataset"], int(r["fold"]), r["variant"]))
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        writer.writeheader()

    for name, tid in DIAG_TASKS:
        task = openml.tasks.get_task(tid, download_data=True, download_qualities=False, download_splits=True)
        d = task.get_dataset()
        X_df, y_s, _, _ = d.get_data(target=task.target_name, dataset_format="dataframe")
        X = X_df.to_numpy(dtype=np.float64)
        y = y_s.astype("float64").to_numpy()
        print(f"== diag {name} X={X.shape}", flush=True)
        for rep in range(2):
            for fold in range(3):
                split_id = rep * 3 + fold
                try:
                    tr, te = task.get_train_test_split_indices(repeat=rep, fold=fold)
                except ValueError:
                    continue
                a, b = train_test_split(tr, test_size=0.2, random_state=0)
                n_fit = int(len(a))
                preds_cache: dict[str, tuple[np.ndarray, int, float]] = {}
                need_blend = ("reg", name, split_id, "B+S6") not in done
                for v in ("B", "S6", "S6_l1", "CAT", "LGB"):
                    need_this = ("reg", name, split_id, v) not in done or (need_blend and v in ("B", "S6"))
                    if not need_this:
                        continue
                    pred, trees, secs = fit_predict_variant("reg", v, X, y, a, b, te, split_id)
                    preds_cache[v] = (pred, trees, secs)
                    if ("reg", name, split_id, v) not in done:
                        metric = eval_metric("reg", y[te], pred)
                        writer.writerow(
                            {
                                "kind": "reg",
                                "dataset": name,
                                "n_fit": n_fit,
                                "fold": split_id,
                                "variant": v,
                                "metric": repr(metric),
                                "trees": trees,
                                "secs": f"{secs:.3f}",
                            }
                        )
                        done.add(("reg", name, split_id, v))
                if need_blend and "B" in preds_cache and "S6" in preds_cache:
                    p_b, tb, sb = preds_cache["B"]
                    p_s, ts, ss = preds_cache["S6"]
                    m_blend = eval_metric("reg", y[te], 0.5 * (p_b + p_s))
                    writer.writerow(
                        {
                            "kind": "reg",
                            "dataset": name,
                            "n_fit": n_fit,
                            "fold": split_id,
                            "variant": "B+S6",
                            "metric": repr(m_blend),
                            "trees": tb + ts,
                            "secs": f"{sb + ss:.3f}",
                        }
                    )
                    done.add(("reg", name, split_id, "B+S6"))
                fh.flush()
                print(
                    f"diag {name:30s} s={split_id}: B={eval_metric('reg', y[te], preds_cache['B'][0]):.5f} "
                    f"S6={eval_metric('reg', y[te], preds_cache['S6'][0]):.5f} "
                    f"CAT={eval_metric('reg', y[te], preds_cache['CAT'][0]):.5f}"
                    if "B" in preds_cache
                    else f"diag {name:30s} s={split_id} cached",
                    flush=True,
                )
    fh.close()


def _stats_vs_b(piv: Any, col: str) -> tuple[float, float, int, int, float, str, float]:
    diff = 100.0 * (piv[col] - piv["B"]) / piv["B"]
    mean_d = float(diff.mean())
    med_d = float(diff.median())
    wins = int((diff < 0.0).sum())
    total = int(len(diff))
    _, p_val = wilcoxon(piv[col].to_numpy(), piv["B"].to_numpy())
    worst_ds = str(diff.idxmax())
    worst_val = float(diff.max())
    return mean_d, med_d, wins, total, float(p_val), worst_ds, worst_val


def cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    frames = [pd.read_csv(f, dtype={"metric": str}) for f in sorted(RAW_DIR.glob("raw_shard*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["metric_f"] = df["metric"].astype(float)
    df = df.drop_duplicates(subset=["kind", "dataset", "fold", "variant"])

    lines: list[str] = []
    sha = prereg_sha256()
    lines.append("# Experimental Report: Oblivious + Oblique (`grow_policy=\"symmetric\"`, S6)\n")
    lines.append("- Pre-registration: [`exp_symmetric_oblique_prereg.md`](exp_symmetric_oblique_prereg.md)")
    lines.append(f"- Pre-registration SHA-256: `{sha}`")
    lines.append("- Hardware / OS: Apple M4, macOS, pinned single-threaded quality runs (`n_jobs=1`), 5 folds\n")

    reg_piv = df[df["kind"] == "reg"].groupby(["dataset", "variant"])["metric_f"].mean().unstack()
    clf_piv = df[df["kind"] == "clf"].groupby(["dataset", "variant"])["metric_f"].mean().unstack()

    speed_data = json.loads((RAW_DIR / "speed.json").read_text(encoding="utf-8"))

    # Verdict calculation per frozen acceptance criteria
    r_s6 = _stats_vs_b(reg_piv, "S6")
    c_s6 = _stats_vs_b(clf_piv, "S6")
    r_bl = _stats_vs_b(reg_piv, "B+S6")
    c_bl = _stats_vs_b(clf_piv, "B+S6")

    a1_reg = (r_s6[1] < -0.5) and (r_s6[4] < 0.05) and (r_s6[6] <= 3.0)
    a1_clf = (c_s6[1] < -0.5) and (c_s6[4] < 0.05) and (c_s6[6] <= 3.0)

    sp_t1 = speed_data["ms_per_tree"]["500000x50"]["T1"]["speedup_B_over_S6"]
    sp_t4 = speed_data["ms_per_tree"]["500000x50"]["T4"]["speedup_B_over_S6"]
    es_t1_ok = speed_data["es_500k"]["T1"]["S6"]["secs"] < speed_data["es_500k"]["T1"]["B"]["secs"]
    es_t4_ok = speed_data["es_500k"]["T4"]["S6"]["secs"] < speed_data["es_500k"]["T4"]["B"]["secs"]
    a2_pass = (r_s6[1] <= 0.5) and (c_s6[1] <= 0.5) and (sp_t1 >= 1.5) and (sp_t4 >= 1.5) and es_t1_ok and es_t4_ok

    b_reg = (r_bl[1] < -0.5) and (r_bl[2] >= 12) and (r_bl[4] < 0.05) and (r_bl[6] <= 1.0)
    b_clf = (c_bl[1] < -0.5) and (c_bl[2] >= 10) and (c_bl[4] < 0.05) and (c_bl[6] <= 1.0)

    lines.append("## 1. Pre-registered Verdict\n")
    lines.append("| Scenario | Task | median % vs B | mean % | wins | Wilcoxon p | worst dataset | Verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    lines.append(f"| A1: S6 solo quality | Reg (16) | {r_s6[1]:+.2f}% | {r_s6[0]:+.2f}% | {r_s6[2]}/{r_s6[3]} | {r_s6[4]:.4g} | {r_s6[5]} ({r_s6[6]:+.2f}%) | **{'PASS' if a1_reg else 'FAIL'}** |")
    lines.append(f"| A1: S6 solo quality | Clf (13) | {c_s6[1]:+.2f}% | {c_s6[0]:+.2f}% | {c_s6[2]}/{c_s6[3]} | {c_s6[4]:.4g} | {c_s6[5]} ({c_s6[6]:+.2f}%) | **{'PASS' if a1_clf else 'FAIL'}** |")
    lines.append(
        f"| A2: S6 speed mode | 500k + Quality | reg {r_s6[1]:+.2f}% / clf {c_s6[1]:+.2f}% | — | — | — | "
        f"500k T1={sp_t1:.2f}x, T4={sp_t4:.2f}x | **{'PASS' if a2_pass else 'FAIL'}** |"
    )
    lines.append(f"| B: B+S6 blend (50/50) | Reg (16) | {r_bl[1]:+.2f}% | {r_bl[0]:+.2f}% | {r_bl[2]}/{r_bl[3]} | {r_bl[4]:.4g} | {r_bl[5]} ({r_bl[6]:+.2f}%) | **{'PASS' if b_reg else 'FAIL'}** |")
    lines.append(f"| B: B+S6 blend (50/50) | Clf (13) | {c_bl[1]:+.2f}% | {c_bl[0]:+.2f}% | {c_bl[2]}/{c_bl[3]} | {c_bl[4]:.4g} | {c_bl[5]} ({c_bl[6]:+.2f}%) | **{'PASS' if b_clf else 'FAIL'}** |\n")

    lines.append("## 2. Exploratory Reference Summary vs B\n")
    lines.append("| Variant | Task | median % vs B | mean % vs B | wins | Wilcoxon p | worst dataset |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for kind_lbl, piv in (("Reg (16)", reg_piv), ("Clf (13)", clf_piv)):
        for col in ("S6_l1", "CAT", "LGB"):
            s = _stats_vs_b(piv, col)
            lines.append(f"| {col} | {kind_lbl} | {s[1]:+.2f}% | {s[0]:+.2f}% | {s[2]}/{s[3]} | {s[4]:.4g} | {s[5]} ({s[6]:+.2f}%) |")
    lines.append("")

    lines.append("## 3. Speed & Early Stopping & Memory\n")
    lines.append("| shape | T | B (ms/t) | S6 (ms/t) | Speedup B/S6 | LGB (ms/t) | CAT (ms/t) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, thm in speed_data["ms_per_tree"].items():
        for th, v in thm.items():
            lines.append(f"| `{k}` | {th} | {v['B']:.2f} | {v['S6']:.2f} | **{v['speedup_B_over_S6']:.2f}x** | {v['LGB']:.2f} | {v['CAT']:.2f} |")
    lines.append("")
    lines.append("| 500k ES Fit | B | S6 | LGB | CAT |")
    lines.append("|---|---|---|---|---|")
    for th, v in speed_data["es_500k"].items():
        lines.append(
            f"| {th} | {v['B']['secs']:.2f}s ({v['B']['trees']}t, {v['B']['nrmse']:.4f}) | "
            f"{v['S6']['secs']:.2f}s ({v['S6']['trees']}t, {v['S6']['nrmse']:.4f}) | "
            f"{v['LGB']['secs']:.2f}s ({v['LGB']['trees']}t, {v['LGB']['nrmse']:.4f}) | "
            f"{v['CAT']['secs']:.2f}s ({v['CAT']['trees']}t, {v['CAT']['nrmse']:.4f}) |"
        )
    lines.append(
        f"\n- Peak RSS (`500k x 50`, 25 trees): `B` = `{speed_data['rss_mb']['B']:.1f} MB` | `S6` = `{speed_data['rss_mb']['S6']:.1f} MB`\n"
    )

    lines.append("## 4. Per-dataset Confirmatory Results\n")
    for title, piv in (("Regression (nRMSE, 5-fold mean)", reg_piv), ("Classification (log-loss, 5-fold mean)", clf_piv)):
        lines.append(f"### {title}\n")
        lines.append("| dataset | B | S6 | B+S6 | S6 vs B % | B+S6 vs B % | S6_l1 | CAT | LGB |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ds_name in piv.index:
            r = piv.loc[ds_name]
            d_s6 = 100.0 * (r["S6"] - r["B"]) / r["B"]
            d_bl = 100.0 * (r["B+S6"] - r["B"]) / r["B"]
            lines.append(
                f"| `{ds_name}` | {r['B']:.5f} | {r['S6']:.5f} | {r['B+S6']:.5f} | {d_s6:+.2f}% | {d_bl:+.2f}% | "
                f"{r['S6_l1']:.5f} | {r['CAT']:.5f} | {r['LGB']:.5f} |"
            )
        lines.append("")

    diag_path = RAW_DIR / "diag.csv"
    if diag_path.exists():
        ddf = pd.read_csv(diag_path, dtype={"metric": str})
        ddf["metric_f"] = ddf["metric"].astype(float)
        ddf = ddf.drop_duplicates(subset=["dataset", "fold", "variant"])
        dpiv = ddf.groupby(["dataset", "variant"])["metric_f"].mean().unstack()
        lines.append("## 5. Exploratory Diagnostic Datasets (NOT part of confirmatory tests)\n")
        lines.append("| dataset | B | S6 | B+S6 | S6 vs B % | B+S6 vs B % | S6_l1 | CAT | LGB |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ds_name in dpiv.index:
            r = dpiv.loc[ds_name]
            d_s6 = 100.0 * (r["S6"] - r["B"]) / r["B"]
            d_bl = 100.0 * (r["B+S6"] - r["B"]) / r["B"]
            lines.append(
                f"| `{ds_name}` | {r['B']:.5f} | {r['S6']:.5f} | {r['B+S6']:.5f} | {d_s6:+.2f}% | {d_bl:+.2f}% | "
                f"{r['S6_l1']:.5f} | {r['CAT']:.5f} | {r['LGB']:.5f} |"
            )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sub.add_parser("speed")
    p_run = sub.add_parser("run")
    p_run.add_argument("--shard", type=int, default=0)
    p_run.add_argument("--nshards", type=int, default=1)
    p_run.add_argument("--folds", type=int, default=5)
    p_diag = sub.add_parser("diag")
    p_diag.add_argument("--folds", type=int, default=5)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "speed":
        cmd_speed()
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "diag":
        cmd_diag(args)
    elif args.cmd == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
