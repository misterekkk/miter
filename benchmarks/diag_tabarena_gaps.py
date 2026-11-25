"""Exploratory diagnosis of the miter-vs-CatBoost gap on miami_housing, concrete, airfoil (parameters only).

Two modes:
  dump <outdir>                 (run in .venv_tabarena; needs openml) export X, y and TabArena outer splits to npz.
  run <datadir> <outdir> <ds> [<ds> ...]   (run in .venv) fit all variants, write <outdir>/<ds>.csv (one row per fit).
  report <outdir>               print mean/sd tables (markdown) from the CSVs.

Protocol: TabArena OpenML task splits (repeat r, fold f of the outer split); inside the outer train set a fixed random
80/20 split (seed = 1000*r+f) provides the early-stopping set for ALL libraries (patience 50, max 10000 trees);
one single model per variant (NOT TabArena's 8-fold bagging). Categorical columns are encoded as numbers
(airfoil attack-angle = its numeric value, miami avno60plus = 0/1), so every library sees plain numeric data.
Metric: RMSE on the original target scale on the outer test fold.
"""

import csv
import os
import sys
import time

import numpy as np

TASKS = {"airfoil": 363612, "concrete": 363625, "miami": 363686}
# (repeats, folds) used per dataset (TabArena has 10x3 splits for small sets, 3x3 for miami)
SPLITS = {
    "airfoil": [(r, f) for r in range(3) for f in range(3)],
    "concrete": [(r, f) for r in range(3) for f in range(3)],
    "miami": [(r, f) for r in range(3) for f in range(3)],
}
ES = 50
MAXT = 10000


def dump(outdir):
    import openml

    here = os.path.dirname(os.path.abspath(__file__))
    openml.config.set_root_cache_directory(here + "/results/raw/openml_cache_pandas2")
    os.makedirs(outdir, exist_ok=True)
    for name, tid in TASKS.items():
        task = openml.tasks.get_task(tid, download_data=False)
        X, y, _, _ = task.get_dataset().get_data(target=task.target_name)
        X = X.apply(
            lambda c: (
                c.astype(str).astype(float)
                if str(c.dtype) == "category"
                else c.astype(float)
            )
        )
        sp = {}
        for r, f in SPLITS[name]:
            tr, te = task.get_train_test_split_indices(repeat=r, fold=f)
            sp[f"tr_{r}_{f}"], sp[f"te_{r}_{f}"] = tr, te
        np.savez(
            f"{outdir}/{name}.npz",
            X=X.to_numpy(),
            y=y.to_numpy(float),
            cols=np.array(X.columns),
            **sp,
        )
        print(name, X.shape)


def variants(name):
    """(label, lib, params, log_target)"""
    v = []
    base = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=20, lambda_l2=1.0)
    v.append(("miter base", "miter", base, False))
    v.append(("miter accurate", "miter", dict(base, preset="accurate"), False))
    v.append(("miter m_mult=4", "miter", dict(base, m_mult=4), False))
    v.append(("miter lr0.03", "miter", dict(base, learning_rate=0.03), False))
    v.append(("miter lr0.02", "miter", dict(base, learning_rate=0.02), False))
    v.append(("miter leaves7", "miter", dict(base, num_leaves=7), False))
    v.append(("miter leaves15", "miter", dict(base, num_leaves=15), False))
    v.append(("miter leaves63", "miter", dict(base, num_leaves=63), False))
    v.append(("miter mdl5", "miter", dict(base, min_data_in_leaf=5), False))
    v.append(("miter mdl40", "miter", dict(base, min_data_in_leaf=40), False))
    v.append(("miter l2=0.1", "miter", dict(base, lambda_l2=0.1), False))
    v.append(("miter l2=3", "miter", dict(base, lambda_l2=3.0), False))
    v.append(("miter l2=10", "miter", dict(base, lambda_l2=10.0), False))
    # feature_fraction / bagging_fraction are accepted by the Python API but absent from src/ (inert), so not tested
    v.append(("miter leaves4", "miter", dict(base, num_leaves=4), False))
    v.append(
        (
            "miter leaves7 mdl40",
            "miter",
            dict(base, num_leaves=7, min_data_in_leaf=40),
            False,
        )
    )
    v.append(
        (
            "miter leaves15 mdl40",
            "miter",
            dict(base, num_leaves=15, min_data_in_leaf=40),
            False,
        )
    )
    v.append(
        (
            "miter leaves7 lr0.02",
            "miter",
            dict(base, num_leaves=7, learning_rate=0.02),
            False,
        )
    )
    v.append(
        (
            "miter leaves7 l2=10",
            "miter",
            dict(base, num_leaves=7, lambda_l2=10.0),
            False,
        )
    )
    v.append(("miter leaves127", "miter", dict(base, num_leaves=127), False))
    v.append(
        (
            "miter leaves255 mdl10",
            "miter",
            dict(base, num_leaves=255, min_data_in_leaf=10),
            False,
        )
    )
    v.append(("miter mdl80", "miter", dict(base, min_data_in_leaf=80), False))
    v.append(
        (
            "miter lr0.02 mdl5",
            "miter",
            dict(base, learning_rate=0.02, min_data_in_leaf=5),
            False,
        )
    )
    lg = dict(learning_rate=0.05, num_leaves=31, min_child_samples=20, reg_lambda=1.0)
    v.append(("lgbm base", "lgbm", lg, False))
    v.append(("lgbm lr0.02", "lgbm", dict(lg, learning_rate=0.02), False))
    v.append(
        ("lgbm depth6 leaves63", "lgbm", dict(lg, max_depth=6, num_leaves=63), False)
    )
    v.append(
        ("lgbm depth4 leaves15", "lgbm", dict(lg, max_depth=4, num_leaves=15), False)
    )
    v.append(
        (
            "lgbm leaves7 mdl40",
            "lgbm",
            dict(lg, num_leaves=7, min_child_samples=40),
            False,
        )
    )
    v.append(("lgbm mdl5", "lgbm", dict(lg, min_child_samples=5), False))
    cb = dict()
    v.append(("cat default (auto lr)", "cat", cb, False))
    v.append(("cat lr0.05", "cat", dict(learning_rate=0.05), False))
    v.append(("cat lr0.02", "cat", dict(learning_rate=0.02), False))
    v.append(
        (
            "cat Plain lr0.05",
            "cat",
            dict(learning_rate=0.05, boosting_type="Plain"),
            False,
        )
    )
    v.append(
        (
            "cat Ordered lr0.05",
            "cat",
            dict(learning_rate=0.05, boosting_type="Ordered"),
            False,
        )
    )
    v.append(
        (
            "cat Lossguide31 lr0.05",
            "cat",
            dict(
                learning_rate=0.05,
                grow_policy="Lossguide",
                max_leaves=31,
                min_data_in_leaf=20,
            ),
            False,
        )
    )
    v.append(
        (
            "cat Depthwise lr0.05",
            "cat",
            dict(learning_rate=0.05, grow_policy="Depthwise"),
            False,
        )
    )
    v.append(("cat depth10 lr0.05", "cat", dict(learning_rate=0.05, depth=10), False))
    v.append(("cat l2=1 lr0.05", "cat", dict(learning_rate=0.05, l2_leaf_reg=1), False))
    v.append(("cat depth8 lr0.05", "cat", dict(learning_rate=0.05, depth=8), False))
    v.append(
        (
            "cat Plain lr0.05 no-bootstrap",
            "cat",
            dict(learning_rate=0.05, boosting_type="Plain", bootstrap_type="No"),
            False,
        )
    )
    if name == "miami":
        for lab, lib, p, _ in list(v):
            if lab in (
                "miter base",
                "miter lr0.02",
                "lgbm base",
                "cat lr0.05",
                "cat default (auto lr)",
                "miter accurate",
            ):
                v.append((lab + " | log-y", lib, p, True))
    return v


def fit_predict(lib, p, Xa, ya, Xv, yv, Xt, seed):
    if lib == "miter":
        from miter import MiterRegressor

        m = MiterRegressor(
            n_estimators=MAXT,
            early_stopping_rounds=ES,
            n_jobs=1,
            random_state=seed,
            **p,
        )
        m.fit(Xa, ya, eval_set=[(Xv, yv)])
        return m.predict(Xt), int(m.best_iteration_)
    if lib == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMRegressor(
            n_estimators=MAXT, n_jobs=1, random_state=seed, verbose=-1, **p
        )
        m.fit(
            Xa,
            ya,
            eval_set=[(Xv, yv)],
            callbacks=[lgb.early_stopping(ES, verbose=False)],
        )
        return m.predict(Xt), int(m.best_iteration_)
    from catboost import CatBoostRegressor

    m = CatBoostRegressor(
        iterations=MAXT,
        od_type="Iter",
        od_wait=ES,
        thread_count=1,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        **p,
    )
    m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return m.predict(Xt), int(m.get_best_iteration())


def run(datadir, outdir, names):
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        d = np.load(f"{datadir}/{name}.npz")
        X, y = d["X"], d["y"]
        path = f"{outdir}/{name}.csv"
        done = set()
        if os.path.exists(path):
            with open(path) as fh:
                done = {(r["variant"], r["rep"], r["fold"]) for r in csv.DictReader(fh)}
        fh = open(path, "a", newline="")
        w = csv.writer(fh)
        if not done:
            w.writerow(
                ["dataset", "variant", "rep", "fold", "rmse", "best_iter", "secs"]
            )
        for r, f in SPLITS[name]:
            tr, te = d[f"tr_{r}_{f}"], d[f"te_{r}_{f}"]
            rng = np.random.RandomState(1000 * r + f)
            perm = rng.permutation(tr)
            nv = int(round(0.2 * len(tr)))
            va, fi = perm[:nv], perm[nv:]
            for lab, lib, p, logy in variants(name):
                if (lab, str(r), str(f)) in done:
                    continue
                t0 = time.time()
                tf = np.log if logy else (lambda a: a)
                pred, bi = fit_predict(
                    lib, p, X[fi], tf(y[fi]), X[va], tf(y[va]), X[te], 1000 * r + f
                )
                if logy:
                    pred = np.exp(pred)
                rmse = float(np.sqrt(np.mean((pred - y[te]) ** 2)))
                w.writerow(
                    [name, lab, r, f, f"{rmse:.6f}", bi, f"{time.time() - t0:.2f}"]
                )
                fh.flush()
            print(name, r, f, "done", flush=True)
        fh.close()


def report(outdir):
    import pandas as pd

    for name in ("concrete", "airfoil", "miami"):
        p = f"{outdir}/{name}.csv"
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = df[df.variant.isin([x[0] for x in variants(name)])]
        g = df.groupby("variant", sort=False)
        t = g.agg(
            n=("rmse", "size"),
            rmse=("rmse", "mean"),
            sd=("rmse", "std"),
            trees=("best_iter", "mean"),
            secs=("secs", "mean"),
        )
        # paired relative difference vs miter base and vs cat default, per split
        piv = df.pivot_table(index=["rep", "fold"], columns="variant", values="rmse")
        for ref in ("miter base", "cat Plain lr0.05"):
            t[f"vs {ref} %"] = [100 * (piv[c] / piv[ref] - 1).mean() for c in t.index]
        print(f"\n### {name}\n")
        t = t.round(4).reset_index()
        print("| " + " | ".join(t.columns) + " |")
        print("|" + "---|" * len(t.columns))
        for row in t.itertuples(index=False):
            print("| " + " | ".join(str(x) for x in row) + " |")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dump":
        dump(sys.argv[2])
    elif mode == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        report(sys.argv[2])
