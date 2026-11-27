"""Classification parity: miter C++ (MiterClassifier, R_uni 2*m0) vs histree prototype, 13 OpenML binary datasets.

Pre-registration: benchmarks/results/m3_clf_parity_prereg.md.
Usage:
  check_m3_clf_parity.py work SHARD NSHARDS FOLDS OUTDIR   # fold-major unit list, skips finished units
  check_m3_clf_parity.py report FOLDS OUTDIR SUMMARY_MD
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import numpy as np

HISTREE_SRC = (str(Path(os.environ.get("HISTREE_DIR", Path(__file__).resolve().parents[2] / "histree"))) + "/src")
sys.path.insert(0, HISTREE_SRC)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_m3_oblique import CLF_DATASETS  # noqa: E402

CFG = dict(
    max_leaves=31,
    min_data_in_leaf=20,
    reg_lambda=1.0,
    learning_rate=0.05,
    n_bins=255,
    objective="binary",
)


def logloss(y, p):
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def make_int_binary():
    """Prototype integer kernel (build_tree_int) with a logistic-loss boosting loop (mirrors ObliqueGBInt.fit and
    ObliqueGBRegressor.fit binary branch: init = logit(mean), g = p - y, h = max(p(1-p), 1e-6))."""
    from histree.gb import oblique as ob
    from histree.gb.oblique_int import ObliqueGBInt, build_tree_int, predict_tree_int

    class IntBinary(ObliqueGBInt):
        def fit(self, X, y, X_val, y_val):
            X = np.asarray(X, float)
            y = np.asarray(y, float)
            n, p = X.shape
            self.n_features_in_ = p
            self._fit_binning(X)
            Xb, _ = self._transform(X)
            Xi = self._codes(X)
            nb = int(max(len(e) for e in self.edges_)) + 1
            m = 2 * int(math.ceil(math.sqrt(p)))
            self.m_ = m
            self.opts_ = np.array(
                [0, 1, 0, 0, 0, 0, 0, int(self.k_proj)], dtype=np.int64
            )
            Xvb, _ = self._transform(X_val)
            Xvi = self._codes(X_val)
            yv = np.asarray(y_val, float)
            p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            self.init_ = math.log(p0 / (1 - p0))
            F = np.full(n, self.init_)
            Fv = np.full(len(yv), self.init_)
            def sig(z):
                return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

            def ll(z, t):
                return float(np.mean(np.logaddexp(0.0, z) - t * z))

            self.trees_, self.val_loss_ = [], []
            best, best_iter, since = np.inf, 0, 0
            for t in range(self.max_trees):
                pr = sig(F)
                g = pr - y
                h = np.maximum(pr * (1.0 - pr), 1e-6)
                self.opts_[2] = self.random_state * 100003 + t + 7
                T = self._alloc_tree()
                log = np.zeros((self.max_leaves, ob.LOG_COLS))
                rv = np.zeros(n)
                n_nodes, n_log, n_leaves = build_tree_int(
                    Xb,
                    Xi,
                    g,
                    h,
                    nb,
                    self.max_leaves,
                    float(self.min_data_in_leaf),
                    self.reg_lambda,
                    self.min_gain_to_split,
                    m,
                    True,
                    self.random_state * 100003 + t,
                    self.opts_,
                    *self._tree_args(T),
                    rv,
                    log,
                )
                F += self.learning_rate * rv
                T["n_nodes"], T["n_leaves"] = int(n_nodes), int(n_leaves)
                self.trees_.append(T)
                Fv += self.learning_rate * predict_tree_int(
                    Xvb, Xvi, *self._tree_args(T)
                )
                vl = ll(Fv, yv)
                self.val_loss_.append(vl)
                if vl < best - 1e-12:
                    best, best_iter, since = vl, t + 1, 0
                else:
                    since += 1
                    if since >= self.early_stopping_rounds:
                        break
            self.best_iteration_ = best_iter
            return self

        def predict_proba(self, X, num_iteration=None):
            z = self.predict(X, num_iteration)
            return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    return IntBinary


def q16r(Xtr, *others):
    lo = np.quantile(Xtr, 0.0001, axis=0)
    hi = np.quantile(Xtr, 0.9999, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    def f(X):
        return np.clip(np.rint((X - lo) / span * 65535.0), 0.0, 65535.0)

    return [f(Xtr)] + [f(o) for o in others]


def run_unit(name, did, task_id, fold, outdir):
    import miter
    import openml
    from histree.gb import oblique as ob
    from sklearn.model_selection import train_test_split

    path = Path(outdir) / f"{name}_{fold}.json"
    if path.exists():
        return
    task = openml.tasks.get_task(
        task_id, download_data=False, download_qualities=False, download_splits=True
    )
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    X = X_df.to_numpy(dtype=np.float64)
    yv = y_s.astype("category").cat.codes.to_numpy(dtype=np.float64)
    p = X.shape[1]
    m0 = int(math.ceil(math.sqrt(p)))
    tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
    a, b = train_test_split(tr, test_size=0.2, random_state=0)
    res = dict(
        name=name,
        did=did,
        fold=fold,
        n=int(X.shape[0]),
        p=int(p),
        nan=bool(np.isnan(X).any()),
        n_train=len(a),
        n_valid=len(b),
        n_test=len(te),
    )

    t0 = time.perf_counter()
    mm = miter.MiterClassifier(
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
    secs = time.perf_counter() - t0
    res["miter"] = dict(
        logloss=logloss(yv[te], mm.predict_proba(X[te])[:, 1]),
        trees_trained=len(mm.train_loss_),
        secs=secs,
    )
    for attr in ("best_iteration_", "best_iteration"):
        if hasattr(mm, attr):
            res["miter"]["best_iter"] = int(getattr(mm, attr))
            break

    if res["nan"]:
        res["proto_float"] = None
        res["proto_int"] = None
    else:
        t0 = time.perf_counter()
        h = ob.ObliqueGBRegressor(
            **CFG,
            selector="R",
            m=2 * m0,
            bin_sample=-3,
            max_trees=5000,
            early_stopping_rounds=50,
            random_state=fold,
        ).fit(X[a], yv[a], X[b], yv[b])
        res["proto_float"] = dict(
            logloss=logloss(yv[te], h.predict_proba(X[te])),
            best_iter=int(h.best_iteration_),
            trees_trained=len(h.trees_),
            secs=time.perf_counter() - t0,
        )
        IB = make_int_binary()
        Xa, Xb_, Xt = q16r(X[a], X[b], X[te])
        t0 = time.perf_counter()
        hi = IB(
            selector="R",
            m=2 * m0,
            max_trees=5000,
            early_stopping_rounds=50,
            max_leaves=31,
            min_data_in_leaf=20,
            reg_lambda=1.0,
            min_gain_to_split=0.0,
            learning_rate=0.05,
            n_bins=255,
            random_state=fold,
        ).fit(Xa, yv[a], Xb_, yv[b])
        res["proto_int"] = dict(
            logloss=logloss(yv[te], hi.predict_proba(Xt)),
            best_iter=int(hi.best_iteration_),
            trees_trained=len(hi.trees_),
            secs=time.perf_counter() - t0,
        )
    path.write_text(json.dumps(res))
    print(
        name,
        fold,
        {
            k: (round(v["logloss"], 5) if isinstance(v, dict) else v)
            for k, v in res.items()
            if k in ("miter", "proto_float", "proto_int")
        },
        flush=True,
    )


def work(shard, nshards, folds, outdir):
    import openml

    Path(outdir).mkdir(parents=True, exist_ok=True)
    suite = openml.study.get_suite(337)
    tmap = {}
    for tid in suite.tasks:
        tm = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        tmap[tm.dataset_id] = tid
    units = [(n, d, f) for f in range(folds) for n, d in CLF_DATASETS]
    for i, (n, d, f) in enumerate(units):
        if i % nshards == shard:
            run_unit(n, d, tmap[d], f, outdir)


def report(folds, outdir, summary):
    from scipy.stats import wilcoxon

    rows = []
    for n, _ in CLF_DATASETS:
        rs = [
            json.loads((Path(outdir) / f"{n}_{f}.json").read_text())
            for f in range(folds)
        ]
        rows.append((n, rs))

    def agg(key):
        mit = np.array([np.mean([r["miter"]["logloss"] for r in rs]) for _, rs in rows])
        pro = np.array([np.mean([r[key]["logloss"] for r in rs]) for _, rs in rows])
        rel = 100.0 * (mit - pro) / pro
        pw = wilcoxon(mit, pro).pvalue
        return mit, pro, rel, pw

    out = []
    for key, label in (
        (
            "proto_float",
            "PRIMARY: histree float oblique.py R_uni (binary, bin_sample=-3)",
        ),
        (
            "proto_int",
            "SECONDARY (exploratory): histree int kernel build_tree_int + logistic loop, Q16r",
        ),
    ):
        mit, pro, rel, pw = agg(key)
        out.append(f"\n### {label}\n")
        out.append(
            "| dataset | miter log-loss | histree log-loss | rel diff | per-fold rel diff | trees miter / histree (mean best/trained) |"
        )
        out.append("|---|---|---|---|---|---|")
        for (n, rs), a_, b_, r_ in zip(rows, mit, pro, rel, strict=True):
            pf = ", ".join(
                f"{100 * (r['miter']['logloss'] - r[key]['logloss']) / r[key]['logloss']:+.2f}"
                for r in rs
            )
            tm = np.mean(
                [r["miter"].get("best_iter", r["miter"]["trees_trained"]) for r in rs]
            )
            th = np.mean([r[key]["best_iter"] for r in rs])
            out.append(
                f"| {n} | {a_:.4f} | {b_:.4f} | {r_:+.2f}% | {pf} | {tm:.0f} / {th:.0f} |"
            )
        out.append(
            f"\n- mean {rel.mean():+.3f}%, **median {np.median(rel):+.3f}%**, miter wins {int(np.sum(mit < pro))}/13, Wilcoxon p = {pw:.4g}; |median| <= 1% : {abs(np.median(rel)) <= 1.0}"
        )
        out.append(
            f"- raw: mean {rel.mean()!r}, median {float(np.median(rel))!r}, p {pw!r}"
        )
    secs = {
        k: sum(r[k]["secs"] for _, rs in rows for r in rs)
        for k in ("miter", "proto_float", "proto_int")
    }
    out.append(
        "\nTotal training seconds (sum over units; run under 3 parallel processes + other load, approximate): "
        + ", ".join(f"{k} {v:.0f}" for k, v in secs.items())
    )
    Path(summary).with_suffix(".body.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    if sys.argv[1] == "work":
        work(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    else:
        report(int(sys.argv[2]), sys.argv[3], sys.argv[4])
