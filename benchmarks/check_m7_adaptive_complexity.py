"""M7: pre-registered benchmark of n-adaptive tree-complexity rules (see results/m7_adaptive_complexity_prereg.md).

Measurement only: rules are applied through explicit MiterRegressor/MiterClassifier arguments; library defaults are not
touched.

    python check_m7_adaptive_complexity.py run --shard 0 --nshards 2 [--only NAME] [--folds 5]
    python check_m7_adaptive_complexity.py report
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_m3_oblique import CLF_DATASETS, REG_DATASETS  # noqa: E402

RAW_DIR = HERE / "results" / "raw" / "m7_adaptive"
LEVELS = [500, 1500, 5000, "native"]
RULE_NAMES = ["R1", "R2", "R3"]
VARIANTS = ["B", "R1", "R2", "R3", "CAT"]
FIELDS = [
    "kind",
    "dataset",
    "level",
    "n_fit",
    "fold",
    "variant",
    "metric",
    "trees",
    "secs",
    "params",
]
SMALL_N = 3000
MID_N = 20000
MAX_NATIVE_FIT = (
    20000  # native cells with n_fit >= this are not run (all rules inactive)
)


def rule_params(rule: str, n: int) -> dict:
    """Parameter overrides of a rule for fit size n (empty = inactive)."""
    p: dict = {}
    if rule == "R1" and n < SMALL_N:
        p = {"num_leaves": 15, "min_data_in_leaf": 40}
    elif rule == "R2" and n < SMALL_N:
        p = {"num_leaves": 7, "min_data_in_leaf": 40}
    elif rule == "R3":
        if n < SMALL_N:
            p.update({"num_leaves": 15, "min_data_in_leaf": 40})
        if n < MID_N:
            p["learning_rate"] = 0.03
    return p


def load_tasks(kind: str):
    import openml

    suite = openml.study.get_suite(336 if kind == "reg" else 337)
    task_map = {}
    for tid in suite.tasks:
        t = openml.tasks.get_task(
            tid, download_data=False, download_qualities=False, download_splits=False
        )
        task_map[t.dataset_id] = tid
    return task_map


def load_dataset(kind: str, did: int, task_map: dict):
    import openml

    task = openml.tasks.get_task(
        task_map[did],
        download_data=False,
        download_qualities=False,
        download_splits=True,
    )
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X_df, y_s, _, _ = d.get_data(target=task.target_name, dataset_format="dataframe")
    X = X_df.to_numpy(dtype=np.float64)
    if kind == "reg":
        y = y_s.astype("float64").to_numpy()
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.float64)
    return task, X, y


def make_split(tr: np.ndarray, level, fold: int):
    if level != "native":
        rng = np.random.default_rng(1000 + fold)
        tr = rng.choice(tr, size=int(round(level / 0.8)), replace=False)
    return train_test_split(tr, test_size=0.2, random_state=0)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_eval(kind, variant, X, y, a, b, te, fold, n_fit):
    """Return (metric, trees, secs, params-string)."""
    t0 = time.perf_counter()
    if variant == "CAT":
        from catboost import CatBoostClassifier, CatBoostRegressor

        common = {
            "iterations": 5000,
            "early_stopping_rounds": 50,
            "thread_count": 1,
            "random_seed": fold,
            "verbose": 0,
        }
        if kind == "reg":
            m = CatBoostRegressor(**common)
            m.fit(X[a], y[a], eval_set=(X[b], y[b]))
            pred = m.predict(X[te])
            metric = float(np.sqrt(np.mean((pred - y[te]) ** 2)) / np.std(y[te]))
        else:
            m = CatBoostClassifier(**common)
            m.fit(X[a], y[a], eval_set=(X[b], y[b]))
            metric = logloss(y[te], m.predict_proba(X[te])[:, 1])
        trees = int(m.get_best_iteration() + 1)
        params = "default"
    else:
        import miter

        over = {} if variant == "B" else rule_params(variant, n_fit)
        kw = {
            "n_estimators": 5000,
            "early_stopping_rounds": 50,
            "n_jobs": 1,
            "random_state": fold,
            **over,
        }
        cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
        m = cls(**kw).fit(X[a], y[a], eval_set=[(X[b], y[b])])
        if kind == "reg":
            pred = m.predict(X[te])
            metric = float(np.sqrt(np.mean((pred - y[te]) ** 2)) / np.std(y[te]))
        else:
            metric = logloss(y[te], m.predict_proba(X[te])[:, 1])
        trees = len(m.train_loss_)
        params = ";".join(f"{k}={v}" for k, v in sorted(over.items())) or "default"
    return metric, trees, time.perf_counter() - t0, params


def variants_to_run(n_fit: int) -> list[str]:
    out = ["B", "CAT"]
    for r in RULE_NAMES:
        active = bool(rule_params(r, n_fit))
        if active or n_fit <= 5000:
            out.append(r)
    return out


def run(args) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"raw_shard{args.shard}.csv"
    done = set()
    for f in RAW_DIR.glob("raw_shard*.csv"):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["kind"], r["dataset"], r["level"], r["fold"], r["variant"]))
    new = not out_path.exists()
    fh = out_path.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    units = []
    for kind, lst in (("reg", REG_DATASETS), ("clf", CLF_DATASETS)):
        for name, did in lst:
            units.append((kind, name, did))
    # greedy balance across shards using a crude cost proxy: native size known only after loading, so use list order
    # interleaving (large datasets are spread through the lists)
    units = [u for i, u in enumerate(units) if i % args.nshards == args.shard]
    if args.only:
        units = [u for u in units if u[1] in args.only]
    maps = {}
    for kind, name, did in units:
        if kind not in maps:
            maps[kind] = load_tasks(kind)
        task, X, y = load_dataset(kind, did, maps[kind])
        print(f"== {kind} {name} X={X.shape}", flush=True)
        for level in LEVELS:
            for fold in range(args.folds):
                tr, te = task.get_train_test_split_indices(repeat=0, fold=fold)
                native_fit = len(train_test_split(tr, test_size=0.2, random_state=0)[0])
                if level != "native" and level >= native_fit:
                    continue
                if level == "native" and native_fit >= MAX_NATIVE_FIT:
                    continue
                a, b = make_split(tr, level, fold)
                n_fit = len(a)
                if level != "native":
                    assert n_fit == level, (n_fit, level)
                for v in variants_to_run(n_fit):
                    key = (kind, name, str(level), str(fold), v)
                    if key in done:
                        continue
                    metric, trees, secs, params = fit_eval(
                        kind, v, X, y, a, b, te, fold, n_fit
                    )
                    w.writerow(
                        {
                            "kind": kind,
                            "dataset": name,
                            "level": level,
                            "n_fit": n_fit,
                            "fold": fold,
                            "variant": v,
                            "metric": repr(metric),
                            "trees": trees,
                            "secs": f"{secs:.3f}",
                            "params": params,
                        }
                    )
                    fh.flush()
                    print(
                        f"{kind} {name} L={level} f={fold} {v}: {metric:.5f} trees={trees} {secs:.1f}s",
                        flush=True,
                    )
    fh.close()


def load_raw():
    import pandas as pd

    frames = [
        pd.read_csv(f, dtype={"metric": str})
        for f in sorted(RAW_DIR.glob("raw_shard*.csv"))
    ]
    df = pd.concat(frames, ignore_index=True)
    df["metric_f"] = df["metric"].astype(float)
    df["level"] = df["level"].astype(str)
    return df


def report(args) -> None:
    import pandas as pd
    from scipy.stats import wilcoxon

    df = load_raw()
    out: list[str] = []
    out.append("# M7 adaptive complexity: generated tables\n")
    out.append(
        f"Raw rows: {len(df)}. Metric: nRMSE (reg) or log-loss (clf), mean over folds; negative % = better.\n"
    )

    # incomplete cells
    cnt = df.groupby(["kind", "dataset", "level", "variant"]).fold.nunique()
    bad = cnt[cnt != args.folds]
    out.append(f"Cells with fold count != {args.folds}: {len(bad)}\n")

    cell = (
        df.groupby(["kind", "dataset", "level", "variant"])
        .agg(
            metric=("metric_f", "mean"),
            n_fit=("n_fit", "first"),
            trees=("trees", "mean"),
            secs=("secs", "mean"),
        )
        .reset_index()
    )
    wide = cell.pivot_table(
        index=["kind", "dataset", "level"], columns="variant", values="metric"
    )
    nfit = cell.groupby(["kind", "dataset", "level"]).n_fit.first()
    wide["n_fit"] = nfit
    wide = wide.reset_index()
    lev_order = {"500": 0, "1500": 1, "5000": 2, "native": 3}
    wide["lo"] = wide["level"].map(lev_order)
    wide = wide.sort_values(["kind", "dataset", "lo"])

    # bit-identity of inactive cells (compare per-fold raw values)
    ident_fail = []
    n_ident = 0
    for r in RULE_NAMES:
        sub = df[df.variant == r]
        for (k, d, lv, fo), g in sub.groupby(["kind", "dataset", "level", "fold"]):
            nf = int(g.n_fit.iloc[0])
            if rule_params(r, nf):
                continue
            base = df[
                (df.kind == k)
                & (df.dataset == d)
                & (df.level == lv)
                & (df.fold == fo)
                & (df.variant == "B")
            ]
            n_ident += 1
            if base.metric.iloc[0] != g.metric.iloc[0]:
                ident_fail.append((r, k, d, lv, fo))
    out.append(
        f"Inactive-rule cells run for real: {n_ident} (per fold); non-bit-identical: {len(ident_fail)} {ident_fail[:5]}\n"
    )

    for kind, title in (
        ("reg", "Regression (nRMSE)"),
        ("clf", "Classification (log-loss)"),
    ):
        w = wide[wide.kind == kind]
        out.append(f"\n## {title}: per (dataset, level) table\n")
        out.append(
            "| dataset | level | n_fit | B | R1 | R2 | R3 | CatBoost | R1 % | R2 % | R3 % | B vs Cat % | R1 vs Cat % | R2 vs Cat % | R3 vs Cat % |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

        def f(x, p=4):
            return "" if pd.isna(x) else f"{x:.{p}f}"

        def pc(x, ref):
            return "" if pd.isna(x) or pd.isna(ref) else f"{100 * (x - ref) / ref:+.2f}"

        for _, r in w.iterrows():
            b = r["B"]
            rr = {
                v: (
                    r[v]
                    if v in r and not pd.isna(r[v])
                    else (b if not rule_params(v, int(r.n_fit)) else float("nan"))
                )
                for v in RULE_NAMES
            }
            out.append(
                f"| {r.dataset} | {r.level} | {int(r.n_fit)} | {f(b)} | "
                + " | ".join(f(rr[v]) for v in RULE_NAMES)
                + f" | {f(r['CAT'])} | "
                + " | ".join(pc(rr[v], b) for v in RULE_NAMES)
                + f" | {pc(b, r['CAT'])} | "
                + " | ".join(pc(rr[v], r["CAT"]) for v in RULE_NAMES)
                + " |"
            )
        out.append(
            "\n(Empty rule cell = inactive and not run for n_fit > 5000, identical to B by construction.)\n"
        )

        out.append(f"\n## {title}: primary test, per-dataset mean over active levels\n")
        out.append(
            "| rule | active levels | datasets | mean % | median % | wins/losses | Wilcoxon p | worst dataset | worst cell | pass crit 1-3 |"
        )
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in RULE_NAMES:
            rows = []
            worst_cell = (-1e9, "")
            for d, g in w.groupby("dataset"):
                ch = []
                for _, x in g.iterrows():
                    if rule_params(r, int(x.n_fit)) and not pd.isna(x[r]):
                        c = 100 * (x[r] - x["B"]) / x["B"]
                        ch.append(c)
                        if c > worst_cell[0]:
                            worst_cell = (c, f"{d}@{x.level}")
                if ch:
                    rows.append((d, float(np.mean(ch)), len(ch)))
            if not rows:
                continue
            v = np.array([x[1] for x in rows])
            a_b = np.array([0.0] * len(v))
            try:
                p = wilcoxon(v, a_b).pvalue
            except ValueError:
                p = float("nan")
            wi = int(np.sum(v < 0))
            lo = int(np.sum(v > 0))
            wd = rows[int(np.argmax(v))]
            crit = bool(np.median(v) < 0 and v.max() < 2.0 and p < 0.05)
            lvls = sorted(
                {x.level for _, x in w.iterrows() if rule_params(r, int(x.n_fit))},
                key=lambda s: lev_order[s],
            )
            out.append(
                f"| {r} | {','.join(lvls)} | {len(v)} | {v.mean():+.2f} | {np.median(v):+.2f} | {wi}/{lo} | {p:.4g} | "
                f"{wd[0]} {wd[1]:+.2f} | {worst_cell[1]} {worst_cell[0]:+.2f} | {crit} |"
            )
            out.append("")  # placeholder replaced below
            out.pop()
        # per-dataset detail
        out.append(
            f"\n### {title}: per-dataset unit values (mean % over active levels)\n"
        )
        out.append("| dataset | R1 | R2 | R3 |")
        out.append("|---|---|---|---|")
        for d, g in w.groupby("dataset", sort=False):
            cells = []
            for r in RULE_NAMES:
                ch = [
                    100 * (x[r] - x["B"]) / x["B"]
                    for _, x in g.iterrows()
                    if rule_params(r, int(x.n_fit)) and not pd.isna(x[r])
                ]
                cells.append(f"{np.mean(ch):+.2f}" if ch else "")
            out.append(f"| {d} | " + " | ".join(cells) + " |")

        # gap to catboost per level (median over datasets)
        out.append(
            f"\n### {title}: gap to CatBoost by level (median over datasets of 100*(x-cat)/cat)\n"
        )
        out.append("| level | datasets | B | R1 | R2 | R3 |")
        out.append("|---|---|---|---|---|---|")
        for lv in ["500", "1500", "5000", "native"]:
            g = w[w.level == lv]
            if g.empty:
                continue
            vals = []
            for v in ["B"] + RULE_NAMES:
                xs = []
                for _, x in g.iterrows():
                    val = x[v] if v == "B" or rule_params(v, int(x.n_fit)) else x["B"]
                    if not pd.isna(val) and not pd.isna(x["CAT"]):
                        xs.append(100 * (val - x["CAT"]) / x["CAT"])
                vals.append(f"{np.median(xs):+.2f}" if xs else "")
            out.append(f"| {lv} | {len(g)} | " + " | ".join(vals) + " |")

    # time / trees
    out.append("\n## Mean fit seconds and trees per variant (all cells)\n")
    t = df.groupby(["kind", "variant"]).agg(
        secs=("secs", "mean"), trees=("trees", "mean"), n=("secs", "size")
    )
    out.append("| kind | variant | mean secs | mean trees | fits |")
    out.append("|---|---|---|---|---|")
    for (k, v), r in t.iterrows():
        out.append(f"| {k} | {v} | {r.secs:.2f} | {r.trees:.0f} | {int(r.n)} |")
    out.append(f"\nTotal sum of per-fit seconds: {df.secs.sum():.0f}\n")

    path = HERE / "results" / "m7_adaptive_complexity_tables_summary.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nWrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--shard", type=int, default=0)
    pr.add_argument("--nshards", type=int, default=1)
    pr.add_argument("--folds", type=int, default=5)
    pr.add_argument("--only", nargs="*")
    pp = sub.add_parser("report")
    pp.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
