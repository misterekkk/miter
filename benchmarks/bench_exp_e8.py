"""E8: data-dependent defaults (results/exp_e8_defaults_prereg.md).

Same protocol as bench_exp_speedups.py (5 folds, early stopping 50, one thread, at most 25,000
rows). Binary tasks also record ROC AUC, TabArena's binary metric. Arms are functions of the task
kind and the number of training rows; an arm that does not apply to a task (e.g. a classification
arm on regression) is not run there.

Usage:
    .venv/bin/python benchmarks/bench_exp_e8.py run --set dev --arms D0 DEPTH_T2000 --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_e8.py summarize --set dev --arms DEPTH_T2000 CAP20K
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench_exp_e7 import CONTROL as E7_CONTROL  # noqa: E402
from bench_exp_e7 import DEV as E7_DEV  # noqa: E402
from bench_exp_external_holdout import _split_cv, eval_metrics, load_dataset  # noqa: E402
from bench_exp_speedups import _fit  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RAW_DIR = HERE / "results" / "raw" / "exp_e8"
FIELDS = ["kind", "dataset", "n_eval", "n_train", "p_num", "p_cat", "fold", "arm", "metric", "auc", "trees",
          "secs"]

Arm = Callable[[str, int], dict[str, Any] | None]


def _depth_by_n(threshold: int) -> Arm:
    return lambda kind, n: {"max_depth": 4} if n < threshold else {}


def _clf_only(params: dict[str, Any]) -> Arm:
    return lambda kind, n: dict(params) if kind != "reg" else None


def _ctr_card(k: int) -> Arm:
    return lambda kind, n: {"max_ctr_complexity": 1, "ctr_n_permutations": 4, "ctr_min_cardinality": k}


ARMS: dict[str, Arm] = {
    "D0": lambda kind, n: {},
    # H1: depth 4 below a training-row threshold.
    "DEPTH_T1000": _depth_by_n(1000),
    "DEPTH_T2000": _depth_by_n(2000),
    "DEPTH_T4000": _depth_by_n(4000),
    # H2: more trees on large data.
    "CAP20K": lambda kind, n: {"n_estimators": 20000},
    "LR_N10K": lambda kind, n: {"learning_rate": 0.1} if n >= 10000 else {},
    # H3: leaf regularisation for classification.
    "MDL5_CLF": _clf_only({"min_data_in_leaf": 5}),
    "MDL20_CLF": _clf_only({"min_data_in_leaf": 20}),
    "L2_10_CLF": _clf_only({"lambda_l2": 10.0}),
    "MSH1_CLF": _clf_only({"min_sum_hessian_in_leaf": 1.0}),
    # H4: CTR on single features above a cardinality threshold.
    "CTR_K16": _ctr_card(16),
    "CTR_K32": _ctr_card(32),
    "CTR_K64": _ctr_card(64),
}

# (kind, name, OpenML dataset id), chosen by the rules in the pre-registration.
GENERAL: list[tuple[str, str, int]] = [
    ("mc", "letter", 6), ("mc", "balance-scale", 11), ("mc", "mfeat-factors", 12), ("clf", "breast-w", 15),
    ("mc", "optdigits", 28), ("mc", "pendigits", 32), ("mc", "vehicle", 54), ("mc", "yeast", 181),
    ("mc", "satimage", 182), ("reg", "yprop_4_1", 416), ("mc", "analcatdata_authorship", 458),
    ("reg", "quake", 550), ("clf", "kc2", 1063), ("clf", "kc1", 1067), ("clf", "banknote-authentication", 1462),
    ("clf", "phoneme", 1489), ("mc", "wall-robot-navigation", 1497), ("mc", "semeion", 1501),
    ("clf", "wdbc", 1510), ("mc", "wine-quality-white", 40498), ("mc", "texture", 40499), ("mc", "shuttle", 40685),
    ("mc", "steel-plates-fault", 40982), ("clf", "madeline", 41144), ("clf", "philippine", 41145),
    ("clf", "sylvine", 41146), ("clf", "MiniBooNE", 41150), ("mc", "volkert", 41166), ("mc", "jannis", 41168),
    ("reg", "Yolanda", 42705), ("reg", "OnlineNewsPopularity", 42724), ("reg", "us_crime", 42730),
    ("reg", "particulate-matter-ukair-2017", 44068), ("reg", "SGEMM_GPU_kernel_performance", 44069),
    ("reg", "pumadyn32nh", 44981),
]
IMBALANCED: list[tuple[str, str, int]] = [
    ("clf", "pc1", 1068), ("clf", "Satellite", 40900), ("clf", "wilt", 40983),
    ("clf", "climate-model-simulation-crashes", 40994), ("clf", "Asteroid_Dataset", 42252),
    ("clf", "Credit_Card_Fraud", 46568), ("clf", "Employee-Turnover-at-TECHCO", 43551), ("clf", "Is_fraud", 46369),
    ("clf", "RelevantImagesDatasetTEST", 42680), ("clf", "Speech", 40910),
    ("clf", "jigsaw_unintended_bias100K", 46654), ("clf", "mammography", 310), ("clf", "mc1", 1056),
    ("clf", "open_payments", 42738), ("clf", "pc2", 1069), ("clf", "sylva_prior", 1040),
    ("clf", "yeast_ml8", 316),
]
HIGH_CARD_EXTRA: list[tuple[str, str, int]] = [
    ("mc", "colleges_aaup", 488), ("reg", "medical_charges", 42130), ("clf", "musk", 1116),
    ("reg", "rainfall_bangladesh", 41539),
]


def _split(pool: list[tuple[str, str, int]], seed: int) -> tuple[list, list]:
    """40% development, 60% confirmation, by a seeded shuffle of the sorted list."""
    items = sorted(pool, key=lambda d: d[2])
    random.Random(seed).shuffle(items)
    k = round(0.4 * len(items))
    return sorted(items[:k], key=lambda d: d[2]), sorted(items[k:], key=lambda d: d[2])


GEN_DEV, GEN_CONF = _split(GENERAL, 2026)
IMB_DEV, IMB_CONF = _split(IMBALANCED, 2027)

SETS: dict[str, list[tuple[str, str, int]]] = {
    "dev": GEN_DEV + IMB_DEV,  # H1-H3 development
    "conf": GEN_CONF + IMB_CONF,  # H1-H3 confirmation
    "ctr_dev": E7_DEV + E7_CONTROL,  # H4 development (seen in E7)
    # H4 confirmation: every fresh dataset with a categorical feature of more than 32 values.
    "ctr_conf": HIGH_CARD_EXTRA + [d for d in IMBALANCED if d[1] in ("RelevantImagesDatasetTEST", "open_payments")],
}


def _csv(set_name: str, arm: str, name: str, fold: int) -> Path:
    return RAW_DIR / set_name / f"{arm}_{name}_f{fold}.csv"


def cmd_run(set_name: str, arms: list[str], shard: int, nshards: int) -> None:
    (RAW_DIR / set_name).mkdir(parents=True, exist_ok=True)
    jobs = [(kind, name, did, fold) for (kind, name, did) in SETS[set_name] for fold in range(5)]
    for i, (kind, name, did, fold) in enumerate(jobs):
        if i % nshards != shard:
            continue
        # D0 first: an arm that changes nothing on this task copies D0's result instead of refitting.
        todo = sorted((a for a in arms if not _csv(set_name, a, name, fold).exists()), key=lambda a: a != "D0")
        if not todo:
            continue
        try:
            X, y, cat_cols, _, _ = load_dataset(kind, did)
            a, b, te = _split_cv(kind, X, y)[fold]
        except Exception as e:  # reported and left out (prereg)
            print(f"FAIL load {name} f{fold}: {type(e).__name__}: {e}", flush=True)
            continue
        for arm in todo:
            params = ARMS[arm](kind, len(a))
            if params is None:
                continue
            d0 = _csv(set_name, "D0", name, fold)
            if arm != "D0" and not params and d0.exists():
                row = next(csv.DictReader(d0.open()))
                row["arm"] = arm
                with _csv(set_name, arm, name, fold).open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerow(row)
                continue
            try:
                pred, trees, secs = _fit(kind, params, X.iloc[a], y[a], X.iloc[b], y[b], X.iloc[te], fold)
                met, auc = eval_metrics(kind, y[te], pred)
            except Exception as e:
                print(f"FAIL fit {name} f{fold} {arm}: {type(e).__name__}: {e}", flush=True)
                continue
            with _csv(set_name, arm, name, fold).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerow({"kind": kind, "dataset": name, "n_eval": len(X), "n_train": len(a),
                            "p_num": X.shape[1] - len(cat_cols), "p_cat": len(cat_cols), "fold": fold, "arm": arm,
                            "metric": f"{met:.12g}", "auc": f"{auc:.12g}", "trees": trees, "secs": f"{secs:.6f}"})
        print(f"DONE {name} f{fold} {todo}", flush=True)


def _rule_metric(df: pd.DataFrame) -> pd.Series:
    """TabArena's error: 1 - AUC for binary tasks, otherwise log-loss / normalised RMSE."""
    return df.apply(lambda r: 1.0 - r.auc if r.kind == "clf" else r.metric, axis=1)


def cmd_summarize(set_name: str, arms: list[str], base: str = "D0") -> None:
    df = pd.concat([pd.read_csv(p) for p in (RAW_DIR / set_name).glob("*.csv")], ignore_index=True)
    df["err"] = _rule_metric(df)
    print(f"{set_name}: error = 1 - AUC (binary), log-loss (multiclass), RMSE / std (regression); negative = better")
    for arm in arms:
        sub = df[df.arm.isin([base, arm])]
        ok = sub.groupby(["dataset", "fold"]).arm.nunique() == 2
        sub = sub.set_index(["dataset", "fold"]).loc[ok[ok].index].reset_index()
        g = sub.groupby(["dataset", "arm"]).agg(err=("err", "mean"), secs=("secs", "sum")).reset_index()
        m = g.pivot(index="dataset", columns="arm", values="err")
        s = g.pivot(index="dataset", columns="arm", values="secs")
        rel = 100.0 * (m[arm] - m[base]) / m[base]
        nz = rel[rel != 0]
        pb = float(wilcoxon(nz, alternative="less").pvalue) if len(nz) > 1 else 1.0
        pw = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 1 else 1.0
        print(f"  {arm:12s} n={len(rel)} wins {int((rel < 0).sum())}, ties {int((rel == 0).sum())}, "
              f"losses {int((rel > 0).sum())}; median {rel.median():+.3f}%, mean {rel.mean():+.2f}%, "
              f"worst {rel.max():+.2f}%; p_better {pb:.3f}, p_worse {pw:.3f}; "
              f"time {s[arm].sum() / s[base].sum():.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--set", choices=list(SETS), required=True)
    r.add_argument("--arms", nargs="+", required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    sm = sp.add_parser("summarize")
    sm.add_argument("--set", choices=list(SETS), required=True)
    sm.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.set, a.arms, a.shard, a.nshards)
    elif a.cmd == "summarize":
        cmd_summarize(a.set, a.arms)
