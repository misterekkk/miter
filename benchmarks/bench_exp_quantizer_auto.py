"""Confirmatory benchmark: QuantizerMode::Auto vs Linear/Quantile (`exp-quantizer-auto`).

Pre-registered in `benchmarks/results/exp_quantizer_auto_prereg.md`.

Usage:
    .venv/bin/python benchmarks/bench_exp_quantizer_auto.py pilot
    .venv/bin/python benchmarks/bench_exp_quantizer_auto.py run --shard 0 --nshards 3
    .venv/bin/python benchmarks/bench_exp_quantizer_auto.py summarize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from bench_exp_native_ctr_v2 import _bootstrap_median_ci, _holm_bonferroni
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "results" / "exp_quantizer_auto_prereg.md"
RAW_DIR = HERE / "results" / "raw" / "exp_quantizer_auto"
SUMMARY_PATH = HERE / "results" / "exp_quantizer_auto_summary.md"
OPENML_CACHE = HERE / "results" / "raw" / "openml_cache_pandas3"
openml.config.set_root_cache_directory(str(OPENML_CACHE))

# (role, kind, name, did)
CONFIRMATORY_DATASETS: list[tuple[str, str, str, int]] = [
    ("conf", "reg", "houses", 46934),
    ("conf", "reg", "miami_housing", 46942),
    ("conf", "reg", "physiochemical_protein", 46949),
    ("conf", "reg", "superconductivity", 46961),
    ("conf", "reg", "wine_quality", 46964),
    ("conf", "reg", "airfoil_self_noise", 46904),
    ("conf", "reg", "concrete_compressive_strength", 46917),
    ("conf", "reg", "QSAR_fish_toxicity", 46954),
    ("conf", "clf", "APSFailure", 46908),
    ("conf", "clf", "GiveMeSomeCredit", 46929),
    ("conf", "clf", "polish_companies_bankruptcy", 46950),
    ("conf", "clf", "heloc", 46932),
    ("conf", "clf", "jm1", 46979),
    ("conf", "clf", "qsar-biodeg", 46952),
    ("conf", "clf", "credit_card_clients_default", 46919),
    ("conf", "clf", "coil2000", 46916),
]

STRESS_DATASETS: list[tuple[str, str, str, int]] = [
    ("stress", "clf", "taiwanese_bankruptcy_prediction", 46962),
    ("stress", "mc", "SDSS17", 46955),
]

ALL_DATASETS = CONFIRMATORY_DATASETS + STRESS_DATASETS
VARIANTS = ["LIN", "AUTO", "QUANT", "S6_LIN", "S6_AUTO"]
FIELDS = [
    "role",
    "kind",
    "dataset",
    "did",
    "n_eval",
    "p_num",
    "p_cat",
    "fold",
    "variant",
    "metric",
    "auc",
    "trees",
    "n_quantile_features",
    "secs",
]


def load_dataset(kind: str, did: int) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    t = d.default_target_attribute
    X_df, y_s, cat_ind, _ = d.get_data(target=t, dataset_format="dataframe")
    X = X_df.copy()
    p_cat = 0
    for col, is_cat in zip(X.columns, cat_ind, strict=True):
        if (
            is_cat
            or isinstance(X[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[col])
            or pd.api.types.is_object_dtype(X[col])
        ):
            X[col] = X[col].astype("category")
            p_cat += 1
        else:
            X[col] = X[col].astype(np.float64)
    X.columns = [str(c) for c in X.columns]
    p_num = X.shape[1] - p_cat

    if kind == "reg":
        y = y_s.astype("float64").to_numpy()
    else:
        y = y_s.astype("category").cat.codes.to_numpy(dtype=np.int64)

    n = len(X)
    if n > 25000:
        if kind == "reg":
            rng = np.random.default_rng(42)
            pick = np.sort(rng.choice(n, size=25000, replace=False))
        else:
            _, pick = train_test_split(np.arange(n), test_size=25000, random_state=42, stratify=y)
            pick = np.sort(pick)
        X = X.iloc[pick].reset_index(drop=True)
        y = y[pick]
    else:
        X = X.reset_index(drop=True)
    return X, y, p_num, p_cat


def eval_metrics(kind: str, y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if kind == "reg":
        return float(np.sqrt(np.mean((pred - y_true) ** 2)) / np.std(y_true)), float("nan")
    if kind == "clf":
        p = np.clip(pred, 1e-15, 1.0 - 1e-15)
        logloss = float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
        auc = float(roc_auc_score(y_true, pred))
        return logloss, auc
    probs = np.clip(pred[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(probs))), float("nan")


def _split_cv(
    kind: str, X: pd.DataFrame, y: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    splitter = (
        KFold(n_splits=5, shuffle=True, random_state=42)
        if kind == "reg"
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_idx, te_idx in splitter.split(X, None if kind == "reg" else y):
        strat = None if kind == "reg" else y[tr_idx]
        a, b = train_test_split(tr_idx, test_size=0.2, random_state=0, stratify=strat)
        out.append((a, b, te_idx))
    return out


def fit_predict_variant(
    kind: str,
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    te: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, int, int, float]:
    import miter

    t0 = time.perf_counter()
    X_tr, X_val, X_te = X.iloc[a], X.iloc[b], X.iloc[te]
    y_tr, y_val = y[a], y[b]
    qmode = "linear" if variant in ("LIN", "S6_LIN") else ("quantile" if variant == "QUANT" else "auto")
    is_sym = variant.startswith("S6")
    kw: dict[str, Any] = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "grow_policy": "symmetric" if is_sym else "leafwise",
        "num_leaves": 31,
        "max_depth": 6 if is_sym else -1,
        "min_data_in_leaf": 1 if is_sym else 20,
        "lambda_l2": 3.0 if is_sym else 1.0,
        "m_mult": 2,
        "k_proj": 0,
        "max_bin": 255,
        "quantizer": qmode,
        "early_stopping_rounds": 50,
        "n_jobs": 1,
        "random_state": fold,
    }
    cls = miter.MiterRegressor if kind == "reg" else miter.MiterClassifier
    m = cls(**kw).fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    if kind == "reg":
        pred = m.predict(X_te)
    elif kind == "clf":
        pred = m.predict_proba(X_te)[:, 1]
    else:
        pred = m.predict_proba(X_te)
    trees = int(m.best_iteration_)
    nq = int(m._model.n_quantile_features)
    return pred, trees, nq, time.perf_counter() - t0


def cmd_pilot() -> None:
    for role, kind, name, did in [
        ("conf", "reg", "concrete_compressive_strength", 46917),
        ("conf", "clf", "GiveMeSomeCredit", 46929),
        ("stress", "clf", "taiwanese_bankruptcy_prediction", 46962),
    ]:
        X, y, p_num, p_cat = load_dataset(kind, did)
        a, b, te = _split_cv(kind, X, y)[0]
        print(
            f"=== PILOT {role} {name} did={did} n={len(X)} p_num={p_num} p_cat={p_cat} ===",
            flush=True,
        )
        for v in VARIANTS:
            pred, trees, nq, secs = fit_predict_variant(kind, v, X, y, a, b, te, 0)
            met, auc = eval_metrics(kind, y[te], pred)
            auc_s = f"{auc:.5f}" if not math.isnan(auc) else "  nan  "
            print(
                f"  {v:8s}: metric={met:.5f} auc={auc_s} trees={trees:4d} nq={nq:2d} secs={secs:.2f}s",
                flush=True,
            )


def cmd_run(shard: int, nshards: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RAW_DIR / f"shard_{shard}_of_{nshards}.csv"
    done: set[tuple[str, int, str]] = set()
    if raw_csv.exists():
        with raw_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["dataset"], int(row["fold"]), row["variant"]))

    write_header = not raw_csv.exists() or raw_csv.stat().st_size == 0
    selected = [d for idx, d in enumerate(ALL_DATASETS) if idx % nshards == shard]

    with raw_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
            fh.flush()

        for role, kind, name, did in selected:
            X, y, p_num, p_cat = load_dataset(kind, did)
            splits = _split_cv(kind, X, y)
            for fold, (a, b, te) in enumerate(splits):
                for v in VARIANTS:
                    if (name, fold, v) in done:
                        continue
                    pred, trees, nq, secs = fit_predict_variant(kind, v, X, y, a, b, te, fold)
                    met, auc = eval_metrics(kind, y[te], pred)
                    writer.writerow(
                        {
                            "role": role,
                            "kind": kind,
                            "dataset": name,
                            "did": did,
                            "n_eval": len(X),
                            "p_num": p_num,
                            "p_cat": p_cat,
                            "fold": fold,
                            "variant": v,
                            "metric": repr(met),
                            "auc": "" if math.isnan(auc) else repr(auc),
                            "trees": trees,
                            "n_quantile_features": nq,
                            "secs": f"{secs:.3f}",
                        }
                    )
                    fh.flush()
                    print(
                        f"[{shard}/{nshards}] {role:6s} {kind:3s} {name:30s} f={fold} {v:7s}: "
                        f"{met:.5f} trees={trees:4d} nq={nq:2d} {secs:5.2f}s",
                        flush=True,
                    )


def _paired_stats(
    names: list[str], agg: dict[tuple[str, str], float], cand: str, base: str
) -> dict[str, Any]:
    deltas = [(agg[(n, cand)] - agg[(n, base)]) / agg[(n, base)] * 100.0 for n in names]
    nz = [d for d in deltas if abs(d) > 1e-12]
    p_val = 1.0 if not nz else float(wilcoxon(nz, alternative="two-sided").pvalue)
    ci_lo, ci_hi = _bootstrap_median_ci(deltas, n_boot=10000, seed=2026)
    return {
        "deltas": deltas,
        "mean": float(statistics.fmean(deltas)),
        "med": float(statistics.median(deltas)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "wins": sum(1 for d in deltas if d < -1e-12),
        "ties": sum(1 for d in deltas if abs(d) <= 1e-12),
        "worst": float(max(deltas)),
        "p_raw": p_val,
    }


def cmd_summarize() -> None:
    sha = hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    for p in sorted(RAW_DIR.glob("shard_*.csv")):
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "role": row["role"],
                        "kind": row["kind"],
                        "dataset": row["dataset"],
                        "did": int(row["did"]),
                        "n_eval": int(row["n_eval"]),
                        "p_num": int(row["p_num"]),
                        "p_cat": int(row["p_cat"]),
                        "fold": int(row["fold"]),
                        "variant": row["variant"],
                        "metric": float(row["metric"]),
                        "auc": float("nan") if row["auc"] == "" else float(row["auc"]),
                        "trees": int(row["trees"]),
                        "n_quantile_features": int(row["n_quantile_features"]),
                        "secs": float(row["secs"]),
                    }
                )
    if not rows:
        raise SystemExit(f"No rows found in {RAW_DIR}")

    agg: dict[tuple[str, str], float] = {}
    agg_secs: dict[tuple[str, str], float] = {}
    agg_nq: dict[tuple[str, str], float] = {}
    agg_one_m_auc: dict[tuple[str, str], float] = {}
    ds_meta: dict[str, dict[str, Any]] = {}

    for _, kind, name, _ in ALL_DATASETS:
        sub_all = [r for r in rows if r["dataset"] == name]
        if not sub_all:
            raise SystemExit(f"Missing dataset {name}")
        ds_meta[name] = sub_all[0]
        for v in VARIANTS:
            sub = [r for r in sub_all if r["variant"] == v]
            if len(sub) != 5:
                raise SystemExit(f"Incomplete runs for {name}/{v}: expected 5 folds, got {len(sub)}")
            agg[(name, v)] = statistics.fmean(r["metric"] for r in sub)
            agg_secs[(name, v)] = statistics.fmean(r["secs"] for r in sub)
            agg_nq[(name, v)] = statistics.fmean(r["n_quantile_features"] for r in sub)
            if kind == "clf":
                agg_one_m_auc[(name, v)] = 1.0 - statistics.fmean(r["auc"] for r in sub)

    conf_names = [n for _, _, n, _ in CONFIRMATORY_DATASETS]
    stress_checks = {}
    for cand, base in [("AUTO", "LIN"), ("S6_AUTO", "S6_LIN")]:
        tb_ll = (agg[("taiwanese_bankruptcy_prediction", cand)] - agg[("taiwanese_bankruptcy_prediction", base)]) / agg[("taiwanese_bankruptcy_prediction", base)] * 100.0
        tb_auc = (agg_one_m_auc[("taiwanese_bankruptcy_prediction", cand)] - agg_one_m_auc[("taiwanese_bankruptcy_prediction", base)]) / agg_one_m_auc[("taiwanese_bankruptcy_prediction", base)] * 100.0
        sd_ll = (agg[("SDSS17", cand)] - agg[("SDSS17", base)]) / agg[("SDSS17", base)] * 100.0
        stress_checks[cand] = (tb_ll, tb_auc, sd_ll, tb_ll < -3.0 and tb_auc < -3.0 and sd_ll < -3.0)

    specs = [("AUTO vs LIN", "AUTO", "LIN"), ("S6_AUTO vs S6_LIN", "S6_AUTO", "S6_LIN")]
    raw_ps: list[float] = []
    stats_list: list[dict[str, Any]] = []
    for _, cand, base in specs:
        st = _paired_stats(conf_names, agg, cand, base)
        raw_ps.append(st["p_raw"])
        st["t_ratio"] = float(
            statistics.median(agg_secs[(n, cand)] / max(agg_secs[(n, base)], 1e-6) for n in conf_names)
        )
        trig_names = [n for n in conf_names if agg_nq[(n, cand)] >= 1.0]
        st["trig_n"] = len(trig_names)
        if trig_names:
            tst = _paired_stats(trig_names, agg, cand, base)
            non_ties = len(trig_names) - tst["ties"]
            st["trig_med"] = tst["med"]
            st["trig_wins"] = tst["wins"]
            st["trig_nonties"] = non_ties
            st["trig_p"] = tst["p_raw"]
        else:
            st["trig_med"] = 0.0
            st["trig_wins"] = 0
            st["trig_nonties"] = 0
            st["trig_p"] = 1.0
        stats_list.append(st)

    holm = _holm_bonferroni(raw_ps)
    lines = [
        "# Confirmatory Summary: Adaptive Quantizer Mode (`exp-quantizer-auto`)",
        "",
        f"- Pre-registration: `benchmarks/results/exp_quantizer_auto_prereg.md` (SHA-256: `{sha}`)",
        "- Scope: 16 fresh confirmatory datasets + 2 diagnostic stress-test datasets × 5 folds (`n_jobs=1`).",
        "",
        "## 1. Confirmatory Scorecard (16 Fresh Datasets; Holm-Bonferroni `m=2`, 10k bootstrap CI)",
        "",
        "| Comparison | Mean % | Median % | 95% Boot CI (Median) | Wins / Ties | Worst % | Time vs Lin | Trig Subgroup (nq>=1) | Stress >3%? | Verdict |",
        "|---|---:|---:|---|---|---:|---:|---|---|---|",
    ]
    for (label, cand, _), st, ph in zip(specs, stats_list, holm, strict=True):
        stress_ok = stress_checks[cand][3]
        trig_rate = (st["trig_wins"] / st["trig_nonties"]) if st["trig_nonties"] > 0 else 0.0
        trig_ok = st["trig_n"] > 0 and st["trig_med"] < -0.25 and trig_rate >= 0.65
        passed = (
            st["mean"] <= 0.0
            and st["med"] <= 0.0
            and trig_ok
            and st["ci_hi"] <= 0.10
            and st["worst"] <= 0.75
            and st["t_ratio"] <= 1.10
            and stress_ok
        )
        verdict = "**PASS**" if passed else "FAIL"
        lines.append(
            f"| {label} | {st['mean']:+.2f}% | {st['med']:+.2f}% | "
            f"[{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%] | {st['wins']}/16 ({st['ties']}t) | "
            f"{st['worst']:+.2f}% | {st['t_ratio']:.2f}x | "
            f"{st['trig_wins']}/{st['trig_nonties']} ({st['trig_n']} ds, med {st['trig_med']:+.2f}%, p={st['trig_p']:.3g}) | "
            f"{'YES' if stress_ok else 'NO'} (p_H={ph:.3g}) | {verdict} |"
        )

    lines += [
        "",
        "## 2. Diagnostic Stress-Test Verification (5-fold mean)",
        "",
        "| dataset | LIN | AUTO | QUANT | S6_LIN | S6_AUTO | AUTO vs LIN | S6_AUTO vs S6_LIN | nq(AUTO) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, kind, name, _ in STRESS_DATASETS:
        lines.append(
            f"| `{name}` ({kind}) | {agg[(name, 'LIN')]:.5f} | {agg[(name, 'AUTO')]:.5f} | "
            f"{agg[(name, 'QUANT')]:.5f} | {agg[(name, 'S6_LIN')]:.5f} | {agg[(name, 'S6_AUTO')]:.5f} | "
            f"{(agg[(name, 'AUTO')] - agg[(name, 'LIN')]) / agg[(name, 'LIN')] * 100.0:+.2f}% | "
            f"{(agg[(name, 'S6_AUTO')] - agg[(name, 'S6_LIN')]) / agg[(name, 'S6_LIN')] * 100.0:+.2f}% | "
            f"{agg_nq[(name, 'AUTO')]:.1f}/{ds_meta[name]['p_num']} |"
        )
        if kind == "clf":
            lines.append(
                f"| `{name}` (1-AUC) | {agg_one_m_auc[(name, 'LIN')]:.5f} | {agg_one_m_auc[(name, 'AUTO')]:.5f} | "
                f"{agg_one_m_auc[(name, 'QUANT')]:.5f} | {agg_one_m_auc[(name, 'S6_LIN')]:.5f} | {agg_one_m_auc[(name, 'S6_AUTO')]:.5f} | "
                f"{(agg_one_m_auc[(name, 'AUTO')] - agg_one_m_auc[(name, 'LIN')]) / agg_one_m_auc[(name, 'LIN')] * 100.0:+.2f}% | "
                f"{(agg_one_m_auc[(name, 'S6_AUTO')] - agg_one_m_auc[(name, 'S6_LIN')]) / agg_one_m_auc[(name, 'S6_LIN')] * 100.0:+.2f}% | - |"
            )

    q_vs_l = _paired_stats(conf_names, agg, "QUANT", "LIN")
    a_vs_q = _paired_stats(conf_names, agg, "AUTO", "QUANT")
    lines += [
        "",
        "## 3. Exploratory Leafwise Quantizer Comparisons (`N=16`)",
        "",
        f"- `QUANT vs LIN`: mean {q_vs_l['mean']:+.2f}%, median {q_vs_l['med']:+.2f}% [{q_vs_l['ci_lo']:+.2f}%, {q_vs_l['ci_hi']:+.2f}%], wins {q_vs_l['wins']}/16, worst {q_vs_l['worst']:+.2f}%, p={q_vs_l['p_raw']:.4g}",
        f"- `AUTO vs QUANT`: mean {a_vs_q['mean']:+.2f}%, median {a_vs_q['med']:+.2f}% [{a_vs_q['ci_lo']:+.2f}%, {a_vs_q['ci_hi']:+.2f}%], wins {a_vs_q['wins']}/16, worst {a_vs_q['worst']:+.2f}%, p={a_vs_q['p_raw']:.4g}",
        "",
        "## 4. Per-dataset Primary Metric & Triggered Quantile Features (5-fold mean)",
        "",
        "| kind | dataset | n | p_num | nq | LIN | AUTO | QUANT | S6_LIN | S6_AUTO | AUTO vs LIN | S6_AUTO vs S6_LIN | LIN t | AUTO t |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, kind, name, _ in CONFIRMATORY_DATASETS:
        m = ds_meta[name]
        lines.append(
            f"| {kind} | `{name}` | {m['n_eval']} | {m['p_num']} | {agg_nq[(name, 'AUTO')]:.1f} | "
            f"{agg[(name, 'LIN')]:.5f} | {agg[(name, 'AUTO')]:.5f} | {agg[(name, 'QUANT')]:.5f} | "
            f"{agg[(name, 'S6_LIN')]:.5f} | {agg[(name, 'S6_AUTO')]:.5f} | "
            f"{(agg[(name, 'AUTO')] - agg[(name, 'LIN')]) / agg[(name, 'LIN')] * 100.0:+.2f}% | "
            f"{(agg[(name, 'S6_AUTO')] - agg[(name, 'S6_LIN')]) / agg[(name, 'S6_LIN')] * 100.0:+.2f}% | "
            f"{agg_secs[(name, 'LIN')]:.2f}s | {agg_secs[(name, 'AUTO')]:.2f}s |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sp = sub.add_parser("run")
    sp.add_argument("--shard", type=int, default=0)
    sp.add_argument("--nshards", type=int, default=1)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "pilot":
        cmd_pilot()
    elif args.cmd == "run":
        cmd_run(args.shard, args.nshards)
    elif args.cmd == "summarize":
        cmd_summarize()


if __name__ == "__main__":
    main()
