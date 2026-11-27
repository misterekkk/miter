"""Exploratory: each manual MITER_V12_T25 config scored as a stand-alone "default" on TabArena lite (51 tasks).

No training: every config was already run (8-fold bagging, fold 0) inside the V12_T25 experiment, and its
per-config result is cached under `ens_v12_t25_lite/<task>/data/MiterV12T25_c<k>_default_BAG_L1/`.
Each config becomes its own method `[New] MITER_<name> (default)` and is compared with the
`(default)` rows of CatBoost, LightGBM and XGBoost, plus paired Wilcoxon tests.

Caveat: the candidate configs were written after earlier TabArena runs, so this is not an
out-of-sample selection. Out-of-sample evidence for the defaults is in `exp_scaling_defaults_summary.md`
and `exp_external_holdout_summary.md`.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/analyze_v12_default_candidates.py
"""

from __future__ import annotations

# ruff: noqa: E402
import gzip
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tabarena_ensemble as ens  # sets TABARENA_CACHE and the OpenML cache dir
from scipy.stats import wilcoxon
from tabarena.contexts import TabArenaContext

ROOT = ens.ROOT / "ens_v12_t25_lite"
OUT = ens.ROOT / "v12_default_candidates_cmp"
SUMMARY_PATH = HERE.parent / "results" / "tabarena_v12_default_candidates_summary.md"

# The manual configs of run_tabarena_v12_51._v12_t25_spec, in order (c1..c6).
CANDIDATES = {
    "c1": ("LW31", "current default (level-wise, 31 leaves, linear quantizer, no CTR)"),
    "c2": ("S6", "grow_policy=symmetric, max_depth=6, min_data_in_leaf=1, lambda_l2=3"),
    "c3": ("S6_QAUTO", "S6 + quantizer=auto"),
    "c4": ("S6_QAUTO_CTR", "S6 + quantizer=auto + max_ctr_complexity=auto + ctr_n_permutations=4"),
    "c5": ("ACCURATE", "preset=accurate"),
    "c6": ("S6_M4_QQ_CTR", "S6 + m_mult=4 + quantizer=quantile + CTR auto, 4 permutations"),
}


def _load(path: Path) -> dict:
    try:
        with gzip.open(path) as f:
            return pickle.load(f)
    except OSError:
        with open(path, "rb") as f:
            return pickle.load(f)


def _rows() -> pd.DataFrame:
    rows = []
    for task_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for ck, (name, _) in CANDIDATES.items():
            hits = list((task_dir / "data" / f"MiterV12T25_{ck}_default_BAG_L1").glob("*/0_0/results.pkl"))
            if not hits:
                print(f"missing {task_dir.name} {ck}", flush=True)
                continue
            d = _load(hits[0])
            ct = f"[New] MITER_{name}"
            rows.append(
                {
                    "dataset": task_dir.name,
                    "fold": 0,
                    "method": f"{ct} (default)",
                    "metric_error": float(d["metric_error"]),
                    "time_train_s": float(d["time_train_s"]),
                    "time_infer_s": float(d["time_infer_s"]),
                    "metric": d["metric"],
                    "problem_type": d["problem_type"],
                    "metric_error_val": float(d["metric_error_val"]),
                    "method_type": "config",
                    "method_subtype": "default",
                    "config_type": ct,
                    "imputed": False,
                    "impute_method": np.nan,
                    "method_metadata": np.nan,
                    "ta_name": ct,
                    "ta_suite": ct,
                }
            )
    return pd.DataFrame(rows)


def _paired(piv: pd.DataFrame, a: str, b: str, names: list[str]) -> str:
    d = piv.loc[[n for n in names if n in piv.index], [a, b]].dropna()
    r = 100.0 * (d[a] - d[b]) / d[b]
    p = float(wilcoxon(r).pvalue) if (r != 0).any() else 1.0
    return (
        f"| {a.replace('[New] ', '')} | {b} | {len(r)} | {int((r < 0).sum())}/{len(r)} | "
        f"{r.median():+.2f}% | {r.mean():+.2f}% | {p:.4f} |"
    )


def main() -> None:
    new = _rows()
    names = sorted(new["dataset"].unique())
    print(f"{len(new)} rows over {len(names)} tasks", flush=True)
    ctx = TabArenaContext()
    tm = ctx.task_metadata.set_index("name")
    ta = ctx.load_results(methods=["LightGBM", "CatBoost", "XGBoost"], download_results=False)
    lines = [
        "# MITER_V12_T25 manual configs as stand-alone defaults, TabArena `lite` (51 tasks)",
        "",
        "Exploratory, from the V12_T25 cache (no new training). Each config is one bagged model "
        "(8 folds, fold 0), the same protocol as the `(default)` rows of the baselines. "
        "The candidates were written after earlier TabArena runs, so this is not out-of-sample.",
        "",
        "| config | settings |",
        "|---|---|",
    ] + [f"| `{n}` | {s} |" for n, s in CANDIDATES.values()] + [""]

    for tag, subset in (
        ("all 51", names),
        ("regression", [n for n in names if tm.loc[n].problem_type == "regression"]),
        ("binary", [n for n in names if tm.loc[n].problem_type == "binary"]),
        ("multiclass", [n for n in names if tm.loc[n].problem_type == "multiclass"]),
    ):
        out = OUT / tag.replace(" ", "_")
        lb, per = ctx.compare(
            output_dir=str(out),
            ta_results=ta,
            new_results=new[new["dataset"].isin(subset)].copy(),
            datasets=subset,
            subset="lite",
            return_results=True,
            plot=False,
            fillna=None,
            calibration_method=None,
        )
        out.mkdir(parents=True, exist_ok=True)
        lb.to_csv(out / "leaderboard.csv", index=False)
        per.to_csv(out / "per_split.csv", index=False)
        lb = lb[lb["method"].str.contains(r"\(default\)|CAT \(tuned|GBM \(tuned|XGB \(tuned", regex=True)]
        lines += [
            f"## gbdt_only, {tag} ({len(subset)} tasks)",
            "",
            "| # | method | elo | elo+ | elo- | rank | winrate |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for i, (_, r) in enumerate(lb.iterrows(), start=1):
            lines.append(
                f"| {i} | {r['method']} | {r['elo']:.4g} | {r['elo+']:.3g} | {r['elo-']:.3g} | "
                f"{r['rank']:.3g} | {r['winrate']:.3g} |"
            )
        piv = per.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
        lines += [
            "",
            "| miter config | baseline | n | wins | median | mean | Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name, _ in CANDIDATES.values():
            m = f"[New] MITER_{name} (default)"
            for b in ("CAT (default)", "GBM (default)", "XGB (default)", "[New] MITER_LW31 (default)"):
                if b != m and b in piv.columns and m in piv.columns:
                    lines.append(_paired(piv, m, b, subset))
        lines.append("")

    t = new.groupby("method")[["time_train_s", "time_infer_s"]].median()
    lines += ["## Median time per task (s, 8-fold bag)", "", "| method | train | infer |", "|---|---:|---:|"]
    lines += [f"| {m} | {r.time_train_s:.1f} | {r.time_infer_s:.2f} |" for m, r in t.iterrows()]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
