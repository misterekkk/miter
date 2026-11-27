"""miter's v1.0 default configuration (symmetric depth 6, m_mult 4: `MiterRegressor()` with no
arguments) on all 51 TabArena `lite` tasks, scored against the `(default)` rows of CatBoost,
LightGBM and XGBoost.

One configuration, no tuning: the `{}` config of the wrapper, i.e. the library defaults. The
summary also lists the single configurations of the V12 run (from its cache) for comparison.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_default_51.py run --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_default_51.py summarize
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_v12_default_candidates as cand  # also sets the TabArena caches
import run_tabarena_ensemble as ens
from miter.tabarena import MiterModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import CustomAGConfigGenerator

EXP_ROOT = ens.ROOT / "default_v1_lite"
SUMMARY_PATH = HERE.parent / "results" / "tabarena_default_v1_51_summary.md"


class MiterDefaultModel(MiterModel):
    ag_key = "MITER_DEFAULT_V1"
    ag_name = "MiterDefaultV1"


def _spec() -> Any:
    cg = CustomAGConfigGenerator(
        model_cls=MiterDefaultModel,
        name="MITER_DEFAULT_V1",
        manual_configs=[{}],
        search_space_func=lambda num_random_configs=0: [],
    )
    return (cg, 0)


def _tasks() -> list[str]:
    tm = TabArenaContext().task_metadata
    return sorted(tm["name"].tolist())


def cmd_run(shard: int, nshards: int) -> None:
    tm = TabArenaContext().task_metadata.set_index("name")
    order = sorted(_tasks(), key=lambda d: -float(tm.loc[d].num_instances) * float(tm.loc[d].n_features) ** 0.5)
    mine = [d for i, d in enumerate(order) if i % nshards == shard]
    print(f"Shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        ctx = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=[_spec()]).build_experiments()
        res = ctx.build_and_run_jobs(
            exps,
            expname=str(EXP_ROOT / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        print(f"DONE dataset={name} jobs={len(res)} wall={time.time() - t0:.1f}s", flush=True)


def _default_rows() -> pd.DataFrame:
    rows = []
    for task_dir in sorted(p for p in EXP_ROOT.iterdir() if p.is_dir()):
        hits = list((task_dir / "data" / "MiterDefaultV1_c1_default_BAG_L1").glob("*/0_0/results.pkl"))
        if not hits:
            print(f"missing {task_dir.name}", flush=True)
            continue
        d = cand._load(hits[0])
        ct = "[New] MITER_DEFAULT_V1"
        rows.append(
            {
                "dataset": task_dir.name, "fold": 0, "method": f"{ct} (default)",
                "metric_error": float(d["metric_error"]), "time_train_s": float(d["time_train_s"]),
                "time_infer_s": float(d["time_infer_s"]), "metric": d["metric"],
                "problem_type": d["problem_type"], "metric_error_val": float(d["metric_error_val"]),
                "method_type": "config", "method_subtype": "default", "config_type": ct,
                "imputed": False, "impute_method": float("nan"), "method_metadata": float("nan"),
                "ta_name": ct, "ta_suite": ct,
            }
        )
    return pd.DataFrame(rows)


def cmd_summarize() -> None:
    new = pd.concat([_default_rows(), cand._rows()], ignore_index=True)
    names = sorted(set(new["dataset"]))
    ctx = TabArenaContext()
    tm = ctx.task_metadata.set_index("name")
    ta = ctx.load_results(methods=["LightGBM", "CatBoost", "XGBoost"], download_results=False)
    lines = [
        "# miter v1.0 default on TabArena `lite` (51 tasks)",
        "",
        "`MITER_DEFAULT_V1` is `MiterRegressor()` / `MiterClassifier()` with no arguments (symmetric "
        "trees of depth 6, `m_mult=4`, `min_data_in_leaf=1`, `lambda_l2=3`, linear quantizer, no CTR): "
        "one configuration, no tuning, 8-fold bagging, fold 0. The `MITER_<name>` rows are the single "
        "configurations of the V12 run (from its cache; `LW31` is the default before v1.0).",
        "",
    ]
    m = "[New] MITER_DEFAULT_V1 (default)"
    for tag, subset in (
        ("all 51", names),
        ("regression", [n for n in names if tm.loc[n].problem_type == "regression"]),
        ("binary", [n for n in names if tm.loc[n].problem_type == "binary"]),
        ("multiclass", [n for n in names if tm.loc[n].problem_type == "multiclass"]),
    ):
        out = ens.ROOT / "default_v1_cmp" / tag.replace(" ", "_")
        lb, per = ctx.compare(
            output_dir=str(out), ta_results=ta, new_results=new[new["dataset"].isin(subset)].copy(),
            datasets=subset, subset="lite", return_results=True, plot=False,
            fillna=None, calibration_method=None,
        )
        out.mkdir(parents=True, exist_ok=True)
        lb.to_csv(out / "leaderboard.csv", index=False)
        per.to_csv(out / "per_split.csv", index=False)
        lb = lb[lb["method"].str.contains(r"\(default\)|\(tuned", regex=True)]
        lines += [f"## gbdt_only, {tag} ({len(subset)} tasks)", "",
                  "| # | method | elo | elo+ | elo- | rank | winrate |", "|---:|---|---:|---:|---:|---:|---:|"]
        for i, (_, r) in enumerate(lb.iterrows(), start=1):
            lines.append(f"| {i} | {r['method']} | {r['elo']:.4g} | {r['elo+']:.3g} | {r['elo-']:.3g} | "
                         f"{r['rank']:.3g} | {r['winrate']:.3g} |")
        piv = per.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
        lines += ["", "| miter | baseline | n | wins | median | mean | Wilcoxon p |", "|---|---|---:|---:|---:|---:|---:|"]
        for b in ("CAT (default)", "GBM (default)", "XGB (default)", "CAT (tuned)", "[New] MITER_LW31 (default)"):
            if b in piv.columns and m in piv.columns:
                lines.append(cand._paired(piv, m, b, subset))
        lines.append("")
    t = new.groupby("method")["time_train_s"].median()
    lines += ["## Median train time per task (s, 8-fold bag, this machine)", "", "| method | train |", "|---|---:|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in t.items()]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--nshards", type=int, required=True)
    sp.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.shard, a.nshards)
    else:
        cmd_summarize()
