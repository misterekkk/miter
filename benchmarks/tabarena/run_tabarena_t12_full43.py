"""MITER_T12 + MITER_BS6 on all TabArena `lite` regression + binary tasks (43); pre-registered in
benchmarks/results/tabarena_t12_full43_prereg.md. Reuses the cache of run_tabarena_ensemble.py.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_t12_full43.py run --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_t12_full43.py summarize
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_tabarena_ensemble as ens  # sets TABARENA_CACHE and the OpenML cache dir
from scipy.stats import wilcoxon
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

SUMMARY_PATH = HERE.parent / "results" / "tabarena_t12_full43_summary.md"
EXP_ROOT = ens.ROOT / "ens_lite"  # same directory: the 20 finished tasks are cache hits
T12 = "[New] MITER_T12 (tuned + ensemble)"
CAT = "CAT (tuned + ensemble)"


def _tasks() -> tuple[list[str], list[str]]:
    tm = TabArenaContext().task_metadata
    reg = sorted(tm[tm.problem_type == "regression"]["name"].tolist())
    binary = sorted(tm[tm.problem_type == "binary"]["name"].tolist())
    return reg, binary


def _models() -> list:
    return [ens._bs6_spec(), ens._t12_spec()]


def _order(names: list[str]) -> list[str]:
    """Heaviest first (rows * sqrt(features)) so shards finish at similar times."""
    tm = TabArenaContext().task_metadata.set_index("name")
    return sorted(
        names,
        key=lambda d: (
            -float(tm.loc[d].num_instances) * float(np.sqrt(tm.loc[d].n_features))
        ),
    )


def cmd_run(shard: int, nshards: int) -> None:
    reg, binary = _tasks()
    order = _order(reg + binary)
    mine = [d for i, d in enumerate(order) if i % nshards == shard]
    print(f"Shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        ctx = (
            TabArenaContext()
        )  # fresh context per task (it narrows its task list after a run)
        exps = TabArenaV0pt1ExperimentBundle(models=_models()).build_experiments()
        res = ctx.build_and_run_jobs(
            exps,
            expname=str(EXP_ROOT / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        print(
            f"DONE dataset={name} jobs={len(res)} wall={time.time() - t0:.1f}s",
            flush=True,
        )


def _new_results(names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        c = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=_models()).build_experiments()
        c.build_and_run_jobs(
            exps,
            expname=str(EXP_ROOT / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        df = c._registered_new_results()
        assert df is not None, name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _leaderboard(
    new: pd.DataFrame, names: list[str], gbdt_only: bool, tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = TabArenaContext()
    methods = (
        ["LightGBM", "CatBoost", "XGBoost"]
        if gbdt_only
        else list(dict.fromkeys(ctx.methods))
    )
    ta = ctx.load_results(methods=methods, download_results=False)
    out = ens.ROOT / "t12_full43_cmp" / tag
    lb, per = ctx.compare(
        output_dir=str(out),
        ta_results=ta,
        new_results=new[new["dataset"].isin(names)].copy(),
        datasets=names,
        subset="lite",
        return_results=True,
        plot=False,
        **({"fillna": None, "calibration_method": None} if gbdt_only else {}),
    )
    out.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out / "leaderboard.csv", index=False)
    per.to_csv(out / "per_split.csv", index=False)
    return lb, per


def _lb_lines(lb: pd.DataFrame, title: str, keep_all: bool) -> list[str]:
    cols = [
        c
        for c in ("elo", "elo+", "elo-", "rank", "winrate", "improvability", "imputed")
        if c in lb.columns
    ]
    lines = [
        f"### {title} (`{len(lb)}` methods)",
        "",
        "| # | method | " + " | ".join(cols) + " |",
        "|---:|---|" + "---:|" * len(cols),
    ]
    for i, (_, r) in enumerate(lb.iterrows(), start=1):
        m = str(r["method"])
        if (
            keep_all
            or "[New]" in m
            or i <= 10
            or any(k in m for k in ("CAT (", "GBM (", "XGB ("))
        ):
            lines.append(
                f"| {i} | {m} | " + " | ".join(f"{r[c]:.4g}" for c in cols) + " |"
            )
    return lines + [""]


def _paired(per: pd.DataFrame, names: list[str], title: str) -> list[str]:
    piv = per.pivot_table(
        index="dataset", columns="method", values="metric_error", aggfunc="mean"
    )
    piv = piv.loc[[d for d in names if d in piv.index], [T12, CAT]].dropna()
    rel = 100.0 * (piv[T12] - piv[CAT]) / piv[CAT]
    wins = int((rel < 0).sum())
    p = float(wilcoxon(rel).pvalue) if (rel != 0).any() else 1.0
    return [
        f"- **{title}** ({len(rel)} tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins {wins}/{len(rel)}, "
        f"median {rel.median():+.2f}%, mean {rel.mean():+.2f}%, Wilcoxon p = {p:.4f}"
    ]


def cmd_summarize() -> None:
    reg, binary = _tasks()
    names = reg + binary
    new = _new_results(names)
    lines = [
        "# MITER_T12 on TabArena `lite`, all regression + binary tasks (43)",
        "",
        "Pre-registered: `tabarena_t12_full43_prereg.md` (SHA-256 in the commit that froze it). "
        "Raw: `benchmarks/results/raw/tabarena/ens_lite/`, `.../t12_full43_cmp/` (gitignored).",
        "",
    ]
    lb_g, per_g = _leaderboard(new, names, True, "gbdt_only_43")
    lines += ["## Primary: gbdt_only pool, 43 tasks", ""] + _lb_lines(
        lb_g, "gbdt_only, 43 tasks", True
    )
    lines += (
        _paired(per_g, names, "all 43")
        + _paired(per_g, reg, "regression 13")
        + _paired(per_g, binary, "binary 30")
    )
    lines += [""]
    for sub, tag in ((reg, "regression 13"), (binary, "binary 30")):
        lb_s, _ = _leaderboard(new, sub, True, f"gbdt_only_{tag.split()[0]}")
        lines += _lb_lines(lb_s, f"gbdt_only, {tag}", True)
    lb_f, _ = _leaderboard(new, names, False, "full_pool_43")
    lines += ["## Secondary: full pool, 43 tasks", ""] + _lb_lines(
        lb_f, "full pool, 43 tasks", False
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY_PATH}")


SUMMARY51_PATH = HERE.parent / "results" / "tabarena_t12_full51_summary.md"


def _multiclass_tasks() -> list[str]:
    tm = TabArenaContext().task_metadata
    return sorted(tm[tm.problem_type == "multiclass"]["name"].tolist())


def cmd_run_multiclass(shard: int, nshards: int) -> None:
    """Addendum (tabarena_t12_full51_prereg.md): the 8 multiclass tasks, miter one-vs-rest."""
    order = _order(_multiclass_tasks())
    mine = [d for i, d in enumerate(order) if i % nshards == shard]
    print(f"Multiclass shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        ctx = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=_models()).build_experiments()
        res = ctx.build_and_run_jobs(
            exps,
            expname=str(EXP_ROOT / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        print(
            f"DONE dataset={name} jobs={len(res)} wall={time.time() - t0:.1f}s",
            flush=True,
        )


def cmd_summarize51() -> None:
    reg, binary = _tasks()
    mc = _multiclass_tasks()
    names = reg + binary + mc
    new = _new_results(names)
    lines = [
        "# MITER_T12 on TabArena `lite`, all 51 tasks (multiclass via one-vs-rest)",
        "",
        "Addendum to `tabarena_t12_full43_prereg.md`: `tabarena_t12_full51_prereg.md`. Multiclass "
        "tasks use miter one-vs-rest (one binary model per class, normalised probabilities), not "
        "native softmax.",
        "",
    ]
    lb_g, per_g = _leaderboard(new, names, True, "gbdt_only_51")
    lines += ["## gbdt_only pool, 51 tasks", ""] + _lb_lines(
        lb_g, "gbdt_only, 51 tasks", True
    )
    lines += (
        _paired(per_g, names, "all 51")
        + _paired(per_g, mc, "multiclass 8 (OvR)")
        + [""]
    )
    lb_m, _ = _leaderboard(new, mc, True, "gbdt_only_multiclass")
    lines += _lb_lines(lb_m, "gbdt_only, multiclass 8 (OvR)", True)
    lb_f, _ = _leaderboard(new, names, False, "full_pool_51")
    lines += ["## full pool, 51 tasks", ""] + _lb_lines(
        lb_f, "full pool, 51 tasks", False
    )
    SUMMARY51_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {SUMMARY51_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--shard", type=int, required=True)
    r.add_argument("--nshards", type=int, required=True)
    sp.add_parser("summarize")
    rm = sp.add_parser("run-multiclass")
    rm.add_argument("--shard", type=int, required=True)
    rm.add_argument("--nshards", type=int, required=True)
    sp.add_parser("summarize51")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.shard, a.nshards)
    elif a.cmd == "run-multiclass":
        cmd_run_multiclass(a.shard, a.nshards)
    elif a.cmd == "summarize51":
        cmd_summarize51()
    else:
        cmd_summarize()
