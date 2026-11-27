"""A speed-up variant of the v1.0 default (bench_exp_speedups.VARIANTS) on all 51 TabArena `lite` tasks,
compared per task with MITER_DEFAULT_V1 (run_tabarena_default_51.py). One configuration, no tuning.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_speedup_51.py run E1 --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_speedup_51.py summarize E1
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import analyze_v12_default_candidates as cand
import run_tabarena_default_51 as base
import run_tabarena_ensemble as ens
from miter.tabarena import MiterModel
from scipy.stats import wilcoxon
from speedup_variants import VARIANTS
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import CustomAGConfigGenerator


def _model(variant: str) -> Any:
    """A module-level subclass per variant (AutoGluon pickles the model, so it must be importable)."""
    name = f"MiterSpeedup{variant}"
    if name not in globals():
        cls = type(name, (MiterModel,), {"ag_key": f"MITER_SPEEDUP_{variant}", "ag_name": name})
        cls.__module__ = __name__
        cls.__qualname__ = name
        globals()[name] = cls
    return globals()[name]


def _spec(variant: str) -> Any:
    cg = CustomAGConfigGenerator(model_cls=_model(variant), name=f"MITER_SPEEDUP_{variant}",
                                 manual_configs=[dict(VARIANTS[variant])],
                                 search_space_func=lambda num_random_configs=0: [])
    return (cg, 0)


def _root(variant: str) -> Path:
    return ens.ROOT / f"speedup_{variant}_lite"


def cmd_run(variant: str, shard: int, nshards: int) -> None:
    tm = TabArenaContext().task_metadata.set_index("name")
    order = sorted(base._tasks(), key=lambda d: -float(tm.loc[d].num_instances) * float(tm.loc[d].n_features) ** 0.5)
    for name in [d for i, d in enumerate(order) if i % nshards == shard]:
        t0 = time.time()
        ctx = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=[_spec(variant)]).build_experiments()
        ctx.build_and_run_jobs(exps, expname=str(_root(variant) / name), new_result_prefix="[New] ",
                               build_kwargs={"dataset_names": [name], "split_indices": "lite"}, debug_mode=True)
        print(f"DONE dataset={name} wall={time.time() - t0:.1f}s", flush=True)


def _errors(root: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        hits = list((task_dir / "data" / f"{prefix}_c1_default_BAG_L1").glob("*/0_0/results.pkl"))
        if hits:
            d = cand._load(hits[0])
            rows.append({"dataset": task_dir.name, "err": float(d["metric_error"]), "secs": float(d["time_train_s"])})
    return pd.DataFrame(rows).set_index("dataset")


def cmd_summarize(variant: str) -> None:
    a = _errors(_root(variant), f"MiterSpeedup{variant}")
    b = _errors(base.EXP_ROOT, "MiterDefaultV1")
    j = a.join(b, lsuffix="_v", rsuffix="_b", how="inner")
    rel = 100.0 * (j.err_v - j.err_b) / j.err_b
    nz = rel[rel != 0]
    p_worse = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) else 1.0
    out = HERE.parent / "results" / f"tabarena_speedup_{variant.lower()}_51_summary.md"
    lines = [f"# TabArena lite, 51 tasks: {variant} {VARIANTS[variant]} vs MITER_DEFAULT_V1", "",
             f"- tasks: {len(rel)}; wins {int((rel < 0).sum())}, ties {int((rel == 0).sum())}, losses {int((rel > 0).sum())}",
             f"- relative error: median {rel.median():+.3f}%, mean {rel.mean():+.3f}%, worst {rel.max():+.2f}%",
             f"- one-sided Wilcoxon (variant worse): p = {p_worse:.4f}",
             f"- train time (8-fold bag, this machine): total {j.secs_v.sum() / j.secs_b.sum():.2f}x, "
             f"median per task {np.median(j.secs_v / j.secs_b):.2f}x", "",
             "| task | default err | variant err | rel | time ratio |", "|---|---:|---:|---:|---:|"]
    for d in rel.sort_values(ascending=False).index:
        lines.append(f"| {d} | {j.err_b[d]:.5g} | {j.err_v[d]:.5g} | {rel[d]:+.2f}% | {j.secs_v[d] / j.secs_b[d]:.2f}x |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("variant")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    s = sp.add_parser("summarize")
    s.add_argument("variant")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.variant, a.shard, a.nshards)
    else:
        cmd_summarize(a.variant)
