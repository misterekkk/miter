"""miter tuned on TabArena `lite` (51 tasks) by the TabArena protocol: the default configuration
plus 99 random configurations from the V12 search space, nothing chosen by hand.

The random configurations are the 24 of E6 (`generate_configs_miter_v12(24, seed=1234)`) followed
by 75 more from the same space (`seed=1235`). They run under the E6 `FULL` model, so the 25
configurations E6 already trained are reused; the E6 summary only ever reads its own 25.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_t100_51.py run --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_t100_51.py summarize
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

from miter.tabarena import generate_configs_miter_v12
from run_tabarena_e6_51 import EXP_ROOT, MiterE6FullModel
from run_tabarena_v12_51 import (
    _leaderboard,
    _order,
    _paired_line,
    _paired_stats,
    _tasks,
)
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import CustomAGConfigGenerator

SUMMARY_PATH = HERE.parent / "results" / "tabarena_t100_51_summary.md"
N_RANDOM = 99
NAME = "[New] MITER"


def configs(num_random_configs: int = N_RANDOM) -> list[dict[str, Any]]:
    out = generate_configs_miter_v12(24, seed=1234) + generate_configs_miter_v12(75, seed=1235)
    assert len({str(c) for c in out}) == len(out)
    return out[:num_random_configs]


def _spec() -> Any:
    cg = CustomAGConfigGenerator(model_cls=MiterE6FullModel, name="MITER_E6_FULL", manual_configs=[{}],
                                 search_space_func=configs)
    return (cg, N_RANDOM)


def _run(name: str) -> pd.DataFrame | None:
    ctx = TabArenaContext()
    exps = TabArenaV0pt1ExperimentBundle(models=[_spec()]).build_experiments()
    ctx.build_and_run_jobs(exps, expname=str(EXP_ROOT / "FULL" / name), new_result_prefix="[New] ",
                           build_kwargs={"dataset_names": [name], "split_indices": "lite"}, debug_mode=True)
    return ctx._registered_new_results()


def cmd_run(shard: int, nshards: int) -> None:
    reg, binary, mc = _tasks()
    mine = [d for i, d in enumerate(_order(reg + binary + mc)) if i % nshards == shard]
    print(f"shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        _run(name)
        print(f"DONE dataset={name} wall={time.time() - t0:.1f}s", flush=True)


def cmd_summarize() -> None:
    reg, binary, mc = _tasks()
    names = reg + binary + mc
    new = pd.concat([df for name in names if (df := _run(name)) is not None], ignore_index=True)
    for col in ("method", "config_type", "ta_name", "ta_suite"):
        if col in new:
            new[col] = new[col].astype(str).str.replace("[New] MITER_E6_FULL", NAME, regex=False)
    lb, per = _leaderboard(new, names, True, "t100_gbdt_only_51")
    lines = ["# miter on TabArena lite (51 tasks): default + 99 random configurations", "",
             "TabArena protocol: the default configuration and random configurations from the V12 search "
             "space (`generate_configs_miter_v12`, seeds 1234 and 1235), none chosen by hand. Baselines: "
             "TabArena's cached LightGBM, CatBoost and XGBoost results.", "",
             "| method | elo | elo+ | elo- | rank | winrate |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['elo']:.0f} | {r['elo+']:.0f} | {r['elo-']:.0f} | {r['rank']:.3f} | "
              f"{r['winrate']:.3f} |" for _, r in lb.iterrows()] + [""]
    for kind in ("tuned + ensemble", "tuned", "default"):
        for other in ("CAT", "GBM", "XGB"):
            st = _paired_stats(per, names, f"{NAME} ({kind})", f"{other} ({kind})")
            lines.append(_paired_line(st, f"{kind}: miter vs {other}"))
        lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    sp.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.shard, a.nshards)
    else:
        cmd_summarize()


if __name__ == "__main__":
    main()
