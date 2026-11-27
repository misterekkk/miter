"""E6-T: tuned TabArena `lite` (51 tasks), V12 search space with and without CTR / quantizer.

Pre-registration: benchmarks/results/exp_e6_options_prereg.md. Both arms use the default
configuration plus 24 random configurations (seed 1234): `FULL` from `generate_configs_miter_v12`,
`LEAN` from the same space with `max_ctr_complexity` fixed at 0 and `quantizer` at "linear".

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_e6_51.py run --arm FULL --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_e6_51.py summarize
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

import run_tabarena_ensemble as ens  # sets TABARENA_CACHE and the OpenML cache dir
from miter.tabarena import MiterModel, generate_configs_miter_v12
from run_tabarena_v12_51 import (
    _leaderboard,
    _order,
    _paired_line,
    _paired_stats,
    _tasks,
)
from scipy.stats import wilcoxon
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import CustomAGConfigGenerator

SUMMARY_PATH = HERE.parent / "results" / "exp_e6_tuned_51_summary.md"
EXP_ROOT = ens.ROOT / "e6_tuned_lite"
N_RANDOM = 24


def generate_configs_lean(num_random_configs: int, seed: int = 1234) -> list[dict[str, Any]]:
    """The V12 space without the CTR and quantizer dimensions (they take their defaults)."""
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        EqualsCondition,
        Float,
        Integer,
    )

    gp = Categorical("grow_policy", ["leafwise", "symmetric"])
    nl = Integer("num_leaves", (4, 128), log=True)
    md = Integer("max_depth", (4, 8))
    space = ConfigurationSpace(
        space=[
            Float("learning_rate", (1e-2, 1e-1), log=True),
            gp,
            nl,
            md,
            Integer("min_data_in_leaf", (1, 64), log=True),
            Float("lambda_l2", (1e-3, 10.0), log=True),
            Categorical("m_mult", [1, 2, 4]),
            Float("feature_fraction", (0.5, 1.0)),
            Categorical("extra_trees", [False, True]),
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 10.0), log=True),
            Float("cat_smooth", (1e-3, 100.0), log=True),
            Integer("max_cat_to_onehot", (4, 64), log=True),
        ],
        seed=seed,
    )
    space.add([EqualsCondition(nl, gp, "leafwise"), EqualsCondition(md, gp, "symmetric")])
    configs = space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    return [{k: v.item() if isinstance(v, np.generic) else v for k, v in dict(c).items() if v is not None}
            for c in configs]


class MiterE6FullModel(MiterModel):
    ag_key = "MITER_E6_FULL"
    ag_name = "MiterE6Full"


class MiterE6LeanModel(MiterModel):
    ag_key = "MITER_E6_LEAN"
    ag_name = "MiterE6Lean"


ARMS: dict[str, tuple[type, Any]] = {
    "FULL": (MiterE6FullModel, lambda num_random_configs=N_RANDOM: generate_configs_miter_v12(num_random_configs, seed=1234)),
    "LEAN": (MiterE6LeanModel, lambda num_random_configs=N_RANDOM: generate_configs_lean(num_random_configs, seed=1234)),
}


def _spec(arm: str) -> Any:
    cls, gen = ARMS[arm]
    cg = CustomAGConfigGenerator(model_cls=cls, name=f"MITER_E6_{arm}", manual_configs=[{}], search_space_func=gen)
    return (cg, N_RANDOM)


def _run(arm: str, name: str) -> pd.DataFrame | None:
    ctx = TabArenaContext()
    exps = TabArenaV0pt1ExperimentBundle(models=[_spec(arm)]).build_experiments()
    ctx.build_and_run_jobs(exps, expname=str(EXP_ROOT / arm / name), new_result_prefix="[New] ",
                           build_kwargs={"dataset_names": [name], "split_indices": "lite"}, debug_mode=True)
    return ctx._registered_new_results()


def cmd_run(arm: str, shard: int, nshards: int) -> None:
    reg, binary, mc = _tasks()
    order = _order(reg + binary + mc)
    mine = [d for i, d in enumerate(order) if i % nshards == shard]
    print(f"{arm} shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        _run(arm, name)
        print(f"DONE arm={arm} dataset={name} wall={time.time() - t0:.1f}s", flush=True)


def cmd_summarize() -> None:
    reg, binary, mc = _tasks()
    names = reg + binary + mc
    new = pd.concat([df for arm in ARMS for name in names if (df := _run(arm, name)) is not None], ignore_index=True)
    _, per = _leaderboard(new, names, True, "e6_gbdt_only_51")
    lines = ["# E6-T: tuned TabArena lite, 51 tasks, LEAN (no CTR / quantizer) vs FULL (V12 space)", "",
             "Pre-registration: `exp_e6_options_prereg.md`. Negative = LEAN better.", ""]
    for kind in ("tuned + ensemble", "tuned"):
        a, b = f"[New] MITER_E6_LEAN ({kind})", f"[New] MITER_E6_FULL ({kind})"
        st = _paired_stats(per, names, a, b)
        piv = per.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
        piv = piv.loc[[d for d in names if d in piv.index], [a, b]].dropna()
        rel = (100.0 * (piv[a] - piv[b]) / piv[b]).to_numpy(dtype=np.float64)
        p_worse = float(wilcoxon(rel, alternative="greater").pvalue) if np.any(rel != 0.0) else 1.0
        lines += [_paired_line(st, f"{kind}: LEAN vs FULL") + f"; one-sided p (LEAN worse) = {p_worse:.4g}"]
        if kind == "tuned + ensemble":
            ok = st["median"] <= 0.25 and p_worse >= 0.05
            lines += ["", f"**Decision (pre-registered, tuned + ensemble): {'REMOVE CTR and quantizer modes' if ok else 'KEEP'}**", ""]
            worst = (100.0 * (piv[a] - piv[b]) / piv[b]).sort_values(ascending=False).head(8)
            lines += ["Largest LEAN losses:", ""] + [f"- {d}: {v:+.2f}%" for d, v in worst.items()] + [""]
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--arm", choices=list(ARMS), required=True)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    sp.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.arm, a.shard, a.nshards)
    else:
        cmd_summarize()


if __name__ == "__main__":
    main()
