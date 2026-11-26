"""MITER_V11_T12 on all 51 TabArena `lite` tasks (native Softmax + CatBoost-style CTRs).

Pre-registered in `benchmarks/results/exp_softmax_tabarena51_prereg.md` (Part B).
Uses a separate experiment directory (`ens_v11_lite/`) so cached `MITER_T12` runs in `ens_lite/`
stay untouched.

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_v11_51.py run --shard 0 --nshards 3
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_v11_51.py summarize
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
from miter.tabarena import MiterModel, generate_configs_miter_v11
from scipy.stats import wilcoxon
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import CustomAGConfigGenerator

SUMMARY_PATH = HERE.parent / "results" / "tabarena_v11_51_summary.md"
EXP_ROOT = ens.ROOT / "ens_v11_lite"
V11_T12 = "[New] MITER_V11_T12 (tuned + ensemble)"
CAT = "CAT (tuned + ensemble)"
OLD_T12 = "[New] MITER_T12 (tuned + ensemble)"

_S6_CFG: dict[str, Any] = {
    "grow_policy": "symmetric",
    "max_depth": 6,
    "min_data_in_leaf": 1,
    "lambda_l2": 3.0,
}
_ACC_CFG: dict[str, Any] = {"preset": "accurate"}
_S6_M4_CFG: dict[str, Any] = {**_S6_CFG, "m_mult": 4}
_S6_CTR_CFG: dict[str, Any] = {
    **_S6_CFG,
    "cat_oblique_ts": True,
    "cat_ctr_mode": "ordered",
    "cat_ctr_pairs": 16,
    "cat_ctr_counter": True,
}


class MiterV11T12Model(MiterModel):
    ag_key = "MITER_V11_T12"
    ag_name = "MiterV11T12"


def _v11_t12_spec() -> Any:
    cg = CustomAGConfigGenerator(
        model_cls=MiterV11T12Model,
        name="MITER_V11_T12",
        manual_configs=[{}, _S6_CFG, _ACC_CFG, _S6_M4_CFG, _S6_CTR_CFG],
        search_space_func=lambda num_random_configs=7: generate_configs_miter_v11(
            num_random_configs, seed=1234
        ),
    )
    return (cg, 7)


def _tasks() -> tuple[list[str], list[str], list[str]]:
    tm = TabArenaContext().task_metadata
    reg = sorted(tm[tm.problem_type == "regression"]["name"].tolist())
    binary = sorted(tm[tm.problem_type == "binary"]["name"].tolist())
    mc = sorted(tm[tm.problem_type == "multiclass"]["name"].tolist())
    return reg, binary, mc


def _order(names: list[str]) -> list[str]:
    """Heaviest first (rows * sqrt(features)) so shards finish at similar times."""
    tm = TabArenaContext().task_metadata.set_index("name")
    return sorted(
        names,
        key=lambda d: -float(tm.loc[d].num_instances) * float(np.sqrt(tm.loc[d].n_features)),
    )


def _bootstrap_median_ci(
    deltas: np.ndarray, *, n_boot: int = 10000, seed: int = 2026
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    meds = np.median(deltas[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_run(shard: int, nshards: int) -> None:
    reg, binary, mc = _tasks()
    order = _order(reg + binary + mc)
    mine = [d for i, d in enumerate(order) if i % nshards == shard]
    print(f"Shard {shard}/{nshards}: {mine}", flush=True)
    for name in mine:
        t0 = time.time()
        ctx = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=[_v11_t12_spec()]).build_experiments()
        res = ctx.build_and_run_jobs(
            exps,
            expname=str(EXP_ROOT / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        print(f"DONE dataset={name} jobs={len(res)} wall={time.time() - t0:.1f}s", flush=True)


def _new_results(names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        c = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=[_v11_t12_spec()]).build_experiments()
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
        ["LightGBM", "CatBoost", "XGBoost"] if gbdt_only else list(dict.fromkeys(ctx.methods))
    )
    ta = ctx.load_results(methods=methods, download_results=False)
    out = ens.ROOT / "v11_51_cmp" / tag
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


def _paired_stats(per: pd.DataFrame, names: list[str], a: str, b: str) -> dict[str, Any]:
    piv = per.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
    piv = piv.loc[[d for d in names if d in piv.index], [a, b]].dropna()
    rel = (100.0 * (piv[a] - piv[b]) / piv[b]).to_numpy(dtype=np.float64)
    wins = int((rel < 0.0).sum())
    p = float(wilcoxon(rel).pvalue) if np.any(rel != 0.0) else 1.0
    ci_lo, ci_hi = _bootstrap_median_ci(rel, n_boot=10000, seed=2026)
    return {
        "n": len(rel),
        "wins": wins,
        "median": float(np.median(rel)),
        "mean": float(np.mean(rel)),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p": p,
        "worst": float(np.max(rel)),
    }


def _paired_line(st: dict[str, Any], title: str) -> str:
    return (
        f"- **{title}** ({st['n']} tasks): wins {st['wins']}/{st['n']}, "
        f"median {st['median']:+.2f}% (95% boot CI [{st['ci_lo']:+.2f}%, {st['ci_hi']:+.2f}%]), "
        f"mean {st['mean']:+.2f}%, Wilcoxon p = {st['p']:.4g}, worst {st['worst']:+.2f}%"
    )


def cmd_summarize() -> None:
    reg, binary, mc = _tasks()
    names = reg + binary + mc
    new = _new_results(names)

    lb_g, per_g = _leaderboard(new, names, True, "gbdt_only_51")
    st_all = _paired_stats(per_g, names, V11_T12, CAT)
    st_reg = _paired_stats(per_g, reg, V11_T12, CAT)
    st_bin = _paired_stats(per_g, binary, V11_T12, CAT)
    st_mc = _paired_stats(per_g, mc, V11_T12, CAT)

    elo_row_v11 = lb_g[lb_g["method"] == V11_T12].iloc[0]
    elo_row_cat = lb_g[lb_g["method"] == CAT].iloc[0]
    elo_gt_cat = float(elo_row_v11["elo"]) > float(elo_row_cat["elo"])

    # Check cached v1.0 OvR results on the 8 multiclass tasks if present in ens_lite.
    mc_vs_ovr_line = "- Multiclass 8 vs cached `MITER_T12 (OvR)`: unavailable (no cached OvR runs)."
    mc_beats_ovr = False
    try:
        from run_tabarena_t12_full43 import _new_results as _old_results

        old_mc = _old_results(mc)
        old_t12 = old_mc[old_mc["method"] == OLD_T12].copy()
        if len(old_t12) > 0:
            v11_mc = new[(new["dataset"].isin(mc)) & (new["method"] == V11_T12)].copy()
            _, per_ovr = _leaderboard(pd.concat([v11_mc, old_t12], ignore_index=True), mc, True, "v11_vs_ovr_mc8")
            st_ovr = _paired_stats(per_ovr, mc, V11_T12, OLD_T12)
            mc_beats_ovr = (st_ovr["wins"] >= 7) or (st_ovr["p"] < 0.05 and st_ovr["median"] < 0.0)
            mc_vs_ovr_line = _paired_line(st_ovr, "MITER_V11_T12 vs cached MITER_T12 (OvR) on multiclass 8")
    except Exception as exc:  # pragma: no cover - optional cache check
        mc_vs_ovr_line = f"- Multiclass 8 vs cached `MITER_T12 (OvR)`: skipped ({exc})."

    beats_cat = (
        elo_gt_cat
        and st_all["p"] < 0.05
        and st_all["median"] < 0.0
        and st_all["ci_hi"] < 0.0
        and st_all["wins"] >= 33
        and mc_beats_ovr
    )
    if beats_cat:
        claim = "**MITER_V11_T12 beats CatBoost (tuned + ensemble) on TabArena lite (51 tasks)** (all 5 anti-fluke criteria met)."
    elif st_all["p"] >= 0.05:
        claim = "**MITER_V11_T12 is statistically level with CatBoost (tuned + ensemble)** (p >= 0.05; anti-fluke headline claim NOT met)."
    elif st_all["median"] < 0.0:
        claim = "**Mixed / partial advantage** (some criteria met, but not all 5 anti-fluke criteria met; do not claim full sweep over CatBoost)."
    else:
        claim = "**Behind CatBoost (tuned + ensemble)**."

    lines = [
        "# MITER_V11_T12 on TabArena `lite`, all 51 tasks (Native Softmax + CatBoost CTRs)",
        "",
        "Pre-registered in `benchmarks/results/exp_softmax_tabarena51_prereg.md` (Part B).",
        "Caveat: TabArena `lite` evaluates **single split (`fold 0`) per dataset**.",
        "",
        "## Anti-Fluke Headline Verdict",
        "",
        f"- **Verdict:** {claim}",
        f"- Criteria: Elo > CAT ({elo_row_v11['elo']:.1f} vs {elo_row_cat['elo']:.1f}: {elo_gt_cat}), "
        f"p < 0.05 ({st_all['p']:.4g}), 95% boot CI < 0 ([{st_all['ci_lo']:+.2f}%, {st_all['ci_hi']:+.2f}%]), "
        f"wins >= 33/51 ({st_all['wins']}/51), MC8 improves over OvR ({mc_beats_ovr}).",
        "",
        "## Primary: gbdt_only pool, 51 tasks",
        "",
    ] + _lb_lines(lb_g, "gbdt_only, 51 tasks", True)
    lines += [
        _paired_line(st_all, "all 51 vs CAT"),
        _paired_line(st_reg, "regression 13 vs CAT"),
        _paired_line(st_bin, "binary 30 vs CAT"),
        _paired_line(st_mc, "multiclass 8 vs CAT"),
        mc_vs_ovr_line,
        "",
    ]

    for sub, tag in ((reg, "regression_13"), (binary, "binary_30"), (mc, "multiclass_8")):
        lb_s, _ = _leaderboard(new, sub, True, f"gbdt_only_{tag}")
        lines += _lb_lines(lb_s, f"gbdt_only, {tag.replace('_', ' ')}", True)

    lb_f, _ = _leaderboard(new, names, False, "full_pool_51")
    lines += ["## Secondary: full pool, 51 tasks", ""] + _lb_lines(lb_f, "full pool, 51 tasks", False)

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")
    print("\n".join(lines[:35]))


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
