"""Official TabArena `lite` ensemble benchmark for miter:
- `MITER_BS6` (2 configs: Leafwise default + Symmetric D=6) across all 20 tasks (13 reg + 7 bin, 0 imputed).
- `MITER_T12` (12 configs: 5 curated + 7 random from search space) across 17 fast/medium tasks (10 reg + 7 bin, 0 imputed).

Usage:
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_ensemble.py run --shard 0 --nshards 4
    .venv_tabarena/bin/python benchmarks/tabarena/run_tabarena_ensemble.py summarize
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results" / "raw" / "tabarena"
SUMMARY_PATH = HERE.parent / "results" / "tabarena_ensemble_lite_summary.md"

os.environ.setdefault(
    "TABARENA_CACHE",
    (str(Path(os.environ.get("HISTREE_DIR", Path(__file__).resolve().parents[3] / "histree"))) + "/research/results/tabarena_cache"),
)
sys.path.insert(0, str(HERE))

import openml

openml.config.set_root_cache_directory(str(HERE.parent / "results" / "raw" / "openml_cache_pandas2"))

import pandas as pd
from miter.tabarena import MiterModel, generate_configs_miter
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.utils.config_utils import ConfigGenerator, CustomAGConfigGenerator

HEAVY_REG_DATASETS = {"QSAR-TID-11", "physiochemical_protein", "superconductivity"}

REG_DATASETS = [
    "airfoil_self_noise",
    "Another-Dataset-on-used-Fiat-500",
    "concrete_compressive_strength",
    "diamonds",
    "Food_Delivery_Time",
    "healthcare_insurance_expenses",
    "houses",
    "miami_housing",
    "physiochemical_protein",
    "QSAR-TID-11",
    "QSAR_fish_toxicity",
    "superconductivity",
    "wine_quality",
]

BIN_DATASETS = [
    "Amazon_employee_access",
    "Bank_Customer_Churn",
    "bank-marketing",
    "credit_card_clients_default",
    "heloc",
    "online_shoppers_intention",
    "blood-transfusion-service-center",
]

ALL_DATASETS = REG_DATASETS + BIN_DATASETS
T12_DATASETS = [d for d in ALL_DATASETS if d not in HEAVY_REG_DATASETS]

BALANCED_ORDER = [
    "QSAR-TID-11",
    "physiochemical_protein",
    "superconductivity",
    "diamonds",
    "houses",
    "miami_housing",
    "Food_Delivery_Time",
    "bank-marketing",
    "Amazon_employee_access",
    "credit_card_clients_default",
    "online_shoppers_intention",
    "heloc",
    "Bank_Customer_Churn",
    "wine_quality",
    "airfoil_self_noise",
    "concrete_compressive_strength",
    "Another-Dataset-on-used-Fiat-500",
    "QSAR_fish_toxicity",
    "healthcare_insurance_expenses",
    "blood-transfusion-service-center",
]

_S6_CFG: dict[str, Any] = {
    "grow_policy": "symmetric",
    "max_depth": 6,
    "min_data_in_leaf": 1,
    "lambda_l2": 3.0,
}
_ACC_CFG: dict[str, Any] = {"preset": "accurate"}
_S6_ACC_CFG: dict[str, Any] = {**_S6_CFG, "m_mult": 4}
_S6_TS_CFG: dict[str, Any] = {**_S6_CFG, "cat_oblique_ts": True}


class MiterBS6Model(MiterModel):
    ag_key = "MITER_BS6"
    ag_name = "MiterBS6"


class MiterT12Model(MiterModel):
    ag_key = "MITER_T12"
    ag_name = "MiterT12"


def _bs6_spec() -> Any:
    cg_bs6 = ConfigGenerator(
        model_cls=MiterBS6Model,
        name="MITER_BS6",
        manual_configs=[{}, _S6_CFG],
        search_space={},
    )
    return (cg_bs6, 0)


def _t12_spec() -> Any:
    cg_t12 = CustomAGConfigGenerator(
        model_cls=MiterT12Model,
        name="MITER_T12",
        manual_configs=[{}, _S6_CFG, _ACC_CFG, _S6_ACC_CFG, _S6_TS_CFG],
        search_space_func=lambda num_random_configs=7: generate_configs_miter(num_random_configs, seed=1234),
    )
    return (cg_t12, 7)


def _models_for(ds_name: str) -> list[Any]:
    if ds_name in HEAVY_REG_DATASETS:
        return [_bs6_spec()]
    return [_bs6_spec(), _t12_spec()]


def _run_datasets(names: list[str]) -> None:
    for name in names:
        t0 = time.time()
        ctx = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=_models_for(name)).build_experiments()
        expdir = ROOT / "ens_lite" / name
        res = ctx.build_and_run_jobs(
            exps,
            expname=str(expdir),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        print(f"DONE dataset={name} jobs={len(res)} wall={time.time() - t0:.1f}s", flush=True)


def cmd_run(shard: int, nshards: int) -> None:
    selected = [d for idx, d in enumerate(BALANCED_ORDER) if idx % nshards == shard]
    print(f"Shard {shard}/{nshards}: {selected}", flush=True)
    _run_datasets(selected)


def _emit_table(
    ctx: TabArenaContext,
    scope_name: str,
    tag_name: str,
    ds_list: list[str],
    methods: list[str] | None,
    new_df: pd.DataFrame,
    out_dir: Path,
    lines: list[str],
) -> None:
    pool_tag = "full_pool" if methods is None else "gbdt_only"
    ml = list(dict.fromkeys(methods if methods else ctx.methods))
    ta = ctx.load_results(methods=ml, download_results=False)
    tag_dir = out_dir / f"{tag_name}_{pool_tag}"
    lb, per = ctx.compare(
        output_dir=str(tag_dir),
        ta_results=ta,
        new_results=new_df[new_df["dataset"].isin(ds_list)].copy(),
        datasets=ds_list,
        subset="lite",
        return_results=True,
        plot=False,
        **({} if methods is None else {"fillna": None, "calibration_method": None}),
    )
    lb.to_csv(tag_dir / "leaderboard.csv", index=False)
    per.to_csv(tag_dir / "per_split.csv", index=False)
    lines.append(f"## {scope_name} — {pool_tag} (`{len(lb)}` methods)")
    lines.append("")
    lines.append("| # | method | elo | rank | winrate | improvability | imputed |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for idx_row, (_, r) in enumerate(lb.iterrows(), start=1):
        m_str = str(r["method"])
        if (
            pool_tag == "gbdt_only"
            or "[New]" in m_str
            or idx_row <= 10
            or any(k in m_str for k in ("CAT (", "GBM (", "XGB (", "REALMLP", "TABM", "TABPFN"))
        ):
            lines.append(
                f"| {idx_row} | {m_str} | {r['elo']:.1f} | {r['rank']:.2f} | {r['winrate']:.3f} | {r['improvability']:.4f} | {r['imputed']:.2f} |"
            )
    lines.append("")


def cmd_summarize() -> None:
    out_dir = ROOT / "ens_lite_cmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_bs6: list[pd.DataFrame] = []
    frames_t12: list[pd.DataFrame] = []
    for name in ALL_DATASETS:
        c = TabArenaContext()
        exps = TabArenaV0pt1ExperimentBundle(models=_models_for(name)).build_experiments()
        c.build_and_run_jobs(
            exps,
            expname=str(ROOT / "ens_lite" / name),
            new_result_prefix="[New] ",
            build_kwargs={"dataset_names": [name], "split_indices": "lite"},
            debug_mode=True,
        )
        df_c = c._registered_new_results()
        assert df_c is not None
        frames_bs6.append(df_c[df_c["method"].astype(str).str.contains("MITER_BS6")])
        if name in T12_DATASETS:
            frames_t12.append(df_c[df_c["method"].astype(str).str.contains("MITER_T12")])

    df_bs6 = pd.concat(frames_bs6, ignore_index=True)
    df_17 = pd.concat([df_bs6[df_bs6["dataset"].isin(T12_DATASETS)], *frames_t12], ignore_index=True)
    ctx_cmp = TabArenaContext()

    lines: list[str] = [
        "# TabArena `lite` Ensemble Evaluation: `MITER_BS6` (2 configs) & `MITER_T12` (12 configs)",
        "",
        "- Protocol: Official `TabArenaV0pt1ExperimentBundle` (`split_indices='lite'`, `subset='lite'`, fold 0, 8-fold bagging, AutoGluon preprocessing, `time_limit=3600s`).",
        "- Exploratory comparison (`subset='lite'` fold 0, 2 or 12 miter configs vs up to 200 configs for cached CatBoost/GBM/XGB).",
        f"- Scopes: `MITER_BS6` across **all 20 tasks** ({len(REG_DATASETS)} reg + {len(BIN_DATASETS)} binary, `0.00` imputed) & **13 reg**; `MITER_BS6 + MITER_T12` across **17 fast/medium tasks** (`0.00` imputed).",
        "",
    ]

    _emit_table(ctx_cmp, "BS6 on 13 Regression Tasks", "bs6_reg13", REG_DATASETS, None, df_bs6, out_dir, lines)
    _emit_table(ctx_cmp, "BS6 on 13 Regression Tasks", "bs6_reg13", REG_DATASETS, ["LightGBM", "CatBoost", "XGBoost"], df_bs6, out_dir, lines)
    _emit_table(ctx_cmp, "BS6 on All 20 Tasks (13 Reg + 7 Binary)", "bs6_all20", ALL_DATASETS, None, df_bs6, out_dir, lines)
    _emit_table(ctx_cmp, "BS6 on All 20 Tasks (13 Reg + 7 Binary)", "bs6_all20", ALL_DATASETS, ["LightGBM", "CatBoost", "XGBoost"], df_bs6, out_dir, lines)
    _emit_table(ctx_cmp, "BS6 + T12 on 17 Fast/Medium Tasks (10 Reg + 7 Binary)", "t12_17", T12_DATASETS, None, df_17, out_dir, lines)
    _emit_table(ctx_cmp, "BS6 + T12 on 17 Fast/Medium Tasks (10 Reg + 7 Binary)", "t12_17", T12_DATASETS, ["LightGBM", "CatBoost", "XGBoost"], df_17, out_dir, lines)

    per17 = pd.read_csv(out_dir / "t12_17_gbdt_only" / "per_split.csv")
    per20 = pd.read_csv(out_dir / "bs6_all20_gbdt_only" / "per_split.csv")
    piv20 = per20.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
    piv17 = per17.pivot_table(index="dataset", columns="method", values="metric_error", aggfunc="mean")
    lines.extend([
        "## Per-dataset Fold-0 (`lite`) Metric Error (RMSE or 1-AUC per TabArena task metric; lower is better)",
        "",
        "| dataset | MITER default | MITER_BS6 (tuned+ens) | MITER_T12 (tuned+ens) | CAT (default) | CAT (tuned+ens) | GBM (tuned+ens) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for ds in ALL_DATASETS:
        row20 = piv20.loc[ds]
        t12_val = f"{piv17.loc[ds, '[New] MITER_T12 (tuned + ensemble)']:.5g}" if ds in piv17.index else "—"
        lines.append(
            f"| `{ds}` | {row20['[New] MITER_BS6 (default)']:.5g} | "
            f"{row20['[New] MITER_BS6 (tuned + ensemble)']:.5g} | {t12_val} | "
            f"{row20['CAT (default)']:.5g} | {row20['CAT (tuned + ensemble)']:.5g} | "
            f"{row20['GBM (tuned + ensemble)']:.5g} |"
        )
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("run")
    sp.add_argument("--shard", type=int, default=0)
    sp.add_argument("--nshards", type=int, default=1)
    sub.add_parser("summarize")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args.shard, args.nshards)
    elif args.cmd == "summarize":
        cmd_summarize()


if __name__ == "__main__":
    main()
