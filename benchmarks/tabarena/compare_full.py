"""Leaderboard on the 13 regression datasets vs the FULL TabArena method pool (official compare() defaults) + GBDT-only view.
Usage: compare_full.py <exp> <outdir> ; reads cached miter results (all jobs must exist), no training."""

# ruff: noqa: E402  (OpenML cache dir and sys.path must be set before tabarena/miter imports)
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(HERE + "/../results/raw/tabarena")
os.environ.setdefault(
    "TABARENA_CACHE",
    "/Users/kacperpiotrowski/Studia/Projekt/histree/research/results/tabarena_cache",
)
sys.path.insert(0, HERE)
import openml

openml.config.set_root_cache_directory(ROOT + "/../openml_cache_pandas2")
import pandas as pd
from miter_tabarena_model import MiterCppAccModel, MiterCppModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

exp, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
names = open(ROOT + "/logs/all13").read().split()
exps = TabArenaV0pt1ExperimentBundle(
    models=[
        (
            (
                MiterCppAccModel
                if os.environ.get("MITER_ACC") == "1"
                else MiterCppModel
            ).config_generator(),
            0,
        )
    ]
).build_experiments()
ctx = TabArenaContext()
res = ctx.build_and_run_jobs(
    exps,
    expname=f"{ROOT}/{exp}",
    new_result_prefix="[New] ",
    build_kwargs=dict(dataset_names=names),
    debug_mode=True,
)
if os.environ.get("EXP_ACC"):  # also score the accurate-preset run (cached) in the same table
    exps2 = TabArenaV0pt1ExperimentBundle(
        models=[(MiterCppAccModel.config_generator(), 0)]
    ).build_experiments()
    res += ctx.build_and_run_jobs(
        exps2,
        expname=f"{ROOT}/{os.environ['EXP_ACC']}",
        new_result_prefix="[New] ",
        build_kwargs=dict(dataset_names=names),
        debug_mode=True,
    )
new = sorted(ctx._new_method_names)
print("jobs", len(res), "new", new, flush=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 500)
for tag, methods in (
    ("full_pool", None),
    ("gbdt_only", ["LightGBM", "CatBoost", "XGBoost"]),
):
    t0 = time.time()
    ml = list(dict.fromkeys((methods if methods else ctx.methods) + new))
    print(tag, len(ml), "methods", flush=True)
    ta = ctx.load_results(methods=ml, download_results=False)
    d = ta[
        ta.duplicated(["dataset", "fold", "method"], keep=False)
        & ta.dataset.isin(names)
    ]
    print(
        "dups on 13 datasets:",
        len(d),
        d.method.value_counts().head(10).to_dict(),
        flush=True,
    )
    lb, per = ctx.compare(
        output_dir=f"{out}/{tag}",
        ta_results=ta,
        datasets=names,
        return_results=True,
        plot=False,
        **({} if methods is None else dict(fillna=None, calibration_method=None)),
    )  # gbdt-only pool lacks the fillna/calibration reference method; official defaults: fillna/calibration "auto"
    lb.to_csv(f"{out}/{tag}_leaderboard.csv", index=False)
    per.to_csv(f"{out}/{tag}_per_split.csv", index=False)
    print(tag, f"{time.time() - t0:.0f}s", lb.shape)
    print(
        lb[
            [
                "method",
                "elo",
                "elo+",
                "elo-",
                "rank",
                "winrate",
                "improvability",
                "imputed",
            ]
        ].to_string(),
        flush=True,
    )
