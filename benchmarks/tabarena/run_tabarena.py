"""Run miter C++ through the TabArena v0.1 protocol (8-fold bagging, AutoGluon preprocessing), then compare with cached GBMs.
Usage: run_tabarena.py <exp> <lite|full> <dataset> [<dataset> ...]
Env: TABARENA_CACHE (cache with LightGBM/CatBoost/XGBoost results), T2_NOCOMPARE=1 to skip compare.
Raw output: benchmarks/results/raw/tabarena/<exp> (gitignored)."""

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
# private OpenML cache: the shared ~/.openml holds pickles written by pandas 3 (other agents), unreadable with TabArena's pandas 2.3.3
import openml

openml.config.set_root_cache_directory(
    os.path.abspath(HERE + "/../results/raw/openml_cache_pandas2")
)
from miter_tabarena_model import MiterCppAccModel, MiterCppModel
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

exp, mode, names = sys.argv[1], sys.argv[2], sys.argv[3:]
experiments = TabArenaV0pt1ExperimentBundle(
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
t0 = time.time()
kw = dict(dataset_names=names)
if mode == "lite":
    kw["split_indices"] = "lite"
res = ctx.build_and_run_jobs(
    experiments,
    expname=f"{ROOT}/{exp}",
    new_result_prefix="[New] ",
    build_kwargs=kw,
    debug_mode=True,
)
print("RUN_DONE", len(res), "jobs", f"{time.time() - t0:.0f}s", flush=True)
if os.environ.get("T2_NOCOMPARE"):
    sys.exit()
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
ta = ctx.load_results(
    methods=["LightGBM", "CatBoost", "XGBoost"] + sorted(ctx._new_method_names)
)
out = f"{ROOT}/{exp}_eval"
lb, per = ctx.compare(
    output_dir=out,
    ta_results=ta,
    datasets=names,
    subset=("lite" if mode == "lite" else None),
    return_results=True,
    plot=False,
    fillna=None,
    calibration_method=None,
)
print(lb.to_string())
per.to_csv(out + "_per_split.csv", index=False)
print("COMPARE_DONE", f"{time.time() - t0:.0f}s", flush=True)
