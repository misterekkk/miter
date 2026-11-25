# Pre-registration: MITER_T12 on all TabArena `lite` regression + binary tasks (43)

Written and hashed before any of the new runs. Purpose: check whether the exploratory result of
`tabarena_ensemble_lite_summary.md` holds on the full set of regression and binary tasks. In that
result, `MITER_T12 (tuned + ensemble)` was #1 in the GBDT pool, with Elo 1228.4 vs CAT
(tuned + ensemble) 1167.4 on 17 tasks.

## Fixed in advance (no change after seeing results)
- **Methods:**
  - `MITER_T12` exactly as in `benchmarks/tabarena/run_tabarena_ensemble.py` at commit `6640218`:
    5 manual configs (default, S6, accurate, S6+accurate, S6+cat_oblique_ts) plus 7 random configs
    from `generate_configs_miter(7, seed=1234)`.
  - `MITER_BS6` (default + S6).
  - miter build: the one installed in `.venv_tabarena`, built from `ebb7d3b` (C++ and Python
    package unchanged in `6640218`; `python/miter/*.py` verified identical to the repo). It is not
    rebuilt during this experiment.
- **Tasks:**
  - All 13 regression and all 30 binary tasks of TabArena v0.1, `lite` (repeat 0, fold 0),
    8-fold bagging, AutoGluon preprocessing, task time limit as in the protocol.
  - The 20 tasks already computed are reused from the cache, not re-run.
  - The 8 multiclass tasks are excluded (miter v1.0 is binary only).
- **Comparison:** `TabArenaContext.compare(subset="lite")` on all 43 tasks.
  - `gbdt_only` pool: LightGBM, CatBoost, XGBoost (all variants) plus the miter methods, with
    `fillna=None` and `calibration_method=None`.
  - `full_pool`: all cached methods, default settings.
  - Elo with its reported confidence bounds.

## Primary question and criterion
- **Primary:** in the 43-task `gbdt_only` pool, is `MITER_T12 (tuned + ensemble)` Elo ≥ `CAT (tuned + ensemble)` Elo?
- **Reported in all cases:**
  - Elo ± bounds for both.
  - Per-task paired comparison of T12 (tuned + ensemble) vs CAT (tuned + ensemble): wins/43,
    median relative error difference, Wilcoxon signed-rank p (two-sided).
  - The same split separately for the regression (13) and binary (30) tasks.
- **Claim rule:**
  - "miter T12 is ahead of CatBoost tuned + ensemble on TabArena reg+binary lite" only if the Elo
    point estimate is higher AND the Wilcoxon p < 0.05 in miter's favour.
  - Otherwise, report as "level with" (p ≥ 0.05) or "behind".
  - No claim about the full TabArena (multiclass missing, single fold).

## Known limitations (stated in advance)
- Single split per task.
- 4 of the 5 manual configs were chosen from earlier miter experiments; part of those used datasets
  that also appear in TabArena (houses, diamonds, wine_quality).
- CatBoost / LightGBM have up to 200 configs vs 12 for miter.
- The machine is shared (timings are approximate).
