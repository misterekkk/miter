# Pre-registration: n-adaptive tree-complexity default (M7 candidate), Stage 1

Frozen before any run of `benchmarks/check_m7_adaptive_complexity.py` (SHA-256 of this file recorded in
`m7_adaptive_complexity_summary.md`). Measurement only: no defaults in `src/` or `python/` are changed; rules are applied
by the benchmark script through explicit constructor arguments.

## Motivation
`diag_tabarena_gaps_summary.md` (exploratory, post hoc): on concrete (n=1030) `num_leaves` 7 / `min_data_in_leaf` 40 removes
almost all of the single-model gap to CatBoost; on miami (n=13776) lr 0.02 / 7 leaves help about 1.5-2 %; on airfoil
(n=1503) smaller trees HURT (leaves7 +3.0 %, leaves4 +8.2 %). A global change is therefore not acceptable; this study tests
n-adaptive rules broadly.

## Problem of the standard suites and the design consequence
In the 16 regression (OpenML suite 336) and 13 classification (suite 337) datasets of `bench_m3_oblique.py` the fit set
(80 % of the outer train fold) has between 3007 (abalone) and about 52 000 rows; only abalone is near the small-n regime.
Run at native size, rules keyed on n < 3000 would be inactive everywhere. To test the rules where they act (and to include
airfoil-like cases where small trees hurt) every dataset is additionally run on random subsamples of its outer training
fold ("levels"). This is a learning-curve design, not new datasets; the subsampled problems are smaller versions of the same
tasks and are not independent of the native ones.

## Candidate rules (n = number of rows given to `fit`, i.e. the fit set WITHOUT the early-stopping set)
Baseline B = current library defaults: lr 0.05, num_leaves 31, min_data_in_leaf 20 (everything else default, m_mult 2).
- **R1**: n < 3000 -> num_leaves 15, min_data_in_leaf 40. (Conservative: concrete leaves15 -2.0 %, mdl40 -2.75 %; airfoil
  not measured for this exact combination.)
- **R2**: n < 3000 -> num_leaves 7, min_data_in_leaf 40. (Strongest on concrete, -3.5 %; riskiest on airfoil, leaves7 +3.0 %.)
- **R3**: R1, plus learning_rate 0.03 for n < 20000 (lr 0.02-0.03 helped miami/concrete by 0.5-0.9 % at no cost on airfoil).
  So for 3000 <= n < 20000 only the lr changes; for n < 3000 leaves 15, mdl 40, lr 0.03.
Thresholds (3000, 20000) were fixed from the diagnosis before running and are not tuned.

## Reference
CatBoost default (`CatBoostRegressor` / `CatBoostClassifier`, auto learning rate, depth 6, `iterations=5000`,
`early_stopping_rounds=50` on the same early-stopping split, `thread_count=1`, `random_seed=fold`), reference column only;
not part of any test. Gap to CatBoost = 100*(miter - cat)/cat.

## Data, levels, splits
- Datasets: the 16 + 13 of `bench_m3_oblique.py` (REG_DATASETS / CLF_DATASETS), OpenML outer split repeat 0, folds 0..4.
- Levels: fit size N in {500, 1500, 5000} plus "native" (full outer train fold). A level is used only if N is smaller than
  the native fit size. For a subsampled level, the outer train indices are subsampled (without replacement,
  `numpy.random.default_rng(1000+fold)`) to round(N/0.8) rows, then split 80/20 with
  `train_test_split(test_size=0.2, random_state=0)`; native uses `train_test_split` on the full outer train fold. Test = full
  outer test fold. Early stopping 50 rounds on the 20 % part, max 5000 trees, `n_jobs=1`, miter `random_state=fold`.
- Native level with fit size >= 20000: all three rules are inactive by construction (identical parameters); these cells
  are not run (cost), and no claim is made about them.
- Inactive cells (rule parameters equal to baseline) are run for real when fit size <= 5000 and must be bit-identical to
  the baseline (test metric equal to all printed digits); a mismatch is reported as a failure of the check.
  Inactive cells with larger n are identical by construction (the script verifies the parameter dicts are equal).
- If the pilot projects more than ~6 h wall time, native levels of the largest classification datasets are dropped, decided
  on pilot time only, not on results, and documented.

## Metrics
- Regression: nRMSE = RMSE / std(y_test) on the outer test fold. Classification: test log-loss (probabilities clipped to
  [1e-15, 1-1e-15]). Per (dataset, level): mean over the 5 folds. Relative change vs baseline = 100*(rule - B)/B
  (negative = better).

## Primary test (separately for regression and classification, separately for each rule)
Active cells of a rule = (dataset, level) cells where the rule changes at least one parameter (R1/R2: fit size < 3000, i.e.
levels 500 and 1500 (and abalone never); R3: fit size < 20000). Unit of analysis = dataset: the mean over its active levels of
the relative change. Statistics over datasets: mean, **median**, wins/losses (rule better/worse than B), two-sided Wilcoxon
signed-rank p on the per-dataset means (rule vs B), worst dataset. Per-(dataset, level) cells are reported too.

## Acceptance criterion (pre-registered)
A rule PASSES for a task type iff ALL hold on the per-dataset units:
1. median relative change < 0 (improvement);
2. no dataset worse than +2.0 %;
3. Wilcoxon p < 0.05;
4. inactive cells that were run are bit-identical to baseline.
A rule passes Stage 1 overall iff it passes for regression AND for classification the median is <= 0 and no dataset is worse
than +2.0 %. Only a rule passing for regression proceeds to Stage 2 (TabArena regression). If several rules pass, the one
with the lowest median regression change goes to Stage 2. No multiplicity correction is applied across the 3 rules
(p-values are reported raw; with 3 rules, a Bonferroni threshold of 0.0167 is also shown but the criterion uses 0.05).
Everything else (per-level breakdowns, CatBoost gap, best/worst cells, runtime/trees) is exploratory; any analysis added after
seeing results is labelled post hoc.

## Known limitations (stated in advance)
Subsampled levels are not independent datasets; 16 regression units and 13 classification units give limited power;
tests of 3 rules are not corrected; all models are single models with early stopping on a 20 % split (not TabArena bagging).
