# Pre-registration: Oblivious + Oblique (`grow_policy="symmetric"`, S6) vs Leaf-wise (`miter default`)

Frozen before writing or running `benchmarks/bench_exp_symmetric_oblique.py` (the SHA-256 of this file is recorded in `benchmarks/results/exp_symmetric_oblique_summary.md`).

## 1. Hypothesis

Symmetric (oblivious) depth-6 trees combined with `miter`'s integer Q16r oblique splits (`R_uni`), axis splits, and K1 categorical splits provide two potential benefits over leaf-wise 31-leaf `miter default`:
1. **Structural regularization and portfolio diversity:** sharing 6 hyperplanes across 64 leaves lowers variance on smooth/noisy tabular targets, either improving single-model accuracy directly (`S6`) or improving accuracy when blended 50/50 with leaf-wise `miter` (`B+S6`).
2. **Large-`n` training speed:** scanning level-by-level across a flat `leaf_ids` array eliminates leaf row partitioning and `work_codes` column copying while maintaining deterministic chunk-ordered summation across OpenMP thread counts.

## 2. Variants and Protocol

All `miter` variants share: `learning_rate=0.05`, `n_estimators=5000`, `early_stopping_rounds=50`, `max_bin=255`, `m_mult=2`, `k_proj=0` (`k ∈ {2, 3}`), `oblique_max_depth=-1`, `n_jobs=1` (for all quality runs), `random_state=fold`. Categorical columns are passed natively via pandas `category` dtype (matching `K1`), and `NaN` values remain native.

### Primary confirmatory variants
- **`B` (baseline):** `grow_policy="leafwise"`, `num_leaves=31`, `min_data_in_leaf=20`, `lambda_l2=1.0`.
- **`S6` (symmetric oblique depth 6):** `grow_policy="symmetric"`, `max_depth=6`, `min_data_in_leaf=1`, `lambda_l2=3.0`.
- **`B+S6` (fixed 50/50 portfolio blend):** average test predictions of `B` and `S6` (`0.5 * (pred_B + pred_S6)` for regression; `0.5 * (proba_B + proba_S6)` for binary class 1 probabilities), with no fitted or tuned weights.

### Exploratory reference columns (outside confirmatory hypothesis tests)
- `S6_l1`: `grow_policy="symmetric"`, `max_depth=6`, `min_data_in_leaf=1`, `lambda_l2=1.0`.
- `CAT`: `CatBoostRegressor` / `CatBoostClassifier` (`iterations=5000`, auto lr, depth 6, `early_stopping_rounds=50`, `thread_count=1`, `random_seed=fold`).
- `LGB`: `LGBMRegressor` / `LGBMClassifier` (`n_estimators=5000`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `reg_lambda=1.0`, `early_stopping_rounds=50`, `n_jobs=1`, `random_state=fold`).

## 3. Datasets, Splits, and Metrics

- **Confirmatory regression (16 datasets, `REG_DATASETS`, OpenML suite 336):** `abalone`, `wine_quality`, `cpu_act`, `yprop_4_1`, `sulfur`, `Brazilian_houses`, `Ailerons`, `MiamiHousing2016`, `pol`, `elevators`, `Bike_Sharing_Demand`, `houses`, `superconduct`, `house_sales`, `house_16H`, `diamonds`. Metric: `nRMSE = RMSE / std(y_test)` on each outer test fold.
- **Confirmatory binary classification (13 datasets, `CLF_DATASETS`, OpenML suite 337):** `eye_movements`, `heloc`, `pol`, `bank-marketing`, `default-of-credit-card-clients`, `MagicTelescope`, `house_16H`, `credit`, `california`, `electricity`, `jannis`, `Diabetes130US`, `MiniBooNE`. Metric: binary log-loss with predicted probabilities clipped to `[1e-15, 1 - 1e-15]`.
- **Splitting protocol (identical to `check_m7_adaptive_complexity.py`):** OpenML outer split repeat 0, folds `0..4` (5 folds per dataset); within each outer training fold, fixed 80/20 split via `train_test_split(..., test_size=0.2, random_state=0)` for fit / early-stopping validation.
- **Unit of analysis:** dataset (mean metric across the 5 folds per dataset). Relative change (%) vs `B` = `100 * (metric_X - metric_B) / metric_B` (negative = lower error = better).
- **Separate exploratory datasets (excluded from all confirmatory tests because the symmetric hypothesis was partially motivated by them):** TabArena tasks `concrete_compressive_strength` (`363625`), `miami_housing` (`363686`), `airfoil_self_noise` (`363612`), 5 folds.
- **Speed and memory benchmarks (synthetic float64):**
  - Shapes `(20000, 50)`, `(100000, 50)`, `(500000, 50)` at `n_jobs=1` and `n_jobs=4` (median of 5 runs after 1 warmup run, `ms/tree` over 25 trees), plus full fit time to early stopping on a synthetic `500000 x 50` regression dataset (`20%` validation split, `early_stopping_rounds=50`) and Peak RSS on `500000 x 50`. Compare `B`, `S6`, `LGB`, `CAT`.

## 4. Primary Acceptance Criteria (PASS/FAIL, evaluated separately per scenario)

- **Scenario A1 (solo quality win of `S6` over `B`), evaluated separately on Regression (16) and Classification (13):**
  1. Median relative change < `-0.5%`;
  2. Two-sided Wilcoxon signed-rank `p < 0.05` on per-dataset fold means;
  3. Worst dataset relative change `<= +3.0%`.
- **Scenario A2 (solo speed mode viability of `S6` over `B`):**
  1. Median relative error change `<= +0.5%` on **both** Regression (16) and Classification (13);
  2. Speedup `>= 1.5x` (`B / S6`) on `500000 x 50` (`ms/tree` at both `T=1` and `T=4`) AND total wall time to early stopping on `500000 x 50` strictly lower than `B`.
- **Scenario B (portfolio diversity win of `B+S6` over `B`), evaluated separately on Regression (16) and Classification (13):**
  1. Median relative change < `-0.5%`;
  2. Wins `>= 12/16` (Regression) or `>= 10/13` (Classification);
  3. Two-sided Wilcoxon signed-rank `p < 0.05`;
  4. Worst dataset relative change `<= +1.0%`.

Any analysis, threshold adjustment, or variant added after observing benchmark results is outside the confirmatory tests and must be labelled post hoc / exploratory.
