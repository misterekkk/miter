# Pre-registration: Native Softmax Multiclass & Full 51-Task TabArena `lite` (`exp-softmax-tabarena51`)

- Branch: `feat-v11-softmax-cat-ctr` (from `main` @ `ef4c18b`)
- Status: **FROZEN BEFORE ANY C++ OR PYTHON CTR / SOFTMAX CODE**

## Part A: Confirmatory Multiclass Softmax vs One-vs-Rest (12 Datasets × 5 Folds)

### A.1 Datasets & Protocol
- 12 OpenML multiclass benchmarks (`5` folds each, `n_jobs = 1`, `random_state = fold`, `20%` inner val split with `random_state = 0` stratified when `min_class_count >= 5`):
  - **8 TabArena `lite` multiclass tasks:** `anneal` (`tid=363614`), `hiva_agnostic` (`363677`), `maternal_health_risk` (`363685`), `MIC` (`363711`), `SDSS17` (`363699`), `splice` (`363702`), `students_dropout_and_academic_success` (`363704`), `website_phishing` (`363707`).
  - **4 OpenML CC18 multiclass benchmarks:** `satimage` (`did=182`, 6 classes), `segment` (`40984`, 7 classes), `pendigits` (`32`, 10 classes), `letter` (`6`, 26 classes).
- Fixed hyperparameters: `n_estimators = 5000`, `learning_rate = 0.05`, `m_mult = 2`, `k_proj = 0`, `max_bin = 255`, `early_stopping_rounds = 50`, `n_jobs = 1`.
- Metric: **Multiclass LogLoss** on test fold probabilities (`clip(p, 1e-15, 1 - 1e-15)`, row-normalized, lower is better). Accuracy reported secondarily.

### A.2 Variants
1. `OVR`: K independent binary `MiterClassifier` models (each with leafwise defaults `num_leaves=31, min_data_in_leaf=20, lambda_l2=1.0`, normalized `p_k / sum p_k`).
2. `SM_B`: Native `Objective::Multiclass` Softmax + `grow_policy="leafwise"`, `num_leaves=31`, `min_data_in_leaf=20`, `lambda_l2=1.0`.
3. `SM_S6`: Native `Objective::Multiclass` Softmax + `grow_policy="symmetric"`, `max_depth=6`, `min_data_in_leaf=1`, `lambda_l2=3.0`.
4. `SM_B+S6`: 50/50 probability average of `SM_B` and `SM_S6`.
5. `CAT`: `CatBoostClassifier(loss_function="MultiClass", iterations=5000, depth=6, early_stopping_rounds=50, thread_count=1)`.
6. `LGB`: `LGBMClassifier(objective="multiclass", n_estimators=5000, learning_rate=0.05, num_leaves=31, min_child_samples=20, reg_lambda=1.0, n_jobs=1)`.

### A.3 Anti-Fluke Confirmatory Rules (Part A)
- **Primary (`SM_B vs OVR`, `N = 12` datasets):** **PASSES** iff **ALL 6 criteria** hold:
  1. Median LogLoss relative Δ `< -2.0%`.
  2. Wins `>= 10/12`.
  3. Two-sided Wilcoxon signed-rank `p < 0.01`.
  4. **95% paired bootstrap CI of median Δ** (10,000 dataset resamples, `seed = 2026`) upper bound strictly `< 0.0%`.
  5. Worst single-dataset LogLoss regression `<= +1.0%`.
  6. Median fit wall-clock time `SM_B <= OVR` (faster or equal).
- **Secondary (`SM_B+S6 vs CAT`):** Claim "ahead of CatBoost MultiClass" ONLY iff `wins >= 9/12`, two-sided Wilcoxon `p < 0.05`, AND 95% paired bootstrap CI `< 0.0%`.

---

## Part B: TabArena `lite` Full 51 Tasks (`MITER_V11_T12`)

### B.1 Protocol
- Method: `MITER_V11_T12` (5 manual configs: default, `S6`, `accurate`, `S6+m4`, `S6+CTR` + 7 random configs from `generate_configs_miter_v11(7, seed=1234)`), 8-fold bagging, AutoGluon preprocessing, `time_limit=3600s`, `split_indices="lite"`, fold 0 across all **51 tasks** (13 reg + 30 binary + 8 multiclass).
- Multiclass uses native `Objective::Multiclass` softmax in `MiterModel`.

### B.2 Strict Claim Rule (Anti-Fluke)
- Primary pools: `TabArenaContext.compare(subset="lite")` on `gbdt_only` (`15` methods, `fillna=None`, `calibration_method=None`) and `full_pool` (`124` methods), reporting Elo ± bootstrap CI bounds and rank.
- Headline claim **"MITER_V11_T12 beats CatBoost (tuned + ensemble) on TabArena lite (51 tasks)"** is permitted **ONLY iff ALL 5 hold**:
  1. `MITER_V11_T12 (tuned + ensemble)` Elo > `CAT (tuned + ensemble)` Elo in `gbdt_only` (51 tasks).
  2. Paired two-sided Wilcoxon `p < 0.05` across all 51 tasks in miter's favour.
  3. 95% paired bootstrap CI (10,000 resamples, `seed = 2026`) of median relative error Δ vs `CAT (tuned + ensemble)` strictly `< 0.0%`.
  4. Wins `>= 33/51`.
  5. On the 8 multiclass tasks, `MITER_V11_T12 (tuned + ensemble)` strictly improves over cached `MITER_T12 (OvR)` (`>= 7/8` wins OR `p < 0.05`).
- Otherwise report accurately as **"level with CatBoost"** (`p >= 0.05` even if Elo is higher) or **"behind"**. Single-split caveat (`fold 0`) must always be stated.
