# Pre-registration: CatBoost-style Categorical CTRs (`exp-cat-ctr`)

- Branch: `feat-v11-softmax-cat-ctr` (from `main` @ `ef4c18b`)
- Status: **FROZEN BEFORE ANY C++ OR PYTHON CTR / SOFTMAX CODE**

## 1. Hypothesis

In `exp-cat-oblique-ts`, 1D OOF target statistics (`cat_oblique_ts=True`) improved high-cardinality categorical datasets (median `-1.82%` for `max_card > 253`), while `MITER_T12` still trailed CatBoost on multi-category interaction tasks (`Amazon_employee_access`, `in_vehicle_coupon_recommendation`, `credit-g`). Adding **CatBoost-style ordered prefix target statistics** (`cat_ctr_mode="ordered"`, 4 deterministic random permutations), **greedy OOF-ranked categorical pair combinations** (`cat_ctr_pairs=16`), and **leave-one-out log-frequency counters** (`cat_ctr_counter=True`) as additional Q16r oblique-eligible columns — while preserving native K1 categorical columns and leaving `cat_oblique_ts=False` bit-identical — significantly reduces test error on categorical tabular benchmarks without statistical fluke.

## 2. Datasets & Protocol (16 OpenML datasets × 5 folds = 80 paired runs/variant)

- Original 12 OpenML categorical datasets from `exp-cat-oblique-ts`:
  - Reg (6): `abalone` (`did=44956`), `diamonds` (`42225`), `Mercedes_Benz` (`42570`), `avocado-sales` (`41210`), `nyc-taxi-green-dec-2016` (`42729`), `colleges` (`42727`).
  - Binary (6): `credit-g` (`31`), `bank-marketing` (`1461`), `adult` (`1590`), `kick` (`41162`), `Amazon_employee_access` (`4135`), `Click_prediction_small` (`42733`).
- Plus 4 TabArena multi-category tasks (`tid` in TabArena / OpenML):
  - `Food_Delivery_Time` (`tid=363672`, Reg)
  - `Bank_Customer_Churn` (`tid=363619`, Binary)
  - `in_vehicle_coupon_recommendation` (`tid=363681`, Binary)
  - `online_shoppers_intention` (`tid=363691`, Binary)
- Split protocol: 5 OpenML folds (`fold = 0..4`), max `25,000` rows subsampled deterministically per dataset (`random_state=42` when `n > 25,000`, stratified for classification), `20%` inner validation split (`train_test_split(..., test_size=0.2, random_state=0)`), `n_jobs = 1`, `random_state = fold`.
- Fixed hyperparameters: `n_estimators = 5000`, `learning_rate = 0.05`, `m_mult = 2`, `k_proj = 0`, `max_bin = 255`, `early_stopping_rounds = 50`.
- Metric: **nRMSE** (`RMSE / std(y_te)`) for regression; **LogLoss** (`clip(p, 1e-15, 1-1e-15)`) for binary classification (lower is better).

## 3. Variants

1. `B`: `grow_policy="leafwise"`, `num_leaves=31`, `min_data_in_leaf=20`, `lambda_l2=1.0`, `cat_oblique_ts=False`
2. `TS1`: `B` + `cat_oblique_ts=True`, `cat_ctr_mode="oof"`, `cat_ctr_pairs=0`, `cat_ctr_counter=False`
3. `CTR`: `B` + `cat_oblique_ts=True`, `cat_ctr_mode="ordered"`, `cat_ctr_pairs=16`, `cat_ctr_counter=True`
4. `CTR_S6`: `grow_policy="symmetric"`, `max_depth=6`, `min_data_in_leaf=1`, `lambda_l2=3.0` + `cat_oblique_ts=True`, `cat_ctr_mode="ordered"`, `cat_ctr_pairs=16`, `cat_ctr_counter=True`
5. `CAT`: `CatBoost` reference (`iterations=5000, depth=6, early_stopping_rounds=50, thread_count=1`).

## 4. Anti-Fluke Confirmatory Rules

Evaluated on per-dataset 5-fold mean relative error delta `(CAND - BASE) / BASE * 100%` (unit = dataset, `N = 16`). Three pre-specified confirmatory comparisons (`CTR vs B`, `CTR_S6 vs B`, `CTR vs TS1`), corrected via **Holm-Bonferroni** across `m = 3`:

A comparison **PASSES** (not statistical noise) iff **ALL 6 criteria** hold:
1. **Effect size**: Median relative Δ `< -0.5%` (for `CTR vs B` and `CTR_S6 vs B`) or `< -0.3%` (for incremental ablation `CTR vs TS1`).
2. **Adjusted significance**: Two-sided Wilcoxon signed-rank `p_holm < 0.05` (Holm-Bonferroni adjusted across the 3 confirmatory comparisons).
3. **Bootstrap CI**: **95% paired bootstrap confidence interval of median relative Δ** (10,000 dataset resamples, `seed = 2026`) has **upper bound strictly `< 0.0%`**.
4. **Consistency**: Wins `>= 11/16`.
5. **Tail safety**: Worst single-dataset regression `<= +1.5%`.
6. **Runtime**: Median per-fit wall time overhead `<= 1.35x` vs baseline (`CTR / B <= 1.35` and `CTR_S6 / B <= 1.35`).
