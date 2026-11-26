# Pre-registration: Native C++20 Online Ordered CTR v2 (`exp-native-ctr-v2`)

- Branch: `feat-v11-softmax-cat-ctr` (code at commit `fabcf52`)
- Status: **FROZEN BEFORE RUNNING `exp-native-ctr-v2`**

## 1. Hypothesis

Adding native C++20 CatBoost-style **online ordered TargetMean CTRs** (`max_ctr_complexity=2` with dynamic on-the-fly category combinations, 1 permutation default `ctr_n_permutations=1`, multi-prior `{0.0, 0.5, 1.0}` in classification, normalized `Counter`, complexity regularization `model_size_reg=0.5`, and cached 1-byte quantized columns) inside symmetric depth-6 trees (`S6_CTR2`) significantly reduces test error on categorical tabular datasets over symmetric depth-6 K1/Oblique alone (`S6`), without statistical fluke, while remaining fast enough for production use.

## 2. Datasets & Protocol (16 OpenML datasets × 5 folds = 80 paired runs/variant)

- Reg (6): `abalone` (`did=44956`), `diamonds` (`42225`), `Mercedes_Benz` (`42570`), `avocado-sales` (`41210`), `nyc-taxi-green-dec-2016` (`42729`), `colleges` (`42727`), plus `Food_Delivery_Time` (`tid=363672`).
- Binary (9): `credit-g` (`31`), `bank-marketing` (`1461`), `adult` (`1590`), `kick` (`41162`), `Amazon_employee_access` (`4135`), `Click_prediction_small` (`42733`), `Bank_Customer_Churn` (`tid=363619`), `in_vehicle_coupon_recommendation` (`tid=363681`), `online_shoppers_intention` (`tid=363691`).
- Split protocol: 5 OpenML folds (`fold = 0..4`), max `25,000` rows subsampled deterministically per dataset (`random_state=42` when `n > 25,000`, stratified for classification), `20%` inner validation split (`train_test_split(..., test_size=0.2, random_state=0)`), `n_jobs = 1`, `random_state = fold`.
- Fixed hyperparameters: `n_estimators = 5000`, `learning_rate = 0.05`, `m_mult = 2`, `k_proj = 0`, `max_bin = 255`, `early_stopping_rounds = 50`.
- Primary metric: **nRMSE** (`RMSE / std(y_te)`) for regression; **LogLoss** (`clip(p, 1e-15, 1-1e-15)`) for binary classification (lower is better).
- Secondary exploratory metric (binary): **`1 - ROC_AUC`** (lower is better).

## 3. Variants

1. `S6`: `grow_policy="symmetric"`, `max_depth=6`, `min_data_in_leaf=1`, `lambda_l2=3.0`, `max_ctr_complexity=0`
2. `S6_CTR1`: `S6` + `max_ctr_complexity=1` (1D native online ordered CTRs)
3. `S6_CTR2`: `S6` + `max_ctr_complexity=2` (1D + dynamic pair native online ordered CTRs)
4. `CAT`: `CatBoost` reference (`iterations=5000, depth=6, early_stopping_rounds=50, thread_count=1`, native `cat_features`).

## 4. Anti-Fluke Confirmatory Rules

Evaluated on per-dataset 5-fold mean relative error delta `(CAND - BASE) / BASE * 100%` (unit = dataset, `N = 16`). Two pre-specified confirmatory comparisons — **Primary: `S6_CTR2 vs S6`** and **Secondary: `S6_CTR1 vs S6`** — corrected via **Holm-Bonferroni** across `m = 2`:

A comparison **PASSES** (not statistical noise) iff **ALL 6 criteria** hold:
1. **Effect size**: Median relative Δ `< -0.5%`.
2. **Adjusted significance**: Two-sided Wilcoxon signed-rank `p_holm < 0.05`.
3. **Bootstrap CI**: **95% paired bootstrap confidence interval of median relative Δ** (10,000 dataset resamples, `seed = 2026`) has **upper bound strictly `< 0.0%`**.
4. **Consistency**: Wins `>= 11/16`.
5. **Tail safety**: Worst single-dataset regression `<= +1.5%`.
6. **Runtime**: Median per-fit wall time ratio vs `S6` `<= 3.0x` (targeting faster than `CAT`).

Exploratory comparisons additionally reported without confirmatory claims: `S6_CTR2 vs CAT`, `S6_CTR2 vs S6_CTR1`, binary `1 - ROC_AUC`, and the high-cardinality subgroup (`max_card > 253`).
