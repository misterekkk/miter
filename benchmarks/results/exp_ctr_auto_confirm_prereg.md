# Pre-registration: Confirmatory Adaptive Native CTR Benchmark (`exp-ctr-auto-confirm`)

- Branch: `feat-v11-softmax-cat-ctr` (code at commit `062cc6b`)
- Status: **FROZEN BEFORE RUNNING `exp-ctr-auto-confirm`**

## 1. Hypothesis

On a completely unseen set of 16 categorical/mixed OpenML datasets (zero overlap with `exp-native-ctr-v2` / `exp-ctr-auto` / `exp-ctr-auto-v2`), both adaptive multi-permutation native CTR configurations — **`S6_AUTO_P4`** (complexity `0/1/2/3`, support gate + sample-size-aware penalty, `4` online permutations) and **`S6_NC_SMART_P4`** (adding gated depth<=1 non-onehot `num x cat` combinations) — significantly outperform `S6` (`max_ctr_complexity=0`) without tail regressions. Head-to-head comparison tests whether `S6_NC_SMART_P4` earns a statistically meaningful gain over simpler `S6_AUTO_P4` to justify its 4 extra controls.

## 2. Unseen Datasets & Protocol (16 datasets × 5 folds = 80 paired runs/variant)

ZERO overlap with any dataset used in `exp-native-ctr-v2`, `exp-ctr-auto`, or `exp-ctr-auto-v2`:
- Binary (12): `Diabetes130US` (`did=46922`), `E-CommereShippingData` (`46924`), `Fitness_Club` (`46927`), `HR_Analytics_Job_Change` (`46935`), `Is-this-a-good-customer` (`46938`), `Marketing_Campaign` (`46940`), `churn` (`46915`), `coil2000` (`46916`), `credit_card_clients_default` (`46919`), `customer_satisfaction_in_airline` (`46920`), `kddcup09_appetency` (`46939`), `qsar-biodeg` (`46952`).
- Multiclass (3): `anneal` (`46906`), `splice` (`46958`), `students_dropout` (`46960`).
- Regression (1): `healthcare_insurance_expenses` (`46931`).

Split protocol: deterministic cap at `25,000` rows (`random_state=42`, stratified for `binary`/`multiclass`), 5 folds (`random_state=42`), inner 20% validation (`random_state=0`), `n_jobs = 1`, `random_state = fold`.
Fixed hyperparameters: `n_estimators=5000, learning_rate=0.05, m_mult=2, k_proj=0, max_bin=255, early_stopping_rounds=50`.
Primary metric: **nRMSE** for regression; **LogLoss** (`clip(p, 1e-15, 1-1e-15)`) for binary/multiclass (lower is better). Secondary exploratory metric (binary): **`1 - ROC_AUC`**.

## 3. Variants (4)

1. `S6`: `grow_policy="symmetric", max_depth=6, min_data_in_leaf=1, lambda_l2=3.0, max_ctr_complexity=0`
2. `S6_AUTO_P4`: `S6` + `max_ctr_complexity="auto", ctr_model_size_reg=0.5, ctr_num_combinations=False, ctr_n_permutations=4`
3. `S6_NC_SMART_P4`: `S6` + `max_ctr_complexity="auto", ctr_model_size_reg=0.5, ctr_num_combinations=True, ctr_n_permutations=4, ctr_num_max_depth=1, ctr_num_skip_onehot=True, ctr_num_gain_margin=1.25, ctr_new_table_penalty=0.25`
4. `CAT`: CatBoost reference (`iterations=5000, learning_rate=0.05, depth=6, early_stopping_rounds=50, thread_count=1`, native `cat_features`).

## 4. Anti-Fluke Confirmatory Rules (Holm-Bonferroni `m = 3`)

Evaluated across `N = 16` datasets using per-dataset 5-fold mean relative error delta `(CAND - BASE) / BASE * 100%`, with Holm-Bonferroni correction across **`m = 3`** hypotheses:
1. `S6_AUTO_P4 vs S6`
2. `S6_NC_SMART_P4 vs S6`
3. `S6_NC_SMART_P4 vs S6_AUTO_P4` (head-to-head complexity test)

A primary comparison (`vs S6`) **PASSES** iff **ALL 6 criteria** hold:
1. Median relative Δ `< -0.25%`;
2. Two-sided Wilcoxon `p_holm < 0.05`;
3. 95% paired bootstrap CI upper bound of median Δ `< 0.0%` (10,000 resamples, `seed = 2026`);
4. Wins `>= 11/16`;
5. Worst single-dataset regression `<= +1.5%`;
6. Median per-fit wall-time ratio vs `S6` `<= 3.0x`.

Head-to-head **`S6_NC_SMART_P4 vs S6_AUTO_P4`** justifies keeping `SMART` over simpler `S6_AUTO_P4` iff:
- Two-sided Wilcoxon `p_holm < 0.05`, Median Δ `< -0.10%`, and Wins `>= 10/16` (otherwise parsimony favors `S6_AUTO_P4`).

Exploratory comparisons additionally reported without confirmatory claims: `vs CAT`, binary `1 - ROC_AUC`, and subgroups (`p_cat`, `max_card > 10`).
