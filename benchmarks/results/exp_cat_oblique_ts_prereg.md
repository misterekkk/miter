# Pre-registration: Categoricals in Oblique Splits via Out-of-Fold Target Statistics (`exp-cat-oblique-ts`)

- Branch: `exp-cat-oblique-ts` (created from `main` @ `11afc68`)
- Status: **FROZEN BEFORE ANY CODE OR MODEL RUN**

## 1. Hypothesis

Native K1 categorical handling keeps categorical columns out of miter's random oblique `R_uni` projections (`k` features sampled only from numeric columns) and caps cardinality at `max_categories = 253` (bucketing rare tail categories into `OTHER`). Computing **deterministic 5-fold out-of-fold smoothed Target Statistics (`a = 10.0`)** per categorical column on the inner training fold and appending them as derived `float64` features (`cat_oblique_ts=True`, keeping native K1 columns alongside) lets categoricals participate directly in `Q16r` oblique projections AND preserves signal above 253 categories without changing `extern/lignum/` or C++ model formats.

We hypothesize that `TS` (`cat_oblique_ts=True`) improves out-of-fold generalization error over `B` (`cat_oblique_ts=False` K1 default), with the largest gains on medium/high-cardinality datasets (`max_card > 50` / `> 253`) and zero degradation on low-cardinality datasets.

## 2. Datasets (12 OpenML Real Categorical Datasets)

Datasets with `n > 30000` are deterministically subsampled to `n = 25000` using `np.random.default_rng(42)` (stratified by target class for binary classification, random permutation for regression). Non-numeric (`category` / `object`) columns stay `category` with missing values preserved.

| Kind | Dataset | OpenML ID | Full `n` | Eval `n` | `p_num` | `p_cat` | `max_card` | `max_card > 50` | `max_card > 253` |
|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| reg | `abalone` | 44956 | 4177 | 4177 | 7 | 1 | 3 | no | no |
| reg | `diamonds` | 42225 | 53940 | 25000 | 6 | 3 | 8 | no | no |
| reg | `Mercedes_Benz_Greener_Manufacturing` | 42570 | 4209 | 4209 | 368 | 8 | 47 | no | no |
| reg | `avocado-sales` | 41210 | 18249 | 18249 | 8 | 3 | 54 | **yes** | no |
| reg | `nyc-taxi-green-dec-2016` | 42729 | 581835 | 25000 | 9 | 9 | 259 | **yes** | **yes** |
| reg | `colleges` | 42727 | 7063 | 7063 | 32 | 12 | 6039 | **yes** | **yes** |
| clf | `credit-g` | 31 | 1000 | 1000 | 7 | 13 | 10 | no | no |
| clf | `bank-marketing` | 1461 | 45211 | 25000 | 7 | 9 | 12 | no | no |
| clf | `adult` | 1590 | 48842 | 25000 | 6 | 8 | 41 | no | no |
| clf | `kick` | 41162 | 72983 | 25000 | 14 | 18 | 1063 | **yes** | **yes** |
| clf | `Amazon_employee_access` | 4135 | 32769 | 25000 | 0 | 9 | 7518 | **yes** | **yes** |
| clf | `Click_prediction_small` | 42733 | 39948 | 25000 | 5 | 6 | 30114 | **yes** | **yes** |

(Note: subsampled datasets compute `max_card` cleanly on the evaluated matrix, reported in the summary alongside full `max_card`.)

## 3. Variants

All miter variants run with `n_estimators = 5000`, `learning_rate = 0.05`, `m_mult = 2`, `k_proj = 0`, `max_bin = 255`, `early_stopping_rounds = 50`, `n_jobs = 1`, `random_state = fold`:

- **`B` (Baseline):** `grow_policy = "leafwise"`, `num_leaves = 31`, `min_data_in_leaf = 20`, `lambda_l2 = 1.0`, `cat_oblique_ts = False`.
- **`TS` (Primary confirmatory):** identical to `B` + `cat_oblique_ts = True`, `cat_ts_smooth = 10.0` (5-fold deterministic OOF-TS on inner `X_tr_inner`, full smoothed map on `X_val` and `X_te`).
- **`TS_S6` (Exploratory):** `grow_policy = "symmetric"`, `max_depth = 6`, `min_data_in_leaf = 1`, `lambda_l2 = 3.0`, `cat_oblique_ts = True`, `cat_ts_smooth = 10.0`.
- **`CAT` (Reference):** `CatBoostRegressor` / `CatBoostClassifier` (`iterations = 5000`, `learning_rate = 0.05` (or CatBoost default when unspecified? Fixed explicitly to `iterations=5000, depth=6, early_stopping_rounds=50, thread_count=1, random_seed=fold, cat_features=...` matching prior benchmark), passing categorical columns directly).
- **`LGB` (Reference):** `LGBMRegressor` / `LGBMClassifier` (`n_estimators = 5000`, `learning_rate = 0.05`, `num_leaves = 31`, `min_child_samples = 20`, `reg_lambda = 1.0`, `max_bin = 255`, `n_jobs = 1`, `random_state = fold`, `deterministic = True`).

## 4. Validation & Leakage-Free Protocol

- Outer CV: 5-fold (`KFold(5, shuffle=True, random_state=42)` for reg; `StratifiedKFold(5, shuffle=True, random_state=42)` for clf).
- Inner early-stopping split: `train_test_split(tr_idx, test_size=0.15, random_state=fold, stratify=y[tr_idx] if clf else None)`.
- Strict zero-leakage: `CatObliqueTSEncoder` fits ONLY on `(X[a], y[a])` (never touches `X[b]` or `X[te]`).
- Primary metric: **nRMSE** (`RMSE / std(y_te)`) for regression, **LogLoss** (`clip(p, 1e-15, 1-1e-15)`) for binary classification (lower is better). Secondary diagnostic: ROC-AUC for classification.

## 5. Confirmatory Decision Rules (`TS` vs `B`, pooled over all 12 datasets)

For each dataset, compute the 5-fold mean metric of `TS` and `B`, and relative delta `(TS - B) / B * 100%` (negative = `TS` better). `TS` is judged **PASS** only if ALL four criteria hold:

1. Pooled median relative delta `< -0.5%`
2. Two-sided Wilcoxon signed-rank `p < 0.05` across the 12 datasets
3. Dataset wins `>= 9 / 12` (`>= 75%`)
4. Worst-case relative delta `<= +1.5%` on every dataset

Exploratory breakdowns will also report:
- High/med-cardinality subgroup (`max_card > 50`, 6 datasets) vs low/med (`max_card <= 50`, 6 datasets), and extreme high-card (`max_card > 253`, 5 datasets).
- `TS` and `TS_S6` vs `CAT` and `LGB`.
