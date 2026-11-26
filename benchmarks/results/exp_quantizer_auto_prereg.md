# Pre-registration: Per-Feature Adaptive Quantizer Mode `auto` (`exp-quantizer-auto`)

- Branch: `feat-v11-softmax-cat-ctr` (code at commit `062cc6b`)
- Status: **FROZEN BEFORE RUNNING `exp-quantizer-auto`**

## 1. Hypothesis

Setting `quantizer="auto"` (Option A: keeps clean features in exact O(1) linear `Q16r` and automatically switches heavy-tailed / multimodal-skewed / clustered features to 16-bit piecewise-linear quantile `Q16q`) strictly avoids quality regressions on clean continuous datasets, eliminates catastrophic `Q16r` compression collapses (`> 3%` improvement on both known heavy-tail/clustered stress-test tasks), and keeps fit wall-time overhead `<= 1.10x` vs `quantizer="linear"`.

## 2. Datasets & Protocol (5 OpenML folds, `n_jobs = 1`, `random_state = fold`)

For any dataset with `n > 25,000`, `25,000` rows are deterministically subsampled (`random_state=42`, stratified for `binary`/`multiclass`). Each fold (`StratifiedKFold`/`KFold(5, shuffle=True, random_state=42)`) splits `20%` inner validation (`random_state=0`). Fixed base parameters: `n_estimators=5000, learning_rate=0.05, m_mult=2, k_proj=0, max_bin=255, early_stopping_rounds=50`.

### 2.1 Confirmatory Suite (16 fresh numeric/continuous-heavy datasets)
- Regression (8): `houses` (`did=46934`), `miami_housing` (`46942`), `physiochemical_protein` (`46949`), `superconductivity` (`46961`), `wine_quality` (`46964`), `airfoil_self_noise` (`46904`), `concrete_compressive_strength` (`46917`), `QSAR_fish_toxicity` (`46954`).
- Binary (8): `APSFailure` (`46908`), `GiveMeSomeCredit` (`46929`), `polish_companies_bankruptcy` (`46950`), `heloc` (`46932`), `jm1` (`46979`), `qsar-biodeg` (`46952`), `credit_card_clients_default` (`46919`), `coil2000` (`46916`).

### 2.2 Diagnostic Stress-Test Suite (2 datasets, reported separately)
- `taiwanese_bankruptcy_prediction` (`did=46962`, binary) — extreme outlier compression case.
- `SDSS17` (`did=46955`, multiclass) — tight redshift cluster compression case.

Primary metric: **nRMSE** (`RMSE / std(y_te)`) for regression; **LogLoss** (`clip(p, 1e-15, 1-1e-15)`) for binary/multiclass (lower is better). Secondary exploratory metric (binary): **`1 - ROC_AUC`**.

## 3. Variants (5)

1. `LIN`: Leafwise (`num_leaves=31, min_data_in_leaf=20, lambda_l2=1.0`), `quantizer="linear"`.
2. `AUTO`: Leafwise, `quantizer="auto"`.
3. `QUANT`: Leafwise, `quantizer="quantile"`.
4. `S6_LIN`: Symmetric (`max_depth=6, min_data_in_leaf=1, lambda_l2=3.0`), `quantizer="linear"`.
5. `S6_AUTO`: Symmetric, `quantizer="auto"`.

## 4. Confirmatory Pass Rules (Holm-Bonferroni `m = 2`)

Evaluated on the 16 confirmatory datasets using per-dataset 5-fold mean relative error delta `(AUTO - LIN) / LIN * 100%` (`AUTO vs LIN` and `S6_AUTO vs S6_LIN`, Holm-Bonferroni corrected over `m = 2`):

A comparison **PASSES** iff **ALL 5 criteria** hold:
1. **Non-inferiority + Triggered Gain**: Overall Mean Δ `<= 0.0%` AND Overall Median Δ `<= 0.0%`; AND on the triggered subgroup (`mean n_quantile_features >= 1`), Median Δ `< -0.25%` with `>= 65%` wins among non-ties (`abs(delta) > 1e-12`);
2. **Bootstrap CI**: 95% paired bootstrap CI upper bound of overall Median Δ `<= +0.10%` (10,000 resamples, `seed = 2026`);
3. **Tail Safety**: Worst single-dataset regression across all 16 confirmatory datasets `<= +0.75%`;
4. **Runtime**: Median per-fit wall-time ratio vs linear `<= 1.10x`;
5. **Stress-Test Verification**: On `taiwanese_bankruptcy_prediction` (both LogLoss and `1 - ROC_AUC`) and `SDSS17` (multiclass LogLoss), error improves by `> 3.0%` over the corresponding linear baseline.
