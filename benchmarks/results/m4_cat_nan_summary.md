# M4 Categorical K1 and Native NaN Benchmark Summary

- Hardware: `arm64` (`macOS-26.6.2-arm64-arm-64bit-Mach-O`)
- Python: `3.13.1`
- Versions: `miter 0.1.0.dev0`, `lightgbm 4.7.0`, `catboost 1.2.10`, `numpy 2.5.5`
- Protocol: single-threaded (`n_jobs=1` / `num_threads=1` / `thread_count=1`). Negative relative difference = lower error = better.

## Part A: 5 TabArena categorical regression datasets x 5 folds (1 thread, ES=50, max_trees=5000)

| dataset | LightGBM | CatBoost | histree K1 | miter K1 | miter vs LGBM | miter vs CB | miter vs histree | ms/tree (L/CB/H/M) |
|---|---|---|---|---|---|---|---|---|
| airfoil_self_noise | 0.2503 | 0.2572 | 0.2185 | 0.2218 | -11.41% | -13.78% | +1.48% | 0.15 / 0.31 / 0.45 / 0.20 |
| Another-Dataset-on-used-Fiat-500 | 0.3884 | 0.3876 | 0.3878 | 0.3853 | -0.81% | -0.59% | -0.63% | 0.25 / 0.71 / 0.69 / 0.29 |
| diamonds | 0.1351 | 0.1338 | 0.1330 | 0.1330 | -1.59% | -0.62% | -0.00% | 1.25 / 11.89 / 6.51 / 4.80 |
| Food_Delivery_Time | 0.8168 | 0.7887 | 0.7892 | 0.7886 | -3.45% | -0.01% | -0.07% | 1.96 / 12.05 / 5.54 / 3.52 |
| healthcare_insurance_expenses | 0.3872 | 0.3785 | 0.3878 | 0.3866 | -0.15% | +2.15% | -0.29% | 0.20 / 0.36 / 0.46 / 0.17 |
- **miter K1 vs LightGBM (5 cat):** mean -3.48%, **median -1.59%**, wins 5/5, Wilcoxon p = 0.0625
- **miter K1 vs CatBoost (5 cat):** mean -2.57%, **median -0.59%**, wins 4/5, Wilcoxon p = 0.4375
- **miter K1 vs histree K1 (5 cat):** mean +0.10%, **median -0.07%**, wins 4/5, Wilcoxon p = 0.625

## Part B: Native NaN handling (4 OpenML reg x 3 folds, 1 thread, ES=50, MCAR 15% & MNAR)

| regime | dataset | LightGBM | CatBoost | miter native NaN | miter median+ind | miter vs LGBM | miter vs CB | miter vs imputed |
|---|---|---|---|---|---|---|---|---|
| mcar | abalone | 0.6922 | 0.7105 | 0.6904 | 0.6917 | -0.26% | -2.82% | -0.18% |
| mcar | wine_quality | 0.8146 | 0.8195 | 0.8150 | 0.8091 | +0.05% | -0.56% | +0.73% |
| mcar | cpu_act | 0.1978 | 0.1768 | 0.1996 | 0.1946 | +0.90% | +12.94% | +2.56% |
| mcar | sulfur | 0.5969 | 0.5854 | 0.5830 | 0.5809 | -2.33% | -0.41% | +0.37% |
- **mcar: miter native NaN vs LightGBM:** mean -0.41%, median -0.10%, wins 2/4
- **mcar: miter native NaN vs CatBoost:** mean +2.29%, median -0.48%, wins 3/4
- **mcar: miter native NaN vs imputed+ind:** mean +0.87%, median +0.55%, wins 1/4
| mnar | abalone | 0.6843 | 0.6946 | 0.6672 | 0.6764 | -2.49% | -3.94% | -1.35% |
| mnar | wine_quality | 0.7600 | 0.7758 | 0.7490 | 0.7559 | -1.45% | -3.45% | -0.92% |
| mnar | cpu_act | 0.1415 | 0.1609 | 0.1510 | 0.1484 | +6.75% | -6.15% | +1.78% |
| mnar | sulfur | 0.5139 | 0.4753 | 0.4664 | 0.4766 | -9.25% | -1.87% | -2.14% |
- **mnar: miter native NaN vs LightGBM:** mean -1.61%, median -1.97%, wins 3/4
- **mnar: miter native NaN vs CatBoost:** mean -3.85%, median -3.70%, wins 4/4
- **mnar: miter native NaN vs imputed+ind:** mean -0.66%, median -1.13%, wins 3/4

## Part C: Single-thread mixed speed (20000 x [30 num + 10 cat], 100 trees, median of 5 runs)

| LightGBM ms/tree | CatBoost ms/tree | miter AX+K1 | miter R_uni+K1 | miter/LGBM | histree K1 |
|---|---|---|---|---|---|
| 3.019 | 14.587 | 2.486 | 5.888 | 1.95x | 10.694 |
