# M5 Performance, OpenMP Scaling, and TabArena Leaderboard Summary

- Hardware: `arm64` (`macOS-26.6.2-arm64-arm-64bit-Mach-O`)
- Python: `3.13.1`
- Versions: `miter 0.1.0.dev0`, `lightgbm 4.7.0`, `catboost 1.2.10`, `numpy 2.5.5`
- Protocol: Part A uses 1 thread; Part B tests 1/2/4/8 threads; Part C uses `n_jobs=4` across 12 TabArena datasets (3 folds, ES=50).

## Part A: Single-thread training speed (100 trees, median of 5 runs, `n_jobs=1`)

| shape | LightGBM ms/tree | CatBoost ms/tree | miter AX | miter default | miter fast | miter acc | miter/LGBM | miter/CB |
|---|---|---|---|---|---|---|---|---|
| 20000 x 16 | 1.514 | 3.051 | 1.112 | 3.563 | 1.802 | 5.807 | 2.35x | 1.17x |
| 20000 x 50 | 4.891 | 7.874 | 3.130 | 7.849 | 4.586 | 12.644 | 1.60x | 1.00x |
| 20000 x 100 | 8.936 | 14.501 | 6.270 | 13.453 | 7.866 | 17.950 | 1.51x | 0.93x |
| 50000 x 50 | 8.385 | 10.712 | 6.048 | 17.745 | 9.247 | 29.552 | 2.12x | 1.66x |
| 10000 x 1024 | 64.989 | 129.160 | 47.210 | 64.395 | 51.611 | 76.339 | 0.99x | 0.50x |

- **Peak RSS on `50000 x 50` (100 trees):** LightGBM `237.5 MB`, CatBoost `179.2 MB`, miter default `74.8 MB`

## Part B: Multithreaded scaling (`1, 2, 4, 8` threads, 100 trees, median of 5 runs)

| dataset | model | T=1 (ms/tree) | T=2 (ms/tree) | T=4 (ms/tree) | T=8 (ms/tree) | Speedup T1/T8 |
|---|---|---|---|---|---|---|
| synth 500k x 50 (reg) | LightGBM | 47.327 | 27.378 | 16.826 | 16.645 | 2.84x |
| synth 500k x 50 (reg) | CatBoost | 60.369 | 32.244 | 19.031 | 16.816 | 3.59x |
| synth 500k x 50 (reg) | miter AX | 45.611 | 27.542 | 18.582 | 17.088 | 2.67x |
| synth 500k x 50 (reg) | miter default | 239.553 | 137.388 | 96.828 | 82.753 | 2.89x |
| covertype 566602 x 10 (clf) | LightGBM | 12.498 | 8.660 | 6.171 | 6.933 | 1.80x |
| covertype 566602 x 10 (clf) | CatBoost | 33.480 | 17.990 | 11.989 | 10.823 | 3.09x |
| covertype 566602 x 10 (clf) | miter AX | 19.690 | 13.615 | 9.023 | 9.190 | 2.14x |
| covertype 566602 x 10 (clf) | miter default | 116.752 | 67.738 | 42.747 | 39.181 | 2.98x |

- Verified byte-for-byte identical JSON model for `500k x 50` (5 trees, `m_mult=2`) across `n_jobs=1` vs `n_jobs=8`.

## Part C: Hard TabArena 12-Dataset Regression Suite (3 folds, ES=50, `n_jobs=4`)

### TabArena-Style Leaderboard Score

| Method | Elo | Win-rate (%) | Avg Rank | Improvability (%) | Normalized Error | Median Train (s) |
|---|---|---|---|---|---|---|
| miter (accurate) | 1258.8 | 82.2% | 1.89 | 0.98% | 0.335 | 1.06 |
| miter (default) | 1103.8 | 63.3% | 2.83 | 1.41% | 0.633 | 0.76 |
| CatBoost (default) | 1095.5 | 62.2% | 2.89 | 1.95% | 0.427 | 2.45 |
| miter (fast) | 1029.8 | 53.3% | 3.33 | 2.05% | 0.815 | 0.37 |
| LightGBM (default) | 788.2 | 22.8% | 4.86 | 4.40% | 0.992 | 0.76 |
| miter (AX) | 723.9 | 16.1% | 5.19 | 4.62% | 1.000 | 0.21 |

### Per-Dataset Mean Test nRMSE (3 folds)

| dataset | LGBM | CatBoost | miter AX | miter fast | miter default | miter acc | def vs LGBM | def vs CB |
|---|---|---|---|---|---|---|---|---|
| airfoil_self_noise (1503x5) | 0.2731 | 0.2749 | 0.2773 | 0.2508 | 0.2455 | 0.2480 | -10.11% | -10.72% |
| Another-Dataset-on-used-Fiat-500 (1538x7) | 0.4034 | 0.3913 | 0.4127 | 0.4059 | 0.4026 | 0.3966 | -0.20% | +2.90% |
| concrete_compressive_strength (1030x8) | 0.2971 | 0.2771 | 0.2988 | 0.2858 | 0.2842 | 0.2821 | -4.33% | +2.55% |
| diamonds (53940x9) | 0.1371 | 0.1342 | 0.1374 | 0.1360 | 0.1353 | 0.1354 | -1.29% | +0.83% |
| Food_Delivery_Time (45451x9) | 0.8184 | 0.7912 | 0.7921 | 0.7927 | 0.7918 | 0.7908 | -3.24% | +0.08% |
| healthcare_insurance_expenses (1338x6) | 0.3813 | 0.3762 | 0.3823 | 0.3802 | 0.3832 | 0.3847 | +0.49% | +1.85% |
| houses (20640x8) | 0.3919 | 0.3828 | 0.3924 | 0.3821 | 0.3775 | 0.3744 | -3.67% | -1.39% |
| miami_housing (13776x15) | 0.2876 | 0.2644 | 0.2897 | 0.2816 | 0.2801 | 0.2766 | -2.59% | +5.95% |
| physiochemical_protein (45730x9) | 0.5961 | 0.6065 | 0.5975 | 0.5700 | 0.5628 | 0.5561 | -5.59% | -7.21% |
| QSAR-TID-11 (5742x1024) | 0.5312 | 0.5233 | 0.5313 | 0.5255 | 0.5256 | 0.5238 | -1.05% | +0.44% |
| QSAR_fish_toxicity (907x6) | 0.6398 | 0.6193 | 0.6388 | 0.6245 | 0.6157 | 0.6110 | -3.77% | -0.58% |
| superconductivity (21263x81) | 0.2823 | 0.2818 | 0.2832 | 0.2799 | 0.2797 | 0.2796 | -0.93% | -0.77% |
- **miter (default) vs LightGBM (default) (12 datasets):** mean -3.02%, **median -2.92%**, wins 11/12, Wilcoxon p = 0.002441
- **miter (default) vs CatBoost (default) (12 datasets):** mean -0.51%, **median +0.26%**, wins 5/12, Wilcoxon p = 0.9097
- **miter (accurate) vs LightGBM (default) (12 datasets):** mean -3.46%, **median -3.60%**, wins 11/12, Wilcoxon p = 0.002441
- **miter (accurate) vs CatBoost (default) (12 datasets):** mean -0.96%, **median +0.02%**, wins 6/12, Wilcoxon p = 0.791
- **miter (fast) vs LightGBM (default) (12 datasets):** mean -2.40%, **median -2.23%**, wins 11/12, Wilcoxon p = 0.003418
- **miter (fast) vs CatBoost (default) (12 datasets):** mean +0.13%, **median +0.63%**, wins 4/12, Wilcoxon p = 0.4238
