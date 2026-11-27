# miter v1.0 default on TabArena `lite` (51 tasks)

`MITER_DEFAULT_V1` is `MiterRegressor()` / `MiterClassifier()` with no arguments (symmetric trees of depth 6, `m_mult=4`, `min_data_in_leaf=1`, `lambda_l2=3`, linear quantizer, no CTR): one configuration, no tuning, 8-fold bagging, fold 0. The `MITER_<name>` rows are the single configurations of the V12 run (from its cache; `LW31` is the default before v1.0).

## gbdt_only, all 51 (51 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1134 | 57.9 | 47.3 | 5.8 | 0.68 |
| 2 | CAT (tuned + ensemble) | 1100 | 61.5 | 57.8 | 6.47 | 0.635 |
| 3 | CAT (tuned) | 1091 | 60 | 57 | 6.67 | 0.622 |
| 4 | [New] MITER_S6_M4_QQ_CTR (default) | 1090 | 54.7 | 48.1 | 6.69 | 0.621 |
| 5 | [New] MITER_S6_QAUTO_CTR (default) | 1064 | 59 | 52.2 | 7.23 | 0.585 |
| 6 | XGB (tuned + ensemble) | 1056 | 45.7 | 46.8 | 7.39 | 0.574 |
| 7 | GBM (tuned) | 1030 | 63.2 | 51.1 | 7.94 | 0.537 |
| 8 | [New] MITER_S6_QAUTO (default) | 1026 | 62.2 | 48.6 | 8.02 | 0.532 |
| 9 | XGB (tuned) | 1013 | 56.3 | 61.7 | 8.31 | 0.512 |
| 10 | [New] MITER_DEFAULT_V1 (default) | 998.3 | 64 | 52.4 | 8.62 | 0.492 |
| 11 | CAT (default) | 996 | 65.5 | 67.4 | 8.67 | 0.489 |
| 12 | [New] MITER_S6 (default) | 991.4 | 61.3 | 60 | 8.76 | 0.482 |
| 13 | [New] MITER_ACCURATE (default) | 970.1 | 66.3 | 72.2 | 9.22 | 0.452 |
| 14 | [New] MITER_LW31 (default) | 930.4 | 54.7 | 60.9 | 10 | 0.397 |
| 15 | XGB (default) | 787.5 | 61.7 | 85.5 | 12.6 | 0.225 |
| 16 | GBM (default) | 722.1 | 66.5 | 77.3 | 13.5 | 0.163 |

| miter | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_DEFAULT_V1 (default) | CAT (default) | 51 | 24/51 | +0.31% | +2.29% | 0.3989 |
| MITER_DEFAULT_V1 (default) | GBM (default) | 51 | 41/51 | -3.23% | -3.59% | 0.0001 |
| MITER_DEFAULT_V1 (default) | XGB (default) | 51 | 37/51 | -1.50% | -1.90% | 0.0144 |
| MITER_DEFAULT_V1 (default) | CAT (tuned) | 51 | 18/51 | +0.75% | +7.15% | 0.0137 |
| MITER_DEFAULT_V1 (default) | [New] MITER_LW31 (default) | 51 | 29/51 | -0.63% | -0.95% | 0.0469 |

## gbdt_only, regression (13 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1218 | 168 | 100 | 4.77 | 0.749 |
| 2 | CAT (tuned + ensemble) | 1208 | 131 | 77.4 | 4.92 | 0.738 |
| 3 | CAT (tuned) | 1152 | 134 | 87.5 | 5.92 | 0.672 |
| 4 | [New] MITER_ACCURATE (default) | 1115 | 111 | 103 | 6.62 | 0.626 |
| 5 | [New] MITER_DEFAULT_V1 (default) | 1087 | 137 | 104 | 7.15 | 0.59 |
| 6 | GBM (tuned) | 1072 | 148 | 121 | 7.46 | 0.569 |
| 7 | [New] MITER_LW31 (default) | 1044 | 100 | 97.3 | 8 | 0.533 |
| 8 | [New] MITER_S6_M4_QQ_CTR (default) | 1040 | 99.2 | 98.8 | 8.08 | 0.528 |
| 9 | [New] MITER_S6_QAUTO (default) | 1029 | 146 | 136 | 8.31 | 0.513 |
| 10 | XGB (tuned + ensemble) | 1009 | 103 | 92.4 | 8.69 | 0.487 |
| 11 | [New] MITER_S6 (default) | 1009 | 136 | 146 | 8.69 | 0.487 |
| 12 | XGB (tuned) | 993.3 | 130 | 122 | 9 | 0.467 |
| 13 | [New] MITER_S6_QAUTO_CTR (default) | 985.4 | 110 | 133 | 9.15 | 0.456 |
| 14 | CAT (default) | 952.8 | 114 | 147 | 9.77 | 0.415 |
| 15 | GBM (default) | 617.7 | 146 | 241 | 14.2 | 0.118 |
| 16 | XGB (default) | 467.6 | 91.7 | 258 | 15.2 | 0.0513 |

| miter | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_DEFAULT_V1 (default) | CAT (default) | 13 | 9/13 | -0.55% | -2.10% | 0.0681 |
| MITER_DEFAULT_V1 (default) | GBM (default) | 13 | 12/13 | -3.08% | -3.92% | 0.0012 |
| MITER_DEFAULT_V1 (default) | XGB (default) | 13 | 13/13 | -3.36% | -4.69% | 0.0002 |
| MITER_DEFAULT_V1 (default) | CAT (tuned) | 13 | 5/13 | +0.28% | +0.58% | 0.2163 |
| MITER_DEFAULT_V1 (default) | [New] MITER_LW31 (default) | 13 | 6/13 | +0.03% | -0.94% | 0.3396 |

## gbdt_only, binary (30 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1115 | 57.7 | 57 | 6.13 | 0.658 |
| 2 | [New] MITER_S6_M4_QQ_CTR (default) | 1113 | 72.8 | 58 | 6.17 | 0.656 |
| 3 | [New] MITER_S6_QAUTO_CTR (default) | 1086 | 74.3 | 64.7 | 6.72 | 0.619 |
| 4 | XGB (tuned + ensemble) | 1079 | 64.3 | 54.9 | 6.87 | 0.609 |
| 5 | CAT (tuned) | 1069 | 84.2 | 98.8 | 7.07 | 0.596 |
| 6 | CAT (tuned + ensemble) | 1068 | 88.2 | 91.8 | 7.1 | 0.593 |
| 7 | CAT (default) | 1044 | 82.4 | 79.1 | 7.6 | 0.56 |
| 8 | [New] MITER_S6_QAUTO (default) | 1027 | 86 | 74.2 | 7.97 | 0.536 |
| 9 | XGB (tuned) | 1015 | 72 | 82.4 | 8.23 | 0.518 |
| 10 | GBM (tuned) | 1001 | 60.2 | 75.7 | 8.53 | 0.498 |
| 11 | [New] MITER_S6 (default) | 978.8 | 82.2 | 90.9 | 9 | 0.467 |
| 12 | [New] MITER_DEFAULT_V1 (default) | 971.8 | 92 | 98.8 | 9.15 | 0.457 |
| 13 | [New] MITER_ACCURATE (default) | 920.8 | 72.1 | 90.7 | 10.2 | 0.386 |
| 14 | [New] MITER_LW31 (default) | 897.6 | 66.7 | 71.9 | 10.7 | 0.354 |
| 15 | XGB (default) | 858.5 | 107 | 88.1 | 11.4 | 0.304 |
| 16 | GBM (default) | 755.6 | 70.6 | 97.4 | 13.1 | 0.191 |

| miter | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_DEFAULT_V1 (default) | CAT (default) | 30 | 12/30 | +0.62% | +4.29% | 0.1241 |
| MITER_DEFAULT_V1 (default) | GBM (default) | 30 | 21/30 | -3.28% | -2.84% | 0.0262 |
| MITER_DEFAULT_V1 (default) | XGB (default) | 30 | 19/30 | -0.93% | -1.46% | 0.2988 |
| MITER_DEFAULT_V1 (default) | CAT (tuned) | 30 | 10/30 | +0.88% | +4.37% | 0.0841 |
| MITER_DEFAULT_V1 (default) | [New] MITER_LW31 (default) | 30 | 17/30 | -0.62% | -1.48% | 0.1048 |

## gbdt_only, multiclass (8 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [New] MITER_S6_QAUTO_CTR (default) | 1131 | 187 | 114 | 6 | 0.667 |
| 2 | GBM (tuned + ensemble) | 1118 | 121 | 75.9 | 6.25 | 0.65 |
| 3 | CAT (tuned) | 1111 | 213 | 154 | 6.38 | 0.642 |
| 4 | [New] MITER_S6_M4_QQ_CTR (default) | 1111 | 239 | 158 | 6.38 | 0.642 |
| 5 | GBM (tuned) | 1105 | 126 | 109 | 6.5 | 0.633 |
| 6 | CAT (tuned + ensemble) | 1098 | 199 | 150 | 6.62 | 0.625 |
| 7 | XGB (tuned + ensemble) | 1067 | 147 | 134 | 7.25 | 0.583 |
| 8 | XGB (tuned) | 1055 | 209 | 169 | 7.5 | 0.567 |
| 9 | [New] MITER_S6_QAUTO (default) | 1043 | 131 | 118 | 7.75 | 0.55 |
| 10 | [New] MITER_S6 (default) | 1030 | 161 | 150 | 8 | 0.533 |
| 11 | [New] MITER_DEFAULT_V1 (default) | 981.6 | 163 | 189 | 9 | 0.467 |
| 12 | [New] MITER_ACCURATE (default) | 947.3 | 206 | 229 | 9.69 | 0.421 |
| 13 | CAT (default) | 885.2 | 113 | 222 | 10.9 | 0.342 |
| 14 | [New] MITER_LW31 (default) | 881.8 | 139 | 194 | 10.9 | 0.338 |
| 15 | XGB (default) | 762.6 | 116 | 206 | 12.9 | 0.208 |
| 16 | GBM (default) | 672.4 | 161 | 334 | 14 | 0.133 |

| miter | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_DEFAULT_V1 (default) | CAT (default) | 8 | 3/8 | +1.14% | +1.93% | 0.4609 |
| MITER_DEFAULT_V1 (default) | GBM (default) | 8 | 8/8 | -4.11% | -5.86% | 0.0078 |
| MITER_DEFAULT_V1 (default) | XGB (default) | 8 | 5/8 | -1.40% | +0.98% | 0.8438 |
| MITER_DEFAULT_V1 (default) | CAT (tuned) | 8 | 3/8 | +5.37% | +28.24% | 0.1094 |
| MITER_DEFAULT_V1 (default) | [New] MITER_LW31 (default) | 8 | 6/8 | -1.15% | +1.05% | 0.5469 |

## Median train time per task (s, 8-fold bag, this machine)

| method | train |
|---|---:|
| [New] MITER_ACCURATE (default) | 5.2 |
| [New] MITER_DEFAULT_V1 (default) | 7.2 |
| [New] MITER_LW31 (default) | 3.4 |
| [New] MITER_S6 (default) | 6.0 |
| [New] MITER_S6_M4_QQ_CTR (default) | 7.5 |
| [New] MITER_S6_QAUTO (default) | 6.5 |
| [New] MITER_S6_QAUTO_CTR (default) | 7.9 |
