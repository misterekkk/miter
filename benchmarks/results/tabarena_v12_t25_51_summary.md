# MITER_V12_T25 on TabArena `lite`, all 51 tasks (Quantizer + Native C++ CTR, 25 configs)

Caveat: TabArena `lite` evaluates **single split (`fold 0`) per dataset** (25 miter configs vs up to 201 for CAT/GBM/XGB).

## Primary: gbdt_only pool, 51 tasks

### gbdt_only, 51 tasks (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V12_T25 (tuned + ensemble) | 1218 | 73.3 | 65.3 | 3.559 | 0.7674 | 0.03234 | 0 |
| 2 | [New] MITER_V12_T25 (tuned) | 1176 | 81.3 | 52.5 | 4.088 | 0.7193 | 0.03656 | 0 |
| 3 | GBM (tuned + ensemble) | 1111 | 45.4 | 37.1 | 4.961 | 0.6399 | 0.04644 | 0 |
| 4 | CAT (tuned + ensemble) | 1081 | 66 | 60.4 | 5.392 | 0.6007 | 0.03487 | 0 |
| 5 | CAT (tuned) | 1076 | 62 | 62.1 | 5.471 | 0.5936 | 0.03148 | 0 |
| 6 | XGB (tuned + ensemble) | 1048 | 44.1 | 44.2 | 5.882 | 0.5561 | 0.05301 | 0 |
| 7 | GBM (tuned) | 1007 | 57.9 | 49.9 | 6.49 | 0.5009 | 0.0568 | 0 |
| 8 | XGB (tuned) | 994.9 | 56.1 | 61.6 | 6.667 | 0.4848 | 0.05539 | 0 |
| 9 | CAT (default) | 966.7 | 56 | 76.2 | 7.078 | 0.4474 | 0.05381 | 0 |
| 10 | [New] MITER_V12_T25 (default) | 897.1 | 61.8 | 68 | 8.059 | 0.3583 | 0.07873 | 0 |
| 11 | XGB (default) | 739.8 | 72.7 | 90.3 | 9.922 | 0.1889 | 0.09102 | 0 |
| 12 | GBM (default) | 684.6 | 75.1 | 73.7 | 10.43 | 0.1426 | 0.1017 | 0 |

### Paired vs CatBoost (tuned + ensemble)

- **all 51 vs CAT** (51 tasks): wins 35/51, median -0.78% (95% boot CI [-1.61%, -0.17%]), mean +1.63%, Wilcoxon p = 0.1386, worst +146.82%
- **regression 13 vs CAT** (13 tasks): wins 11/13, median -0.78% (95% boot CI [-1.82%, -0.17%]), mean -0.49%, Wilcoxon p = 0.1465, worst +4.39%
- **binary 30 vs CAT** (30 tasks): wins 19/30, median -0.61% (95% boot CI [-1.98%, +0.20%]), mean -1.52%, Wilcoxon p = 0.2894, worst +11.22%
- **multiclass 8 vs CAT** (8 tasks): wins 5/8, median -1.65% (95% boot CI [-5.98%, +7.82%]), mean +16.85%, Wilcoxon p = 0.9453, worst +146.82%

### Paired vs cached MITER_V11_T12 (tuned + ensemble)

- **all 51 vs MITER_V11_T12** (51 tasks): wins 34/51, median -0.58% (95% boot CI [-1.03%, -0.16%]), mean -0.06%, Wilcoxon p = 0.01169, worst +51.31%
- **regression 13 vs MITER_V11_T12** (13 tasks): wins 12/13, median -0.58% (95% boot CI [-0.96%, -0.16%]), mean -0.70%, Wilcoxon p = 0.0004883, worst +0.01%
- **binary 30 vs MITER_V11_T12** (30 tasks): wins 17/30, median -0.40% (95% boot CI [-1.19%, +0.14%]), mean -0.83%, Wilcoxon p = 0.2129, worst +8.06%
- **multiclass 8 vs MITER_V11_T12** (8 tasks): wins 5/8, median -1.56% (95% boot CI [-4.21%, +1.56%]), mean +3.89%, Wilcoxon p = 0.4609, worst +51.31%

### gbdt_only, regression 13 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V12_T25 (tuned + ensemble) | 1334 | 200 | 100.4 | 2.846 | 0.8322 | 0.009115 | 0 |
| 2 | [New] MITER_V12_T25 (tuned) | 1266 | 201.4 | 107.9 | 3.538 | 0.7692 | 0.01079 | 0 |
| 3 | GBM (tuned + ensemble) | 1212 | 121.5 | 100.5 | 4.154 | 0.7133 | 0.01887 | 0 |
| 4 | CAT (tuned + ensemble) | 1174 | 131.2 | 96 | 4.615 | 0.6713 | 0.01413 | 0 |
| 5 | CAT (tuned) | 1113 | 114.2 | 101.4 | 5.385 | 0.6014 | 0.01452 | 0 |
| 6 | GBM (tuned) | 1042 | 115.3 | 143.7 | 6.308 | 0.5175 | 0.02784 | 0 |
| 7 | [New] MITER_V12_T25 (default) | 999.6 | 103 | 114.7 | 6.846 | 0.4685 | 0.0292 | 0 |
| 8 | XGB (tuned + ensemble) | 975 | 99.6 | 116.3 | 7.154 | 0.4406 | 0.02621 | 0 |
| 9 | XGB (tuned) | 975 | 118.1 | 139.2 | 7.154 | 0.4406 | 0.02589 | 0 |
| 10 | CAT (default) | 923.5 | 113.1 | 151.8 | 7.769 | 0.3846 | 0.04048 | 0 |
| 11 | GBM (default) | 548.6 | 117.5 | 191.8 | 10.85 | 0.1049 | 0.05837 | 0 |
| 12 | XGB (default) | 436.9 | 89.3 | 221.1 | 11.38 | 0.05594 | 0.06572 | 0 |

### gbdt_only, binary 30 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V12_T25 (tuned + ensemble) | 1210 | 101.6 | 79.6 | 3.6 | 0.7636 | 0.02391 | 0 |
| 2 | [New] MITER_V12_T25 (tuned) | 1155 | 90.4 | 67.4 | 4.3 | 0.7 | 0.03086 | 0 |
| 3 | GBM (tuned + ensemble) | 1082 | 47.3 | 49.8 | 5.333 | 0.6061 | 0.04308 | 0 |
| 4 | XGB (tuned + ensemble) | 1082 | 52.6 | 53.5 | 5.333 | 0.6061 | 0.05 | 0 |
| 5 | CAT (tuned) | 1062 | 78.5 | 91.2 | 5.633 | 0.5788 | 0.03751 | 0 |
| 6 | CAT (tuned + ensemble) | 1051 | 79.6 | 92.4 | 5.8 | 0.5636 | 0.03988 | 0 |
| 7 | CAT (default) | 1016 | 74.5 | 81.1 | 6.333 | 0.5152 | 0.03743 | 0 |
| 8 | XGB (tuned) | 1002 | 78.7 | 79.2 | 6.533 | 0.497 | 0.05496 | 0 |
| 9 | GBM (tuned) | 973.5 | 61.2 | 81.6 | 6.967 | 0.4576 | 0.05669 | 0 |
| 10 | [New] MITER_V12_T25 (default) | 863.2 | 73 | 84.8 | 8.533 | 0.3152 | 0.08161 | 0 |
| 11 | XGB (default) | 807.5 | 107.1 | 106 | 9.233 | 0.2515 | 0.08604 | 0 |
| 12 | GBM (default) | 695.1 | 78.7 | 105.2 | 10.4 | 0.1455 | 0.09854 | 0 |

### gbdt_only, multiclass 8 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V12_T25 (tuned) | 1170 | 312.3 | 143.3 | 4.188 | 0.7102 | 0.09978 | 0 |
| 2 | [New] MITER_V12_T25 (tuned + ensemble) | 1142 | 302.2 | 141.1 | 4.562 | 0.6761 | 0.1017 | 0 |
| 3 | GBM (tuned + ensemble) | 1119 | 119.8 | 83.7 | 4.875 | 0.6477 | 0.1039 | 0 |
| 4 | CAT (tuned) | 1110 | 220.3 | 148.6 | 5 | 0.6364 | 0.03644 | 0 |
| 5 | GBM (tuned) | 1110 | 128 | 121.4 | 5 | 0.6364 | 0.1043 | 0 |
| 6 | CAT (tuned + ensemble) | 1101 | 190.2 | 156.8 | 5.125 | 0.625 | 0.04976 | 0 |
| 7 | XGB (tuned + ensemble) | 1050 | 109.2 | 103.8 | 5.875 | 0.5568 | 0.1079 | 0 |
| 8 | XGB (tuned) | 1015 | 152 | 154.2 | 6.375 | 0.5114 | 0.1049 | 0 |
| 9 | [New] MITER_V12_T25 (default) | 881.7 | 189.8 | 264.2 | 8.25 | 0.3409 | 0.1484 | 0 |
| 10 | CAT (default) | 842.5 | 136.7 | 284.8 | 8.75 | 0.2955 | 0.1369 | 0 |
| 11 | GBM (default) | 742.5 | 214.9 | 407.1 | 9.875 | 0.1932 | 0.1841 | 0 |
| 12 | XGB (default) | 716.9 | 94.4 | 183.3 | 10.12 | 0.1705 | 0.1508 | 0 |

## Secondary: full pool, 51 tasks

### full pool, 51 tasks (`121` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2091 | 173.8 | 113.5 | 6.333 | 0.9556 | 0.03645 | 0 |
| 2 | TA-LIMIX-2 (default) | 2061 | 166.5 | 109.6 | 7.157 | 0.9487 | 0.02974 | 0 |
| 3 | Chakra-Tab (full) | 1969 | 118.7 | 76.9 | 10.28 | 0.9226 | 0.05738 | 0 |
| 4 | TA-SELDON (default) | 1960 | 186.4 | 133.5 | 10.65 | 0.9196 | 0.04833 | 0 |
| 5 | Chakra-Tab (medium) | 1940 | 102.8 | 64.4 | 11.47 | 0.9127 | 0.06029 | 0 |
| 6 | TA-KUMO-TABULAR-MEDIUM (default) | 1936 | 146.7 | 100.8 | 11.65 | 0.9113 | 0.05609 | 0 |
| 7 | TA-TABPFN-3.5 (default) | 1882 | 144.2 | 93.9 | 14.15 | 0.8904 | 0.07414 | 0 |
| 8 | TabFM+ | 1858 | 125.9 | 107.2 | 15.34 | 0.8805 | 0.07075 | 0 |
| 9 | TA-SELDON-MINI (default) | 1839 | 111.6 | 78.6 | 16.34 | 0.8721 | 0.08427 | 0 |
| 10 | INPRIORA (default) | 1833 | 101.8 | 94.5 | 16.71 | 0.8691 | 0.07667 | 0 |
| 35 | [New] MITER_V12_T25 (tuned + ensemble) | 1441 | 45.8 | 46.7 | 49.38 | 0.5968 | 0.1669 | 0 |
| 40 | [New] MITER_V12_T25 (tuned) | 1414 | 53.1 | 45 | 52.5 | 0.5708 | 0.1711 | 0 |
| 44 | CAT (tuned) | 1395 | 45.6 | 39 | 54.69 | 0.5526 | 0.1652 | 0 |
| 45 | GBM (tuned + ensemble) | 1388 | 35.3 | 36.8 | 55.54 | 0.5455 | 0.1757 | 0 |
| 47 | CAT (tuned + ensemble) | 1386 | 38.1 | 35.7 | 55.74 | 0.5439 | 0.1675 | 0 |
| 54 | XGB (tuned + ensemble) | 1351 | 37.6 | 42.4 | 59.93 | 0.5089 | 0.1819 | 0 |
| 57 | CAT (default) | 1339 | 36 | 43.2 | 61.43 | 0.4964 | 0.1814 | 0 |
| 58 | GBM (tuned) | 1335 | 41.4 | 41.6 | 61.89 | 0.4926 | 0.1842 | 0 |
| 60 | XGB (tuned) | 1328 | 43.6 | 47 | 62.65 | 0.4863 | 0.1838 | 0 |
| 75 | [New] MITER_V12_T25 (default) | 1259 | 56.4 | 58.5 | 71.1 | 0.4158 | 0.2034 | 0 |
| 90 | XGB (default) | 1180 | 46.4 | 52.3 | 80.43 | 0.3381 | 0.2137 | 0 |
| 99 | GBM (default) | 1147 | 53.8 | 47.1 | 84.19 | 0.3068 | 0.2203 | 0 |

