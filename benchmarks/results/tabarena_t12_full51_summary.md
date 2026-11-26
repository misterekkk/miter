# MITER_T12 on TabArena `lite`, all 51 tasks (multiclass via one-vs-rest)

Addendum to `tabarena_t12_full43_prereg.md`: `tabarena_t12_full51_prereg.md`. Multiclass tasks use miter one-vs-rest (one binary model per class, normalised probabilities), not native softmax.

## gbdt_only pool, 51 tasks

### gbdt_only, 51 tasks (`15` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_T12 (tuned + ensemble) | 1152 | 52 | 41.9 | 5.225 | 0.6982 | 0.04653 | 0 |
| 2 | GBM (tuned + ensemble) | 1141 | 62.4 | 49.6 | 5.412 | 0.6849 | 0.04354 | 0 |
| 3 | CAT (tuned + ensemble) | 1099 | 73 | 60.5 | 6.176 | 0.6303 | 0.03187 | 0 |
| 4 | CAT (tuned) | 1095 | 67.1 | 54.3 | 6.255 | 0.6246 | 0.02849 | 0 |
| 5 | [New] MITER_T12 (tuned) | 1079 | 60.7 | 43.4 | 6.559 | 0.6029 | 0.05271 | 0 |
| 6 | XGB (tuned + ensemble) | 1070 | 46.7 | 49.9 | 6.725 | 0.591 | 0.05009 | 0 |
| 7 | GBM (tuned) | 1030 | 63.8 | 50.6 | 7.49 | 0.5364 | 0.05395 | 0 |
| 8 | XGB (tuned) | 1015 | 57 | 64.2 | 7.784 | 0.5154 | 0.05247 | 0 |
| 9 | [New] MITER_BS6 (tuned + ensemble) | 1011 | 57.2 | 43.1 | 7.873 | 0.5091 | 0.06384 | 0 |
| 10 | CAT (default) | 1005 | 69.8 | 71.7 | 7.98 | 0.5014 | 0.05089 | 0 |
| 11 | [New] MITER_BS6 (tuned) | 996.8 | 58.6 | 58.4 | 8.147 | 0.4895 | 0.06581 | 0 |
| 12 | [New] MITER_BS6 (default) | 908.2 | 49.3 | 62.5 | 9.824 | 0.3697 | 0.07663 | 0 |
| 13 | [New] MITER_T12 (default) | 908.2 | 49.3 | 62.5 | 9.824 | 0.3697 | 0.07663 | 0 |
| 14 | XGB (default) | 777.4 | 66.6 | 79.2 | 11.94 | 0.2185 | 0.08811 | 0 |
| 15 | GBM (default) | 711.4 | 77.1 | 82.4 | 12.78 | 0.1583 | 0.09897 | 0 |

- **all 51** (51 tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins 28/51, median -0.29%, mean +3.35%, Wilcoxon p = 0.3937
- **multiclass 8 (OvR)** (8 tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins 3/8, median +4.45%, mean +19.49%, Wilcoxon p = 0.1094

### gbdt_only, multiclass 8 (OvR) (`15` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1175 | 202.2 | 100.6 | 4.875 | 0.7232 | 0.09229 | 0 |
| 2 | CAT (tuned + ensemble) | 1160 | 233.1 | 136.2 | 5.125 | 0.7054 | 0.03833 | 0 |
| 3 | CAT (tuned) | 1160 | 319.8 | 159.3 | 5.125 | 0.7054 | 0.02491 | 0 |
| 4 | GBM (tuned) | 1160 | 220.7 | 124.7 | 5.125 | 0.7054 | 0.09279 | 0 |
| 5 | XGB (tuned + ensemble) | 1095 | 259.5 | 142.5 | 6.25 | 0.625 | 0.09633 | 0 |
| 6 | XGB (tuned) | 1074 | 297.5 | 173.9 | 6.625 | 0.5982 | 0.09327 | 0 |
| 7 | [New] MITER_T12 (tuned) | 1041 | 134.4 | 163.2 | 7.25 | 0.5536 | 0.1293 | 0 |
| 8 | [New] MITER_T12 (tuned + ensemble) | 1031 | 129.8 | 161.9 | 7.438 | 0.5402 | 0.1254 | 0 |
| 9 | CAT (default) | 947.8 | 154.2 | 172.2 | 9 | 0.4286 | 0.1257 | 0 |
| 10 | [New] MITER_BS6 (tuned + ensemble) | 917.2 | 100.5 | 210.1 | 9.562 | 0.3884 | 0.1454 | 0 |
| 11 | [New] MITER_BS6 (tuned) | 899.8 | 155 | 326.7 | 9.875 | 0.3661 | 0.1546 | 0 |
| 12 | [New] MITER_BS6 (default) | 878.4 | 179.5 | 217.7 | 10.25 | 0.3393 | 0.1421 | 0 |
| 13 | [New] MITER_T12 (default) | 878.4 | 179.5 | 217.7 | 10.25 | 0.3393 | 0.1421 | 0 |
| 14 | XGB (default) | 833.7 | 114 | 161.8 | 11 | 0.2857 | 0.1395 | 0 |
| 15 | GBM (default) | 749.1 | 215.9 | 433.7 | 12.25 | 0.1964 | 0.1737 | 0 |

## full pool, 51 tasks

### full pool, 51 tasks (`124` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2092 | 178.2 | 113.4 | 6.353 | 0.9565 | 0.03645 | 0 |
| 2 | TA-LIMIX-2 (default) | 2062 | 170.4 | 109.4 | 7.216 | 0.9495 | 0.02974 | 0 |
| 3 | Chakra-Tab (full) | 1973 | 118.3 | 75 | 10.23 | 0.925 | 0.05738 | 0 |
| 4 | TA-SELDON (default) | 1960 | 190.5 | 138.8 | 10.75 | 0.9208 | 0.04833 | 0 |
| 5 | Chakra-Tab (medium) | 1943 | 105.4 | 62.3 | 11.47 | 0.9149 | 0.06029 | 0 |
| 6 | TA-KUMO-TABULAR-MEDIUM (default) | 1937 | 149.4 | 102.2 | 11.71 | 0.913 | 0.05609 | 0 |
| 7 | TA-TABPFN-3.5 (default) | 1884 | 146.4 | 94.8 | 14.19 | 0.8928 | 0.07414 | 0 |
| 8 | TabFM+ | 1861 | 129.2 | 109.1 | 15.36 | 0.8832 | 0.07075 | 0 |
| 9 | TA-SELDON-MINI (default) | 1842 | 112.6 | 76.2 | 16.36 | 0.8751 | 0.08427 | 0 |
| 10 | INPRIORA (default) | 1834 | 100.1 | 94 | 16.82 | 0.8714 | 0.07667 | 0 |
| 42 | CAT (tuned) | 1397 | 45.5 | 39 | 55.47 | 0.5571 | 0.1652 | 0 |
| 43 | GBM (tuned + ensemble) | 1392 | 33.5 | 37.2 | 55.99 | 0.5529 | 0.1757 | 0 |
| 45 | CAT (tuned + ensemble) | 1388 | 40.8 | 35.8 | 56.52 | 0.5486 | 0.1675 | 0 |
| 46 | [New] MITER_T12 (tuned + ensemble) | 1382 | 40.9 | 47.7 | 57.26 | 0.5426 | 0.1803 | 0 |
| 53 | XGB (tuned + ensemble) | 1354 | 37.9 | 41.6 | 60.77 | 0.514 | 0.1819 | 0 |
| 55 | [New] MITER_T12 (tuned) | 1349 | 41.5 | 44.5 | 61.38 | 0.5091 | 0.1847 | 0 |
| 56 | CAT (default) | 1341 | 37.2 | 44.9 | 62.33 | 0.5014 | 0.1814 | 0 |
| 58 | GBM (tuned) | 1337 | 43.2 | 43.2 | 62.89 | 0.4968 | 0.1842 | 0 |
| 61 | XGB (tuned) | 1330 | 44.2 | 49.1 | 63.76 | 0.4897 | 0.1838 | 0 |
| 64 | [New] MITER_BS6 (tuned + ensemble) | 1310 | 50.1 | 49.7 | 66.25 | 0.4695 | 0.1948 | 0 |
| 66 | [New] MITER_BS6 (tuned) | 1302 | 51.5 | 51.3 | 67.23 | 0.4616 | 0.1962 | 0 |
| 78 | [New] MITER_T12 (default) | 1254 | 61.3 | 52.4 | 73.35 | 0.4118 | 0.2038 | 0 |
| 79 | [New] MITER_BS6 (default) | 1254 | 61.3 | 52.4 | 73.35 | 0.4118 | 0.2038 | 0 |
| 93 | XGB (default) | 1179 | 48.1 | 53.2 | 82.45 | 0.3378 | 0.2137 | 0 |
| 102 | GBM (default) | 1144 | 53.3 | 48.2 | 86.54 | 0.3046 | 0.2203 | 0 |

