# MITER_T12 on TabArena `lite`, all regression + binary tasks (43)

Pre-registered: `tabarena_t12_full43_prereg.md` (SHA-256 in the commit that froze it). Raw: `benchmarks/results/raw/tabarena/ens_lite/`, `.../t12_full43_cmp/` (gitignored).

## Primary: gbdt_only pool, 43 tasks

### gbdt_only, 43 tasks (`15` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_T12 (tuned + ensemble) | 1177 | 60.7 | 56.8 | 4.814 | 0.7276 | 0.03186 | 0 |
| 2 | GBM (tuned + ensemble) | 1136 | 58.8 | 50.3 | 5.512 | 0.6777 | 0.03447 | 0 |
| 3 | CAT (tuned + ensemble) | 1089 | 97.9 | 65.9 | 6.372 | 0.6163 | 0.03066 | 0 |
| 4 | [New] MITER_T12 (tuned) | 1086 | 63.4 | 61 | 6.43 | 0.6121 | 0.03847 | 0 |
| 5 | CAT (tuned) | 1084 | 90.5 | 62.5 | 6.465 | 0.6096 | 0.02916 | 0 |
| 6 | XGB (tuned + ensemble) | 1066 | 54.2 | 59.2 | 6.814 | 0.5847 | 0.04149 | 0 |
| 7 | [New] MITER_BS6 (tuned + ensemble) | 1028 | 61.7 | 61 | 7.558 | 0.5316 | 0.04867 | 0 |
| 8 | CAT (default) | 1016 | 82.4 | 67.9 | 7.791 | 0.515 | 0.03698 | 0 |
| 9 | [New] MITER_BS6 (tuned) | 1014 | 61.6 | 67.6 | 7.826 | 0.5125 | 0.0493 | 0 |
| 10 | GBM (tuned) | 1008 | 62.3 | 55.2 | 7.93 | 0.505 | 0.04673 | 0 |
| 11 | XGB (tuned) | 1005 | 62.6 | 57.8 | 8 | 0.5 | 0.04488 | 0 |
| 12 | [New] MITER_BS6 (default) | 912.5 | 53.5 | 57.7 | 9.744 | 0.3754 | 0.06446 | 0 |
| 13 | [New] MITER_T12 (default) | 912.5 | 53.5 | 57.7 | 9.744 | 0.3754 | 0.06446 | 0 |
| 14 | XGB (default) | 763.6 | 79.9 | 90 | 12.12 | 0.206 | 0.07855 | 0 |
| 15 | GBM (default) | 701.2 | 67.1 | 84.5 | 12.88 | 0.1512 | 0.08507 | 0 |

- **all 43** (43 tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins 25/43, median -0.41%, mean +0.34%, Wilcoxon p = 0.9096
- **regression 13** (13 tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins 9/13, median -0.59%, mean +0.06%, Wilcoxon p = 0.4548
- **binary 30** (30 tasks): T12 (tuned+ens) vs CAT (tuned+ens): wins 16/30, median -0.16%, mean +0.47%, Wilcoxon p = 0.6408

### gbdt_only, regression 13 (`15` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_T12 (tuned) | 1250 | 157.5 | 74.7 | 4.346 | 0.761 | 0.01182 | 0 |
| 2 | [New] MITER_T12 (tuned + ensemble) | 1247 | 154.3 | 79 | 4.385 | 0.7582 | 0.01172 | 0 |
| 3 | GBM (tuned + ensemble) | 1217 | 174.9 | 111 | 4.846 | 0.7253 | 0.0162 | 0 |
| 4 | CAT (tuned + ensemble) | 1169 | 151.7 | 113.5 | 5.615 | 0.6703 | 0.01147 | 0 |
| 5 | CAT (tuned) | 1123 | 144.1 | 113.8 | 6.385 | 0.6154 | 0.01186 | 0 |
| 6 | [New] MITER_BS6 (tuned + ensemble) | 1101 | 134.5 | 100.5 | 6.769 | 0.5879 | 0.01671 | 0 |
| 7 | [New] MITER_BS6 (tuned) | 1077 | 113.2 | 97.5 | 7.192 | 0.5577 | 0.01651 | 0 |
| 8 | GBM (tuned) | 1057 | 137.7 | 137.1 | 7.538 | 0.533 | 0.02518 | 0 |
| 9 | XGB (tuned + ensemble) | 989.6 | 113.6 | 107.9 | 8.692 | 0.4505 | 0.02357 | 0 |
| 10 | XGB (tuned) | 985 | 133.5 | 125.9 | 8.769 | 0.4451 | 0.02325 | 0 |
| 11 | [New] MITER_BS6 (default) | 982.7 | 90.9 | 96.9 | 8.808 | 0.4423 | 0.02653 | 0 |
| 12 | [New] MITER_T12 (default) | 982.7 | 90.9 | 96.9 | 8.808 | 0.4423 | 0.02653 | 0 |
| 13 | CAT (default) | 932.4 | 118.9 | 182.4 | 9.615 | 0.3846 | 0.03786 | 0 |
| 14 | GBM (default) | 499 | 113.1 | 193.7 | 13.85 | 0.08242 | 0.05577 | 0 |
| 15 | XGB (default) | 388 | 97 | 245.6 | 14.38 | 0.04396 | 0.06313 | 0 |

### gbdt_only, binary 30 (`15` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_T12 (tuned + ensemble) | 1162 | 75.9 | 54.4 | 5 | 0.7143 | 0.04059 | 0 |
| 2 | GBM (tuned + ensemble) | 1117 | 52.7 | 59.2 | 5.8 | 0.6571 | 0.04239 | 0 |
| 3 | XGB (tuned + ensemble) | 1106 | 63.8 | 60.2 | 6 | 0.6429 | 0.04926 | 0 |
| 4 | CAT (tuned) | 1079 | 83.5 | 93.6 | 6.5 | 0.6071 | 0.03665 | 0 |
| 5 | CAT (tuned + ensemble) | 1069 | 92.2 | 92.8 | 6.7 | 0.5929 | 0.03898 | 0 |
| 6 | CAT (default) | 1054 | 77.2 | 88.2 | 7 | 0.5714 | 0.03659 | 0 |
| 7 | [New] MITER_T12 (tuned) | 1036 | 71.9 | 75 | 7.333 | 0.5476 | 0.05001 | 0 |
| 8 | XGB (tuned) | 1020 | 75.4 | 80.4 | 7.667 | 0.5238 | 0.05426 | 0 |
| 9 | [New] MITER_BS6 (tuned + ensemble) | 1008 | 98.6 | 88.2 | 7.9 | 0.5071 | 0.06251 | 0 |
| 10 | GBM (tuned) | 997.6 | 59.2 | 79.2 | 8.1 | 0.4929 | 0.05606 | 0 |
| 11 | [New] MITER_BS6 (tuned) | 997.6 | 97.7 | 90.3 | 8.1 | 0.4929 | 0.06351 | 0 |
| 12 | [New] MITER_T12 (default) | 890.2 | 61.5 | 71.4 | 10.15 | 0.3464 | 0.08089 | 0 |
| 13 | [New] MITER_BS6 (default) | 890.2 | 61.5 | 71.4 | 10.15 | 0.3464 | 0.08089 | 0 |
| 14 | XGB (default) | 832.9 | 98.3 | 90.4 | 11.13 | 0.2762 | 0.08523 | 0 |
| 15 | GBM (default) | 741.4 | 75.5 | 107.1 | 12.47 | 0.181 | 0.09776 | 0 |

## Secondary: full pool, 43 tasks

### full pool, 43 tasks (`124` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2087 | 175.4 | 137.2 | 6.698 | 0.9537 | 0.03236 | 0 |
| 2 | TA-LIMIX-2 (default) | 2068 | 161.4 | 120.2 | 7.256 | 0.9491 | 0.02878 | 0 |
| 3 | TA-SELDON (default) | 1990 | 143.3 | 110.9 | 9.86 | 0.928 | 0.04404 | 0 |
| 4 | Chakra-Tab (full) | 1980 | 102.9 | 82.6 | 10.27 | 0.9247 | 0.05455 | 0 |
| 5 | TA-KUMO-TABULAR-MEDIUM (default) | 1948 | 128.5 | 110.1 | 11.53 | 0.9144 | 0.05265 | 0 |
| 6 | Chakra-Tab (medium) | 1946 | 96.9 | 75.6 | 11.65 | 0.9134 | 0.0578 | 0 |
| 7 | TA-TABPFN-3.5 (default) | 1885 | 132.5 | 97.9 | 14.45 | 0.8906 | 0.07183 | 0 |
| 8 | TabFM+ | 1866 | 135.9 | 102 | 15.48 | 0.8823 | 0.0676 | 0 |
| 9 | INPRIORA (default) | 1853 | 104.5 | 74.5 | 16.14 | 0.8769 | 0.07461 | 0 |
| 10 | AutoGluon 1.6 (noncommercial, 4h) | 1842 | 84.8 | 59.5 | 16.77 | 0.8718 | 0.07635 | 0 |
| 42 | [New] MITER_T12 (tuned + ensemble) | 1398 | 49.9 | 47 | 55.88 | 0.5538 | 0.1696 | 0 |
| 45 | CAT (tuned) | 1392 | 53.3 | 54.6 | 56.6 | 0.5479 | 0.1632 | 0 |
| 46 | GBM (tuned + ensemble) | 1392 | 41.3 | 46.3 | 56.71 | 0.5471 | 0.1688 | 0 |
| 49 | CAT (tuned + ensemble) | 1382 | 49.9 | 51.7 | 57.8 | 0.5382 | 0.1645 | 0 |
| 53 | [New] MITER_T12 (tuned) | 1360 | 49.2 | 50.1 | 60.57 | 0.5157 | 0.1743 | 0 |
| 54 | XGB (tuned + ensemble) | 1358 | 48.2 | 51.3 | 60.78 | 0.514 | 0.1751 | 0 |
| 56 | CAT (default) | 1356 | 46.2 | 45.6 | 61 | 0.5122 | 0.1707 | 0 |
| 59 | XGB (tuned) | 1330 | 55.1 | 56.4 | 64.3 | 0.4853 | 0.1777 | 0 |
| 61 | GBM (tuned) | 1327 | 43.7 | 43 | 64.73 | 0.4818 | 0.1787 | 0 |
| 62 | [New] MITER_BS6 (tuned + ensemble) | 1321 | 50.1 | 56.9 | 65.38 | 0.4766 | 0.1841 | 0 |
| 64 | [New] MITER_BS6 (tuned) | 1317 | 51.5 | 60.4 | 65.99 | 0.4716 | 0.1845 | 0 |
| 77 | [New] MITER_T12 (default) | 1257 | 57.7 | 56.8 | 73.51 | 0.4105 | 0.1948 | 0 |
| 78 | [New] MITER_BS6 (default) | 1257 | 57.7 | 56.8 | 73.51 | 0.4105 | 0.1948 | 0 |
| 94 | XGB (default) | 1179 | 61.3 | 59.2 | 83.05 | 0.333 | 0.2065 | 0 |
| 101 | GBM (default) | 1140 | 39.3 | 43.3 | 87.5 | 0.2967 | 0.21 | 0 |

