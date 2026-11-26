# MITER_V11_T12 on TabArena `lite`, all 51 tasks (Native Softmax + CatBoost CTRs)

Pre-registered in `benchmarks/results/exp_softmax_tabarena51_prereg.md` (Part B).
Caveat: TabArena `lite` evaluates **single split (`fold 0`) per dataset**.

## Anti-Fluke Headline Verdict

- **Verdict:** **MITER_V11_T12 is statistically level with CatBoost (tuned + ensemble)** (p >= 0.05; anti-fluke headline claim NOT met).
- Criteria: Elo > CAT (1156.2 vs 1101.8: True), p < 0.05 (0.5738), 95% boot CI < 0 ([-1.24%, +0.65%]), wins >= 33/51 (27/51), MC8 improves over OvR (True).

## Primary: gbdt_only pool, 51 tasks

### gbdt_only, 51 tasks (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V11_T12 (tuned + ensemble) | 1156 | 61.8 | 47.4 | 4.284 | 0.7014 | 0.0376 | 0 |
| 2 | GBM (tuned + ensemble) | 1134 | 54.7 | 39.3 | 4.588 | 0.6738 | 0.04456 | 0 |
| 3 | CAT (tuned + ensemble) | 1102 | 71.2 | 58.7 | 5.059 | 0.631 | 0.0327 | 0 |
| 4 | CAT (tuned) | 1091 | 65.9 | 60 | 5.216 | 0.6168 | 0.02935 | 0 |
| 5 | [New] MITER_V11_T12 (tuned) | 1073 | 65.4 | 55.4 | 5.49 | 0.5918 | 0.04614 | 0 |
| 6 | XGB (tuned + ensemble) | 1061 | 48 | 44.8 | 5.667 | 0.5758 | 0.05116 | 0 |
| 7 | GBM (tuned) | 1020 | 58 | 51 | 6.294 | 0.5187 | 0.05497 | 0 |
| 8 | XGB (tuned) | 1008 | 58.1 | 59.8 | 6.471 | 0.5027 | 0.05356 | 0 |
| 9 | CAT (default) | 983.8 | 59.4 | 67.9 | 6.843 | 0.4688 | 0.05181 | 0 |
| 10 | [New] MITER_V11_T12 (default) | 908.1 | 65.7 | 75.1 | 7.951 | 0.3681 | 0.07692 | 0 |
| 11 | XGB (default) | 758.3 | 80.8 | 83.8 | 9.804 | 0.1996 | 0.08911 | 0 |
| 12 | GBM (default) | 703.8 | 78.5 | 82.2 | 10.33 | 0.1515 | 0.09993 | 0 |

- **all 51 vs CAT** (51 tasks): wins 27/51, median -0.03% (95% boot CI [-1.24%, +0.65%]), mean +1.19%, Wilcoxon p = 0.5738, worst +63.13%
- **regression 13 vs CAT** (13 tasks): wins 7/13, median -0.53% (95% boot CI [-1.24%, +0.28%]), mean +0.22%, Wilcoxon p = 0.8394, worst +5.10%
- **binary 30 vs CAT** (30 tasks): wins 16/30, median -0.03% (95% boot CI [-1.43%, +3.33%]), mean -0.47%, Wilcoxon p = 0.7611, worst +17.23%
- **multiclass 8 vs CAT** (8 tasks): wins 4/8, median +2.46% (95% boot CI [-1.84%, +7.12%]), mean +8.98%, Wilcoxon p = 0.4609, worst +63.13%
- **MITER_V11_T12 vs cached MITER_T12 (OvR) on multiclass 8** (8 tasks): wins 7/8, median -0.64% (95% boot CI [-6.69%, -0.13%]), mean -5.12%, Wilcoxon p = 0.07812, worst +2.37%

### gbdt_only, regression 13 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1243 | 158.1 | 99.9 | 3.692 | 0.7552 | 0.01607 | 0 |
| 2 | CAT (tuned + ensemble) | 1211 | 153.7 | 106.5 | 4.077 | 0.7203 | 0.01134 | 0 |
| 3 | [New] MITER_V11_T12 (tuned + ensemble) | 1199 | 147.3 | 94.8 | 4.231 | 0.7063 | 0.01327 | 0 |
| 4 | [New] MITER_V11_T12 (tuned) | 1152 | 162.7 | 130 | 4.846 | 0.6503 | 0.01501 | 0 |
| 5 | CAT (tuned) | 1140 | 135.7 | 101 | 5 | 0.6364 | 0.01173 | 0 |
| 6 | GBM (tuned) | 1073 | 135.5 | 149.8 | 5.923 | 0.5524 | 0.02505 | 0 |
| 7 | [New] MITER_V11_T12 (default) | 1010 | 110.9 | 124.9 | 6.769 | 0.4755 | 0.0264 | 0 |
| 8 | XGB (tuned) | 1004 | 118.9 | 108.9 | 6.846 | 0.4685 | 0.02312 | 0 |
| 9 | XGB (tuned + ensemble) | 998.4 | 99.3 | 83.2 | 6.923 | 0.4615 | 0.02344 | 0 |
| 10 | CAT (default) | 956.4 | 118.7 | 147.8 | 7.462 | 0.4126 | 0.03773 | 0 |
| 11 | GBM (default) | 562.7 | 112.2 | 182.2 | 10.85 | 0.1049 | 0.05564 | 0 |
| 12 | XGB (default) | 451.2 | 98.5 | 226.8 | 11.38 | 0.05594 | 0.06301 | 0 |

### gbdt_only, binary 30 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V11_T12 (tuned + ensemble) | 1160 | 87.2 | 62 | 4.167 | 0.7121 | 0.03352 | 0 |
| 2 | GBM (tuned + ensemble) | 1111 | 49.8 | 46.1 | 4.867 | 0.6485 | 0.04404 | 0 |
| 3 | XGB (tuned + ensemble) | 1093 | 66.7 | 55.6 | 5.133 | 0.6242 | 0.05099 | 0 |
| 4 | CAT (tuned) | 1071 | 76.2 | 75 | 5.467 | 0.5939 | 0.03804 | 0 |
| 5 | CAT (tuned + ensemble) | 1066 | 85 | 85 | 5.533 | 0.5879 | 0.04032 | 0 |
| 6 | [New] MITER_V11_T12 (tuned) | 1033 | 78.8 | 73.6 | 6.05 | 0.5409 | 0.04808 | 0 |
| 7 | CAT (default) | 1032 | 73.2 | 75.8 | 6.067 | 0.5394 | 0.03809 | 0 |
| 8 | XGB (tuned) | 1010 | 72.5 | 84.9 | 6.4 | 0.5091 | 0.05604 | 0 |
| 9 | GBM (tuned) | 986.9 | 54.8 | 85.1 | 6.767 | 0.4758 | 0.05772 | 0 |
| 10 | [New] MITER_V11_T12 (default) | 880.4 | 78.5 | 91.1 | 8.35 | 0.3318 | 0.08259 | 0 |
| 11 | XGB (default) | 829.1 | 103.7 | 104.6 | 9.033 | 0.2697 | 0.08686 | 0 |
| 12 | GBM (default) | 728.6 | 78.1 | 114 | 10.17 | 0.1667 | 0.09931 | 0 |

### gbdt_only, multiclass 8 (`12` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [New] MITER_V11_T12 (tuned) | 1153 | 185.9 | 113.4 | 4.438 | 0.6875 | 0.08944 | 0 |
| 2 | CAT (tuned) | 1139 | 259.6 | 158.9 | 4.625 | 0.6705 | 0.02541 | 0 |
| 3 | [New] MITER_V11_T12 (tuned + ensemble) | 1126 | 218.9 | 120.6 | 4.812 | 0.6534 | 0.09243 | 0 |
| 4 | CAT (tuned + ensemble) | 1121 | 229.5 | 156.1 | 4.875 | 0.6477 | 0.03883 | 0 |
| 5 | GBM (tuned + ensemble) | 1112 | 151.8 | 103.6 | 5 | 0.6364 | 0.0928 | 0 |
| 6 | GBM (tuned) | 1104 | 141.5 | 128 | 5.125 | 0.625 | 0.09329 | 0 |
| 7 | XGB (tuned + ensemble) | 1069 | 130.7 | 122 | 5.625 | 0.5795 | 0.09683 | 0 |
| 8 | XGB (tuned) | 1034 | 176.8 | 148.7 | 6.125 | 0.5341 | 0.09376 | 0 |
| 9 | [New] MITER_V11_T12 (default) | 871.8 | 195.3 | 253.1 | 8.375 | 0.3295 | 0.1377 | 0 |
| 10 | CAT (default) | 841.7 | 124.6 | 296 | 8.75 | 0.2955 | 0.1261 | 0 |
| 11 | GBM (default) | 713.7 | 199.8 | 396.6 | 10.12 | 0.1705 | 0.1742 | 0 |
| 12 | XGB (default) | 713.7 | 83.8 | 173.2 | 10.12 | 0.1705 | 0.14 | 0 |

## Secondary: full pool, 51 tasks

### full pool, 51 tasks (`121` methods)

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2092 | 173.4 | 111.1 | 6.294 | 0.9559 | 0.03645 | 0 |
| 2 | TA-LIMIX-2 (default) | 2060 | 167.2 | 107.2 | 7.176 | 0.9485 | 0.02974 | 0 |
| 3 | Chakra-Tab (full) | 1970 | 118.6 | 76.7 | 10.23 | 0.9231 | 0.05738 | 0 |
| 4 | TA-SELDON (default) | 1961 | 185.4 | 134.8 | 10.61 | 0.9199 | 0.04833 | 0 |
| 5 | Chakra-Tab (medium) | 1940 | 105.7 | 64.3 | 11.45 | 0.9129 | 0.06029 | 0 |
| 6 | TA-KUMO-TABULAR-MEDIUM (default) | 1936 | 148.6 | 99.2 | 11.63 | 0.9114 | 0.05609 | 0 |
| 7 | TA-TABPFN-3.5 (default) | 1882 | 144 | 92.4 | 14.13 | 0.8906 | 0.07414 | 0 |
| 8 | TabFM+ | 1858 | 127 | 106.6 | 15.34 | 0.8805 | 0.07075 | 0 |
| 9 | TA-SELDON-MINI (default) | 1840 | 111.2 | 78.4 | 16.26 | 0.8728 | 0.08427 | 0 |
| 10 | INPRIORA (default) | 1833 | 102 | 94.2 | 16.67 | 0.8694 | 0.07667 | 0 |
| 42 | [New] MITER_V11_T12 (tuned + ensemble) | 1402 | 41.4 | 42.8 | 53.77 | 0.5602 | 0.1733 | 0 |
| 43 | CAT (tuned) | 1397 | 45 | 37.8 | 54.43 | 0.5547 | 0.1652 | 0 |
| 44 | GBM (tuned + ensemble) | 1390 | 34.4 | 37.1 | 55.17 | 0.5486 | 0.1757 | 0 |
| 45 | CAT (tuned + ensemble) | 1388 | 39.6 | 36.5 | 55.4 | 0.5467 | 0.1675 | 0 |
| 52 | [New] MITER_V11_T12 (tuned) | 1360 | 42.6 | 45.8 | 58.71 | 0.5191 | 0.1793 | 0 |
| 54 | XGB (tuned + ensemble) | 1352 | 38.7 | 42.4 | 59.72 | 0.5107 | 0.1819 | 0 |
| 57 | CAT (default) | 1340 | 35.8 | 44.3 | 61.2 | 0.4984 | 0.1814 | 0 |
| 58 | GBM (tuned) | 1336 | 41.9 | 42.6 | 61.7 | 0.4942 | 0.1842 | 0 |
| 60 | XGB (tuned) | 1330 | 44.1 | 46.8 | 62.45 | 0.4879 | 0.1838 | 0 |
| 75 | [New] MITER_V11_T12 (default) | 1260 | 56.8 | 60.2 | 70.99 | 0.4167 | 0.2034 | 0 |
| 90 | XGB (default) | 1181 | 46.4 | 52.5 | 80.31 | 0.3391 | 0.2137 | 0 |
| 99 | GBM (default) | 1148 | 53.3 | 48.1 | 84.09 | 0.3076 | 0.2203 | 0 |

