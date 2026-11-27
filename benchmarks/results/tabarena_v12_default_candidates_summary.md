# MITER_V12_T25 manual configs as stand-alone defaults, TabArena `lite` (51 tasks)

Exploratory, from the V12_T25 cache (no new training). Each config is one bagged model (8 folds, fold 0), the same protocol as the `(default)` rows of the baselines. The candidates were written after earlier TabArena runs, so this is not out-of-sample.

| config | settings |
|---|---|
| `LW31` | current default (level-wise, 31 leaves, linear quantizer, no CTR) |
| `S6` | grow_policy=symmetric, max_depth=6, min_data_in_leaf=1, lambda_l2=3 |
| `S6_QAUTO` | S6 + quantizer=auto |
| `S6_QAUTO_CTR` | S6 + quantizer=auto + max_ctr_complexity=auto + ctr_n_permutations=4 |
| `ACCURATE` | preset=accurate |
| `S6_M4_QQ_CTR` | S6 + m_mult=4 + quantizer=quantile + CTR auto, 4 permutations |

## gbdt_only, all 51 (51 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1135 | 56 | 45.8 | 5.47 | 0.681 |
| 2 | CAT (tuned + ensemble) | 1099 | 61.4 | 58.4 | 6.14 | 0.633 |
| 3 | [New] MITER_S6_M4_QQ_CTR (default) | 1092 | 55.6 | 50.1 | 6.27 | 0.623 |
| 4 | CAT (tuned) | 1090 | 59.7 | 54.6 | 6.31 | 0.62 |
| 5 | [New] MITER_S6_QAUTO_CTR (default) | 1065 | 59.1 | 53 | 6.79 | 0.586 |
| 6 | XGB (tuned + ensemble) | 1058 | 44.9 | 46 | 6.94 | 0.576 |
| 7 | GBM (tuned) | 1031 | 61.6 | 51.8 | 7.47 | 0.538 |
| 8 | [New] MITER_S6_QAUTO (default) | 1026 | 62 | 53.9 | 7.56 | 0.532 |
| 9 | XGB (tuned) | 1011 | 54 | 61 | 7.86 | 0.51 |
| 10 | CAT (default) | 994.4 | 64 | 68.7 | 8.2 | 0.486 |
| 11 | [New] MITER_S6 (default) | 993.4 | 64.3 | 59.2 | 8.22 | 0.485 |
| 12 | [New] MITER_ACCURATE (default) | 971.5 | 70.4 | 74.9 | 8.65 | 0.454 |
| 13 | [New] MITER_LW31 (default) | 928.8 | 54.8 | 63.4 | 9.47 | 0.395 |
| 14 | XGB (default) | 783.9 | 63.8 | 86.9 | 11.9 | 0.221 |
| 15 | GBM (default) | 719.3 | 64.9 | 74.9 | 12.7 | 0.161 |

| miter config | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_LW31 (default) | CAT (default) | 51 | 22/51 | +0.88% | +3.28% | 0.0583 |
| MITER_LW31 (default) | GBM (default) | 51 | 42/51 | -1.72% | -2.64% | 0.0000 |
| MITER_LW31 (default) | XGB (default) | 51 | 35/51 | -1.35% | -1.04% | 0.0227 |
| MITER_S6 (default) | CAT (default) | 51 | 22/51 | +0.21% | +2.27% | 0.2567 |
| MITER_S6 (default) | GBM (default) | 51 | 40/51 | -3.03% | -3.59% | 0.0000 |
| MITER_S6 (default) | XGB (default) | 51 | 36/51 | -1.55% | -1.92% | 0.0130 |
| MITER_S6 (default) | [New] MITER_LW31 (default) | 51 | 31/51 | -0.38% | -0.96% | 0.0635 |
| MITER_S6_QAUTO (default) | CAT (default) | 51 | 26/51 | -0.10% | +1.37% | 0.9551 |
| MITER_S6_QAUTO (default) | GBM (default) | 51 | 42/51 | -3.18% | -4.42% | 0.0000 |
| MITER_S6_QAUTO (default) | XGB (default) | 51 | 38/51 | -2.17% | -2.78% | 0.0006 |
| MITER_S6_QAUTO (default) | [New] MITER_LW31 (default) | 51 | 31/51 | -0.90% | -1.76% | 0.0120 |
| MITER_S6_QAUTO_CTR (default) | CAT (default) | 51 | 31/51 | -0.42% | -0.46% | 0.2091 |
| MITER_S6_QAUTO_CTR (default) | GBM (default) | 51 | 45/51 | -4.04% | -5.91% | 0.0000 |
| MITER_S6_QAUTO_CTR (default) | XGB (default) | 51 | 42/51 | -3.06% | -4.27% | 0.0000 |
| MITER_S6_QAUTO_CTR (default) | [New] MITER_LW31 (default) | 51 | 33/51 | -1.36% | -3.02% | 0.0008 |
| MITER_ACCURATE (default) | CAT (default) | 51 | 22/51 | +0.73% | +3.02% | 0.1132 |
| MITER_ACCURATE (default) | GBM (default) | 51 | 41/51 | -2.26% | -2.89% | 0.0001 |
| MITER_ACCURATE (default) | XGB (default) | 51 | 34/51 | -1.62% | -1.28% | 0.0270 |
| MITER_ACCURATE (default) | [New] MITER_LW31 (default) | 51 | 32/51 | -0.11% | -0.26% | 0.0224 |
| MITER_S6_M4_QQ_CTR (default) | CAT (default) | 51 | 32/51 | -0.35% | -0.45% | 0.1711 |
| MITER_S6_M4_QQ_CTR (default) | GBM (default) | 51 | 47/51 | -3.59% | -5.92% | 0.0000 |
| MITER_S6_M4_QQ_CTR (default) | XGB (default) | 51 | 43/51 | -3.35% | -4.28% | 0.0000 |
| MITER_S6_M4_QQ_CTR (default) | [New] MITER_LW31 (default) | 51 | 36/51 | -1.41% | -3.05% | 0.0007 |

## gbdt_only, regression (13 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1222 | 168 | 98.6 | 4.46 | 0.753 |
| 2 | CAT (tuned + ensemble) | 1212 | 128 | 75.8 | 4.62 | 0.742 |
| 3 | CAT (tuned) | 1156 | 125 | 79.1 | 5.54 | 0.676 |
| 4 | [New] MITER_ACCURATE (default) | 1120 | 115 | 109 | 6.15 | 0.632 |
| 5 | GBM (tuned) | 1070 | 145 | 124 | 7.08 | 0.566 |
| 6 | [New] MITER_S6_M4_QQ_CTR (default) | 1057 | 95.5 | 101 | 7.31 | 0.549 |
| 7 | [New] MITER_LW31 (default) | 1044 | 99.5 | 102 | 7.54 | 0.533 |
| 8 | [New] MITER_S6_QAUTO (default) | 1036 | 149 | 132 | 7.69 | 0.522 |
| 9 | [New] MITER_S6 (default) | 1019 | 142 | 150 | 8 | 0.5 |
| 10 | XGB (tuned + ensemble) | 1011 | 95.2 | 85.9 | 8.15 | 0.489 |
| 11 | XGB (tuned) | 998 | 127 | 116 | 8.38 | 0.473 |
| 12 | [New] MITER_S6_QAUTO_CTR (default) | 993.7 | 104 | 126 | 8.46 | 0.467 |
| 13 | CAT (default) | 958.9 | 116 | 146 | 9.08 | 0.423 |
| 14 | GBM (default) | 622.6 | 138 | 229 | 13.3 | 0.121 |
| 15 | XGB (default) | 480.4 | 95.1 | 254 | 14.2 | 0.0549 |

| miter config | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_LW31 (default) | CAT (default) | 13 | 8/13 | -0.59% | -1.19% | 0.3054 |
| MITER_LW31 (default) | GBM (default) | 13 | 13/13 | -2.66% | -3.06% | 0.0002 |
| MITER_LW31 (default) | XGB (default) | 13 | 13/13 | -2.67% | -3.81% | 0.0002 |
| MITER_S6 (default) | CAT (default) | 13 | 7/13 | -0.25% | -1.82% | 0.2439 |
| MITER_S6 (default) | GBM (default) | 13 | 11/13 | -2.88% | -3.65% | 0.0024 |
| MITER_S6 (default) | XGB (default) | 13 | 13/13 | -2.78% | -4.41% | 0.0002 |
| MITER_S6 (default) | [New] MITER_LW31 (default) | 13 | 7/13 | -0.16% | -0.65% | 0.4548 |
| MITER_S6_QAUTO (default) | CAT (default) | 13 | 8/13 | -0.23% | -1.94% | 0.2163 |
| MITER_S6_QAUTO (default) | GBM (default) | 13 | 11/13 | -2.79% | -3.77% | 0.0024 |
| MITER_S6_QAUTO (default) | XGB (default) | 13 | 12/13 | -2.98% | -4.54% | 0.0005 |
| MITER_S6_QAUTO (default) | [New] MITER_LW31 (default) | 13 | 6/13 | +0.09% | -0.79% | 0.5879 |
| MITER_S6_QAUTO_CTR (default) | CAT (default) | 13 | 8/13 | -0.33% | -1.30% | 0.1909 |
| MITER_S6_QAUTO_CTR (default) | GBM (default) | 13 | 11/13 | -2.82% | -3.13% | 0.0017 |
| MITER_S6_QAUTO_CTR (default) | XGB (default) | 13 | 12/13 | -2.99% | -3.87% | 0.0005 |
| MITER_S6_QAUTO_CTR (default) | [New] MITER_LW31 (default) | 13 | 4/13 | +0.23% | -0.05% | 0.7869 |
| MITER_ACCURATE (default) | CAT (default) | 13 | 8/13 | -0.83% | -1.49% | 0.2163 |
| MITER_ACCURATE (default) | GBM (default) | 13 | 13/13 | -2.82% | -3.36% | 0.0002 |
| MITER_ACCURATE (default) | XGB (default) | 13 | 13/13 | -2.90% | -4.10% | 0.0002 |
| MITER_ACCURATE (default) | [New] MITER_LW31 (default) | 13 | 10/13 | -0.25% | -0.30% | 0.0134 |
| MITER_S6_M4_QQ_CTR (default) | CAT (default) | 13 | 9/13 | -0.28% | -1.27% | 0.1677 |
| MITER_S6_M4_QQ_CTR (default) | GBM (default) | 13 | 12/13 | -2.90% | -3.11% | 0.0012 |
| MITER_S6_M4_QQ_CTR (default) | XGB (default) | 13 | 13/13 | -3.48% | -3.85% | 0.0002 |
| MITER_S6_M4_QQ_CTR (default) | [New] MITER_LW31 (default) | 13 | 6/13 | +0.05% | -0.03% | 0.7869 |

## gbdt_only, binary (30 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1115 | 57 | 54.3 | 5.8 | 0.657 |
| 2 | [New] MITER_S6_M4_QQ_CTR (default) | 1110 | 75.1 | 59.6 | 5.9 | 0.65 |
| 3 | [New] MITER_S6_QAUTO_CTR (default) | 1088 | 86.6 | 69.3 | 6.32 | 0.62 |
| 4 | XGB (tuned + ensemble) | 1082 | 62.6 | 51.9 | 6.43 | 0.612 |
| 5 | CAT (tuned) | 1066 | 82.5 | 92 | 6.73 | 0.59 |
| 6 | CAT (tuned + ensemble) | 1064 | 89.5 | 92.7 | 6.77 | 0.588 |
| 7 | CAT (default) | 1043 | 80.5 | 79.1 | 7.2 | 0.557 |
| 8 | [New] MITER_S6_QAUTO (default) | 1024 | 95.5 | 80.8 | 7.57 | 0.531 |
| 9 | XGB (tuned) | 1011 | 68.6 | 82.5 | 7.83 | 0.512 |
| 10 | GBM (tuned) | 1003 | 55.5 | 72.5 | 8 | 0.5 |
| 11 | [New] MITER_S6 (default) | 978.4 | 88.2 | 94 | 8.48 | 0.465 |
| 12 | [New] MITER_ACCURATE (default) | 920.4 | 73.9 | 89.7 | 9.62 | 0.385 |
| 13 | [New] MITER_LW31 (default) | 893.6 | 70.6 | 74 | 10.1 | 0.349 |
| 14 | XGB (default) | 855.1 | 103 | 86.5 | 10.8 | 0.3 |
| 15 | GBM (default) | 747.5 | 68.2 | 94.7 | 12.4 | 0.183 |

| miter config | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_LW31 (default) | CAT (default) | 30 | 9/30 | +4.06% | +5.81% | 0.0066 |
| MITER_LW31 (default) | GBM (default) | 30 | 22/30 | -1.41% | -1.51% | 0.0113 |
| MITER_LW31 (default) | XGB (default) | 30 | 18/30 | -0.39% | -0.09% | 0.5291 |
| MITER_S6 (default) | CAT (default) | 30 | 11/30 | +1.11% | +4.26% | 0.0667 |
| MITER_S6 (default) | GBM (default) | 30 | 21/30 | -3.54% | -2.87% | 0.0164 |
| MITER_S6 (default) | XGB (default) | 30 | 17/30 | -1.04% | -1.50% | 0.2894 |
| MITER_S6 (default) | [New] MITER_LW31 (default) | 30 | 18/30 | -0.66% | -1.50% | 0.0919 |
| MITER_S6_QAUTO (default) | CAT (default) | 30 | 13/30 | +0.54% | +3.07% | 0.3387 |
| MITER_S6_QAUTO (default) | GBM (default) | 30 | 23/30 | -4.33% | -3.98% | 0.0019 |
| MITER_S6_QAUTO (default) | XGB (default) | 30 | 19/30 | -1.41% | -2.61% | 0.0497 |
| MITER_S6_QAUTO (default) | [New] MITER_LW31 (default) | 30 | 19/30 | -1.35% | -2.54% | 0.0164 |
| MITER_S6_QAUTO_CTR (default) | CAT (default) | 30 | 17/30 | -0.28% | -0.14% | 0.7457 |
| MITER_S6_QAUTO_CTR (default) | GBM (default) | 30 | 26/30 | -5.26% | -6.61% | 0.0000 |
| MITER_S6_QAUTO_CTR (default) | XGB (default) | 30 | 23/30 | -3.24% | -5.24% | 0.0010 |
| MITER_S6_QAUTO_CTR (default) | [New] MITER_LW31 (default) | 30 | 24/30 | -2.22% | -4.82% | 0.0002 |
| MITER_ACCURATE (default) | CAT (default) | 30 | 9/30 | +2.70% | +5.59% | 0.0113 |
| MITER_ACCURATE (default) | GBM (default) | 30 | 21/30 | -1.22% | -1.73% | 0.0248 |
| MITER_ACCURATE (default) | XGB (default) | 30 | 17/30 | -0.60% | -0.30% | 0.6554 |
| MITER_ACCURATE (default) | [New] MITER_LW31 (default) | 30 | 18/30 | -0.09% | -0.24% | 0.1631 |
| MITER_S6_M4_QQ_CTR (default) | CAT (default) | 30 | 17/30 | -0.26% | -0.21% | 0.7151 |
| MITER_S6_M4_QQ_CTR (default) | GBM (default) | 30 | 27/30 | -5.34% | -6.69% | 0.0000 |
| MITER_S6_M4_QQ_CTR (default) | XGB (default) | 30 | 23/30 | -3.21% | -5.35% | 0.0005 |
| MITER_S6_M4_QQ_CTR (default) | [New] MITER_LW31 (default) | 30 | 24/30 | -2.42% | -4.95% | 0.0001 |

## gbdt_only, multiclass (8 tasks)

| # | method | elo | elo+ | elo- | rank | winrate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1120 | 111 | 70.5 | 5.88 | 0.652 |
| 2 | [New] MITER_S6_QAUTO_CTR (default) | 1120 | 189 | 116 | 5.88 | 0.652 |
| 3 | CAT (tuned) | 1113 | 205 | 152 | 6 | 0.643 |
| 4 | [New] MITER_S6_M4_QQ_CTR (default) | 1113 | 260 | 160 | 6 | 0.643 |
| 5 | GBM (tuned) | 1106 | 119 | 104 | 6.12 | 0.634 |
| 6 | CAT (tuned + ensemble) | 1099 | 194 | 142 | 6.25 | 0.625 |
| 7 | XGB (tuned + ensemble) | 1066 | 148 | 132 | 6.88 | 0.58 |
| 8 | XGB (tuned) | 1053 | 208 | 169 | 7.12 | 0.562 |
| 9 | [New] MITER_S6_QAUTO (default) | 1043 | 160 | 139 | 7.31 | 0.549 |
| 10 | [New] MITER_S6 (default) | 1030 | 176 | 158 | 7.56 | 0.531 |
| 11 | [New] MITER_ACCURATE (default) | 950 | 202 | 224 | 9.06 | 0.424 |
| 12 | [New] MITER_LW31 (default) | 886.7 | 143 | 197 | 10.2 | 0.344 |
| 13 | CAT (default) | 868.1 | 117 | 236 | 10.5 | 0.321 |
| 14 | XGB (default) | 749.2 | 119 | 225 | 12.2 | 0.196 |
| 15 | GBM (default) | 685 | 167 | 339 | 13 | 0.143 |

| miter config | baseline | n | wins | median | mean | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| MITER_LW31 (default) | CAT (default) | 8 | 5/8 | -1.32% | +1.03% | 0.7422 |
| MITER_LW31 (default) | GBM (default) | 8 | 7/8 | -2.26% | -6.17% | 0.0781 |
| MITER_LW31 (default) | XGB (default) | 8 | 4/8 | -0.26% | -0.09% | 0.7422 |
| MITER_S6 (default) | CAT (default) | 8 | 4/8 | +0.23% | +1.48% | 0.6406 |
| MITER_S6 (default) | GBM (default) | 8 | 8/8 | -4.92% | -6.21% | 0.0078 |
| MITER_S6 (default) | XGB (default) | 8 | 6/8 | -1.44% | +0.52% | 0.6406 |
| MITER_S6 (default) | [New] MITER_LW31 (default) | 8 | 6/8 | -0.89% | +0.58% | 0.5469 |
| MITER_S6_QAUTO (default) | CAT (default) | 8 | 5/8 | -0.72% | +0.39% | 1.0000 |
| MITER_S6_QAUTO (default) | GBM (default) | 8 | 8/8 | -4.85% | -7.14% | 0.0078 |
| MITER_S6_QAUTO (default) | XGB (default) | 8 | 7/8 | -2.78% | -0.59% | 0.1953 |
| MITER_S6_QAUTO (default) | [New] MITER_LW31 (default) | 8 | 6/8 | -0.95% | -0.42% | 0.4609 |
| MITER_S6_QAUTO_CTR (default) | CAT (default) | 8 | 6/8 | -1.00% | -0.29% | 0.7422 |
| MITER_S6_QAUTO_CTR (default) | GBM (default) | 8 | 8/8 | -4.99% | -7.78% | 0.0078 |
| MITER_S6_QAUTO_CTR (default) | XGB (default) | 8 | 7/8 | -3.35% | -1.26% | 0.1953 |
| MITER_S6_QAUTO_CTR (default) | [New] MITER_LW31 (default) | 8 | 5/8 | -1.55% | -1.10% | 0.4609 |
| MITER_ACCURATE (default) | CAT (default) | 8 | 5/8 | -1.28% | +0.72% | 0.8438 |
| MITER_ACCURATE (default) | GBM (default) | 8 | 7/8 | -2.32% | -6.48% | 0.0781 |
| MITER_ACCURATE (default) | XGB (default) | 8 | 4/8 | -0.33% | -0.40% | 0.7422 |
| MITER_ACCURATE (default) | [New] MITER_LW31 (default) | 8 | 4/8 | -0.03% | -0.30% | 0.5781 |
| MITER_S6_M4_QQ_CTR (default) | CAT (default) | 8 | 6/8 | -1.56% | -0.01% | 0.7422 |
| MITER_S6_M4_QQ_CTR (default) | GBM (default) | 8 | 8/8 | -4.44% | -7.59% | 0.0078 |
| MITER_S6_M4_QQ_CTR (default) | XGB (default) | 8 | 7/8 | -3.42% | -0.95% | 0.1953 |
| MITER_S6_M4_QQ_CTR (default) | [New] MITER_LW31 (default) | 8 | 6/8 | -1.44% | -0.80% | 0.3828 |

## Median time per task (s, 8-fold bag)

| method | train | infer |
|---|---:|---:|
| [New] MITER_ACCURATE (default) | 5.2 | 0.08 |
| [New] MITER_LW31 (default) | 3.4 | 0.07 |
| [New] MITER_S6 (default) | 6.0 | 0.14 |
| [New] MITER_S6_M4_QQ_CTR (default) | 7.5 | 0.25 |
| [New] MITER_S6_QAUTO (default) | 6.5 | 0.19 |
| [New] MITER_S6_QAUTO_CTR (default) | 7.9 | 0.25 |
