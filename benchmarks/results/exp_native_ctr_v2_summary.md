# Confirmatory Summary: Native C++20 Online Ordered CTR v2 (`exp-native-ctr-v2`)

Pre-registered in `benchmarks/results/exp_native_ctr_v2_prereg.md` (16 datasets x 5 folds = 80 paired runs/variant).

## 1. Anti-Fluke Confirmatory Scorecard (Holm-Bonferroni m=2, 10k bootstrap CI)

| Comparison | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | p_Holm | Worst % | Time vs S6 | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Primary: S6_CTR2 vs S6 | -0.05% | [-1.25%, +0.20%] | 8/16 | 0.2769 | 0.2769 | +1.42% | 2.24x | FAIL |
| Secondary: S6_CTR1 vs S6 | -0.31% | [-0.74%, -0.05%] | 12/16 | 0.008545 | 0.01709 | +1.02% | 1.41x | FAIL |

## 2. Exploratory Comparisons (Primary metric unless noted)

| Comparison | Subgroup | N | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |
|---|---|---:|---:|---|---:|---:|---:|
| S6 vs CAT | 16 | 16 | +0.09% | [-0.73%, +1.30%] | 8/16 | 0.7436 | +26.34% |
| S6_CTR1 vs CAT | 16 | 16 | -0.16% | [-1.02%, +0.59%] | 9/16 | 0.6685 | +8.65% |
| S6_CTR2 vs CAT | 16 | 16 | -0.13% | [-1.15%, +0.96%] | 8/16 | 0.7057 | +2.80% |
| S6_CTR2 vs S6_CTR1 | 16 | 16 | -0.05% | [-0.27%, +0.26%] | 9/16 | 0.7615 | +1.85% |
| S6_CTR2 vs S6 (1-AUC) | 9 | 9 | -1.24% | [-6.01%, +0.87%] | 7/9 | 0.1641 | +5.21% |
| S6_CTR2 vs CAT (1-AUC) | 9 | 9 | +2.03% | [-2.39%, +5.88%] | 3/9 | 0.25 | +6.50% |
| S6_CTR2 vs S6 (max_card>253) | 5 | 5 | -0.52% | [-19.40%, +0.75%] | 3/5 | 0.4375 | +0.75% |
| S6_CTR2 vs CAT (max_card>253) | 5 | 5 | +0.12% | [-0.75%, +1.83%] | 2/5 | 1 | +1.83% |

## 3. Per-dataset Primary Metric & Wall Time (5-fold mean; lower is better)

| kind | dataset | n | p_cat | max_card | S6 | S6_CTR1 | S6_CTR2 | CAT | CTR2 vs S6 | S6 t | CTR2 t | CAT t | mean tbls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reg | `abalone` | 4177 | 1 | 3 | 0.65272 | 0.65272 | 0.65272 | 0.67354 | +0.00% | 0.24s | 0.13s | 0.63s | 0.0 |
| reg | `diamonds` | 25000 | 3 | 8 | 0.13780 | 0.13686 | 0.13784 | 0.13882 | +0.03% | 1.23s | 2.74s | 7.84s | 12.0 |
| reg | `Mercedes_Benz` | 4209 | 8 | 47 | 0.65579 | 0.65545 | 0.65635 | 0.65348 | +0.09% | 1.32s | 1.41s | 2.95s | 50.6 |
| reg | `avocado-sales` | 18249 | 3 | 54 | 0.34348 | 0.34249 | 0.34519 | 0.35937 | +0.50% | 9.18s | 12.57s | 26.53s | 5.4 |
| reg | `nyc-taxi-green-dec-2016` | 25000 | 9 | 240 | 0.74380 | 0.71473 | 0.70201 | 0.74736 | -5.62% | 1.25s | 3.95s | 3.82s | 46.8 |
| reg | `colleges` | 7063 | 12 | 6039 | 0.62468 | 0.63106 | 0.62936 | 0.63176 | +0.75% | 3.53s | 3.62s | 14.05s | 72.2 |
| reg | `Food_Delivery_Time` | 25000 | 3 | 1320 | 0.79894 | 0.80065 | 0.80140 | 0.79902 | +0.31% | 1.49s | 1.69s | 9.43s | 7.4 |
| clf | `credit-g` | 1000 | 13 | 10 | 0.50723 | 0.50513 | 0.51446 | 0.50623 | +1.42% | 0.06s | 0.15s | 0.74s | 152.8 |
| clf | `bank-marketing` | 25000 | 9 | 12 | 0.19864 | 0.19635 | 0.19415 | 0.19684 | -2.26% | 1.61s | 3.64s | 13.62s | 89.6 |
| clf | `adult` | 25000 | 8 | 41 | 0.28611 | 0.28453 | 0.28427 | 0.27902 | -0.64% | 1.32s | 6.11s | 15.23s | 125.8 |
| clf | `kick` | 25000 | 18 | 872 | 0.30064 | 0.29990 | 0.29909 | 0.30136 | -0.52% | 1.39s | 7.62s | 20.57s | 231.6 |
| clf | `Amazon_employee_access` | 25000 | 9 | 6520 | 0.18747 | 0.16122 | 0.15109 | 0.14838 | -19.40% | 0.82s | 5.28s | 14.59s | 119.8 |
| clf | `Click_prediction_small` | 25000 | 6 | 18858 | 0.42288 | 0.41417 | 0.41412 | 0.41363 | -2.07% | 0.57s | 3.05s | 7.56s | 34.0 |
| clf | `Bank_Customer_Churn` | 10000 | 4 | 3 | 0.33127 | 0.33127 | 0.33097 | 0.32783 | -0.09% | 0.31s | 0.26s | 0.94s | 9.6 |
| clf | `in_vehicle_coupon_recommendation` | 12684 | 22 | 25 | 0.51017 | 0.50853 | 0.51119 | 0.49728 | +0.20% | 1.75s | 11.82s | 19.87s | 575.2 |
| clf | `online_shoppers_intention` | 12330 | 6 | 20 | 0.22431 | 0.22366 | 0.22336 | 0.22596 | -0.42% | 0.69s | 0.95s | 2.63s | 57.4 |

## 4. Exploratory Binary Classification ROC-AUC (higher is better)

| dataset | S6 | S6_CTR1 | S6_CTR2 | CAT |
|---|---:|---:|---:|---:|
| `credit-g` | 0.77964 | 0.77848 | 0.76817 | 0.77698 |
| `bank-marketing` | 0.93371 | 0.93524 | 0.93664 | 0.93509 |
| `adult` | 0.92389 | 0.92485 | 0.92494 | 0.92777 |
| `kick` | 0.76629 | 0.76681 | 0.76748 | 0.76229 |
| `Amazon_employee_access` | 0.78139 | 0.84785 | 0.86524 | 0.87272 |
| `Click_prediction_small` | 0.67411 | 0.69384 | 0.69371 | 0.69684 |
| `Bank_Customer_Churn` | 0.86879 | 0.86879 | 0.86892 | 0.87153 |
| `in_vehicle_coupon_recommendation` | 0.82192 | 0.82331 | 0.82037 | 0.83134 |
| `online_shoppers_intention` | 0.93340 | 0.93414 | 0.93422 | 0.93208 |
