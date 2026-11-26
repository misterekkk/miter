# Confirmatory Summary: CatBoost-style Categorical CTRs (`exp-cat-ctr`)

Pre-registered in `benchmarks/results/exp_cat_ctr_prereg.md` (16 datasets x 5 folds = 80 paired runs/variant).

## 1. Anti-Fluke Confirmatory Scorecard (Holm-Bonferroni m=3, 10k bootstrap CI)

| Comparison | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | p_Holm | Worst % | Time vs B | Verdict |
|---|---:|---|---:|---:|---:|---:|---:|---|
| CTR vs B | +1.16% | [+0.40%, +2.09%] | 4/16 | 0.05768 | 0.1154 | +9.61% | 1.82x (1.82x vs base) | FAIL |
| CTR_S6 vs B | -0.00% | [-1.09%, +1.23%] | 8/16 | 0.9399 | 0.9399 | +10.41% | 2.59x (2.59x vs base) | FAIL |
| CTR vs TS1 | +1.02% | [+0.42%, +2.15%] | 1/16 | 0.001678 | 0.005035 | +9.44% | 1.82x (1.34x vs base) | FAIL |

## 2. Reference Comparison vs CatBoost (Exploratory)

| Comparison | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |
|---|---:|---|---:|---:|---:|
| B vs CAT | +0.27% | [-0.62%, +1.63%] | 8/16 | 0.4037 | +24.28% |
| TS1 vs CAT | +0.60% | [-0.53%, +1.98%] | 6/16 | 0.4954 | +8.12% |
| CTR vs CAT | +1.14% | [+0.62%, +3.41%] | 2/16 | 0.007629 | +5.97% |
| CTR_S6 vs CAT | +0.71% | [-0.06%, +1.72%] | 4/16 | 0.1297 | +4.44% |

## 3. Per-dataset Results (5-fold mean; lower is better)

| kind | dataset | n | p_cat | B | TS1 | CTR | CTR_S6 | CAT | CTR vs B % | B time | CTR time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reg | `abalone` | 4177 | 1 | 0.65920 | 0.65991 | 0.66182 | 0.65983 | 0.67465 | +0.40% | 0.07s | 0.07s |
| reg | `diamonds` | 25000 | 3 | 0.14120 | 0.13979 | 0.14067 | 0.13829 | 0.13893 | -0.37% | 0.68s | 0.69s |
| reg | `Mercedes_Benz` | 4209 | 8 | 0.66326 | 0.66592 | 0.67218 | 0.66028 | 0.65301 | +1.34% | 1.17s | 1.29s |
| reg | `avocado-sales` | 18249 | 3 | 0.33942 | 0.33996 | 0.37204 | 0.37477 | 0.36899 | +9.61% | 3.68s | 2.66s |
| reg | `nyc-taxi-green-dec-2016` | 25000 | 9 | 0.74708 | 0.73741 | 0.74299 | 0.71601 | 0.75449 | -0.55% | 0.45s | 1.07s |
| reg | `colleges` | 7063 | 12 | 0.62183 | 0.62385 | 0.63076 | 0.62967 | 0.62933 | +1.44% | 1.58s | 1.96s |
| reg | `Food_Delivery_Time` | 25000 | 3 | 0.79876 | 0.80397 | 0.80735 | 0.80753 | 0.79910 | +1.08% | 0.49s | 0.64s |
| clf | `credit-g` | 1000 | 13 | 0.51308 | 0.50960 | 0.52501 | 0.51937 | 0.50658 | +2.33% | 0.02s | 0.09s |
| clf | `bank-marketing` | 25000 | 9 | 0.19572 | 0.19561 | 0.19981 | 0.19717 | 0.19644 | +2.09% | 0.64s | 1.47s |
| clf | `adult` | 25000 | 8 | 0.28489 | 0.28500 | 0.29285 | 0.29101 | 0.27930 | +2.79% | 0.64s | 1.30s |
| clf | `kick` | 25000 | 18 | 0.30246 | 0.30328 | 0.30372 | 0.30178 | 0.30071 | +0.42% | 0.82s | 1.56s |
| clf | `Amazon_employee_access` | 25000 | 9 | 0.18448 | 0.16049 | 0.15730 | 0.15503 | 0.14844 | -14.73% | 0.38s | 0.85s |
| clf | `Click_prediction_small` | 25000 | 6 | 0.42436 | 0.41545 | 0.41591 | 0.41404 | 0.41368 | -1.99% | 0.58s | 1.08s |
| clf | `Bank_Customer_Churn` | 10000 | 4 | 0.33677 | 0.33453 | 0.33862 | 0.33311 | 0.32747 | +0.55% | 0.16s | 0.28s |
| clf | `in_vehicle_coupon_recommendation` | 12684 | 22 | 0.49793 | 0.50868 | 0.52112 | 0.51975 | 0.49831 | +4.66% | 1.12s | 3.14s |
| clf | `online_shoppers_intention` | 12330 | 6 | 0.22475 | 0.22496 | 0.22755 | 0.22452 | 0.22615 | +1.25% | 0.38s | 0.59s |

## 4. Exploratory Binary Classification ROC-AUC (higher is better)

| dataset | B | TS1 | CTR | CTR_S6 | CAT |
|---|---:|---:|---:|---:|---:|
| `credit-g` | 0.77581 | 0.77762 | 0.75240 | 0.76869 | 0.77840 |
| `bank-marketing` | 0.93545 | 0.93524 | 0.93225 | 0.93425 | 0.93543 |
| `adult` | 0.92465 | 0.92459 | 0.92090 | 0.92148 | 0.92770 |
| `kick` | 0.76130 | 0.75954 | 0.75906 | 0.76396 | 0.76394 |
| `Amazon_employee_access` | 0.79141 | 0.84833 | 0.84799 | 0.85476 | 0.87188 |
| `Click_prediction_small` | 0.67015 | 0.69022 | 0.69157 | 0.69611 | 0.69681 |
| `Bank_Customer_Churn` | 0.86404 | 0.86543 | 0.86239 | 0.86800 | 0.87161 |
| `in_vehicle_coupon_recommendation` | 0.83196 | 0.82323 | 0.81231 | 0.81245 | 0.83071 |
| `online_shoppers_intention` | 0.93336 | 0.93341 | 0.93236 | 0.93326 | 0.93212 |
