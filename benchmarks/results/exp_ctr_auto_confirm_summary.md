# Confirmatory Summary: Adaptive Native CTR (`exp-ctr-auto-confirm`)

- Pre-registration: `benchmarks/results/exp_ctr_auto_confirm_prereg.md` (SHA-256: `6fe4239438c0f6d802548b8b25f83489f3119c7a2a8c6cc79769518d5b2cd48a`)
- Scope: 16 completely unseen OpenML datasets × 5 folds = 80 paired runs/variant (`n_jobs=1`).

## 1. Anti-Fluke Confirmatory Scorecard (Holm-Bonferroni `m=3`, 10k bootstrap CI)

| Comparison | Mean % | Median % | 95% Boot CI (Median) | Wins | Wilcoxon p | p_Holm | Worst % | Time vs S6 | Verdict |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| Primary 1: S6_AUTO_P4 vs S6 | -0.39% | -0.22% | [-0.74%, +0.12%] | 11/16 | 0.2078 | 0.2411 | +8.81% | 1.34x | FAIL |
| Primary 2: S6_NC_SMART_P4 vs S6 | -0.44% | -0.37% | [-0.72%, +0.01%] | 10/16 | 0.1205 | 0.2411 | +8.81% | 1.45x | FAIL |
| Head-to-head: S6_NC_SMART_P4 vs S6_AUTO_P4 | -0.05% | +0.00% | [-0.05%, +0.00%] | 6/16 | 0.07812 | 0.2344 | +0.05% | 1.45x | FAIL (PARSIMONY -> S6_AUTO_P4) |

## 2. Exploratory Comparisons (Primary metric unless noted)

| Comparison | N | Mean % | Median % | 95% Boot CI | Wins | Wilcoxon p | Worst % |
|---|---:|---:|---:|---|---:|---:|---:|
| S6 vs CAT | 16 | -1.20% | -0.06% | [-2.20%, +0.47%] | 8/16 | 0.5966 | +8.17% |
| S6_AUTO_P4 vs CAT | 16 | -1.72% | -0.30% | [-2.54%, +0.15%] | 10/16 | 0.08325 | +1.50% |
| S6_NC_SMART_P4 vs CAT | 16 | -1.77% | -0.28% | [-2.54%, +0.15%] | 10/16 | 0.07391 | +1.50% |
| S6_AUTO_P4 vs S6 (1-AUC) | 12 | -2.23% | -0.78% | [-1.71%, +0.08%] | 9/12 | 0.1099 | +1.84% |
| S6_NC_SMART_P4 vs S6 (1-AUC) | 12 | -2.43% | -0.93% | [-1.75%, -0.02%] | 9/12 | 0.09229 | +1.79% |
| S6_AUTO_P4 vs CAT (1-AUC) | 12 | -0.61% | -0.27% | [-1.85%, +2.07%] | 7/12 | 0.8501 | +4.87% |
| S6_NC_SMART_P4 vs CAT (1-AUC) | 12 | -0.83% | -0.27% | [-1.88%, +1.66%] | 7/12 | 0.7334 | +3.40% |
| S6_AUTO_P4 vs S6 (max_card>10) | 9 | -0.28% | -0.35% | [-1.09%, +0.45%] | 6/9 | 0.4258 | +0.81% |
| S6_NC_SMART_P4 vs S6 (max_card>10) | 9 | -0.37% | -0.40% | [-1.04%, +0.33%] | 6/9 | 0.1641 | +0.81% |

## 3. Per-dataset Primary Metric, Wall Time & CTR Tables (5-fold mean; lower is better)

| kind | dataset | n | p_cat | max_card | S6 | S6_AUTO_P4 | SMART_P4 | CAT | AUTO_P4 vs S6 | SMART vs S6 | S6 t | AUTO_P4 t | SMART t | CAT t | tbl:AUTO_P4 | tbl:SMART |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clf | `Diabetes130US` | 25000 | 39 | 601 | 0.28646 | 0.28532 | 0.28532 | 0.28490 | -0.40% | -0.40% | 0.73s | 15.20s | 15.62s | 21.28s | 250.6 | 250.6 |
| clf | `E-CommereShippingData` | 10999 | 4 | 5 | 0.50389 | 0.50378 | 0.50395 | 0.50510 | -0.02% | +0.01% | 0.17s | 0.28s | 0.33s | 0.81s | 17.2 | 21.8 |
| clf | `Fitness_Club` | 1500 | 3 | 7 | 0.47757 | 0.47551 | 0.47551 | 0.47635 | -0.43% | -0.43% | 0.04s | 0.05s | 0.05s | 0.21s | 8.0 | 8.0 |
| clf | `HR_Analytics_Job_Change` | 19158 | 10 | 123 | 0.43619 | 0.43454 | 0.43423 | 0.43416 | -0.38% | -0.45% | 0.23s | 2.13s | 1.95s | 3.85s | 214.0 | 198.2 |
| clf | `Is-this-a-good-customer` | 1723 | 8 | 22 | 0.32371 | 0.32464 | 0.32464 | 0.32315 | +0.29% | +0.29% | 0.04s | 0.05s | 0.05s | 0.23s | 11.2 | 11.2 |
| clf | `Marketing_Campaign` | 2240 | 9 | 663 | 0.27164 | 0.27069 | 0.27069 | 0.27776 | -0.35% | -0.35% | 0.23s | 0.30s | 0.30s | 0.66s | 12.0 | 12.0 |
| clf | `churn` | 5000 | 4 | 51 | 0.14774 | 0.14767 | 0.14710 | 0.14973 | -0.05% | -0.43% | 0.38s | 0.51s | 0.54s | 1.29s | 21.4 | 24.8 |
| clf | `coil2000` | 9822 | 5 | 39 | 0.20173 | 0.20262 | 0.20239 | 0.20150 | +0.45% | +0.33% | 0.74s | 0.93s | 1.01s | 1.79s | 31.4 | 33.6 |
| clf | `credit_card_clients_default` | 25000 | 3 | 7 | 0.42214 | 0.42174 | 0.42173 | 0.42317 | -0.09% | -0.10% | 0.81s | 1.06s | 1.18s | 3.45s | 11.6 | 12.2 |
| clf | `customer_satisfaction_in_airline` | 25000 | 16 | 6 | 0.12574 | 0.11435 | 0.11428 | 0.11624 | -9.06% | -9.11% | 3.09s | 19.17s | 21.77s | 29.66s | 891.6 | 894.6 |
| clf | `kddcup09_appetency` | 25000 | 38 | 9531 | 0.07828 | 0.07685 | 0.07662 | 0.07602 | -1.83% | -2.12% | 2.63s | 10.28s | 14.56s | 40.16s | 94.2 | 107.4 |
| clf | `qsar-biodeg` | 1054 | 5 | 11 | 0.30282 | 0.30528 | 0.30528 | 0.32061 | +0.81% | +0.81% | 0.22s | 0.21s | 0.22s | 0.57s | 3.0 | 3.0 |
| mc | `anneal` | 898 | 32 | 8 | 0.03835 | 0.04173 | 0.04173 | 0.04577 | +8.81% | +8.81% | 1.15s | 0.95s | 0.93s | 47.39s | 40.6 | 40.6 |
| mc | `splice` | 3190 | 60 | 6 | 0.11167 | 0.10847 | 0.10847 | 0.11783 | -2.87% | -2.87% | 1.23s | 11.84s | 11.76s | 142.89s | 1289.2 | 1289.2 |
| mc | `students_dropout` | 4424 | 17 | 46 | 0.54199 | 0.53606 | 0.53633 | 0.55517 | -1.09% | -1.04% | 1.15s | 3.71s | 3.71s | 13.47s | 500.6 | 486.2 |
| reg | `healthcare_insurance_expenses` | 1338 | 3 | 4 | 0.38501 | 0.38501 | 0.38501 | 0.37931 | +0.00% | +0.00% | 0.03s | 0.03s | 0.03s | 0.09s | 0.0 | 0.0 |

## 4. Exploratory Binary Classification ROC-AUC (higher is better)

| dataset | S6 | S6_AUTO_P4 | SMART_P4 | CAT |
|---|---:|---:|---:|---:|
| `Diabetes130US` | 0.64553 | 0.65150 | 0.65150 | 0.65007 |
| `E-CommereShippingData` | 0.73977 | 0.73903 | 0.73894 | 0.74673 |
| `Fitness_Club` | 0.80948 | 0.81133 | 0.81133 | 0.81107 |
| `HR_Analytics_Job_Change` | 0.79964 | 0.80267 | 0.80321 | 0.80445 |
| `Is-this-a-good-customer` | 0.72714 | 0.73185 | 0.73185 | 0.72949 |
| `Marketing_Campaign` | 0.89227 | 0.89289 | 0.89289 | 0.88538 |
| `churn` | 0.92076 | 0.92086 | 0.92146 | 0.92251 |
| `coil2000` | 0.75802 | 0.75358 | 0.75370 | 0.75847 |
| `credit_card_clients_default` | 0.79053 | 0.79128 | 0.79129 | 0.78808 |
| `customer_satisfaction_in_airline` | 0.99024 | 0.99182 | 0.99183 | 0.99164 |
| `kddcup09_appetency` | 0.80577 | 0.81887 | 0.82139 | 0.82727 |
| `qsar-biodeg` | 0.93579 | 0.93509 | 0.93509 | 0.92896 |
