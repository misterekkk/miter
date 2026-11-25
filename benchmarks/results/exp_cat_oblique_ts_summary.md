# Experimental Report: Categoricals in Oblique Splits via OOF Target Statistics (`exp-cat-oblique-ts`)

- Pre-registration: [`exp_cat_oblique_ts_prereg.md`](exp_cat_oblique_ts_prereg.md)
- Pre-registration SHA-256: `6594bf6736280041468cb59b958e1fc587e7bec8d8c7e4582fe071cacf2a9be0`
- Rows collected: `300 / 300` (12 datasets × 5 folds × 5 variants, `n_jobs=1`)

## 1. Pre-registered Confirmatory Verdict (`TS` vs `B`, 12 datasets)

| Comparison | Scope | median % vs B | mean % vs B | wins | Wilcoxon p | worst dataset % | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **TS vs B** | All (12) | -0.04% | -1.38% | 7/12 | 0.5186 | +1.22% | **FAIL** |

## 2. Exploratory Subgroup Breakdowns & Reference Comparisons

| Variant / Slice | Scope | median % | mean % | wins | Wilcoxon p | worst % |
|---|---|---:|---:|---:|---:|---:|
| TS vs B | High-card (`max_card > 253`) (5) | -1.82% | -3.06% | 3/5 | 0.3125 | +0.40% |
| TS vs B | Med/High-card (`max_card > 50`) (6) | -0.72% | -2.49% | 3/6 | 0.4375 | +0.40% |
| TS vs B | Low/Med-card (`max_card <= 50`) (6) | -0.04% | -0.28% | 4/6 | 0.8438 | +1.22% |
| TS vs B | Regression (6) (6) | +0.15% | -0.69% | 3/6 | 0.8438 | +0.38% |
| TS vs B | Classification (6) (6) | -0.09% | -2.07% | 4/6 | 0.5625 | +1.22% |
| TS vs CAT | All (12) | +0.05% | +0.49% | 6/12 | 0.7334 | +10.09% |
| TS_S6 vs CAT | All (12) | -0.05% | +0.01% | 6/12 | 0.5693 | +9.84% |
| TS vs LGB | All (12) | -0.60% | -0.80% | 8/12 | 0.2334 | +1.76% |
| TS_S6 vs LGB | All (12) | -0.79% | -1.26% | 9/12 | 0.05225 | +1.45% |
| TS_S6 vs B | All (12) | -0.91% | -1.84% | 9/12 | 0.04248 | +1.61% |

## 3. Per-dataset Results (5-fold mean; lower metric is better)

| kind | dataset | n | p_cat | max_card | B | TS | TS vs B % | TS_S6 | CAT | LGB | B time | TS time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reg | `abalone` | 4177 | 1 | 3 (3) | 0.66267 | 0.66258 | -0.01% | 0.65746 | 0.67372 | 0.67592 | 0.07s | 0.07s |
| reg | `diamonds` | 25000 | 3 | 8 (8) | 0.14181 | 0.13758 | -2.99% | 0.13628 | 0.13819 | 0.14303 | 0.75s | 0.89s |
| reg | `Mercedes_Benz` | 4209 | 8 | 47 (47) | 0.66277 | 0.66488 | +0.32% | 0.65405 | 0.65238 | 0.66217 | 1.24s | 1.20s |
| reg | `avocado-sales` | 18249 | 3 | 54 (54) | 0.33434 | 0.33562 | +0.38% | 0.33972 | 0.35668 | 0.33822 | 3.88s | 3.80s |
| reg | `nyc-taxi-green-dec-2016` | 25000 | 9 | 259 (240) | 0.74128 | 0.72475 | -2.23% | 0.72147 | 0.74655 | 0.72874 | 0.97s | 1.68s |
| reg | `colleges` | 7063 | 12 | 6039 (6039) | 0.62396 | 0.62629 | +0.37% | 0.62663 | 0.63506 | 0.63036 | 1.33s | 2.14s |
| clf | `credit-g` | 1000 | 13 | 10 (10) | 0.50968 | 0.51590 | +1.22% | 0.50442 | 0.49897 | 0.50697 | 0.03s | 0.05s |
| clf | `bank-marketing` | 25000 | 9 | 12 (12) | 0.19517 | 0.19492 | -0.12% | 0.19530 | 0.19592 | 0.19917 | 0.65s | 0.99s |
| clf | `adult` | 25000 | 8 | 41 (41) | 0.28418 | 0.28400 | -0.06% | 0.28391 | 0.27844 | 0.27985 | 0.78s | 0.98s |
| clf | `kick` | 25000 | 18 | 1063 (872) | 0.30212 | 0.30333 | +0.40% | 0.30178 | 0.30113 | 0.30347 | 0.83s | 0.95s |
| clf | `Amazon_employee_access` | 25000 | 9 | 7518 (6520) | 0.18310 | 0.16109 | -12.02% | 0.16072 | 0.14632 | 0.16829 | 0.44s | 0.50s |
| clf | `Click_prediction_small` | 25000 | 6 | 30114 (18858) | 0.42295 | 0.41527 | -1.82% | 0.41408 | 0.41306 | 0.41124 | 0.41s | 0.65s |

## 4. Exploratory Classification ROC-AUC (5-fold mean, higher is better)

| dataset | B | TS | TS_S6 | CAT | LGB |
|---|---:|---:|---:|---:|---:|
| `credit-g` | 0.77179 | 0.77610 | 0.77921 | 0.78643 | 0.77638 |
| `bank-marketing` | 0.93596 | 0.93584 | 0.93591 | 0.93604 | 0.93338 |
| `adult` | 0.92490 | 0.92520 | 0.92523 | 0.92816 | 0.92701 |
| `kick` | 0.76198 | 0.75914 | 0.76280 | 0.76316 | 0.75872 |
| `Amazon_employee_access` | 0.79610 | 0.84941 | 0.85168 | 0.87730 | 0.82449 |
| `Click_prediction_small` | 0.67368 | 0.69122 | 0.69469 | 0.69798 | 0.70730 |
