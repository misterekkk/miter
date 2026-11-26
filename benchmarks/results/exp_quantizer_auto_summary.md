# Confirmatory Summary: Adaptive Quantizer Mode (`exp-quantizer-auto`)

- Pre-registration: `benchmarks/results/exp_quantizer_auto_prereg.md` (SHA-256: `8033d6f57c65a2bcbcd37349ab6a682b1558e1b921ed5a98f546ec54ac436264`)
- Scope: 16 fresh confirmatory datasets + 2 diagnostic stress-test datasets × 5 folds (`n_jobs=1`).

## 1. Confirmatory Scorecard (16 Fresh Datasets; Holm-Bonferroni `m=2`, 10k bootstrap CI)

| Comparison | Mean % | Median % | 95% Boot CI (Median) | Wins / Ties | Worst % | Time vs Lin | Trig Subgroup (nq>=1) | Stress >3%? | Verdict |
|---|---:|---:|---|---|---:|---:|---|---|---|
| AUTO vs LIN | +0.18% | +0.00% | [+0.00%, +0.07%] | 2/16 (8t) | +3.54% | 0.99x | 1/6 (6 ds, med +0.10%, p=0.312) | YES (p_H=0.547) | FAIL |
| S6_AUTO vs S6_LIN | -0.16% | +0.00% | [-0.15%, +0.00%] | 6/16 (8t) | +0.12% | 0.99x | 5/6 (6 ds, med -0.19%, p=0.0625) | YES (p_H=0.109) | FAIL |

## 2. Diagnostic Stress-Test Verification (5-fold mean)

| dataset | LIN | AUTO | QUANT | S6_LIN | S6_AUTO | AUTO vs LIN | S6_AUTO vs S6_LIN | nq(AUTO) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `taiwanese_bankruptcy_prediction` (clf) | 0.09265 | 0.08938 | 0.08717 | 0.08903 | 0.08295 | -3.53% | -6.83% | 46.0/94 |
| `taiwanese_bankruptcy_prediction` (1-AUC) | 0.06865 | 0.05599 | 0.05440 | 0.06626 | 0.05579 | -18.44% | -15.80% | - |
| `SDSS17` (mc) | 0.08709 | 0.07928 | 0.07938 | 0.08613 | 0.07844 | -8.96% | -8.93% | 1.0/9 |

## 3. Exploratory Leafwise Quantizer Comparisons (`N=16`)

- `QUANT vs LIN`: mean +0.62%, median +0.14% [-0.33%, +1.17%], wins 5/16, worst +4.28%, p=0.1928
- `AUTO vs QUANT`: mean -0.43%, median -0.07% [-0.69%, +0.18%], wins 10/16, worst +0.93%, p=0.2744

## 4. Per-dataset Primary Metric & Triggered Quantile Features (5-fold mean)

| kind | dataset | n | p_num | nq | LIN | AUTO | QUANT | S6_LIN | S6_AUTO | AUTO vs LIN | S6_AUTO vs S6_LIN | LIN t | AUTO t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reg | `houses` | 20640 | 8 | 0.0 | 0.37164 | 0.37164 | 0.37600 | 0.37000 | 0.37000 | +0.00% | +0.00% | 1.47s | 1.45s |
| reg | `miami_housing` | 13776 | 14 | 0.0 | 0.26987 | 0.26987 | 0.27058 | 0.26181 | 0.26181 | +0.00% | +0.00% | 1.06s | 1.06s |
| reg | `physiochemical_protein` | 25000 | 9 | 0.0 | 0.59202 | 0.59202 | 0.58983 | 0.59791 | 0.59791 | +0.00% | +0.00% | 2.87s | 2.85s |
| reg | `superconductivity` | 21263 | 81 | 4.6 | 0.27306 | 0.27322 | 0.27306 | 0.27409 | 0.27290 | +0.06% | -0.43% | 7.70s | 8.18s |
| reg | `wine_quality` | 6497 | 11 | 0.0 | 0.70254 | 0.70254 | 0.70549 | 0.70688 | 0.70688 | +0.00% | +0.00% | 0.65s | 0.65s |
| reg | `airfoil_self_noise` | 1503 | 4 | 0.0 | 0.22137 | 0.22137 | 0.23085 | 0.20772 | 0.20772 | +0.00% | +0.00% | 0.49s | 0.49s |
| reg | `concrete_compressive_strength` | 1030 | 8 | 0.2 | 0.25884 | 0.25693 | 0.25692 | 0.25135 | 0.25074 | -0.74% | -0.25% | 0.13s | 0.13s |
| reg | `QSAR_fish_toxicity` | 907 | 6 | 0.0 | 0.61817 | 0.61817 | 0.62770 | 0.60954 | 0.60954 | +0.00% | +0.00% | 0.02s | 0.02s |
| clf | `APSFailure` | 25000 | 170 | 122.2 | 0.02139 | 0.02214 | 0.02194 | 0.02124 | 0.02123 | +3.54% | -0.02% | 2.88s | 2.63s |
| clf | `GiveMeSomeCredit` | 25000 | 10 | 2.0 | 0.17745 | 0.17601 | 0.17634 | 0.17647 | 0.17606 | -0.81% | -0.23% | 0.45s | 0.46s |
| clf | `polish_companies_bankruptcy` | 5910 | 64 | 57.6 | 0.11253 | 0.11305 | 0.11461 | 0.10527 | 0.10351 | +0.47% | -1.67% | 0.51s | 0.52s |
| clf | `heloc` | 10459 | 23 | 0.0 | 0.54923 | 0.54923 | 0.54989 | 0.54596 | 0.54596 | +0.00% | +0.00% | 0.21s | 0.21s |
| clf | `jm1` | 10885 | 21 | 4.4 | 0.42848 | 0.42851 | 0.42669 | 0.43082 | 0.43018 | +0.01% | -0.15% | 0.36s | 0.30s |
| clf | `qsar-biodeg` | 1054 | 36 | 0.2 | 0.30786 | 0.30843 | 0.30684 | 0.30282 | 0.30320 | +0.19% | +0.12% | 0.07s | 0.06s |
| clf | `credit_card_clients_default` | 25000 | 20 | 7.2 | 0.42294 | 0.42354 | 0.42363 | 0.42214 | 0.42222 | +0.14% | +0.02% | 0.55s | 0.47s |
| clf | `coil2000` | 9822 | 80 | 0.0 | 0.20525 | 0.20525 | 0.20526 | 0.20173 | 0.20173 | +0.00% | +0.00% | 0.50s | 0.50s |
