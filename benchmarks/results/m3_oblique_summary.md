# M3 Oblique Booster Benchmark Summary

- Hardware: `arm64` (`macOS-26.6.2-arm64-arm-64bit-Mach-O`)
- Python: `3.13.1`
- Versions: `miter 0.1.0.dev0`, `lightgbm 4.7.0`, `numpy 2.5.5`
- Protocol: single-threaded (`n_jobs=1` / `num_threads=1`). Negative relative difference = lower error = better.
- Provenance note: Parts A, B, and the first 10 datasets of Part C come from the initial `bench_m3_oblique.py` run; the last 3 datasets of Part C (`jannis`, `Diabetes130US`, `MiniBooNE`) were run immediately afterwards on the same build/environment after fixing the `jannis` dataset ID typo (`44121 -> 45021`). Summary relative differences and Wilcoxon p-values in Parts B and C are computed from the 4-decimal table values.

## Part A: Single-thread training speed (100 trees, median of 5 runs)

| shape | LightGBM ms/tree | miter AX | miter default (2m0) | miter/LGBM | miter fast | miter acc (4m0) | histree ObliqueGBInt |
|---|---|---|---|---|---|---|---|
| 20000 x 16 | 1.536 | 1.121 | 3.527 | 2.30x | 1.785 | 5.970 | 6.024 |
| 20000 x 50 | 4.596 | 3.150 | 7.825 | 1.70x | 4.449 | 12.437 | 14.692 |
| 20000 x 100 | 8.881 | 6.181 | 12.391 | 1.40x | 7.798 | 17.829 | 24.123 |
| 50000 x 50 | 8.424 | 6.067 | 17.726 | 2.10x | 9.217 | 28.619 | 34.816 |

## Part B: 16 OpenML regression datasets x 5 folds (1 thread, ES=50, max_trees=5000)

| dataset | LightGBM nRMSE | miter R_uni nRMSE | histree T7 Q16r_int | miter vs LGBM | miter vs histree T7 | ms/tree (L/M) |
|---|---|---|---|---|---|---|
| abalone | 0.6768 | 0.6632 | 0.6640 | -2.01% | -0.12% | 0.49 / 0.58 |
| wine_quality | 0.7287 | 0.7001 | 0.7058 | -3.92% | -0.80% | 0.35 / 0.89 |
| cpu_act | 0.1213 | 0.1142 | 0.1175 | -5.88% | -2.87% | 0.75 / 1.79 |
| yprop_4_1 | 0.9587 | 0.9568 | 0.9559 | -0.21% | +0.09% | 1.35 / 2.82 |
| sulfur | 0.4454 | 0.3682 | 0.3808 | -17.33% | -3.31% | 0.64 / 1.20 |
| Brazilian_houses | 0.0869 | 0.0505 | 0.0447 | -41.86% | +13.09% | 0.60 / 1.96 |
| Ailerons | 0.3927 | 0.3876 | 0.3880 | -1.30% | -0.09% | 0.81 / 2.85 |
| MiamiHousing2016 | 0.2656 | 0.2596 | 0.2612 | -2.25% | -0.61% | 0.63 / 1.66 |
| pol | 0.1107 | 0.0997 | 0.0965 | -9.90% | +3.37% | 0.64 / 3.28 |
| elevators | 0.3065 | 0.2901 | 0.2903 | -5.37% | -0.08% | 0.60 / 2.01 |
| Bike_Sharing_Demand | 0.5464 | 0.5394 | 0.5388 | -1.29% | +0.10% | 0.44 / 1.36 |
| houses | 0.3836 | 0.3683 | 0.3680 | -4.00% | +0.09% | 0.57 / 1.75 |
| superconduct | 0.2693 | 0.2663 | 0.2666 | -1.13% | -0.14% | 3.07 / 6.24 |
| house_sales | 0.3315 | 0.3263 | 0.3277 | -1.56% | -0.41% | 0.89 / 2.72 |
| house_16H | 0.6813 | 0.6594 | 0.6702 | -3.21% | -1.61% | 1.73 / 4.35 |
| diamonds | 0.2291 | 0.2278 | 0.2275 | -0.57% | +0.11% | 1.18 / 4.38 |

- **miter vs LightGBM (16 reg):** mean -6.36%, **median -2.74%**, wins 16/16, Wilcoxon p = 3.052e-05
- **miter vs histree T7 Q16r_int (16 reg):** mean +0.42%, **median -0.11%**, wins 10/16, Wilcoxon p = 0.2242

## Part C: 13 OpenML binary classification datasets (fold 0, 1 thread, ES=50, max_trees=5000)

| dataset | LightGBM log-loss | miter R_uni log-loss | rel diff | ms/tree (L/M) |
|---|---|---|---|---|
| eye_movements | 0.6189 | 0.6065 | -2.00% | 0.79 / 1.55 |
| heloc | 0.5633 | 0.5596 | -0.65% | 1.21 / 1.98 |
| pol | 0.0409 | 0.0372 | -8.95% | 0.70 / 2.95 |
| bank-marketing | 0.4120 | 0.4081 | -0.96% | 0.71 / 1.38 |
| default-of-credit-card-clients | 0.5580 | 0.5559 | -0.38% | 1.71 / 2.58 |
| MagicTelescope | 0.3128 | 0.2961 | -5.35% | 0.77 / 1.97 |
| house_16H | 0.2841 | 0.2786 | -1.93% | 1.20 / 2.16 |
| credit | 0.4760 | 0.4777 | +0.36% | 1.17 / 2.43 |
| california | 0.2043 | 0.1976 | -3.26% | 0.77 / 2.09 |
| electricity | 0.2490 | 0.2289 | -8.07% | 0.81 / 3.04 |
| jannis | 0.4433 | 0.4358 | -1.70% | 5.97 / 11.63 |
| Diabetes130US | 0.6602 | 0.6610 | +0.12% | 3.07 / 7.69 |
| MiniBooNE | 0.1544 | 0.1521 | -1.51% | 7.51 / 14.97 |

- **miter vs LightGBM (13 clf, fold 0):** mean -2.64%, **median -1.69%**, wins 11/13, Wilcoxon p = 0.001221
