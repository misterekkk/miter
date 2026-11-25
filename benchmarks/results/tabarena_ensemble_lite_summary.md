# TabArena `lite` Ensemble Evaluation: `MITER_BS6` (2 configs) & `MITER_T12` (12 configs)

- Protocol: Official `TabArenaV0pt1ExperimentBundle` (`split_indices='lite'`, `subset='lite'`, fold 0, 8-fold bagging, AutoGluon preprocessing, `time_limit=3600s`).
- Exploratory comparison (`subset='lite'` fold 0, 2 or 12 miter configs vs up to 200 configs for cached CatBoost/GBM/XGB).
- Scopes: `MITER_BS6` across **all 20 tasks** (13 reg + 7 binary, `0.00` imputed) & **13 reg**; `MITER_BS6 + MITER_T12` across **17 fast/medium tasks** (`0.00` imputed).

## BS6 on 13 Regression Tasks — full_pool (`117` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TA-LIMIX-2 (default) | 2696.4 | 3.38 | 0.979 | 0.0029 | 0.00 |
| 2 | TA-KUMO-TABULAR (default) | 2672.2 | 3.69 | 0.977 | 0.0113 | 0.00 |
| 3 | Chakra-Tab (full) | 2536.9 | 5.88 | 0.958 | 0.0233 | 0.00 |
| 4 | TA-KUMO-TABULAR-MEDIUM (default) | 2531.0 | 6.00 | 0.957 | 0.0233 | 0.00 |
| 5 | Chakra-Tab (medium) | 2450.4 | 7.73 | 0.942 | 0.0256 | 0.00 |
| 6 | TA-SELDON (default) | 2396.1 | 9.08 | 0.930 | 0.0323 | 0.00 |
| 7 | TA-SELDON-MINI (default) | 2332.0 | 10.85 | 0.915 | 0.0368 | 0.00 |
| 8 | TabFM+ | 2318.9 | 11.23 | 0.912 | 0.0360 | 0.00 |
| 9 | TA-TABPFN-3.5 (default) | 2313.8 | 11.38 | 0.910 | 0.0422 | 0.00 |
| 10 | INPRIORA (default) | 2291.1 | 12.08 | 0.905 | 0.0324 | 0.00 |
| 18 | TA-TABPFN-3.5-FAST (default) | 2051.6 | 20.85 | 0.829 | 0.0583 | 0.00 |
| 22 | TA-TABPFN-3 (default) | 1932.4 | 26.38 | 0.781 | 0.0575 | 0.00 |
| 24 | REALTABPFN-V2.5 (tuned + ensemble) | 1898.4 | 28.15 | 0.766 | 0.0623 | 0.00 |
| 26 | TABPFN-V2.6 (default) | 1884.1 | 28.92 | 0.759 | 0.0681 | 0.00 |
| 29 | REALTABPFN-V2.5 (tuned) | 1771.6 | 35.62 | 0.702 | 0.0710 | 0.00 |
| 33 | TABM-A (tuned + ensemble) | 1730.0 | 38.38 | 0.678 | 0.0843 | 0.00 |
| 34 | REALMLP_GPU (tuned + ensemble) | 1705.7 | 40.08 | 0.663 | 0.0897 | 0.00 |
| 37 | REALTABPFN-V2.5 (default) | 1669.8 | 42.69 | 0.641 | 0.0845 | 0.00 |
| 42 | TABM-A (tuned) | 1593.7 | 48.62 | 0.590 | 0.0955 | 0.00 |
| 43 | REALMLP_GPU (tuned) | 1583.4 | 49.46 | 0.582 | 0.1005 | 0.00 |
| 44 | GBM (tuned + ensemble) | 1568.5 | 50.69 | 0.572 | 0.1178 | 0.00 |
| 46 | CAT (tuned + ensemble) | 1533.9 | 53.62 | 0.546 | 0.1161 | 0.00 |
| 48 | CAT (tuned) | 1529.4 | 54.00 | 0.543 | 0.1164 | 0.00 |
| 49 | TA-TABM (tuned + ensemble) | 1527.6 | 54.15 | 0.542 | 0.1043 | 0.00 |
| 51 | [New] MITER_BS6 (tuned + ensemble) | 1508.9 | 55.77 | 0.528 | 0.1199 | 0.00 |
| 53 | [New] MITER_BS6 (tuned) | 1501.9 | 56.38 | 0.523 | 0.1198 | 0.00 |
| 55 | TA-TABM (tuned) | 1482.7 | 58.08 | 0.508 | 0.1080 | 0.00 |
| 56 | GBM (tuned) | 1482.7 | 58.08 | 0.508 | 0.1252 | 0.00 |
| 59 | XGB (tuned + ensemble) | 1449.0 | 61.08 | 0.482 | 0.1248 | 0.00 |
| 60 | XGB (tuned) | 1444.7 | 61.46 | 0.479 | 0.1247 | 0.00 |
| 63 | [New] MITER_BS6 (default) | 1421.7 | 63.54 | 0.461 | 0.1278 | 0.00 |
| 64 | TABPFNV2_GPU (tuned + ensemble) | 1420.0 | 63.69 | 0.460 | 0.1106 | 0.46 |
| 65 | CAT (default) | 1418.3 | 63.85 | 0.458 | 0.1354 | 0.00 |
| 73 | TABM-A (default) | 1324.7 | 72.31 | 0.385 | 0.1285 | 0.00 |
| 74 | TABPFNV2_GPU (tuned) | 1316.1 | 73.08 | 0.379 | 0.1247 | 0.46 |
| 75 | TA-TABM (default) | 1304.9 | 74.08 | 0.370 | 0.1301 | 0.00 |
| 81 | TABPFNV2_GPU (default) | 1268.2 | 77.31 | 0.342 | 0.1358 | 0.46 |
| 82 | REALMLP_GPU (default) | 1265.5 | 77.54 | 0.340 | 0.1450 | 0.00 |
| 88 | GBM (default) | 1212.2 | 82.08 | 0.301 | 0.1517 | 0.00 |
| 92 | XGB (default) | 1180.2 | 84.69 | 0.279 | 0.1587 | 0.00 |

## BS6 on 13 Regression Tasks — gbdt_only (`12` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1246.7 | 3.62 | 0.762 | 0.0150 | 0.00 |
| 2 | CAT (tuned + ensemble) | 1215.0 | 4.00 | 0.727 | 0.0103 | 0.00 |
| 3 | CAT (tuned) | 1155.9 | 4.77 | 0.657 | 0.0107 | 0.00 |
| 4 | [New] MITER_BS6 (tuned + ensemble) | 1144.5 | 4.92 | 0.643 | 0.0155 | 0.00 |
| 5 | [New] MITER_BS6 (tuned) | 1127.7 | 5.15 | 0.622 | 0.0153 | 0.00 |
| 6 | GBM (tuned) | 1077.7 | 5.85 | 0.559 | 0.0240 | 0.00 |
| 7 | [New] MITER_BS6 (default) | 1033.1 | 6.46 | 0.503 | 0.0254 | 0.00 |
| 8 | XGB (tuned + ensemble) | 1004.6 | 6.85 | 0.469 | 0.0224 | 0.00 |
| 9 | XGB (tuned) | 1004.6 | 6.85 | 0.469 | 0.0221 | 0.00 |
| 10 | CAT (default) | 969.4 | 7.31 | 0.427 | 0.0367 | 0.00 |
| 11 | GBM (default) | 566.0 | 10.85 | 0.105 | 0.0547 | 0.00 |
| 12 | XGB (default) | 454.6 | 11.38 | 0.056 | 0.0620 | 0.00 |

## BS6 on All 20 Tasks (13 Reg + 7 Binary) — full_pool (`121` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2384.3 | 4.45 | 0.971 | 0.0215 | 0.00 |
| 2 | TA-LIMIX-2 (default) | 2365.0 | 4.80 | 0.968 | 0.0070 | 0.00 |
| 3 | TA-KUMO-TABULAR-MEDIUM (default) | 2204.2 | 8.70 | 0.936 | 0.0323 | 0.00 |
| 4 | Chakra-Tab (full) | 2187.9 | 9.20 | 0.932 | 0.0337 | 0.00 |
| 5 | TA-SELDON (default) | 2184.7 | 9.30 | 0.931 | 0.0364 | 0.00 |
| 6 | Chakra-Tab (medium) | 2155.6 | 10.25 | 0.923 | 0.0352 | 0.00 |
| 7 | TabFM+ | 2151.1 | 10.40 | 0.922 | 0.0411 | 0.00 |
| 8 | TA-TABPFN-3.5 (default) | 2136.7 | 10.90 | 0.917 | 0.0427 | 0.00 |
| 9 | AutoGluon 1.6 (noncommercial, 4h) | 2074.0 | 13.25 | 0.898 | 0.0392 | 0.00 |
| 10 | TA-SELDON-MINI (default) | 2050.1 | 14.22 | 0.890 | 0.0460 | 0.00 |
| 18 | TA-TABPFN-3.5-FAST (default) | 1931.1 | 19.75 | 0.844 | 0.0583 | 0.00 |
| 22 | TA-TABPFN-3 (default) | 1804.1 | 27.05 | 0.783 | 0.0648 | 0.00 |
| 23 | REALTABPFN-V2.5 (tuned + ensemble) | 1762.1 | 29.85 | 0.760 | 0.0703 | 0.00 |
| 24 | TABPFN-V2.6 (default) | 1721.0 | 32.80 | 0.735 | 0.0742 | 0.00 |
| 27 | REALTABPFN-V2.5 (tuned) | 1669.8 | 36.80 | 0.702 | 0.0771 | 0.00 |
| 30 | TABM-A (tuned + ensemble) | 1635.2 | 39.70 | 0.677 | 0.0858 | 0.00 |
| 31 | REALMLP_GPU (tuned + ensemble) | 1614.2 | 41.55 | 0.662 | 0.0866 | 0.00 |
| 32 | REALTABPFN-V2.5 (default) | 1609.1 | 42.00 | 0.658 | 0.0864 | 0.00 |
| 39 | TABM-A (tuned) | 1549.0 | 47.65 | 0.611 | 0.0934 | 0.00 |
| 40 | REALMLP_GPU (tuned) | 1532.0 | 49.33 | 0.597 | 0.0966 | 0.00 |
| 41 | TA-TABM (tuned + ensemble) | 1499.8 | 52.60 | 0.570 | 0.1039 | 0.00 |
| 42 | CAT (tuned + ensemble) | 1497.6 | 52.83 | 0.568 | 0.1067 | 0.00 |
| 43 | CAT (tuned) | 1496.4 | 52.95 | 0.567 | 0.1062 | 0.00 |
| 45 | GBM (tuned + ensemble) | 1483.0 | 54.35 | 0.555 | 0.1140 | 0.00 |
| 51 | TA-TABM (tuned) | 1450.0 | 57.85 | 0.526 | 0.1089 | 0.00 |
| 56 | [New] MITER_BS6 (tuned + ensemble) | 1419.6 | 61.15 | 0.499 | 0.1300 | 0.00 |
| 58 | GBM (tuned) | 1416.2 | 61.52 | 0.496 | 0.1224 | 0.00 |
| 59 | XGB (tuned + ensemble) | 1415.9 | 61.55 | 0.495 | 0.1202 | 0.00 |
| 60 | CAT (default) | 1414.1 | 61.75 | 0.494 | 0.1217 | 0.00 |
| 61 | [New] MITER_BS6 (tuned) | 1413.2 | 61.85 | 0.493 | 0.1309 | 0.00 |
| 62 | XGB (tuned) | 1406.4 | 62.60 | 0.487 | 0.1213 | 0.00 |
| 64 | TABPFNV2_GPU (tuned + ensemble) | 1385.5 | 64.90 | 0.468 | 0.1154 | 0.45 |
| 71 | [New] MITER_BS6 (default) | 1341.9 | 69.72 | 0.427 | 0.1380 | 0.00 |
| 74 | TABPFNV2_GPU (tuned) | 1324.4 | 71.65 | 0.411 | 0.1256 | 0.45 |
| 75 | TA-TABM (default) | 1323.1 | 71.80 | 0.410 | 0.1255 | 0.00 |
| 77 | TABM-A (default) | 1311.7 | 73.05 | 0.400 | 0.1268 | 0.00 |
| 78 | REALMLP_GPU (default) | 1308.1 | 73.45 | 0.396 | 0.1334 | 0.00 |
| 80 | TABPFNV2_GPU (default) | 1293.2 | 75.08 | 0.383 | 0.1336 | 0.45 |
| 95 | GBM (default) | 1189.6 | 85.95 | 0.292 | 0.1501 | 0.00 |
| 97 | XGB (default) | 1181.1 | 86.80 | 0.285 | 0.1548 | 0.00 |

## BS6 on All 20 Tasks (13 Reg + 7 Binary) — gbdt_only (`12` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GBM (tuned + ensemble) | 1175.7 | 4.15 | 0.714 | 0.0208 | 0.00 |
| 2 | CAT (tuned + ensemble) | 1164.5 | 4.30 | 0.700 | 0.0109 | 0.00 |
| 3 | CAT (tuned) | 1142.5 | 4.60 | 0.673 | 0.0104 | 0.00 |
| 4 | [New] MITER_BS6 (tuned + ensemble) | 1096.9 | 5.25 | 0.614 | 0.0375 | 0.00 |
| 5 | [New] MITER_BS6 (tuned) | 1079.7 | 5.50 | 0.591 | 0.0385 | 0.00 |
| 6 | GBM (tuned) | 1035.7 | 6.15 | 0.532 | 0.0306 | 0.00 |
| 7 | XGB (tuned + ensemble) | 1022.1 | 6.35 | 0.514 | 0.0274 | 0.00 |
| 8 | XGB (tuned) | 1001.7 | 6.65 | 0.486 | 0.0285 | 0.00 |
| 9 | CAT (default) | 998.3 | 6.70 | 0.482 | 0.0308 | 0.00 |
| 10 | [New] MITER_BS6 (default) | 963.6 | 7.20 | 0.436 | 0.0470 | 0.00 |
| 11 | GBM (default) | 669.4 | 10.50 | 0.136 | 0.0623 | 0.00 |
| 12 | XGB (default) | 649.8 | 10.65 | 0.123 | 0.0672 | 0.00 |

## BS6 + T12 on 17 Fast/Medium Tasks (10 Reg + 7 Binary) — full_pool (`124` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TA-KUMO-TABULAR (default) | 2375.6 | 4.82 | 0.969 | 0.0231 | 0.00 |
| 2 | TA-LIMIX-2 (default) | 2346.5 | 5.41 | 0.964 | 0.0083 | 0.00 |
| 3 | TA-SELDON (default) | 2211.8 | 8.94 | 0.935 | 0.0368 | 0.00 |
| 4 | TA-KUMO-TABULAR-MEDIUM (default) | 2198.9 | 9.35 | 0.932 | 0.0341 | 0.00 |
| 5 | TabFM+ | 2193.5 | 9.53 | 0.931 | 0.0405 | 0.00 |
| 6 | Chakra-Tab (full) | 2181.2 | 9.94 | 0.927 | 0.0356 | 0.00 |
| 7 | TA-TABPFN-3.5 (default) | 2159.1 | 10.71 | 0.921 | 0.0443 | 0.00 |
| 8 | Chakra-Tab (medium) | 2151.0 | 11.00 | 0.919 | 0.0373 | 0.00 |
| 9 | AutoGluon 1.6 (noncommercial, 4h) | 2100.3 | 12.94 | 0.903 | 0.0383 | 0.00 |
| 10 | TABFM (default) | 2070.7 | 14.18 | 0.893 | 0.0556 | 0.00 |
| 18 | TA-TABPFN-3.5-FAST (default) | 1954.0 | 19.76 | 0.847 | 0.0602 | 0.00 |
| 22 | TA-TABPFN-3 (default) | 1820.7 | 27.76 | 0.782 | 0.0669 | 0.00 |
| 23 | REALTABPFN-V2.5 (tuned + ensemble) | 1789.4 | 29.94 | 0.765 | 0.0714 | 0.00 |
| 26 | TABPFN-V2.6 (default) | 1762.9 | 31.88 | 0.749 | 0.0734 | 0.00 |
| 27 | REALTABPFN-V2.5 (tuned) | 1679.5 | 38.65 | 0.694 | 0.0788 | 0.00 |
| 31 | TABM-A (tuned + ensemble) | 1659.6 | 40.41 | 0.680 | 0.0855 | 0.00 |
| 32 | REALMLP_GPU (tuned + ensemble) | 1643.5 | 41.88 | 0.668 | 0.0864 | 0.00 |
| 33 | REALTABPFN-V2.5 (default) | 1639.1 | 42.29 | 0.664 | 0.0874 | 0.00 |
| 35 | TABM-A (tuned) | 1573.5 | 48.71 | 0.612 | 0.0920 | 0.00 |
| 36 | REALMLP_GPU (tuned) | 1571.0 | 48.97 | 0.610 | 0.0959 | 0.00 |
| 42 | CAT (tuned) | 1539.8 | 52.24 | 0.583 | 0.1036 | 0.00 |
| 43 | [New] MITER_T12 (tuned + ensemble) | 1539.2 | 52.29 | 0.583 | 0.1128 | 0.00 |
| 44 | CAT (tuned + ensemble) | 1535.6 | 52.68 | 0.580 | 0.1046 | 0.00 |
| 45 | [New] MITER_T12 (tuned) | 1519.3 | 54.44 | 0.566 | 0.1116 | 0.00 |
| 46 | TA-TABM (tuned + ensemble) | 1511.5 | 55.29 | 0.559 | 0.1056 | 0.00 |
| 48 | GBM (tuned + ensemble) | 1494.9 | 57.12 | 0.544 | 0.1154 | 0.00 |
| 52 | TA-TABM (tuned) | 1458.9 | 61.18 | 0.511 | 0.1103 | 0.00 |
| 54 | TABPFNV2_GPU (tuned + ensemble) | 1457.6 | 61.32 | 0.510 | 0.1095 | 0.35 |
| 57 | CAT (default) | 1449.6 | 62.24 | 0.502 | 0.1200 | 0.00 |
| 59 | [New] MITER_BS6 (tuned + ensemble) | 1439.3 | 63.41 | 0.493 | 0.1328 | 0.00 |
| 62 | [New] MITER_BS6 (tuned) | 1430.9 | 64.38 | 0.485 | 0.1340 | 0.00 |
| 64 | XGB (tuned + ensemble) | 1429.1 | 64.59 | 0.483 | 0.1215 | 0.00 |
| 65 | GBM (tuned) | 1415.0 | 66.21 | 0.470 | 0.1253 | 0.00 |
| 66 | XGB (tuned) | 1412.8 | 66.47 | 0.468 | 0.1230 | 0.00 |
| 73 | TABPFNV2_GPU (tuned) | 1384.0 | 69.79 | 0.441 | 0.1215 | 0.35 |
| 76 | TABPFNV2_GPU (default) | 1348.1 | 73.94 | 0.407 | 0.1309 | 0.35 |
| 77 | [New] MITER_T12 (default) | 1347.1 | 74.06 | 0.406 | 0.1424 | 0.00 |
| 78 | [New] MITER_BS6 (default) | 1347.1 | 74.06 | 0.406 | 0.1424 | 0.00 |
| 80 | REALMLP_GPU (default) | 1338.4 | 75.06 | 0.398 | 0.1335 | 0.00 |
| 81 | TA-TABM (default) | 1335.8 | 75.35 | 0.396 | 0.1267 | 0.00 |
| 85 | TABM-A (default) | 1317.3 | 77.47 | 0.378 | 0.1283 | 0.00 |
| 100 | XGB (default) | 1195.9 | 90.53 | 0.272 | 0.1577 | 0.00 |
| 102 | GBM (default) | 1187.7 | 91.35 | 0.265 | 0.1544 | 0.00 |

## BS6 + T12 on 17 Fast/Medium Tasks (10 Reg + 7 Binary) — gbdt_only (`15` methods)

| # | method | elo | rank | winrate | improvability | imputed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [New] MITER_T12 (tuned + ensemble) | 1228.4 | 4.29 | 0.765 | 0.0211 | 0.00 |
| 2 | [New] MITER_T12 (tuned) | 1191.4 | 4.85 | 0.725 | 0.0196 | 0.00 |
| 3 | CAT (tuned) | 1171.1 | 5.18 | 0.702 | 0.0100 | 0.00 |
| 4 | CAT (tuned + ensemble) | 1167.4 | 5.24 | 0.697 | 0.0111 | 0.00 |
| 5 | GBM (tuned + ensemble) | 1135.6 | 5.76 | 0.660 | 0.0252 | 0.00 |
| 6 | [New] MITER_BS6 (tuned + ensemble) | 1068.3 | 6.94 | 0.576 | 0.0433 | 0.00 |
| 7 | [New] MITER_BS6 (tuned) | 1043.7 | 7.38 | 0.544 | 0.0446 | 0.00 |
| 8 | XGB (tuned + ensemble) | 999.5 | 8.18 | 0.487 | 0.0317 | 0.00 |
| 9 | CAT (default) | 996.2 | 8.24 | 0.483 | 0.0319 | 0.00 |
| 10 | GBM (tuned) | 966.4 | 8.76 | 0.445 | 0.0367 | 0.00 |
| 11 | XGB (tuned) | 952.9 | 9.00 | 0.429 | 0.0331 | 0.00 |
| 12 | [New] MITER_BS6 (default) | 906.1 | 9.79 | 0.372 | 0.0546 | 0.00 |
| 13 | [New] MITER_T12 (default) | 906.1 | 9.79 | 0.372 | 0.0546 | 0.00 |
| 14 | XGB (default) | 653.1 | 13.12 | 0.134 | 0.0732 | 0.00 |
| 15 | GBM (default) | 613.8 | 13.47 | 0.109 | 0.0699 | 0.00 |

## Per-dataset Fold-0 (`lite`) Metric Error (RMSE or 1-AUC per TabArena task metric; lower is better)

| dataset | MITER default | MITER_BS6 (tuned+ens) | MITER_T12 (tuned+ens) | CAT (default) | CAT (tuned+ens) | GBM (tuned+ens) |
|---|---:|---:|---:|---:|---:|---:|
| `airfoil_self_noise` | 1.2983 | 1.2312 | 1.2248 | 1.4748 | 1.1814 | 1.3396 |
| `Another-Dataset-on-used-Fiat-500` | 707.77 | 694.08 | 687.74 | 699.31 | 699.19 | 694.78 |
| `concrete_compressive_strength` | 4.386 | 4.3556 | 4.3361 | 4.5421 | 4.3463 | 4.2318 |
| `diamonds` | 525.38 | 524.68 | 526.35 | 529.75 | 529.3 | 522.9 |
| `Food_Delivery_Time` | 7.4283 | 7.4293 | 7.4246 | 7.4187 | 7.4049 | 7.4147 |
| `healthcare_insurance_expenses` | 4177.1 | 4122.2 | 4091.2 | 4110.3 | 4115.5 | 4101.1 |
| `houses` | 0.21255 | 0.2116 | 0.21099 | 0.21331 | 0.21251 | 0.21322 |
| `miami_housing` | 86202 | 83827 | 83790 | 81043 | 79487 | 81592 |
| `physiochemical_protein` | 3.3756 | 3.3846 | — | 3.5566 | 3.4176 | 3.4165 |
| `QSAR-TID-11` | 0.8349 | 0.83479 | — | 0.84791 | 0.84342 | 0.81975 |
| `QSAR_fish_toxicity` | 0.90608 | 0.89573 | 0.89674 | 0.91758 | 0.91322 | 0.9151 |
| `superconductivity` | 9.2891 | 9.2847 | — | 9.2638 | 9.2594 | 9.1389 |
| `wine_quality` | 0.60821 | 0.60936 | 0.60217 | 0.61179 | 0.59406 | 0.5944 |
| `Amazon_employee_access` | 0.20668 | 0.20752 | 0.14712 | 0.12915 | 0.12618 | 0.15136 |
| `Bank_Customer_Churn` | 0.13012 | 0.12586 | 0.1231 | 0.12329 | 0.12361 | 0.12574 |
| `bank-marketing` | 0.22743 | 0.22485 | 0.22505 | 0.22543 | 0.22512 | 0.22583 |
| `credit_card_clients_default` | 0.20832 | 0.20767 | 0.207 | 0.21021 | 0.21023 | 0.2094 |
| `heloc` | 0.20431 | 0.20332 | 0.20246 | 0.2019 | 0.20076 | 0.20196 |
| `online_shoppers_intention` | 0.071727 | 0.071628 | 0.071193 | 0.073401 | 0.073278 | 0.071579 |
| `blood-transfusion-service-center` | 0.31561 | 0.31123 | 0.29228 | 0.29412 | 0.28649 | 0.27807 |

