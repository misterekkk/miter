# TabArena protocol, miter C++ default (M5) - 13 regression datasets

Headline = **post-fix** run (axis binner rebuilt with the research quantile rule). The pre-fix run is kept for reference.
All numbers below are produced from files in `benchmarks/results/raw/tabarena/` (gitignored): `cmp_post/*`, `cmp_prefix/*`, `analysis_post.json`, `analysis_prefix.json`.

## Provenance
- TabArena `tabarena 0.1.1.dev20261117092751`, `autogluon.tabular 1.6.4b20261118`, bencheval `0.1.1.dev20261117092751`, pandas 2.3.3, numpy 2.5.5, Python 3.12 (venv `.venv_tabarena`, site-packages of histree's tabarena venv + miter installed from this repo).
- Protocol: TabArena v0.1 bundle (`TabArenaV0pt1ExperimentBundle`), their splits (all repeats/folds: 9 splits for n>=2500, 30 for small sets), 8-fold bagging, AutoGluon default preprocessing, RMSE. Task time limit forwarded: TabArena sets 3600 s per model (log line `ag.max_time_limit=3600`); the child's limit is passed to miter's `time_limit`. All jobs succeeded (0 failures in the logs); I did not check whether any child stopped because of the limit.
- miter config: lr 0.05, 31 leaves, min_data_in_leaf 20, lambda_l2 1.0, m_mult 2, k_proj 0, early stopping 50 on the bagging X_val, n_estimators 10000, n_jobs 1, random_state = model seed. Native K1 categoricals (pandas `category` columns) and native NaN handling; no ordinal coding or imputation. Wrapper: `benchmarks/tabarena/miter_tabarena_model.py` (model persisted through miter JSON because `_core.Model` is not picklable).
- Commits: post-fix build installed from the working tree at the time of install (HEAD then `d1d48d9` plus uncommitted src/core changes, since committed). Current `git rev-parse HEAD` = `ff9d80f` (clean tree except untracked `benchmarks/tabarena/`). Whether the installed build equals HEAD byte for byte was NOT verified. Pre-fix build: installed when HEAD was `6bbe399` (old binning). The pre-fix run is a different build than the post-fix one in several respects possibly (binner fix, plus other uncommitted src/core edits present at post-fix install time; I cannot separate them).
- Reviewer note (main session): the only uncommitted `src/` edits that could have been present at post-fix install time are those committed in `d1d48d9` (memset through `void*`, `__extension__` typedef for `unsigned __int128`, a `static_assert`) — none changes numerical behaviour. So the post-fix build is behaviourally the axis-binner fix `876f133` on top of M5; the pre-fix/post-fix difference is the binner fix.
- Hardware: Apple M4, 10 cores, 16 GB; runs under `nice -n 10`, 3 processes in parallel plus two other benchmark agents on the same machine, so timings are approximate (loaded machine).
- Baseline artifacts used: the TabArena cache (`histree/research/results/tabarena_cache`) holds 32 artifact dates `tabarena-2025-06-12` ... `tabarena-2026-11-16`; e.g. LightGBM and XGBoost come from `tabarena-2025-06-12`, CatBoost from `tabarena-2026-07-13`. Per-method artifact dates are in `method_info.csv` under `cmp_post/full_pool/`.
- Elapsed (wall, own runs): pre-fix three groups 1153 s, 914 s, 1657 s (3 parallel); post-fix three groups 1217 s, 1010 s, 1086 s (3 parallel). Plus 1 smoke test on 2 datasets (3 s).
- Download of all methods' cached results via `TabArenaContext.load_results(download_results="auto")` (official downloader, normal channel): 145 s; cache grew from 11.7 MB to 103.4 MB (`du -sk` 11736 -> 105912 KB, includes writes by the compare runs); 68 methods, 94,050 result rows, 118 method variants. Nothing was trained on binary/multiclass.

## Post-fix check of the binner
Heavy-tailed column (lognormal, n=20000): 972 distinct Q16r codes, `Dataset.num_bins_for_feature` = 76 axis bins (74 edges), i.e. fewer bins than distinct codes.

## Headline: post-fix leaderboard, full method pool (13 regression datasets)
Official `ctx.compare(...)` defaults (fillna "auto", calibration "auto"), miter C++ added as one new method, pool = every method returned by `TabArenaContext.methods` (68 methods, all their default/tuned/tuned+ens variants available) = 115 rows. miter: Elo 1379.9 (+65.7/-63.1), rank-by-Elo position 63/115, mean rank 61.95, winrate 0.465, improvability 0.1204.
Elo is relative to this loaded pool on these 13 tasks and calibrated by compare() defaults; it is NOT the public leaderboard value.

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---|---|---|---|---|---|---|---|---|
| 1 | TA-KUMO-TABULAR (default) | 2473.0 | 258.1 | 188.6 | 4.0 | 0.973 | 0.013 | 0.0 |
| 2 | TA-LIMIX-2 (default) | 2424.2 | 331.9 | 191.9 | 4.9 | 0.966 | 0.006 | 0.0 |
| 3 | Chakra-Tab (full) | 2314.4 | 197.8 | 141.6 | 7.2 | 0.946 | 0.023 | 0.0 |
| 4 | Chakra-Tab (medium) | 2312.9 | 190.4 | 137.0 | 7.2 | 0.945 | 0.023 | 0.0 |
| 5 | TA-KUMO-TABULAR-MEDIUM (default) | 2295.7 | 205.7 | 135.8 | 7.7 | 0.941 | 0.023 | 0.0 |
| 6 | INPRIORA (default) | 2226.1 | 239.1 | 147.7 | 9.6 | 0.924 | 0.025 | 0.0 |
| 7 | TA-SELDON (default) | 2206.4 | 162.0 | 83.4 | 10.2 | 0.919 | 0.030 | 0.0 |
| 8 | TA-TABPFN-3.5 (default) | 2166.2 | 202.5 | 112.7 | 11.5 | 0.907 | 0.033 | 0.0 |
| 9 | TA-SELDON-MINI (default) | 2143.1 | 164.9 | 66.9 | 12.4 | 0.900 | 0.035 | 0.0 |
| 10 | TA-KUMO-TABULAR-SMALL (default) | 2126.1 | 137.7 | 79.2 | 13.0 | 0.895 | 0.033 | 0.0 |
| 11 | TabFM+ | 2085.2 | 161.0 | 98.4 | 14.5 | 0.881 | 0.036 | 0.0 |
| 12 | AutoGluon 1.6 (noncommercial, 4h) | 2077.0 | 154.4 | 104.4 | 14.8 | 0.879 | 0.040 | 0.0 |
| 13 | CAUSILO (default) | 2055.8 | 172.1 | 101.7 | 15.7 | 0.871 | 0.039 | 0.0 |
| 14 | TABABAX (default) | 2033.8 | 182.6 | 91.3 | 16.6 | 0.863 | 0.039 | 0.0 |
| 15 | AutoGluon 1.6 (extreme, 4h) | 2032.5 | 140.9 | 79.7 | 16.7 | 0.863 | 0.045 | 0.0 |
| 16 | TABFM (default) | 2010.6 | 155.7 | 72.4 | 17.6 | 0.854 | 0.047 | 0.0 |
| 17 | MITRA_V2 (default) | 1998.0 | 172.4 | 103.8 | 18.2 | 0.849 | 0.039 | 0.0 |
| 18 | TA-TABPFN-3.5-FAST (default) | 1962.6 | 148.9 | 93.3 | 19.8 | 0.835 | 0.051 | 0.0 |
| 19 | TA-EXAONE-TABULAR (default) | 1895.4 | 109.7 | 67.7 | 23.2 | 0.806 | 0.057 | 0.0 |
| 20 | Xiaomi-TabLDM+ | 1885.2 | 164.9 | 100.1 | 23.7 | 0.801 | 0.051 | 0.0 |
| 21 | TA-TABPFN-3 (default) | 1811.2 | 171.1 | 112.8 | 27.9 | 0.764 | 0.055 | 0.0 |
| 22 | AutoGluon 1.5 (extreme, 4h) | 1810.9 | 116.6 | 65.7 | 27.9 | 0.764 | 0.069 | 0.0 |
| 23 | XIAOMI-TABLDM (default) | 1793.9 | 204.6 | 162.0 | 28.9 | 0.755 | 0.058 | 0.0 |
| 24 | TA-NORI-30M (default) | 1785.5 | 133.2 | 65.8 | 29.5 | 0.750 | 0.065 | 0.0 |
| 25 | REALTABPFN-V2.5 (tuned + ensemble) | 1751.7 | 110.1 | 78.4 | 31.6 | 0.731 | 0.064 | 0.0 |
| 26 | TABPFN-V2.6 (default) | 1751.0 | 88.7 | 40.3 | 31.7 | 0.731 | 0.069 | 0.0 |
| 27 | TABDPT_GPU (tuned + ensemble) | 1740.8 | 132.4 | 66.1 | 32.3 | 0.725 | 0.072 | 0.0 |
| 28 | TA-NORI (default) | 1706.1 | 122.3 | 68.0 | 34.7 | 0.704 | 0.074 | 0.0 |
| 29 | TABICLV2 (default) | 1704.8 | 192.1 | 131.0 | 34.8 | 0.704 | 0.069 | 0.0 |
| 30 | TABDPT_GPU (tuned) | 1689.4 | 89.7 | 52.4 | 35.9 | 0.694 | 0.076 | 0.0 |
| 31 | TABDPT-1.3 (default) | 1678.8 | 196.8 | 115.3 | 36.6 | 0.687 | 0.074 | 0.0 |
| 32 | REALMLP_GPU (tuned + ensemble) | 1664.8 | 75.6 | 52.4 | 37.7 | 0.678 | 0.079 | 0.0 |
| 33 | REALTABPFN-V2.5 (tuned) | 1651.1 | 107.8 | 88.7 | 38.7 | 0.669 | 0.071 | 0.0 |
| 34 | TABM-A (tuned + ensemble) | 1648.0 | 72.0 | 60.2 | 38.9 | 0.667 | 0.082 | 0.0 |
| 35 | TABH2O (default) | 1639.9 | 212.6 | 126.4 | 39.5 | 0.662 | 0.077 | 0.0 |
| 36 | TABDPT_TURBO (default) | 1626.1 | 155.3 | 76.7 | 40.6 | 0.653 | 0.081 | 0.0 |
| 37 | AutoGluon 1.4 (best, 4h) | 1610.9 | 69.2 | 54.8 | 41.8 | 0.642 | 0.092 | 0.0 |
| 38 | TABDPT_GPU (default) | 1603.9 | 120.3 | 66.9 | 42.3 | 0.637 | 0.084 | 0.0 |
| 39 | REALMLP_GPU (tuned) | 1570.3 | 71.2 | 52.2 | 45.1 | 0.613 | 0.089 | 0.0 |
| 40 | REALTABPFN-V2.5 (default) | 1562.7 | 109.2 | 109.5 | 45.7 | 0.608 | 0.085 | 0.0 |
| 41 | MNCA_GPU (tuned + ensemble) | 1558.5 | 102.1 | 109.6 | 46.0 | 0.605 | 0.100 | 0.0 |
| 42 | MLP-A (tuned + ensemble) | 1546.3 | 80.6 | 69.5 | 47.1 | 0.596 | 0.093 | 0.0 |
| 43 | TABPACK (default) | 1527.8 | 105.2 | 102.4 | 48.6 | 0.582 | 0.095 | 0.0 |
| 44 | CAT (tuned + ensemble) | 1506.9 | 72.6 | 64.5 | 50.5 | 0.566 | 0.107 | 0.0 |
| 45 | TABM-A (tuned) | 1501.7 | 85.1 | 93.3 | 50.9 | 0.562 | 0.096 | 0.0 |
| 46 | GBM (tuned + ensemble) | 1498.8 | 61.8 | 68.3 | 51.2 | 0.560 | 0.111 | 0.0 |
| 47 | XRFM_GPU (tuned + ensemble) | 1490.4 | 89.1 | 100.4 | 51.9 | 0.554 | 0.102 | 0.0 |
| 48 | CAT (tuned) | 1485.0 | 76.1 | 74.1 | 52.4 | 0.549 | 0.108 | 0.0 |
| 49 | TA-TABM (tuned + ensemble) | 1474.1 | 100.9 | 73.8 | 53.3 | 0.541 | 0.097 | 0.0 |
| 50 | TA-ILTM (tuned + ensemble) | 1457.8 | 61.7 | 65.6 | 54.8 | 0.528 | 0.120 | 0.0 |
| 51 | CHIMERA (tuned + ensemble) | 1440.8 | 119.7 | 115.9 | 56.3 | 0.515 | 0.119 | 0.0 |
| 52 | GBM (tuned) | 1436.4 | 76.6 | 77.9 | 56.7 | 0.511 | 0.116 | 0.0 |
| 53 | XGB (tuned + ensemble) | 1428.8 | 54.7 | 55.2 | 57.4 | 0.505 | 0.116 | 0.0 |
| 54 | TA-TABSWIFT (default) | 1423.0 | 139.6 | 120.8 | 58.0 | 0.500 | 0.101 | 0.0 |
| 55 | LIGHTPFN (default) | 1421.4 | 122.3 | 97.5 | 58.1 | 0.499 | 0.112 | 0.0 |
| 56 | XGB (tuned) | 1407.5 | 66.7 | 63.3 | 59.4 | 0.488 | 0.117 | 0.0 |
| 57 | CHIMERA (tuned) | 1406.5 | 105.9 | 124.6 | 59.5 | 0.487 | 0.122 | 0.0 |
| 58 | XRFM_GPU (tuned) | 1405.3 | 73.1 | 86.7 | 59.6 | 0.486 | 0.109 | 0.0 |
| 59 | CAT (default) | 1400.3 | 77.2 | 102.7 | 60.1 | 0.482 | 0.122 | 0.0 |
| 60 | TA-TABM (tuned) | 1398.8 | 95.6 | 82.6 | 60.2 | 0.481 | 0.106 | 0.0 |
| 61 | MNCA_GPU (tuned) | 1394.4 | 93.3 | 102.3 | 60.6 | 0.477 | 0.119 | 0.0 |
| 62 | MLP-A (tuned) | 1388.0 | 113.2 | 91.5 | 61.2 | 0.472 | 0.110 | 0.0 |
| **63** | **[New] MITER_CPP (default)** | 1379.9 | 65.7 | 63.1 | 61.9 | 0.465 | 0.120 | 0.0 |
| 64 | TA-ILTM (tuned) | 1351.0 | 62.3 | 69.4 | 64.6 | 0.442 | 0.120 | 0.0 |
| 65 | CHIMERA (default) | 1345.7 | 119.9 | 130.3 | 65.1 | 0.437 | 0.130 | 0.0 |
| 66 | TABPFNV2_GPU (tuned + ensemble) | 1341.6 | 169.9 | 141.1 | 65.5 | 0.434 | 0.115 | 0.5 |
| 67 | PRISMBOOST (tuned + ensemble) | 1300.6 | 70.7 | 74.0 | 69.3 | 0.401 | 0.135 | 0.0 |
| 68 | TA-TABM (default) | 1294.6 | 104.0 | 99.0 | 69.9 | 0.396 | 0.121 | 0.0 |
| 69 | MNCA_GPU (default) | 1290.3 | 79.6 | 84.4 | 70.3 | 0.392 | 0.133 | 0.0 |
| 70 | TABM-A (default) | 1285.3 | 100.3 | 103.4 | 70.7 | 0.388 | 0.123 | 0.0 |
| 71 | MITRA_GPU (default) | 1285.1 | 141.7 | 134.3 | 70.8 | 0.388 | 0.130 | 0.5 |
| 72 | SAP-RPT-OSS (default) | 1284.0 | 162.7 | 171.5 | 70.9 | 0.387 | 0.134 | 0.0 |
| 73 | TA-LIMIX (default) | 1274.0 | 182.7 | 161.1 | 71.8 | 0.379 | 0.130 | 0.0 |
| 74 | TABPFNV2_GPU (tuned) | 1254.8 | 159.8 | 155.0 | 73.5 | 0.364 | 0.125 | 0.5 |
| 75 | CTB (tuned + ensemble) | 1246.7 | 96.2 | 146.3 | 74.3 | 0.357 | 0.151 | 0.0 |
| 76 | NN_TORCH (tuned + ensemble) | 1244.3 | 104.1 | 127.3 | 74.5 | 0.355 | 0.136 | 0.0 |
| 77 | XT (tuned + ensemble) | 1241.5 | 107.2 | 124.2 | 74.7 | 0.353 | 0.154 | 0.0 |
| 78 | PRISMBOOST (tuned) | 1238.4 | 68.6 | 70.8 | 75.0 | 0.351 | 0.142 | 0.0 |
| 79 | REALMLP_GPU (default) | 1231.8 | 100.4 | 93.3 | 75.6 | 0.346 | 0.132 | 0.0 |
| 80 | XT (tuned) | 1221.4 | 99.8 | 121.7 | 76.5 | 0.337 | 0.156 | 0.0 |
| 81 | GBM (default) | 1217.4 | 35.2 | 56.2 | 76.9 | 0.334 | 0.139 | 0.0 |
| 82 | CTB (tuned) | 1207.5 | 100.1 | 149.0 | 77.8 | 0.327 | 0.155 | 0.0 |
| 83 | PRISMBOOST (default) | 1205.5 | 91.9 | 92.1 | 77.9 | 0.325 | 0.146 | 0.0 |
| 84 | TABPFNV2_GPU (default) | 1203.8 | 130.8 | 174.7 | 78.1 | 0.324 | 0.138 | 0.5 |
| 85 | XGB (default) | 1196.3 | 85.3 | 97.7 | 78.7 | 0.318 | 0.146 | 0.0 |
| 86 | PB (tuned + ensemble) | 1190.9 | 64.3 | 105.0 | 79.2 | 0.314 | 0.156 | 0.0 |
| 87 | NN_TORCH (tuned) | 1190.1 | 92.5 | 126.4 | 79.3 | 0.313 | 0.143 | 0.0 |
| 88 | MLP-A (default) | 1168.4 | 77.7 | 91.0 | 81.1 | 0.297 | 0.144 | 0.0 |
| 89 | RF (tuned + ensemble) | 1164.4 | 76.6 | 125.3 | 81.5 | 0.294 | 0.164 | 0.0 |
| 90 | EBM (tuned + ensemble) | 1161.3 | 146.1 | 200.5 | 81.7 | 0.292 | 0.168 | 0.0 |
| 91 | XRFM_GPU (default) | 1158.8 | 113.8 | 118.8 | 81.9 | 0.290 | 0.162 | 0.0 |
| 92 | TA-TABSTAR-V1.1 (default) | 1143.1 | 102.2 | 149.6 | 83.3 | 0.278 | 0.166 | 0.0 |
| 93 | PB (tuned) | 1135.6 | 58.6 | 107.8 | 83.9 | 0.273 | 0.166 | 0.0 |
| 94 | RF (tuned) | 1125.3 | 80.8 | 123.4 | 84.7 | 0.266 | 0.168 | 0.0 |
| 95 | EBM (tuned) | 1117.5 | 144.5 | 197.2 | 85.3 | 0.260 | 0.174 | 0.0 |
| 96 | XT (default) | 1070.0 | 96.5 | 140.6 | 89.0 | 0.228 | 0.179 | 0.0 |
| 97 | EBM (default) | 1053.3 | 135.1 | 189.9 | 90.2 | 0.218 | 0.182 | 0.0 |
| 98 | NN_TORCH (default) | 1037.6 | 80.8 | 143.3 | 91.3 | 0.208 | 0.176 | 0.0 |
| 99 | TA-APLR (tuned + ensemble) | 1036.6 | 110.4 | 147.4 | 91.4 | 0.207 | 0.183 | 0.0 |
| 100 | FASTAI (tuned + ensemble) | 1027.6 | 107.1 | 148.1 | 92.0 | 0.202 | 0.177 | 0.0 |
| 101 | TABICL_GPU (default) | 1000.0 | 51.5 | 113.1 | 93.8 | 0.186 | 0.183 | 1.0 |
| 102 | RF (default) | 1000.0 | 51.5 | 113.1 | 93.8 | 0.186 | 0.183 | 0.0 |
| 103 | TA-APLR (tuned) | 991.5 | 110.6 | 159.9 | 94.4 | 0.181 | 0.191 | 0.0 |
| 104 | CTB (default) | 984.2 | 112.4 | 216.4 | 94.8 | 0.177 | 0.207 | 0.0 |
| 105 | FASTAI (tuned) | 980.5 | 112.0 | 135.6 | 95.1 | 0.175 | 0.182 | 0.0 |
| 106 | PB (default) | 932.7 | 93.6 | 142.7 | 97.9 | 0.150 | 0.200 | 0.0 |
| 107 | KNN (tuned + ensemble) | 896.0 | 176.9 | 203.9 | 99.8 | 0.133 | 0.231 | 0.0 |
| 108 | FASTAI (default) | 865.6 | 122.0 | 184.1 | 101.3 | 0.120 | 0.221 | 0.0 |
| 109 | TA-ILTM (default) | 860.1 | 79.6 | 134.3 | 101.6 | 0.118 | 0.212 | 0.0 |
| 110 | TA-APLR (default) | 850.2 | 132.0 | 225.0 | 102.0 | 0.114 | 0.234 | 0.0 |
| 111 | KNN (tuned) | 795.4 | 180.0 | 223.8 | 104.3 | 0.094 | 0.253 | 0.0 |
| 112 | KNN (default) | 687.3 | 203.0 | 362.1 | 107.7 | 0.064 | 0.321 | 0.0 |
| 113 | LR (tuned + ensemble) | 453.6 | 129.9 | 384.5 | 112.0 | 0.026 | 0.389 | 0.0 |
| 114 | LR (tuned) | 415.7 | 155.5 | 489.9 | 112.5 | 0.022 | 0.391 | 0.0 |
| 115 | LR (default) | 197.5 | 165.0 | 467.1 | 114.2 | 0.007 | 0.413 | 0.0 |

## Secondary: post-fix, GBDT-only pool (LightGBM, CatBoost, XGBoost variants + miter)
Here compare() was run with fillna=None, calibration_method=None (the default fillna reference method is not in this pool), so Elo is NOT comparable to the table above. miter: Elo 1017.4 (+80.0/-92.4), rank-by-Elo position 6/10, mean rank 5.51, winrate 0.499, improvability 0.0227.

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---|---|---|---|---|---|---|---|---|
| 1 | CAT (tuned + ensemble) | 1255.5 | 106.3 | 68.5 | 3.0 | 0.783 | 0.007 | 0.0 |
| 2 | GBM (tuned + ensemble) | 1225.9 | 88.7 | 73.0 | 3.2 | 0.752 | 0.013 | 0.0 |
| 3 | CAT (tuned) | 1174.2 | 109.7 | 75.3 | 3.8 | 0.693 | 0.008 | 0.0 |
| 4 | GBM (tuned) | 1093.5 | 80.1 | 92.1 | 4.7 | 0.594 | 0.019 | 0.0 |
| 5 | XGB (tuned + ensemble) | 1035.1 | 73.8 | 69.0 | 5.3 | 0.521 | 0.019 | 0.0 |
| **6** | **[New] MITER_CPP (default)** | 1017.4 | 80.0 | 92.4 | 5.5 | 0.499 | 0.023 | 0.0 |
| 7 | CAT (default) | 996.5 | 110.4 | 133.4 | 5.7 | 0.473 | 0.027 | 0.0 |
| 8 | XGB (tuned) | 985.7 | 61.9 | 71.4 | 5.9 | 0.460 | 0.020 | 0.0 |
| 9 | GBM (default) | 646.0 | 72.5 | 117.2 | 8.8 | 0.137 | 0.046 | 0.0 |
| 10 | XGB (default) | 570.3 | 72.8 | 107.9 | 9.2 | 0.090 | 0.053 | 0.0 |

## Pre-fix leaderboard (old binning), for reference
Full pool: miter Elo 1378.5 (+62.6/-68.3), rank-by-Elo position 63/115, mean rank 62.07, winrate 0.464, improvability 0.1205. GBDT-only (fillna None): miter Elo 1020.9 (+86.8/-102.9), rank-by-Elo position 6/10, mean rank 5.47, winrate 0.503, improvability 0.0228.


<details><summary>Pre-fix GBDT-only table</summary>

| # | method | elo | elo+ | elo- | rank | winrate | improvability | imputed |
|---|---|---|---|---|---|---|---|---|
| 1 | CAT (tuned + ensemble) | 1251.7 | 104.6 | 68.7 | 3.0 | 0.780 | 0.007 | 0.0 |
| 2 | GBM (tuned + ensemble) | 1225.5 | 87.9 | 72.3 | 3.2 | 0.752 | 0.013 | 0.0 |
| 3 | CAT (tuned) | 1170.6 | 109.5 | 76.6 | 3.8 | 0.689 | 0.008 | 0.0 |
| 4 | GBM (tuned) | 1096.6 | 82.5 | 94.3 | 4.6 | 0.598 | 0.019 | 0.0 |
| 5 | XGB (tuned + ensemble) | 1035.5 | 69.6 | 68.0 | 5.3 | 0.521 | 0.019 | 0.0 |
| **6** | **[New] MITER_CPP (default)** | 1020.9 | 86.8 | 102.9 | 5.5 | 0.503 | 0.023 | 0.0 |
| 7 | CAT (default) | 994.4 | 108.3 | 132.0 | 5.8 | 0.470 | 0.027 | 0.0 |
| 8 | XGB (tuned) | 985.6 | 61.0 | 71.4 | 5.9 | 0.460 | 0.020 | 0.0 |
| 9 | GBM (default) | 647.6 | 72.3 | 117.5 | 8.8 | 0.137 | 0.046 | 0.0 |
| 10 | XGB (default) | 571.7 | 71.8 | 102.6 | 9.2 | 0.090 | 0.053 | 0.0 |

</details>

## Per-dataset mean RMSE (mean over splits), post-fix miter C++
Prototype = histree numba prototype from `histree/research/results/t2/{T2A_full,T2A_tid11,T2K1}_eval_per_split.csv` (RUNI on the 6 no-cat datasets, K1 on the 7 cat datasets; other protocol settings as in its notes: ordinal/imputation vs native). Same splits.

| dataset | n splits | miter C++ post-fix | prototype | CAT def | CAT tuned+ens | GBM def | GBM tuned+ens | XGB def | XGB tuned+ens |
|---|---|---|---|---|---|---|---|---|---|
| Another-Dataset-on-used-Fiat-500 | 30 | 738.0545 | 738.4444 | 737.8291 | 732.6861 | 746.0224 | 729.4338 | 754.6256 | 737.1119 |
| Food_Delivery_Time | 9 | 7.3850 | 7.3848 | 7.3791 | 7.3671 | 7.6156 | 7.3741 | 7.3972 | 7.4005 |
| QSAR-TID-11 | 9 | 0.7706 | 0.7703 | 0.7740 | 0.7712 | 0.7721 | 0.7558 | 0.7865 | 0.7599 |
| QSAR_fish_toxicity | 30 | 0.8782 | 0.8773 | 0.8769 | 0.8744 | 0.8938 | 0.8829 | 0.9046 | 0.8793 |
| airfoil_self_noise | 30 | 1.4078 | 1.4112 | 1.5783 | 1.3300 | 1.5538 | 1.4512 | 1.5494 | 1.4426 |
| concrete_compressive_strength | 30 | 4.3631 | 4.3524 | 4.2017 | 4.2018 | 4.4838 | 4.2116 | 4.7546 | 4.2222 |
| diamonds | 9 | 523.9331 | 525.5949 | 520.3787 | 520.6668 | 532.1269 | 519.0275 | 538.9908 | 528.1893 |
| healthcare_insurance_expenses | 30 | 4599.9270 | 4603.3047 | 4528.1672 | 4522.3862 | 4610.3775 | 4511.9040 | 4671.9858 | 4519.5821 |
| houses | 9 | 0.2111 | 0.2110 | 0.2111 | 0.2108 | 0.2171 | 0.2112 | 0.2152 | 0.2146 |
| miami_housing | 9 | 84027.5802 | 84033.2773 | 79760.3760 | 78405.0988 | 85625.2883 | 79611.9796 | 86503.1774 | 80415.7368 |
| physiochemical_protein | 9 | 3.3509 | 3.3499 | 3.5213 | 3.3821 | 3.4769 | 3.3844 | 3.5130 | 3.3902 |
| superconductivity | 9 | 9.3639 | 9.3715 | 9.3661 | 9.3441 | 9.4115 | 9.2591 | 9.4499 | 9.3402 |
| wine_quality | 9 | 0.6149 | 0.6153 | 0.6219 | 0.6044 | 0.6279 | 0.6081 | 0.6211 | 0.6103 |

## miter C++ (post-fix) vs others: relative RMSE gain per dataset-mean, over 13 datasets (+ = miter better)
| comparison (miter C++ vs X) | mean gain | median gain | wins | Wilcoxon p |
|---|---|---|---|---|
| [Proto] numba (RUNI on 6 no-cat sets, K1 | +0.03% | +0.01% | 7/13 | 0.4973 |
| CAT (default) | +0.42% | -0.03% | 5/13 | 0.7354 |
| CAT (tuned) | -1.52% | -0.31% | 2/13 | 0.0266 |
| CAT (tuned + ensemble) | -1.67% | -0.63% | 2/13 | 0.0081 |
| GBM (default) | +2.36% | +1.87% | 13/13 | 0.0002 |
| GBM (tuned + ensemble) | -1.00% | -1.13% | 4/13 | 0.0803 |
| XGB (default) | +3.10% | +2.20% | 13/13 | 0.0002 |
| XGB (tuned + ensemble) | -0.45% | -0.13% | 6/13 | 0.5879 |

Same for pre-fix miter C++:

| comparison (miter C++ vs X) | mean gain | median gain | wins | Wilcoxon p |
|---|---|---|---|---|
| [Proto] numba (RUNI on 6 no-cat sets, K1 | +0.02% | -0.03% | 5/13 | 0.7869 |
| CAT (default) | +0.41% | +0.02% | 7/13 | 1.0 |
| CAT (tuned) | -1.53% | -0.27% | 3/13 | 0.0327 |
| CAT (tuned + ensemble) | -1.68% | -0.68% | 2/13 | 0.0105 |
| GBM (default) | +2.35% | +1.78% | 13/13 | 0.0002 |
| GBM (tuned + ensemble) | -1.01% | -1.13% | 4/13 | 0.0681 |
| XGB (default) | +3.09% | +2.24% | 13/13 | 0.0002 |
| XGB (tuned + ensemble) | -0.46% | -0.08% | 6/13 | 0.5879 |

## Fix effect (pre -> post), per-dataset mean RMSE gain of post over pre (+ = post better)
- Another-Dataset-on-used-Fiat-500: -0.048%
- Food_Delivery_Time: +0.024%
- QSAR-TID-11: +0.000%
- QSAR_fish_toxicity: +0.056%
- airfoil_self_noise: -0.008%
- concrete_compressive_strength: -0.202%
- diamonds: +0.185%
- healthcare_insurance_expenses: +0.155%
- houses: -0.090%
- miami_housing: +0.083%
- physiochemical_protein: -0.023%
- superconductivity: -0.110%
- wine_quality: +0.094%
mean +0.009%, median +0.000%, post better on 6/13 datasets.

## Median training time per split (8 bagged children, s, loaded machine, approximate)
|                                  |   pre |   post |   proto |
|:---------------------------------|------:|-------:|--------:|
| Another-Dataset-on-used-Fiat-500 |   0.5 |    0.4 |     2   |
| Food_Delivery_Time               |   9.2 |    8.9 |    27.6 |
| QSAR-TID-11                      | 130.4 |  134.2 |   488.5 |
| QSAR_fish_toxicity               |   0.2 |    0.2 |     0.7 |
| airfoil_self_noise               |   4.4 |    4   |    11.6 |
| concrete_compressive_strength    |   1.2 |    1.2 |     3.5 |
| diamonds                         |  18.6 |   15   |    42.3 |
| healthcare_insurance_expenses    |   0.2 |    0.2 |     1.3 |
| houses                           |  12.9 |   11.5 |    24.2 |
| miami_housing                    |  10   |    9.4 |    30.8 |
| physiochemical_protein           |  66.7 |   62.2 |   107.2 |
| superconductivity                |  60.3 |   53.4 |   123.8 |
| wine_quality                     |   4.7 |    4.3 |    17.5 |

## What still differs from the official leaderboard / what was not measured
- Regression subset only: 13 of the 51 TabArena tasks; Elo/rank/winrate on this subset are not the public numbers.
- miter has the DEFAULT config only (no tuned, no tuned+ensemble variants), while the other methods have default/tuned/tuned+ens variants.
- Timings: Apple M4, loaded (3 parallel jobs + other agents), not comparable to the official 4h8c cluster timings.
- Artifacts: baselines come from 32 different artifact dates; 'tabarena-2025-06-12' vs 2026-xx; whether all were measured under the same TabArena revision as the installed code was not checked.
- Some methods in the pool have imputed (filled) results on some tasks (column `imputed`), e.g. TabPFNv2/Mitra with 0.46 imputed share; this is compare()'s default behaviour.
- The prototype vs miter C++ comparison mixes: different engine (numba vs C++), categorical handling (native K1 vs native K1 in both for 7 cat datasets; no-cat datasets identical preprocessing), NaN handling (median impute + indicator vs native).
- Not measured: tuning, multi-thread runs, binary/multiclass, hold-out of the 3600 s time limit (whether any child hit it), per-child stop reasons.
- Exploratory results; no pre-registration for this comparison.
