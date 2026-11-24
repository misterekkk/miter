# M2 AX Booster Benchmark Summary

- Hardware: `arm64` (`macOS-26.6.2-arm64-arm-64bit-Mach-O`)
- Python: `3.13.1`
- Versions: `miter 0.1.0.dev0`, `lightgbm 4.7.0`, `numpy 2.5.5`
- Protocol: single-threaded (`n_jobs=1` / `num_threads=1`), median of 5 runs for timing.

## Part A: Single-thread training speed (100 trees, median of 5 runs)

| shape | LightGBM ms/tree | miter ms/tree (total) | miter / LGBM | miter prep ms | miter loop ms/tree | histree AX ms/tree |
|---|---|---|---|---|---|---|
| 20000 x 16 | 1.512 | 1.109 | 0.73x | 9.5 | 1.015 | 1.874 |
| 20000 x 50 | 4.592 | 3.150 | 0.69x | 30.2 | 2.848 | 5.279 |
| 20000 x 100 | 8.869 | 6.251 | 0.70x | 60.0 | 5.651 | 10.965 |
| 50000 x 50 | 8.447 | 6.053 | 0.72x | 71.0 | 5.343 | 10.277 |

## Part B: OpenML fold 0 quality & speed parity (1 thread, ES=50, max_trees=3000)

| dataset | kind | metric | LightGBM | miter AX | histree AX | trees (L/M/H) | ms/tree (L/M) |
|---|---|---|---|---|---|---|---|
| cpu_act | reg | nRMSE | 0.1082 | 0.1086 | 0.1130 | 481/475/497 | 0.81 / 0.70 |
| elevators | reg | nRMSE | 0.3031 | 0.3041 | 0.3063 | 959/840/761 | 0.58 / 0.64 |
| pol | reg | nRMSE | 0.0945 | 0.0962 | 0.0981 | 1171/2169/1542 | 0.64 / 1.33 |
| houses | reg | nRMSE | 0.3803 | 0.3806 | 0.3777 | 1126/1030/972 | 0.55 / 0.47 |
| sulfur | reg | nRMSE | 0.4233 | 0.3891 | 0.3849 | 909/893/510 | 0.39 / 0.30 |
| california | clf | log-loss | 0.2043 | 0.2041 | 0.2097 | 570/638/554 | 0.76 / 0.75 |
| electricity | clf | log-loss | 0.2490 | 0.2441 | 0.2416 | 1715/1837/2594 | 0.82 / 1.08 |
| MagicTelescope | clf | log-loss | 0.3128 | 0.3116 | 0.3077 | 274/317/258 | 0.80 / 0.61 |
