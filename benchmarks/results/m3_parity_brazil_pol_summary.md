# M3 parity check: Brazilian_houses and pol, miter C++ vs live histree T7 Q16r_int

## Provenance
- hardware/OS: macOS-26.6.2-arm64-arm-64bit-Mach-O / arm, python 3.13.1
- numpy 2.5.5, scipy 1.18.1, sklearn 1.9.1, numba 0.68.1, miter 0.1.0.dev0
- miter source commit: d1b4d12 (branch m5-performance); histree: 9320e45
- seeds: seed = fold + 100*s, s=0..4; both methods get the same random_state, same outer folds, inner split train_test_split(random_state=0)
- histree gets Q16r codes computed as in t7_int.py (np.quantile 1e-4/0.9999 of inner-train part, rint, clip); miter quantizes internally from raw X

## Brazilian_houses
- miter elapsed 79s, histree elapsed 170s (under load, approximate)

| fold | seed | miter nRMSE | histree nRMSE | rel diff % | miter trees | histree trees |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.116315 | 0.111538 | +4.28 | 1069 | 1463 |
| 0 | 100 | 0.120920 | 0.107830 | +12.14 | 841 | 3675 |
| 0 | 200 | 0.122495 | 0.114570 | +6.92 | 714 | 1514 |
| 0 | 300 | 0.111803 | 0.110129 | +1.52 | 1420 | 2572 |
| 0 | 400 | 0.122880 | 0.114700 | +7.13 | 946 | 2996 |
| 1 | 1 | 0.034240 | 0.032685 | +4.76 | 1934 | 1075 |
| 1 | 101 | 0.037958 | 0.034135 | +11.20 | 1315 | 1307 |
| 1 | 201 | 0.037580 | 0.032842 | +14.43 | 1460 | 1024 |
| 1 | 301 | 0.039470 | 0.035407 | +11.48 | 940 | 2055 |
| 1 | 401 | 0.035952 | 0.031857 | +12.85 | 1352 | 1309 |
| 2 | 2 | 0.044741 | 0.033412 | +33.91 | 389 | 1305 |
| 2 | 102 | 0.049362 | 0.030580 | +61.42 | 233 | 1720 |
| 2 | 202 | 0.044292 | 0.030359 | +45.89 | 290 | 1458 |
| 2 | 302 | 0.057695 | 0.030770 | +87.51 | 188 | 2029 |
| 2 | 402 | 0.055730 | 0.032893 | +69.43 | 188 | 703 |
| 3 | 3 | 0.037984 | 0.024299 | +56.32 | 1356 | 2270 |
| 3 | 103 | 0.031906 | 0.031518 | +1.23 | 2121 | 1862 |
| 3 | 203 | 0.036417 | 0.029783 | +22.28 | 1954 | 1558 |
| 3 | 303 | 0.039089 | 0.028002 | +39.59 | 1305 | 1805 |
| 3 | 403 | 0.036984 | 0.036193 | +2.18 | 1564 | 1230 |
| 4 | 4 | 0.019218 | 0.021343 | -9.95 | 1188 | 942 |
| 4 | 104 | 0.019338 | 0.020499 | -5.66 | 1580 | 1048 |
| 4 | 204 | 0.020955 | 0.022170 | -5.48 | 1046 | 1316 |
| 4 | 304 | 0.021105 | 0.022316 | -5.43 | 1155 | 923 |
| 4 | 404 | 0.020403 | 0.021991 | -7.22 | 1631 | 1036 |

| stat | miter | histree |
|---|---|---|
| mean | 0.052593 | 0.045673 |
| median | 0.037984 | 0.031857 |
| seed std (ddof=1, within fold, mean over folds) | 0.003309 | 0.002183 |

| per-fold mean over seeds | miter | histree | rel diff % |
|---|---|---|---|
| 0 | 0.118882 | 0.111753 | +6.38 |
| 1 | 0.037040 | 0.033385 | +10.95 |
| 2 | 0.050364 | 0.031602 | +59.37 |
| 3 | 0.036476 | 0.029959 | +21.75 |
| 4 | 0.020204 | 0.021664 | -6.74 |

- paired rel diff: mean +18.91%, median +11.20%; mean-of-nRMSE ratio +15.15%
- miter wins (lower nRMSE): 5/25; Wilcoxon p = 8.804e-05
- seed-to-seed std of rel diff: 26.59 pct points

| fold | cached T7 clean|Q16r_int | live histree (seed=fold) | abs diff |
|---|---|---|---|
| 0 | 0.111538 | 0.111538 | 0 |
| 1 | 0.032685 | 0.032685 | 0 |
| 2 | 0.033412 | 0.033412 | 0 |
| 3 | 0.024299 | 0.024299 | 0 |
| 4 | 0.021343 | 0.021343 | 0 |

## pol
- miter elapsed 186s, histree elapsed 250s (under load, approximate)

| fold | seed | miter nRMSE | histree nRMSE | rel diff % | miter trees | histree trees |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.088945 | 0.086935 | +2.31 | 1300 | 1623 |
| 0 | 100 | 0.087401 | 0.088452 | -1.19 | 2427 | 1735 |
| 0 | 200 | 0.086065 | 0.088859 | -3.14 | 1549 | 1442 |
| 0 | 300 | 0.086813 | 0.087774 | -1.10 | 1528 | 1231 |
| 0 | 400 | 0.083663 | 0.085842 | -2.54 | 1275 | 1152 |
| 1 | 1 | 0.106212 | 0.103819 | +2.31 | 1608 | 1058 |
| 1 | 101 | 0.110843 | 0.105106 | +5.46 | 606 | 1471 |
| 1 | 201 | 0.107026 | 0.103784 | +3.12 | 1089 | 1366 |
| 1 | 301 | 0.103127 | 0.106195 | -2.89 | 1638 | 1410 |
| 1 | 401 | 0.105725 | 0.110164 | -4.03 | 1463 | 636 |
| 2 | 2 | 0.104674 | 0.101603 | +3.02 | 1274 | 1363 |
| 2 | 102 | 0.100589 | 0.103899 | -3.19 | 1443 | 1790 |
| 2 | 202 | 0.105173 | 0.104946 | +0.22 | 969 | 1320 |
| 2 | 302 | 0.102938 | 0.105633 | -2.55 | 1505 | 1115 |
| 2 | 402 | 0.104902 | 0.099148 | +5.80 | 1833 | 1265 |
| 3 | 3 | 0.102050 | 0.097261 | +4.92 | 2013 | 2192 |
| 3 | 103 | 0.101088 | 0.099325 | +1.77 | 1844 | 1849 |
| 3 | 203 | 0.100786 | 0.101829 | -1.02 | 1759 | 1826 |
| 3 | 303 | 0.100741 | 0.099894 | +0.85 | 1916 | 1834 |
| 3 | 403 | 0.102110 | 0.099823 | +2.29 | 1636 | 1797 |
| 4 | 4 | 0.096836 | 0.092837 | +4.31 | 2227 | 1624 |
| 4 | 104 | 0.092373 | 0.093125 | -0.81 | 2131 | 1705 |
| 4 | 204 | 0.092793 | 0.094487 | -1.79 | 1912 | 1681 |
| 4 | 304 | 0.093351 | 0.094593 | -1.31 | 1512 | 1904 |
| 4 | 404 | 0.097343 | 0.093190 | +4.46 | 1879 | 1828 |

| stat | miter | histree |
|---|---|---|
| mean | 0.098543 | 0.097941 |
| median | 0.100786 | 0.099325 |
| seed std (ddof=1, within fold, mean over folds) | 0.001939 | 0.001792 |

| per-fold mean over seeds | miter | histree | rel diff % |
|---|---|---|---|
| 0 | 0.086577 | 0.087572 | -1.14 |
| 1 | 0.106587 | 0.105814 | +0.73 |
| 2 | 0.103655 | 0.103046 | +0.59 |
| 3 | 0.101355 | 0.099626 | +1.74 |
| 4 | 0.094539 | 0.093646 | +0.95 |

- paired rel diff: mean +0.61%, median +0.22%; mean-of-nRMSE ratio +0.61%
- miter wins (lower nRMSE): 12/25; Wilcoxon p = 0.381
- seed-to-seed std of rel diff: 3.04 pct points

| fold | cached T7 clean|Q16r_int | live histree (seed=fold) | abs diff |
|---|---|---|---|
| 0 | 0.086935 | 0.086935 | 0 |
| 1 | 0.103819 | 0.103819 | 0 |
| 2 | 0.101603 | 0.101603 | 0 |
| 3 | 0.097261 | 0.097261 | 0 |
| 4 | 0.092837 | 0.092837 | 0 |


## Provenance addendum
- Hardware: Apple M4 (10 cores), macOS (Darwin 25.6.0); lightgbm 4.7.0; miter 0.1.0.dev0 installed from branch m5-performance (commit d1b4d12 at task start; repo HEAD was c03d299 when this summary was finalized, installed wheel was not rebuilt); histree commit 9320e45.
- All runs 1 thread, `nice -n 10`, with another benchmark agent running: timings are approximate. Elapsed per job is shown above (miter 79s/186s, histree 170s/250s for 25 runs on Brazilian_houses/pol).
- Scripts: `benchmarks/check_m3_parity_brazil_pol.py` (main), `benchmarks/check_m3_parity_brazil_rootcause.py` (ablation). Raw: `benchmarks/results/raw/m3_parity_*.json`, `m3_rootcause_Brazilian_houses.json` (gitignored).

## Verdict
- pol: miter vs live histree T7 Q16r_int = +0.61% mean, +0.22% median, 12/25 wins, Wilcoxon p = 0.381. Within seed noise (std of paired rel diff 3.04 pp). The +3.37% in m3_oblique_summary Part B came from a single seed per fold (random_state = fold).
- Brazilian_houses: miter is WORSE: +18.91% mean, +11.20% median, 5/25 wins, p = 8.8e-05. Not seed noise (seed std of nRMSE 0.0033 miter / 0.0022 histree; per-fold means differ by up to +59% on fold 2). The Part B figure (+13.09%) used seed = fold only.
- Cached T7 numbers ARE reproduced exactly by the live histree runs (seed = fold): abs diff 0 on all 5 folds for both datasets (tables above). Histree fold 0 = 0.1115 is a genuinely hard fold (miter also 0.1189 mean): fold 0 does not drive the gap (+6.4% there; fold 2 does: +59%).

## Root cause (Brazilian_houses): axis bin edges, not Q16r, RNG, or the oblique evaluator
Checked step by step (scratch scripts, fold 2 unless noted):
1. Q16r codes: lo is bit-identical to the prototype (np.quantile); hi differs because miter scales to 65534 (code 65535 reserved for NaN; `src/data/quantizer.hpp:9`) and uses floor(v+0.5) vs rint. Codes differ by at most 1 in 5-6% of cells (Brazilian_houses 4873-9575 of 85536; pol ~22000 of 390000). Feeding histree miter's codes instead: nRMSE 0.0325/0.0313/0.0302 vs 0.0334/0.0306/0.0304 with proto codes (seeds 2,102,202; trees 1356-1710), so codes are NOT the cause.
2. Quantile thresholds: lo identical, hi equivalent up to the 65534/65535 scale (relative 1.5e-5).
3. Tree building: RNG streams differ (numba np.random vs counter-based SplitMix64), so cuts cannot match one-to-one. Distributions do: 1-tree stump, 40 seeds, train MSE miter 0.234833 vs histree 0.235197; 31-leaf tree 0.024284 vs 0.024384; validation MSE after one 31-leaf tree 0.016730 vs 0.016812; oblique fraction of nodes 0.129 vs 0.136. Train loss (mean of 12 seeds) matches through 300 trees (100 trees: 0.004945 vs 0.005145).
4. But validation loss diverges after ~50 trees at equal train loss (12 seeds, 100 trees: 0.001732 miter vs 0.001564 histree; 200 trees: 0.001573 vs 0.001161). Same with axis-only models (miter m_mult=0 vs histree AX): train 0.004187 vs 0.005164, val 0.004674 vs 0.002446 at 200 trees: miter fits the training tail better and generalizes worse.
5. Axis bin resolution differs. Brazilian_houses fold 2, bins per feature (miter / histree): f0 233/150, f1 10/8, f2 10/8, f3 11/10, f4 229/116, f5 176/141, f6 255/180, f7 207/114. Miter `src/data/axis_binner.cpp:50-53`: if distinct codes <= max_bin, one bin per distinct code (no min-data-in-bin), otherwise greedy equal-count over distinct codes. Histree `histree/src/histree/gb/oblique.py:1111-1112`: `np.unique(np.quantile(X[:, j], linspace(0,1,256)[1:-1]))` collapses rare tail values into one bin, so axis cuts cannot isolate a few extreme rows (a regularizer). Brazilian_houses has a heavy-tailed target and rare extreme feature values.
6. Causal test (script `check_m3_parity_brazil_rootcause.py`, 25 paired runs): histree T7 Q16r_int with miter's AxisBinner edges injected (everything else unchanged):
   - mean nRMSE: histree own edges 0.045673, histree + miter edges 0.052339, miter 0.052593; medians 0.031857 / 0.040065 / 0.037984.
   - histree + miter edges vs own edges: +17.43% mean, +9.81% median, 5/25 wins, p = 1.83e-05.
   - miter vs histree + miter edges: +1.48% mean, -1.24% median, 14/25 wins, p = 1 (indistinguishable).
   - Per-fold mean (own / miter edges / miter): f0 0.11175/0.11917/0.11888; f1 0.03339/0.03664/0.03704; f2 0.03160/0.04852/0.05036; f3 0.02996/0.03685/0.03648; f4 0.02166/0.02051/0.02020 (on fold 4 finer bins help: miter better than histree by 6.7%).
   So the Brazilian_houses gap is explained by the axis binner; miter's oblique code is not implicated.

## Not measured / caveats
- Only Brazilian_houses and pol were run; whether the axis-binner effect explains the other 14 datasets of Part B was not tested (it could go either way, as fold 4 shows).
- No per-node cut-by-cut identity check was possible (different RNG); equivalence was shown only in distribution (stumps, one tree, train-loss curves).
- The ablation injects miter's edges into histree; the reverse (histree-style coarse edges inside miter) was not run because src/ must not be modified. Hence causality is shown in one direction.
- Miter's float32 gradient storage and 64-bin range = pmax-pmin+1 (histree: pmax-pmin with clamp) were noticed but not isolated; their effect is below what the ablation resolves.
- Whether coarse binning is "better" in general or only on heavy-tailed data is untested; this is a design decision for the author (miter's binning follows LightGBM-style exact distinct bins).
