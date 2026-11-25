# Brazilian_houses residual gap: one-component swaps in histree T7 toward miter

## Provenance
- hardware/OS: Apple M4 (10 cores), macOS-26.6.2-arm64-arm-64bit-Mach-O, python 3.13.1
- numpy 2.5.5, scipy 1.18.1, sklearn 1.9.1, numba 0.68.1, miter 0.1.0.dev0
- miter commit ff9d80f (with axis-binning fix 876f133); histree commit 9320e45
- 1 thread per process, <=2 processes in parallel, nice -n 10, another benchmark running concurrently (timings approximate)
- elapsed (sum of raw files, s): combo_f32+q16m+rng1 272, f32 124, miter 69, orig 132, q16m 132, rng1 120
- orig/miter s=0..4 are the raw files of m3_parity_brazil_pol; s=5..9 were run by this script (m3_residual_*_s5-10.json)

## s=0..4 (25 pairs)

| variant | mean nRMSE | median nRMSE | mean trees | vs orig: mean% / median% / wins / p | miter vs variant: mean% / median% / miter wins / p |
|---|---|---|---|---|---|
| orig | 0.045673 | 0.031857 | 1608 | - | +5.04 / +3.28 / 10/25 / 0.0219 |
| miter | 0.047283 | 0.034460 | 1582 | +5.04 / +3.28 / 10/25 / 0.0219 | - |
| combo_f32+q16m+rng1 | 0.046150 | 0.031512 | 1514 | +1.36 / +1.19 / 10/25 / 0.381 | +3.93 / +4.20 / 9/25 / 0.0367 |
| f32 | 0.045554 | 0.031993 | 1576 | -0.37 / -0.00 / 13/25 / 0.596 | +5.52 / +2.97 / 10/25 / 0.0236 |
| q16m | 0.045587 | 0.031331 | 1877 | -0.43 / +0.03 / 12/25 / 0.979 | +5.56 / +4.17 / 10/25 / 0.0275 |
| rng1 | 0.046075 | 0.031573 | 1568 | +0.50 / -0.19 / 13/25 / 0.937 | +4.86 / +3.41 / 8/25 / 0.0588 |

(wins in 'vs orig' = variant has lower nRMSE than orig; wins in 'miter vs variant' = miter lower. Rel diff = (a-b)/b with a = first-named.)

| fold | miter mean | orig mean | rel diff of means % | paired rel diff mean % +- se |
|---|---|---|---|---|
| 0 | 0.112730 | 0.111753 | +0.87 | +0.93 +- 1.38 |
| 1 | 0.034349 | 0.033385 | +2.89 | +2.96 +- 1.74 |
| 2 | 0.033929 | 0.031602 | +7.36 | +7.62 +- 3.47 |
| 3 | 0.033686 | 0.029959 | +12.44 | +13.32 +- 6.24 |
| 4 | 0.021721 | 0.021664 | +0.26 | +0.36 +- 2.06 |
- miter vs orig seed-to-seed std of rel diff: 8.65 pct points; standard error of mean rel diff: 1.73

## s=0..9 (50 pairs)

| variant | mean nRMSE | median nRMSE | mean trees | vs orig: mean% / median% / wins / p | miter vs variant: mean% / median% / miter wins / p |
|---|---|---|---|---|---|
| orig | 0.046607 | 0.033006 | 1670 | - | +3.48 / +1.78 / 19/50 / 0.0141 |
| miter | 0.047851 | 0.033981 | 1510 | +3.48 / +1.78 / 19/50 / 0.0141 | - |
| combo_f32+q16m+rng1 | 0.046407 | 0.031877 | 1761 | -0.15 / +0.73 / 22/50 / 0.878 | +3.96 / +3.86 / 16/50 / 0.00261 |

(wins in 'vs orig' = variant has lower nRMSE than orig; wins in 'miter vs variant' = miter lower. Rel diff = (a-b)/b with a = first-named.)

| fold | miter mean | orig mean | rel diff of means % | paired rel diff mean % +- se |
|---|---|---|---|---|
| 0 | 0.114758 | 0.112678 | +1.85 | +1.88 +- 0.97 |
| 1 | 0.034721 | 0.034586 | +0.39 | +0.82 +- 2.55 |
| 2 | 0.034382 | 0.032382 | +6.18 | +6.50 +- 3.55 |
| 3 | 0.033948 | 0.031821 | +6.69 | +8.68 +- 5.91 |
| 4 | 0.021449 | 0.021570 | -0.56 | -0.49 +- 1.31 |
- miter vs orig seed-to-seed std of rel diff: 10.81 pct points; standard error of mean rel diff: 1.53

## Implementation differences found by reading code (miter C++ vs ObliqueGBInt)

(P = histree/src/histree/gb/oblique_int.py, M = miter/src)

| # | difference | miter | prototype | tested? |
|---|---|---|---|---|
| a | gradient/hessian precision | float32 g,h: core/booster.cpp:128-129, core/objective.cpp:161 (accumulated in double: core/oblique.cpp:166,352) | float64: P:333 | yes, `f32` |
| b | oblique 64-bin range | `range = pmax - pmin + 1`: core/oblique.cpp:336, 589-590; bin index include/oblique.hpp:25-29 | `rng = smax - smin`, max element clamped into bin 63: P:37,46-48; `rng<=0` skip P:40 | yes, `rng1` |
| c | Q16 code scale/rounding | 65534, floor(v+0.5), clip [0,65534] (65535 = NaN): data/quantizer.hpp:9,24-26 | 65535, rint, clip [0,65535] (check_m3_parity_brazil_pol.py::q16r_proto) | yes, `q16m` |
| d | RNG | counter-based per (seed, tree, node, cand): core/oblique.cpp:515-516, sample_k_distinct + next_normal | numba np.random seeded per tree: P:150, 88, 93, 96 | cannot be matched |
| e | m | m_mult*ceil(sqrt(p_elig)), p_elig excludes constant features: core/oblique.cpp:445-452 | m = 2*ceil(sqrt(p)) all features | no effect here (8 features, 0 constant, no NaN in Brazilian_houses) |
| f | early stopping | vloss < best-1e-12, patience 50, truncate to best_iter: core/booster.cpp:174-185 | same rule: P:339-344 | identical |
| g | init score, leaf value, lr | mean (compute_init_score), lr*(-G/(H+lam)): core/tree_learner.cpp:596 | mean P:330, same leaf formula P:283 | identical up to (a) |
| h | min_sum_hessian | 1e-3 (params.hpp:18), irrelevant for h=1 | none | no effect (hessian = 1, min_data 20) |
| i | min_gain | gain > 0 per bin: core/oblique.cpp:44 | best > mingain: P:182 | identical |
| j | NaN handling | sentinel, nan_go_left: core/oblique.cpp:60-118 | none | no NaN in this dataset |
| k | oblique vs axis choice | oblique if gain > axis gain: core/tree_learner.cpp:338 | og > ag: P:131 | identical |

Plausibility ranking of remaining untested sources: d (RNG stream: different random candidate sets, cannot be matched) >> a, b, c (measured, no effect) > everything else (identical or inert on this dataset).

## Verdict

- No single swap (f32, rng1, q16m) or the combination of all three moved the prototype significantly (|mean change| <= 0.5%, p >= 0.59 for singles; combo +1.36% on 25 pairs, p=0.38; -0.15% on 50 pairs, p=0.88). Miter stays worse than every variant on s=0..4 (+4.9% to +5.6%, p 0.024-0.059; one variant, rng1, at p=0.059).
- The combo (50 pairs) is still +3.96% worse than miter-vs-combo with p=0.0026, so the swaps did NOT bring the prototype down to miter's level. Cause not found among a, b, c.
- Adding 5 seeds (s=5..9) shrank the miter-vs-orig gap from +5.04% (p=0.022) to +3.48% (p=0.014, 19/50 miter wins); the gap is concentrated in folds 2 and 3 (+6.2%, +6.7% of fold means), folds 0, 1, 4 are within +-2%. Seed-to-seed std of the paired rel diff is 8.65 (25 pairs) / 10.81 (50 pairs) pct points vs 2.70 on pol and a 16-dataset (5-fold, seed=fold) max |diff| of 3.34%, so Brazilian_houses is an unusually noisy dataset (small test-set rmse scale on fold 3/4, tree counts vary 700-3800), but the residual is still significant at 50 pairs, so "pure RNG noise" is not established (the pairs share folds, so seeds are not independent across the 5 folds; p-values treat 50 pairs as independent and are optimistic).
- Not measured: variants (a), (b), (c) individually on s=5..9; swapping miter toward the prototype (e.g. a C++ build with float64 g/h or the prototype range convention); RNG-stream swap (feeding identical candidate sets to both); other datasets with the swaps; timing.
- Proposed fix for miter: none justified by these data. Cheapest next causal test would be injecting identical (features, weights) candidate sequences into both implementations (e.g. from a Python-side RNG exposed to a test hook) which needs a src/ change and was out of scope.
