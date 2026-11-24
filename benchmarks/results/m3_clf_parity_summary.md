# M3 classification parity: miter C++ vs histree prototype (13 OpenML binary datasets x 5 folds)

Confirmatory comparison; pre-registration `benchmarks/results/m3_clf_parity_prereg.md`, SHA-256 `cef442c99cb67e283c38e813f88f008c481e7d6e4ee915d5c891ee414451891d` (frozen before any run; a 1-unit timing smoke test (MagicTelescope fold 0, 6 s for all three models) was run after freezing and before the full run, and its result was not used; the fold-count rule was not needed because the full 65-unit run took roughly 10-12 minutes wall-clock).

- Hardware: Apple M4, 10 cores, 16 GiB, `macOS-26.6.2-arm64-arm-64bit-Mach-O`; 3 parallel worker processes under `nice -n 10` (other benchmark agent running concurrently), 1 thread each (OMP/NUMBA threads = 1).
- Versions: miter 0.1.0.dev0 (installed from branch m5-performance, repo commit d1b4d12), numpy 2.5.5, numba 0.68.1, scipy 1.18.1, scikit-learn 1.9.1, openml 0.15.1, Python 3.13.1; histree repo HEAD 9320e45 (untracked extra file oblique_rl.py not used).
- Script: `benchmarks/check_m3_clf_parity.py` (raw per-unit JSON in the session scratchpad `/private/tmp/claude-501/-Users-kacperpiotrowski-Studia-Projekt-miter/1f6c6a51-2d6a-4aff-a079-ed693b1ffe42/scratchpad/res`, not committed).
- Settings (both): lr 0.05, 31 leaves, min_data_in_leaf 20, l2 1.0, 255 bins, m = 2*ceil(sqrt(p)), k in {2,3}, ES 50 on validation log-loss, max 5000 trees, 1 thread, seed = fold; outer OpenML split fold f, inner `train_test_split(test_size=0.2, random_state=0)` (not stratified); test log-loss clipped to [1e-15, 1-1e-15].
- Rel diff = 100*(miter - histree)/histree on the per-dataset mean over 5 folds; negative = miter better.
- Prototype: `ObliqueGBInt` is regression-only (L2 loop), so PRIMARY = float `histree.gb.oblique.ObliqueGBRegressor(objective="binary", selector="R", m=2*m0, bin_sample=-3)` (B4 configuration, raw inputs, float projections, no quantization). SECONDARY (exploratory, not in H) = prototype integer kernel `build_tree_int` + Q16r inputs, driven by a logistic-loss loop written in the check script (not an existing prototype classifier; init = logit(mean), g = p - y, h = max(p(1-p), 1e-6)).
- No dataset contains NaN. "trees" columns are the mean best iteration.

## Results

### PRIMARY: histree float oblique.py R_uni (binary, bin_sample=-3)

| dataset | miter log-loss | histree log-loss | rel diff | per-fold rel diff | best iter miter / histree (mean over folds) |
|---|---|---|---|---|---|
| eye_movements | 0.6093 | 0.6112 | -0.31% | -1.06, -0.13, -0.37, -0.04, +0.07 | 398 / 463 |
| heloc | 0.5530 | 0.5531 | -0.01% | -0.17, +0.16, +0.03, -0.40, +0.31 | 76 / 79 |
| pol | 0.0375 | 0.0366 | +2.26% | -5.02, -3.96, +5.16, +18.89, +4.24 | 388 / 384 |
| bank-marketing | 0.4156 | 0.4157 | -0.03% | +0.12, -0.27, +0.88, +0.17, -1.06 | 105 / 105 |
| default-of-credit-card-clients | 0.5502 | 0.5512 | -0.18% | -0.17, +0.26, -0.53, +0.14, -0.61 | 82 / 70 |
| MagicTelescope | 0.2942 | 0.2936 | +0.20% | +0.17, -0.45, -0.44, +1.36, +0.34 | 287 / 288 |
| house_16H | 0.2744 | 0.2724 | +0.75% | +0.28, +1.72, +0.57, +1.41, -0.17 | 244 / 221 |
| credit | 0.4788 | 0.4759 | +0.60% | +0.80, +0.49, +0.80, +0.54, +0.40 | 91 / 87 |
| california | 0.2201 | 0.2191 | +0.42% | +0.45, -0.01, +0.45, +0.46, +0.74 | 545 / 549 |
| electricity | 0.2298 | 0.2300 | -0.08% | +1.31, -0.49, -1.84, +0.56, +0.08 | 2442 / 2410 |
| jannis | 0.4407 | 0.4429 | -0.50% | -0.41, -0.52, -0.84, -0.59, -0.12 | 795 / 703 |
| Diabetes130US | 0.6585 | 0.6584 | +0.01% | +0.03, +0.00, +0.11, -0.06, -0.02 | 93 / 82 |
| MiniBooNE | 0.1489 | 0.1486 | +0.16% | +0.74, -0.30, -0.17, +0.75, -0.23 | 842 / 729 |

- mean +0.255%, **median +0.012%**, miter wins 6/13, Wilcoxon p = 0.6848; |median| <= 1% : True
- raw: mean np.float64(0.2547651195758091), median 0.011642197339189088, p np.float64(0.684814453125)

### SECONDARY (exploratory): histree int kernel build_tree_int + logistic loop, Q16r

| dataset | miter log-loss | histree log-loss | rel diff | per-fold rel diff | best iter miter / histree (mean over folds) |
|---|---|---|---|---|---|
| eye_movements | 0.6093 | 0.6115 | -0.35% | -1.00, +1.07, +0.18, -2.30, +0.26 | 398 / 429 |
| heloc | 0.5530 | 0.5510 | +0.36% | +0.35, +0.86, +0.27, -0.18, +0.49 | 76 / 77 |
| pol | 0.0375 | 0.0367 | +2.18% | -6.73, +1.35, +2.27, +15.63, +4.13 | 388 / 381 |
| bank-marketing | 0.4156 | 0.4158 | -0.07% | -0.60, -0.30, +1.03, -0.04, -0.45 | 105 / 104 |
| default-of-credit-card-clients | 0.5502 | 0.5515 | -0.24% | -0.27, +0.13, -0.66, -0.03, -0.34 | 82 / 84 |
| MagicTelescope | 0.2942 | 0.2931 | +0.36% | +1.09, +0.56, -1.09, -0.95, +2.34 | 287 / 291 |
| house_16H | 0.2744 | 0.2741 | +0.12% | -1.01, +1.13, +0.85, +0.83, -1.13 | 244 / 242 |
| credit | 0.4788 | 0.4786 | +0.05% | -0.17, +0.12, +0.18, +0.19, -0.08 | 91 / 85 |
| california | 0.2201 | 0.2219 | -0.83% | -1.38, -0.56, +0.21, -1.96, -0.33 | 545 / 494 |
| electricity | 0.2298 | 0.2300 | -0.09% | +1.03, -0.65, -1.84, +1.68, -0.69 | 2442 / 2490 |
| jannis | 0.4407 | 0.4412 | -0.12% | -0.24, -0.29, +0.22, -0.27, -0.02 | 795 / 747 |
| Diabetes130US | 0.6585 | 0.6584 | +0.01% | +0.03, +0.01, +0.04, -0.03, -0.01 | 93 / 85 |
| MiniBooNE | 0.1489 | 0.1486 | +0.16% | -0.12, -1.09, +0.25, +1.46, +0.36 | 842 / 837 |

- mean +0.118%, **median +0.008%**, miter wins 6/13, Wilcoxon p = 0.8926; |median| <= 1% : True
- raw: mean np.float64(0.11845017741778203), median 0.007711812133130089, p np.float64(0.892578125)

Total training seconds (sum over units; run under 3 parallel processes + other load, approximate): miter 242, proto_float 461, proto_int 471

## Verdict

H (|median rel diff| <= 1%) is **supported** for the primary comparison (median +0.012%, 5 folds, 13 datasets) and also holds for the exploratory integer-kernel comparison (median +0.008%). Means are +0.255% / +0.118%; Wilcoxon p = 0.68 / 0.89 (no detectable systematic difference); miter wins 6/13 in both.

## Notable discrepancies

- **pol** is the only dataset beyond 1% (primary +2.26%, secondary +2.18%): very noisy (per-fold -5.0% ... +18.9% primary; log-loss 0.023-0.047), driven by fold 3 (miter 0.02784 vs 0.02342). Single-run variance of this dataset is large; no seed repeats were done, so whether it is noise or a small systematic difference is unresolved.
- house_16H +0.75% and credit +0.60% (primary) are consistently positive across folds (credit 5/5 folds positive, house_16H 4/5), but shrink to +0.12% / +0.05% against the integer-kernel prototype, so the gap is vs the float prototype path rather than vs Q16r int.
- jannis -0.50% (5/5 folds negative, primary) and california -0.83% (secondary, 4/5 folds negative) are the largest in miter's favour.
- Early-stopping iterations are close for most datasets; miter stops later than the primary prototype on jannis (795 vs 703) and MiniBooNE (842 vs 729).

## Not measured

- AUC, accuracy or calibration; only log-loss.
- Seed variance: one seed (= fold) per unit, no repeated seeds, so per-dataset differences below a few percent cannot be called signal (pol shows fold-level swings of +-5-19%).
- Exact bitwise/algorithmic equivalence of RNG streams (the two implementations use different random projections; only statistical parity is tested).
- Timing: training seconds are recorded in the raw JSON but were taken under 3 parallel processes plus another agent's load and are approximate; no speed claim is made.
- Stratified inner split (prototype B4 used stratified; this check uses the Part C non-stratified split), categorical/NaN data, regression, other hyperparameters.
- Prototype classification results from research/results/b4 were not used for the comparison (cached reference only; all prototype numbers here are from a live run).
