# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds

Relative error per dataset (mean over folds), negative = better; time = total fit seconds (one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.

| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2 | 32 | 2 | 22 | +0.000% | +0.318% | +4.11% | 0.0654 | 0.76x | 0.97x |

## Per dataset (mean metric over folds)

| dataset | BASE | E2 |
|---|---:|---:|
| GesturePhaseSegmentationProcessed | 0.83624 | 0.85757 |
| Moneyball | 0.24504 | 0.24504 |
| analcatdata_dmft | 1.7785 | 1.7785 |
| car | 0.021095 | 0.021095 |
| cars | 0.22756 | 0.22756 |
| cpu_activity | 0.11864 | 0.1194 |
| credit-approval | 0.32095 | 0.32095 |
| cylinder-bands | 0.45334 | 0.45334 |
| dna | 0.11799 | 0.11799 |
| dresses-sales | 0.67929 | 0.67929 |
| energy_efficiency | 0.03884 | 0.03884 |
| eucalyptus | 0.75601 | 0.75601 |
| first-order-theorem-proving | 1.1219 | 1.1219 |
| forest_fires | 1.0133 | 1.0133 |
| fps_benchmark | 0.0056929 | 0.0054541 |
| geographical_origin_of_music | 0.84099 | 0.84099 |
| grid_stability | 0.17777 | 0.18 |
| kin8nm | 0.31555 | 0.32327 |
| kings_county | 0.30989 | 0.30805 |
| naval_propulsion_plant | 0.033063 | 0.034421 |
| nomao | 0.075891 | 0.076084 |
| ozone-level-8hr | 0.14338 | 0.14338 |
| sarcos | 0.12184 | 0.12495 |
| segment | 0.17345 | 0.17345 |
| sick | 0.029093 | 0.029093 |
| socmob | 0.55604 | 0.55604 |
| solar_flare | 0.95105 | 0.95105 |
| space_ga | 0.49826 | 0.49826 |
| spambase | 0.13061 | 0.13061 |
| student_performance_por | 0.82769 | 0.82769 |
| vowel | 0.20329 | 0.20329 |
| wave_energy | 0.39568 | 0.40025 |

## Verdict (pre-registered rule, `exp_speedups_e2_prereg.md`)

**PASS on all criteria, so `oblique_sample_rows=65536` becomes the default.** This run uses the
final code: the parallel, chunk-ordered exact refit, commit `766fd0f`.

- **A, held-out (32 datasets, stress setting 4096):** median +0.000% (≤ +0.25%), one-sided
  p_worse 0.065 (≥ 0.05), worst +4.11% (≤ +5%). Total time 0.76×.
- **B, TabArena (51 tasks, stress setting 4096):** median +0.000%, p_worse 0.106
  (`tabarena_speedup_e2_51_summary.md`).
- **C, speed (`exp_speedups_e2_speed_summary.md`):** synthetic 500k × 50, one thread 0.42× (≤ 0.7×).
  Also 0.50× on 10 threads, and 0.40× / 0.42× on covertype.

The sequential first version (`*_e2_seq_*`) gave identical quality numbers. Its speed on 10 threads
was 0.59× / 0.69×, which is why the refit was parallelised. The stress setting lost on APSFailure
(+8.9%) and SDSS17 (+3.3%) on TabArena; the adopted 65,536-row sample is 16× larger.
