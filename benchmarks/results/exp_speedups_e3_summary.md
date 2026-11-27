# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds

Relative error per dataset (mean over folds), negative = better; time = total fit seconds (one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.

| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 | 32 | 1 | 31 | +0.000% | -0.000% | +0.00% | 1.0000 | 0.92x | 0.94x |

## Per dataset (mean metric over folds)

| dataset | BASE | E3 |
|---|---:|---:|
| GesturePhaseSegmentationProcessed | 0.83624 | 0.83624 |
| Moneyball | 0.24504 | 0.24504 |
| analcatdata_dmft | 1.7785 | 1.7785 |
| car | 0.021095 | 0.021095 |
| cars | 0.22756 | 0.22756 |
| cpu_activity | 0.11864 | 0.11864 |
| credit-approval | 0.32095 | 0.32095 |
| cylinder-bands | 0.45334 | 0.45334 |
| dna | 0.11799 | 0.11799 |
| dresses-sales | 0.67929 | 0.67929 |
| energy_efficiency | 0.03884 | 0.03884 |
| eucalyptus | 0.75601 | 0.75601 |
| first-order-theorem-proving | 1.1219 | 1.1219 |
| forest_fires | 1.0133 | 1.0133 |
| fps_benchmark | 0.0056929 | 0.0056929 |
| geographical_origin_of_music | 0.84099 | 0.84099 |
| grid_stability | 0.17777 | 0.17777 |
| kin8nm | 0.31555 | 0.31555 |
| kings_county | 0.30989 | 0.30989 |
| naval_propulsion_plant | 0.033063 | 0.033063 |
| nomao | 0.075891 | 0.075891 |
| ozone-level-8hr | 0.14338 | 0.14338 |
| sarcos | 0.12184 | 0.12184 |
| segment | 0.17345 | 0.17345 |
| sick | 0.029093 | 0.029093 |
| socmob | 0.55604 | 0.55604 |
| solar_flare | 0.95105 | 0.95105 |
| space_ga | 0.49826 | 0.49826 |
| spambase | 0.13061 | 0.13061 |
| student_performance_por | 0.82769 | 0.82769 |
| vowel | 0.20329 | 0.20329 |
| wave_energy | 0.39568 | 0.39568 |

## Verdict (pre-registered rule, `exp_speedups_e3_prereg.md`)

**PASS on all criteria, so `efb=True` becomes the default.**

- **A, held-out:** median +0.000%, worst +0.00%; 160/160 folds bit-identical to BASE.
- **B, TabArena** (`tabarena_speedup_e3_51_summary.md`): 51/51 tasks identical to DEF2.
  AutoGluon keeps categories as categories, so there are few sparse numeric columns.
- **C, speed** (`exp_speedups_e3_speed_summary.md`): covertype one thread 22.53 s → 10.51 s
  (0.47×), 10 threads 3.99 s → 2.58 s (0.65×).
- **No-op case:** synthetic 500k × 50 1.00× on both 1 and 10 threads (limit 1.05×).
