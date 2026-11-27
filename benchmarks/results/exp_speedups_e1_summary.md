# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds

Relative error per dataset (mean over folds), negative = better; time = total fit seconds (one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.

| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 32 | 5 | 24 | +0.000% | +0.170% | +10.37% | 0.6797 | 0.93x | 0.99x |

## Per dataset (mean metric over folds)

| dataset | BASE | E1 |
|---|---:|---:|
| GesturePhaseSegmentationProcessed | 0.83624 | 0.83624 |
| Moneyball | 0.24504 | 0.24504 |
| analcatdata_dmft | 1.7785 | 1.7785 |
| car | 0.021095 | 0.021095 |
| cars | 0.22756 | 0.25116 |
| cpu_activity | 0.11864 | 0.11864 |
| credit-approval | 0.32095 | 0.32095 |
| cylinder-bands | 0.45334 | 0.44687 |
| dna | 0.11799 | 0.11799 |
| dresses-sales | 0.67929 | 0.67929 |
| energy_efficiency | 0.03884 | 0.038496 |
| eucalyptus | 0.75601 | 0.75652 |
| first-order-theorem-proving | 1.1219 | 1.1219 |
| forest_fires | 1.0133 | 1.0133 |
| fps_benchmark | 0.0056929 | 0.0054902 |
| geographical_origin_of_music | 0.84099 | 0.84099 |
| grid_stability | 0.17777 | 0.17777 |
| kin8nm | 0.31555 | 0.31555 |
| kings_county | 0.30989 | 0.31264 |
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

## Verdict (pre-registered rule, `exp_speedups_e1_prereg.md`)

**FAIL on criterion A:**
- The worst dataset, `cars`, is +10.37% (limit +5%).
- The median (+0.000%) and the one-sided Wilcoxon (p = 0.68) pass.
- Sanity check passed: BASE reproduced the `S6_M4` metrics of `exp_scaling_defaults` part 3
  exactly (160/160 folds).

`cars` has binary make/type columns. A 0/1 term inside a projection shifts the oblique threshold
separately for the two groups, which axis splits do not reproduce. The hypothesis that such terms
are redundant is therefore false. E1 helped where binary columns are noise (`fps_benchmark`
−3.56% at 0.73× time, `cylinder-bands` −1.43%).

The TabArena and speed measurements were stopped once A had failed, so criteria B and C were not
evaluated. `oblique_min_distinct` stays an option, default 0. The author also prefers binary
features to stay in projections by design.
