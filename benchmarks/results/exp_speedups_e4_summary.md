# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds

Relative error per dataset (mean over folds), negative = better; time = total fit seconds (one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.

| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E4A | 32 | 4 | 25 | +0.000% | +0.007% | +2.10% | 0.6562 | 0.92x | 0.94x |
| E4B | 32 | 10 | 6 | +0.011% | -0.008% | +1.96% | 0.1702 | 0.90x | 0.94x |

## Per dataset (mean metric over folds)

| dataset | BASE | E4A | E4B |
|---|---:|---:|---:|
| GesturePhaseSegmentationProcessed | 0.83624 | 0.83624 | 0.83765 |
| Moneyball | 0.24504 | 0.24504 | 0.24317 |
| analcatdata_dmft | 1.7785 | 1.7785 | 1.7785 |
| car | 0.021095 | 0.021095 | 0.021095 |
| cars | 0.22756 | 0.23234 | 0.23092 |
| cpu_activity | 0.11864 | 0.11864 | 0.11978 |
| credit-approval | 0.32095 | 0.32095 | 0.32221 |
| cylinder-bands | 0.45334 | 0.44967 | 0.45114 |
| dna | 0.11799 | 0.11799 | 0.11799 |
| dresses-sales | 0.67929 | 0.67929 | 0.67929 |
| energy_efficiency | 0.03884 | 0.038446 | 0.038684 |
| eucalyptus | 0.75601 | 0.75836 | 0.75995 |
| first-order-theorem-proving | 1.1219 | 1.1219 | 1.1222 |
| forest_fires | 1.0133 | 1.0133 | 1.0209 |
| fps_benchmark | 0.0056929 | 0.0056641 | 0.0053308 |
| geographical_origin_of_music | 0.84099 | 0.84099 | 0.83706 |
| grid_stability | 0.17777 | 0.17777 | 0.1782 |
| kin8nm | 0.31555 | 0.31555 | 0.31657 |
| kings_county | 0.30989 | 0.31033 | 0.30887 |
| naval_propulsion_plant | 0.033063 | 0.033063 | 0.032829 |
| nomao | 0.075891 | 0.075891 | 0.07614 |
| ozone-level-8hr | 0.14338 | 0.14338 | 0.14178 |
| sarcos | 0.12184 | 0.12184 | 0.12216 |
| segment | 0.17345 | 0.17345 | 0.17512 |
| sick | 0.029093 | 0.029093 | 0.02941 |
| socmob | 0.55604 | 0.55604 | 0.55604 |
| solar_flare | 0.95105 | 0.95105 | 0.95105 |
| space_ga | 0.49826 | 0.49826 | 0.49788 |
| spambase | 0.13061 | 0.13061 | 0.13051 |
| student_performance_por | 0.82769 | 0.82769 | 0.83391 |
| vowel | 0.20329 | 0.20329 | 0.20727 |
| wave_energy | 0.39568 | 0.39568 | 0.39699 |

## Verdict (pre-registered rule, exp_speedups_e4_prereg.md)

Both arms **fail**: neither improves the held-out median (E4A +0.000% with 25/32 ties, E4B +0.011%;
one-sided "better" p ≈ 0.34 and ≈ 0.83, far above the Holm thresholds). Neither is worse on
TabArena (E4A p_worse 0.14, E4B median −0.004%, p_worse 0.65), so both are harmless, but not gains.
`oblique_binary_penalty` and `oblique_gain_margin` stay options with default 0.
