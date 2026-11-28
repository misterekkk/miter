# Speed-up experiments vs BASE (v1.0 default), 32 held-out OpenML datasets x 5 folds

Relative error per dataset (mean over folds), negative = better; time = total fit seconds (one thread). `p_worse`: one-sided Wilcoxon that the variant is worse.

| variant | n | wins | ties | median | mean | worst | p_worse | time (total) | time (median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E6_ACC | 32 | 16 | 5 | -0.000% | -0.331% | +4.56% | 0.8708 | 1.21x | 1.23x |
| E6_FAST | 32 | 7 | 6 | +0.620% | +0.858% | +12.37% | 0.0204 | 0.59x | 0.63x |

## Per dataset (mean metric over folds)

| dataset | BASE | E6_ACC | E6_FAST |
|---|---:|---:|---:|
| GesturePhaseSegmentationProcessed | 0.83624 | 0.82859 | 0.84186 |
| Moneyball | 0.24504 | 0.24191 | 0.24692 |
| analcatdata_dmft | 1.7785 | 1.7785 | 1.7785 |
| car | 0.021095 | 0.021095 | 0.021095 |
| cars | 0.22756 | 0.22865 | 0.23404 |
| cpu_activity | 0.11864 | 0.11633 | 0.12218 |
| credit-approval | 0.32095 | 0.3257 | 0.32388 |
| cylinder-bands | 0.45334 | 0.45592 | 0.43275 |
| dna | 0.11799 | 0.11799 | 0.11799 |
| dresses-sales | 0.67929 | 0.67929 | 0.67929 |
| energy_efficiency | 0.03884 | 0.038594 | 0.038374 |
| eucalyptus | 0.75601 | 0.76661 | 0.75245 |
| first-order-theorem-proving | 1.1219 | 1.1191 | 1.1234 |
| forest_fires | 1.0133 | 1.0195 | 1.0354 |
| fps_benchmark | 0.0056929 | 0.0059527 | 0.0048935 |
| geographical_origin_of_music | 0.84099 | 0.83898 | 0.84354 |
| grid_stability | 0.17777 | 0.17798 | 0.17913 |
| kin8nm | 0.31555 | 0.31152 | 0.33376 |
| kings_county | 0.30989 | 0.3101 | 0.30873 |
| naval_propulsion_plant | 0.033063 | 0.031047 | 0.037023 |
| nomao | 0.075891 | 0.075857 | 0.075349 |
| ozone-level-8hr | 0.14338 | 0.14222 | 0.14497 |
| sarcos | 0.12184 | 0.12045 | 0.12373 |
| segment | 0.17345 | 0.17104 | 0.17666 |
| sick | 0.029093 | 0.030043 | 0.028407 |
| socmob | 0.55604 | 0.55604 | 0.55604 |
| solar_flare | 0.95105 | 0.95105 | 0.95105 |
| space_ga | 0.49826 | 0.48902 | 0.50533 |
| spambase | 0.13061 | 0.12988 | 0.13138 |
| student_performance_por | 0.82769 | 0.83271 | 0.84677 |
| vowel | 0.20329 | 0.19243 | 0.22843 |
| wave_energy | 0.39568 | 0.39568 | 0.39825 |
