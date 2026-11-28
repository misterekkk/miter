# E7 amendment 1: a categorical development pool for phase A

Written before any run on the datasets below or on the control set `C`.

## Why

Phase A on the 32 held-out datasets is finished: CTR on single features (`max_ctr_complexity=1`)
removes the fps_benchmark failure (worst dataset +2.9% instead of +158.8%), and leaves models without
categorical features bit-identical. It neither helps nor hurts on the 17 datasets with categorical
features (7 wins, 3 ties, 7 losses, median 0.00%). Their categorical features have low cardinality,
where the K1 splits already work, so they do not test what E7 is about.

The other development data named in the pre-registration (the datasets of `exp_ctr_auto`,
`exp_ctr_auto_v2`, `exp_ctr_auto_confirm`, `exp_native_ctr_v2`, `exp_cat_ctr`) are all TabArena
datasets. Using them would make phase C on TabArena no test at all, so they are **not** used.

## The development pool

Chosen mechanically from all active OpenML datasets:
1. 2,000 to 200,000 rows, 5 to 500 features;
2. at least 3 nominal features, and a nominal feature with at least 20 distinct values;
3. not in TabArena, not in an earlier miter experiment, not in the control set `C`;
4. same data under another name removed: adult-census, adult_income_p, ada_prior (= adult),
   uci_diabetes_p (= Diabetes130US), Kaggle_bike_sharing_demand_challange (= Bike_Sharing),
   ozone_level (= ozone-level-8hr), internet_usage (= kdd_internet_usage); KDDCup09_churn and
   KDDCup09_upselling share their features with kddcup09_appetency (TabArena) and are removed;
5. a default target attribute is required (nfl_games has none).

| OpenML id | name | task | rows | features | categorical | max cardinality |
|---:|---|---|---:|---:|---:|---:|
| 42125 | employee_salaries | regression | 9228 | 12 | 4 | 37 |
| 41160 | rl | binary | 31406 | 22 | 14 | 2580 |
| 41275 | wine-reviews | regression | 129971 | 12 | 6 | 16757 |
| 981 | kdd_internet_usage | binary | 10108 | 68 | 68 | 129 |
| 993 | kdd_ipums_la_97-small | binary | 7019 | 60 | 27 | 191 |
| 473 | cjs | multiclass | 2796 | 34 | 2 | 57 |
| 4552 | BachChoralHarmony | multiclass (102) | 5665 | 16 | 14 | 62 |
| 23513 | KDD98 | binary | 191260 | 478 | 132 | 25847 |

## Arms on the pool (development, all may be looked at)

- `D0`: the current default.
- `E7_C1`: `max_ctr_complexity=1`.
- `E7_C1P4`: `max_ctr_complexity=1, ctr_n_permutations=4`.
- `E7_AUTO`: `max_ctr_complexity="auto", ctr_n_permutations=4` (the configuration that failed on
  fps_benchmark).

Protocol as the held-out harness (5 folds, early stopping 50, one thread), runner
`benchmarks/bench_exp_e7.py`. Further candidates, including a fix of the combinations, may be added
in phase A. The decision rules for adopting `D1` are unchanged and are applied only to phases B and C.
