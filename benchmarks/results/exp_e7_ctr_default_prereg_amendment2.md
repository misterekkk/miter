# E7 amendment 2: the phase-B candidate, frozen

Written before any run on the control set `C`.

## Phase A results (development data)

Categorical development pool (amendment 1), 7 datasets with 5 folds (BachChoralHarmony left out:
classes with a single row cannot be split by the stratified folds), relative error vs `D0`:

| arm | wins / losses | median | worst | time |
|---|---|---:|---:|---:|
| `E7_C1` (`max_ctr_complexity=1`) | 4 / 3 | −4.02% | +1.77% | 1.43× |
| `E7_C1P4` (`max_ctr_complexity=1, ctr_n_permutations=4`) | 5 / 2 | −3.92% | +1.21% | 1.35× |
| `E7_AUTO` (`"auto"`, 4 permutations) | 4 / 3 | −4.22% | +1.97% | 4.08× |

Per dataset (`E7_C1P4`): cjs −86.9%, rl −33.9%, employee_salaries −20.9%, wine-reviews −3.9%,
kdd_internet_usage −0.03%, KDD98 +0.15%, kdd_ipums_la_97-small +1.21%.

On the 32 held-out datasets `E7_C1P4` is neutral: 9 wins, 16 ties, 7 losses, median 0.00%, worst
+2.46% (no failure like `"auto"` on fps_benchmark, +158.8%). Combinations add nothing on the pool
and cost 4× the time, so they stay out of the candidate.

## Frozen

- **`D1` = the current default with `max_ctr_complexity=1, ctr_n_permutations=4`.**
- Code: `main` at the commit of this amendment (models without CTR identical to 0.1.0 code plus
  E2/E3).
- Phase B runs `D0`, `D1` and the CatBoost reference (`CAT`: fixed learning rate 0.05, depth 6,
  early stopping 50, as in every earlier held-out comparison) on the 27 datasets of `C`, with
  `benchmarks/bench_exp_e7.py run --set control --variants D0 D1 CAT`. Decision rules unchanged.
