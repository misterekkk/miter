# E7: CTR splits in the default — result

Pre-registration: `exp_e7_ctr_default_prereg.md` (+ amendments 1 and 2). Runner:
`benchmarks/bench_exp_e7.py`. Candidate `D1` = default + `max_ctr_complexity=1, ctr_n_permutations=4`.

**Decision: `D1` is not adopted; CTRs stay off by default.** Phase C (TabArena) is not run.

## Phase B: control set C (27 datasets × 5 folds), relative error vs `D0`

| arm | wins / ties / losses | median | mean | worst | p (better) | time |
|---|---|---:|---:|---:|---:|---:|
| `D1` | 7 / 13 / 7 | +0.000% | −1.67% | +9.50% | 0.40 | 1.10× |
| CatBoost (reference) | 11 / 0 / 16 | +0.128% | +1.61% | +112.7% | 0.75 | 2.50× |

Rules: rule 1 fails (median 0, p = 0.40; worst +9.50% > +5%); rule 3 passes (1.10×). Rules 2 and 4
are not needed.

Per dataset (`D1`, CatBoost): auction_verification −40.1% (−63.5%), sensory −9.8% (−8.0%),
house_prices_nominal −6.0% (−6.5%), okcupid-stem −1.2%, Allstate_Claims_Severity −0.9%; 13 ties
(all categorical features at or below `max_cat_to_onehot`, or K1 already used); losses
black_friday +0.1%, seattlecrime6 +0.2%, boston +0.3%, electricity +0.3%, Australian +1.0%,
MIP-2016-regression +1.8%, SAT11-HAND-runtime-regression +9.5%.

The worst case, SAT11-HAND-runtime-regression, has a single categorical feature (`algorithm`, 15
values): K1 splits on it exactly, and the CTR split replaces them with a noisier statistic.

## Phase A (development data)

- 32 held-out datasets: `max_ctr_complexity=1` neutral (7 / 18 / 7, median 0.00%, worst +2.85%);
  the old `"auto"` with combinations failed on fps_benchmark (+158.8%) because combination CTRs win
  the greedy level choice over exact K1 splits and the trees end up worse even on training data.
- Categorical development pool (amendment 1, 7 datasets): CTR on single features helps strongly on
  high-cardinality data (cjs −86.9%, rl −33.9%, employee_salaries −20.9%, wine-reviews −3.9%).

## What it means

- CTRs help a lot on some datasets with high-cardinality categorical features and cost little
  elsewhere, but not consistently enough to be a default under the pre-registered rule. They stay a
  tuning dimension, where the choice is made on validation data (E6: removing them costs 49 Elo).
- A follow-up would restrict CTRs to high-cardinality features (where K1 has too few rows per
  category), on fresh data: the control set C has now been seen and cannot confirm a rule derived
  from it.
- The current default (no CTR) beats the CatBoost reference on C: 16/27 datasets, in 0.4× its time.
