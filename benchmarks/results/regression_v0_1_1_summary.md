# Regression check before 0.1.1

Code: `main` at `750ae40` (E2 and E3 adopted, options removed by E1/E4/E6 and the cleanup, faster
histograms). Question: did anything change the models or their quality, other than what the
pre-registered experiments adopted on purpose?

## Default configuration

- **32 held-out OpenML datasets × 5 folds** (`bench_exp_speedups.py`, variant `R011` = library
  defaults): all 160 folds have the same metric (12 significant digits) and the same tree count as
  `E3`, the run that adopted EFB. They also equal `BASE`, the 0.1.0 default: no held-out dataset
  has more than 65,536 rows, so E2's row sampling never applies.
- **51 TabArena lite tasks:** the `{}` configuration of the E6 runs (FULL and LEAN arms) has
  exactly the same error as `MITER_SPEEDUP_E3` and `MITER_SPEEDUP_DEF2` on all 51 tasks. Against
  `MITER_DEFAULT_V1` (0.1.0), 49 tasks are identical; the two that differ have more than 65,536
  rows, where E2's row sampling applies: customer_satisfaction_in_airline (130k rows, −2.41%) and
  GiveMeSomeCredit (150k rows, +0.01%).

## Elo of the default: 998 (0.1.0) vs 956 (E6 table)

The same results get a different Elo because Elo is fitted to the whole table. With the same
baselines, the current default scores 1001 and the 0.1.0 default 998. Adding the strong tuned miter
entries above them lowers both, and CatBoost's default with them:

| table | miter default | CatBoost default |
|---|---:|---:|
| baselines + 0.1.0 default | 998 | 996 |
| baselines + current default | 1001 | 996 |
| baselines + both defaults (current / 0.1.0) | 1001 / 998 | 999 |
| everything: V12_T25, E6 FULL and LEAN, 0.1.0 default (current / 0.1.0) | 947 / 944 | 963 |

Head to head, the current default and the 0.1.0 default both beat CatBoost's default on 24/51 tasks
(median +0.31%, p = 0.40): a tie, as before.

## Tuned: E6 FULL vs the old MITER_V12_T25 (indicative)

| | E6 FULL | V12_T25 | paired (E6 FULL vs V12_T25) |
|---|---:|---:|---|
| tuned + ensemble, Elo (common table) | 1166 | 1202 | 18/51 wins, median +0.14%, p = 0.023 |
| tuned, Elo (common table) | 1080 | 1146 | 22/51 wins, median +0.26%, p = 0.10 |

This is not a regression of the code:

- **The configuration sets differ.** V12_T25 is 6 hand-picked configurations (several symmetric
  ones with CTRs and the auto / quantile quantizers) plus 19 random ones. E6 FULL is `{}` plus 24
  random ones from the same space and seed; the first 19 random configurations are identical.
- **The shared configurations are as good as before.** Over r1–r19 × 51 tasks (969 pairs), the
  current code against the V12 run: median relative error −0.0005%, mean −0.007%, better in 50.05%
  of pairs; per configuration the median stays within ±0.25%.
- The models of the shared configurations are not bit-identical to the V12 run (965 of 969 pairs
  differ). Between the two runs, multiclass moved from one-vs-rest to native softmax, the CTR code
  changed (priors, removed gating) and E2 applies above 65,536 rows. Since the quality is
  unchanged, the exact commit was not bisected.

For a TabArena submission, the V12_T25 configuration list is the stronger one. Its `ACC`
configuration used `preset="accurate"`, which no longer exists; its equivalent is `m_mult=8`.

## Training speed and bit-identity on CI

`bench.yml` run 46129068560, base `dddd1a5` (0.1.0 code plus E2/E3, before the speed-ups) vs
`main`, 100 trees, median of 5 interleaved runs, seconds (change vs base):

| data | threads | x86_64 (AMD EPYC 7763) | aarch64 (Neoverse-N2) |
|---|---:|---:|---:|
| covertype | 1 | 12.94 (−23.9%) | 9.94 (−29.4%) |
| covertype | 4 | 6.88 (−18.1%) | 3.06 (−26.4%) |
| 500000x50 | 1 | 18.00 (−8.9%) | 14.08 (−18.9%) |
| 500000x50 | 4 | 9.19 (−5.8%) | 4.36 (−13.2%) |
| 100000x20 | 1 | 3.19 (−12.1%) | 2.74 (−16.1%) |
| 100000x20 | 4 | 1.97 (−2.6%) | 1.01 (−5.3%) |
| 200000x8 | 1 | 3.30 (−8.1%) | 2.54 (−12.7%) |
| 200000x8 | 4 | 2.23 (−3.5%) | 0.96 (−7.5%) |
| 50000x100 | 1 | 6.41 (−2.6%) | 5.80 (−6.8%) |
| 50000x100 | 4 | 2.95 (+2.4%, spread ±16%) | 1.63 (−1.9%) |

Models and predictions are identical to the base on both machines (54 of 54, 1 and 4 threads). The
one slower x86 cell is within its run-to-run spread; the earlier runs on other x86 machines
(Xeon 6975P-C, EPYC 9V74) measured that cell 1.8% faster.
