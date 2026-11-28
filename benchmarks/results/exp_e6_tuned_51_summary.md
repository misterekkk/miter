# E6-T: tuned TabArena lite, 51 tasks, LEAN (no CTR / quantizer) vs FULL (V12 space)

Pre-registration: `exp_e6_options_prereg.md` and `exp_e6_options_prereg_amendment1.md`. Negative = LEAN better.

| method | elo | elo+ | elo- | rank |
|---|---:|---:|---:|---:|
| [New] MITER_E6_FULL (tuned + ensemble) | 1180 | 64 | 48 | 4.794 |
| [New] MITER_E6_LEAN (tuned + ensemble) | 1131 | 50 | 54 | 5.627 |
| GBM (tuned + ensemble) | 1099 | 55 | 45 | 6.216 |
| [New] MITER_E6_FULL (tuned) | 1085 | 67 | 46 | 6.480 |
| CAT (tuned + ensemble) | 1078 | 63 | 55 | 6.608 |
| CAT (tuned) | 1068 | 63 | 60 | 6.804 |
| [New] MITER_E6_LEAN (tuned) | 1051 | 56 | 44 | 7.127 |
| XGB (tuned + ensemble) | 1022 | 46 | 54 | 7.696 |
| GBM (tuned) | 1000 | 55 | 51 | 8.118 |
| XGB (tuned) | 981 | 54 | 61 | 8.490 |
| CAT (default) | 974 | 58 | 69 | 8.608 |
| [New] MITER_E6_FULL (default) | 956 | 62 | 49 | 8.961 |
| [New] MITER_E6_LEAN (default) | 956 | 62 | 49 | 8.961 |
| XGB (default) | 741 | 70 | 92 | 12.392 |
| GBM (default) | 678 | 77 | 85 | 13.118 |

- **tuned + ensemble: LEAN vs FULL** (51 tasks): wins 20/51, median +0.14% (95% boot CI [-0.07%, +0.53%]), mean +1.56%, Wilcoxon p = 0.0429, worst +23.69%; one-sided p (LEAN worse) = 0.02145

- rule 1 (median / Wilcoxon): fail
- rule 2 (amendment 1, Elo LEAN 1131 >= FULL 1180): fail

**Decision (pre-registered, tuned + ensemble): KEEP**

Largest LEAN losses:

- APSFailure: +23.69%
- taiwanese_bankruptcy_prediction: +21.83%
- Amazon_employee_access: +14.41%
- SDSS17: +11.01%
- polish_companies_bankruptcy: +8.44%
- kddcup09_appetency: +3.48%
- customer_satisfaction_in_airline: +3.07%
- hazelnut-spread-contaminant-detection: +2.27%

- **tuned: LEAN vs FULL** (51 tasks): wins 20/51, median +0.15% (95% boot CI [-0.38%, +0.71%]), mean +1.55%, Wilcoxon p = 0.2722, worst +36.32%; one-sided p (LEAN worse) = 0.1361
