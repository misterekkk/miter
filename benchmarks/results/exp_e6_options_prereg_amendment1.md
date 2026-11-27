# E6 pre-registration, amendment 1: TabArena rank guard for E6-T

Amends `exp_e6_options_prereg.md` (sha256 13bd679813d108b9541b014ea26c8ff36c8d135ee86bf69a4ae68bbb8c2b18ec).
Written and hashed before any E6-T (tuned) result exists: at this point only the E6-A / E6-F
held-out runs have finished; the tuned FULL and LEAN arms have not started.

## Reason
The maintainer's goal is the TabArena ranking. The original E6-T rule looks at the median
relative error and a Wilcoxon test, which describe the typical task; the leaderboard Elo comes
from wins and losses against all methods on every task, so a few tasks with large CTR gains can
move it while the median does not.

## Change (E6-T only; E6-A and E6-F are unchanged)
CTR and the quantizer modes are removed only if **both** hold:
1. the original rule: tuned + ensemble, `LEAN` vs `FULL`, median relative error ≤ +0.25% and
   one-sided Wilcoxon (`LEAN` worse) p ≥ 0.05;
2. **new:** in the TabArena `lite` gbdt-only leaderboard of the 51 tasks (the methods of
   `run_tabarena_v12_51.py`: LightGBM, CatBoost, XGBoost and both E6 arms), the Elo of
   `MITER_E6_LEAN (tuned + ensemble)` is not lower than the Elo of
   `MITER_E6_FULL (tuned + ensemble)` (point estimates).

Otherwise CTR and the quantizer modes stay. Both Elo values, their confidence intervals and the
ranks are reported either way.
