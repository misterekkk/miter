# Pre-registration addendum: the 8 multiclass TabArena tasks with miter one-vs-rest

This addendum extends `tabarena_t12_full43_prereg.md`, whose primary question and claim rule are
unchanged. It was written and hashed before any multiclass run.

## Fixed in advance
- **Methods:**
  - `MITER_T12` and `MITER_BS6` with exactly the same configs as in the 43-task run.
  - Multiclass goes through `miter.tabarena.MiterModel` one-vs-rest: one binary miter model per
    class, each with its own early stopping on the bagging fold's validation set. The task time
    limit is split evenly over the classes. A class missing from a fold gets a constant
    probability. Probabilities are normalised to sum to 1.
  - This is not native softmax, and it is reported as "miter OvR".
- **Build:**
  - miter is installed into `.venv_tabarena` from `main` after the 43-task run has finished.
  - `src/`, `include/` and `python/` are unchanged since `ebb7d3b` except `python/miter/tabarena.py`,
    which gains only the one-vs-rest branch. The regression and binary code paths are unchanged.
- **Tasks:** the 8 TabArena v0.1 multiclass tasks, `lite` (repeat 0, fold 0), same protocol:
  anneal, maternal_health_risk, website_phishing, MIC, splice, hiva_agnostic,
  students_dropout_and_academic_success, SDSS17.
- **Comparison:**
  - Same as the main pre-registration: `compare(subset="lite")`, `gbdt_only` (fillna/calibration
    None) and full pool, on all 51 tasks.
  - Also the 8 multiclass tasks on their own.
  - Paired T12 (tuned + ensemble) vs CAT (tuned + ensemble): wins, median, Wilcoxon p.

## Reporting rule
- The 51-task numbers are always reported with "multiclass via one-vs-rest".
- The 43-task primary result stays the pre-registered headline. The 51-task result is secondary.
- No claim of a "full TabArena" result is made from a single fold.
