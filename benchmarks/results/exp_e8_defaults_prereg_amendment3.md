# E8 amendment 3: secondary candidates

Written while the confirmation runs of amendment 2 were in progress, before any of their results
were looked at.

## Why

The selection rule (lowest median over the changed datasets) compares medians over different numbers
of datasets (`DEPTH_T1000`: 4, `DEPTH_T4000`: 11), and it ignores the worst case: it chose
`MSHFIX01_CLF` (worst +10.1%) over the fix alone, `MSHFIX` (worst +1.05%). The development data also
show that the stronger hessian bound helps multiclass (analcatdata_authorship −13.7%,
wall-robot-navigation −13.3%, shuttle −12.0%) and hurts imbalanced binary tasks on AUC
(Credit_Card_Fraud +10.1%, wilt +4.7%): the right bound depends on the task type.

## Added to the confirmation (secondary)

- **H1b `DEPTH_T4000`:** depth 4 when the training split has fewer than 4,000 rows.
- **H3b `MSHFIX`:** `enforce_min_hessian=True` at the default bound, all tasks (the bug fix alone).
- **H3c `MSH_ADAPT`:** `enforce_min_hessian=True`; `min_sum_hessian_in_leaf=0.1` for multiclass,
  the default bound for binary and regression.

They run on `conf` and the held-out set with the same rules as the primary candidates. The Holm
correction covers all seven confirmation tests (four primary, three secondary). If both a primary
and a secondary candidate of one hypothesis pass, the one with the lower median on `conf` is
adopted; the other is reported.
