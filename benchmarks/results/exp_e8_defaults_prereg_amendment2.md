# E8 amendment 2: candidates frozen

Written before any run on `conf`, `ctr_conf` or the held-out check.

## Development results (all folds complete)

Relative error vs `D0` over the datasets each arm changes (`dev`: 21 datasets, 17 of them
classification; `ctr_dev`: 34, BachChoralHarmony cannot be split by the stratified folds).

| arm | changed | median | wins | worst | time |
|---|---:|---:|---:|---:|---:|
| `DEPTH_T1000` | 4 | −1.13% | 2/4 | +2.86% | 0.93× |
| `DEPTH_T2000` | 7 | −0.00% | 4/7 | +2.86% | 0.91× |
| `DEPTH_T4000` | 11 | −0.21% | 7/11 | +3.02% | 0.84× |
| `CAP20K` | 2 | −0.70% | 1/2 | +1.62% | 1.57× |
| `LR_N10K` | 8 | −1.23% | 5/8 | +1.28% | 0.88× |
| `MDL5_CLF` | 17 | −0.59% | 10/17 | +4.54% | 0.81× |
| `MDL20_CLF` | 17 | −0.34% | 9/17 | +9.11% | 0.75× |
| `L2_10_CLF` | 17 | −0.12% | 10/17 | +11.22% | 1.49× |
| `MSH1_CLF` | 0 | void (the bound was ignored, amendment 1) | | | |
| `MSHFIX` | 12 | −0.60% | 8/12 | +1.05% | 1.03× |
| `MSHFIX01_CLF` | 17 | −0.80% | 11/17 | +10.10% | 0.66× |
| `MSHFIX1_CLF` | 17 | +1.90% | 5/17 | +19.33% | 0.48× |
| `CTR_K16` | 12 | −2.44% | 9/12 | +0.82% | 1.03× |
| `CTR_K32` | 10 | −2.26% | 8/10 | +1.27% | 1.02× |
| `CTR_K64` | 9 | −0.45% | 6/9 | +1.15% | 1.02× |

Stronger leaf constraints hurt the imbalanced binary datasets on AUC (Credit_Card_Fraud, wilt,
RelevantImages) and help multiclass; the fix at the default bound (`MSHFIX`) is the mildest.

## Frozen (selection rule of the pre-registration: lowest median over changed datasets)

- **H1:** `DEPTH_T1000` (depth 4 when the training split has fewer than 1,000 rows).
- **H2:** `LR_N10K` (`learning_rate=0.1` when the training split has at least 10,000 rows).
- **H3:** `MSHFIX01_CLF` (classification: `enforce_min_hessian=True, min_sum_hessian_in_leaf=0.1`).
- **H4:** `CTR_K16` (`max_ctr_complexity=1, ctr_n_permutations=4, ctr_min_cardinality=16`).

Code: branch `exp/e8` at the commit of this amendment. Confirmation: `D0` and the H1–H3 candidates
on `conf`, `D0` and `CTR_K16` on `ctr_conf`; then the held-out check (rule 3) on the 32 held-out
datasets, set `heldout` in the runner, same metric. Rules unchanged.
