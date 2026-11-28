# Pre-registration: E7, CTR splits in the default

Written and hashed before any run on the control set below.

## Why

On TabArena lite the default (`max_ctr_complexity=0`, no CTR splits) ties CatBoost's default
(24/51 wins, median +0.31%, p = 0.40). On the 32-dataset held-out set it beats it (23/32, median
−1.6%, p = 0.005). The largest TabArena losses are on categorical or high-cardinality data, where
CatBoost uses CTRs by default (Amazon_employee_access +66%, in_vehicle_coupon +15%,
kddcup09_appetency +15%, ...). The held-out set has few such datasets.

CTRs were kept off because `max_ctr_complexity="auto"` failed on the held-out set (ablation
`AB_CTR`: median 0.000%, mean +4.7%, worst +158.8% on fps_benchmark). A first look at
fps_benchmark (fold 0) shows:
- complexity 1 improves on no CTR (test RMSE 0.262 vs 0.290);
- complexity 2 and 3, which `"auto"` picks there, are worse **on the training data** too (train RMSE
  0.353 vs 0.175 for complexity 1), so the cause is not overfitting.

The library is young; the cause is to be found and fixed before CTRs can be a default.

## Phases

### Phase A: diagnose and fix (development data only)

- **Allowed data:** the 32 held-out datasets (`bench_exp_scaling_defaults.PART3_DATASETS`) and the
  datasets of earlier CTR experiments (`exp_ctr_auto`, `exp_ctr_auto_v2`,
  `exp_ctr_auto_confirm`, `exp_native_ctr_v2`, `exp_cat_ctr`). All of them have been seen before.
- **What may change:** the CTR code, the `"auto"` rule, and the CTR parameters of the candidate
  (`max_ctr_complexity`, `ctr_n_permutations`, `ctr_prior_weight`, `ctr_model_size_reg`,
  `ctr_max_borders`).
- **Constraint:** models trained without CTR (`max_ctr_complexity=0`) must stay bit-identical
  (`benchmarks/ci/model_hashes.py`, CI A/B).
- **Output:** one candidate configuration `D1` (the current default with CTR parameters set),
  and the code commit it runs on. Both are fixed in an amendment, committed before Phase B
  starts. If no candidate is found that beats `D0` on the development data, E7 stops here and
  CTRs stay off.

### Phase B: confirmation on an unseen control set

- **Control set `C`:** chosen mechanically, listed below. The rule:
  1. every dataset in the OpenML suites AMLB classification (271) and regression (269),
     OpenML-CC18 (99), OpenML-CTR23 (353), and the Grinsztajn categorical suites (334, 335);
  2. with at least one categorical feature (OpenML's symbolic features, minus a nominal target);
  3. with 500 to 200,000 rows and at most 500 features;
  4. not in TabArena (`EXCLUDED_NAMES`) and not used in any earlier miter experiment;
  5. removed as the same data under another name or version: Mercedes_Benz_Greener_Manufacturing
     (= Mercedes_Benz), house_sales (= kings_county), default-of-credit-card-clients
     (= credit_card_clients_default), Bike_Sharing_Demand (used in the M3 parity checks), and the
     second versions of electricity (44156) and Allstate_Claims_Severity (45046).
- **Protocol:** as the held-out harness (`bench_exp_speedups.py`): 5 folds, early stopping 50
  rounds on a validation split, one thread, metrics as in `eval_metrics` (RMSE / log-loss).
- **Arms:** `D0` = the current default, `D1` = the candidate. CatBoost's default is run as a
  reference (reported, not part of the rules).

### Phase C: TabArena, default only

Only if Phase B adopts `D1`: `D1` as a single configuration on the 51 TabArena lite tasks
(`run_tabarena_speedup_51.py`), against the cached current default (`DEF2`).

## Decision rules

`D1` becomes the default only if all of the following hold.

1. **Control set `C` (primary):** relative error per dataset (mean over folds), `D1` vs `D0`:
   - median < 0 and one-sided Wilcoxon (`D1` better) p < 0.05;
   - worst dataset ≤ +5%.
2. **Held-out 32 (no harm; development data, so not a test of benefit):**
   - median ≤ +0.25%, one-sided Wilcoxon (`D1` worse) p ≥ 0.05, worst ≤ +5%.
3. **Training time:** total over `C` ≤ 1.5 × `D0`.
4. **TabArena (Phase C):** median ≤ +0.25% and one-sided Wilcoxon (`D1` worse) p ≥ 0.05 against
   `DEF2`, and the Elo of `D1` (default) not lower than `DEF2` (default) in the table with the
   GBDT baselines.

Otherwise CTRs stay off by default. Results are reported whatever the outcome.

## Control set C (27 datasets)

| OpenML id | name | task | rows | features | categorical |
|---:|---|---|---:|---:|---:|
| 3 | kr-vs-kp | binary | 3196 | 37 | 36 |
| 23 | cmc | multiclass | 1473 | 10 | 7 |
| 50 | tic-tac-toe | binary | 958 | 10 | 9 |
| 151 | electricity | binary | 45312 | 9 | 1 |
| 531 | boston | regression | 506 | 14 | 2 |
| 546 | sensory | regression | 576 | 12 | 11 |
| 1480 | ilpd | binary | 583 | 11 | 1 |
| 4534 | PhishingWebsites | binary | 11055 | 31 | 30 |
| 40966 | MiceProtein | multiclass | 1080 | 82 | 4 |
| 40981 | Australian | binary | 690 | 15 | 8 |
| 41143 | jasmine | binary | 2984 | 145 | 136 |
| 41540 | black_friday | regression | 166821 | 10 | 4 |
| 41980 | SAT11-HAND-runtime-regression | regression | 4440 | 117 | 1 |
| 42563 | house_prices_nominal | regression | 1460 | 80 | 43 |
| 42571 | Allstate_Claims_Severity | regression | 188318 | 131 | 116 |
| 42734 | okcupid-stem | multiclass | 50789 | 20 | 17 |
| 43071 | MIP-2016-regression | regression | 1090 | 145 | 1 |
| 44055 | analcatdata_supreme | regression | 4052 | 8 | 5 |
| 44056 | visualizing_soil | regression | 8641 | 5 | 1 |
| 44958 | auction_verification | regression | 2043 | 8 | 2 |
| 44974 | video_transcoding | regression | 68784 | 19 | 2 |
| 45012 | fifa | regression | 19178 | 29 | 1 |
| 45035 | albert | binary | 58252 | 32 | 10 |
| 45038 | road-safety | binary | 111762 | 33 | 3 |
| 45039 | compas-two-years | binary | 4966 | 12 | 8 |
| 45041 | topo_2_1 | regression | 8885 | 256 | 3 |
| 45043 | seattlecrime6 | regression | 52031 | 5 | 2 |

Row and feature counts are OpenML's metadata. "categorical" counts OpenML's nominal features, some of
which may load as numeric or have few levels; the set is fixed by the rule, not by what CTRs will do
on it. If a dataset fails to load or train, it is reported and left out, and the rules use the rest.
