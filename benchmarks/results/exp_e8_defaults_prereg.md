# Pre-registration: E8, data-dependent defaults

Written and hashed before any E8 run. Runner: `benchmarks/bench_exp_e8.py` (branch `exp/e8`).
Protocol as the held-out harness: 5 folds, early stopping 50 rounds on a validation split, one
thread, at most 25,000 rows per dataset.

## Hypotheses and arms

**H1, depth by data size.** Default (depth 6) vs depth 4 when the training split has fewer rows
than T. Arms `DEPTH_T1000`, `DEPTH_T2000`, `DEPTH_T4000`. Evidence so far (held-out ablation, seen
data): depth 4 better at n ≤ 1500 (10/14 wins, median −0.79%), worse above 8,000 rows (2/10).

**H2, more trees on large data.** 14 of 160 held-out folds hit the 5,000-tree cap. Arms `CAP20K`
(`n_estimators=20000`; early stopping ends the rest) and `LR_N10K` (`learning_rate=0.1` when the
training split has at least 10,000 rows).

**H3, leaf regularisation for classification.** On TabArena the default ties or loses to CatBoost on
classification, most on imbalanced binary tasks (minority ≤ 10%: 3/9 wins, median +2.2%), and wins
on regression (9/13). Classification-only arms (regression unchanged): `MDL5_CLF`
(`min_data_in_leaf=5`), `MDL20_CLF` (20), `L2_10_CLF` (`lambda_l2=10`), `MSH1_CLF`
(`min_sum_hessian_in_leaf=1.0`).

**H4, CTR only above a cardinality threshold.** E7: CTR on single features helps on high-cardinality
features (cjs −87%, rl −34%, employee_salaries −21%, auction_verification −40%) and hurts where K1
already splits a feature of few values (SAT11, 15 values: +9.5%). New option `ctr_min_cardinality`
(default 0, models without it unchanged: 54/54 hashes identical to `main`). Arms `CTR_K16`, `CTR_K32`,
`CTR_K64`: `max_ctr_complexity=1, ctr_n_permutations=4, ctr_min_cardinality=K`.

## Data

All sets are disjoint from TabArena, the 32 held-out datasets, and every earlier experiment, except
H4's development set, which is E7's (seen).

- **General pool (35):** every dataset of the OpenML suites AMLB (269, 271), OpenML-CC18 (99),
  OpenML-CTR23 (353) and Grinsztajn (334, 335) not used before, with at most 500 features and at most
  50 classes; same data removed (pol, elevators, house_16H, covertype; mfeat-factors kept of the six
  mfeat views; ada, derived from adult).
- **Imbalanced binary pool (17):** active OpenML binary datasets, 1,000–200,000 rows, ≤ 500 features,
  minority class ≤ 10% (OpenML metadata), not used before; same data removed (GiveMeSomeCredit samples,
  the second Credit_Card_Fraud and CreditCardSubset, dis (thyroid, as sick), insurance_company (as
  coil2000)), time series removed (art_daily_nojump4); plus the suites' pc1, Satellite, wilt,
  climate-model-simulation-crashes.
- Each pool is split 40% development / 60% confirmation by a seeded shuffle (seeds 2026, 2027):
  - `dev` (21): balance-scale, pendigits, satimage, yprop_4_1, analcatdata_authorship, quake, kc2,
    wall-robot-navigation, shuttle, steel-plates-fault, philippine, MiniBooNE, us_crime,
    SGEMM_GPU_kernel_performance, pc1, Speech, wilt, RelevantImagesDatasetTEST,
    Employee-Turnover-at-TECHCO, Credit_Card_Fraud, jigsaw_unintended_bias100K.
  - `conf` (31): letter, mfeat-factors, breast-w, optdigits, vehicle, yeast, kc1,
    banknote-authentication, phoneme, semeion, wdbc, wine-quality-white, texture, madeline, sylvine,
    volkert, jannis, Yolanda, OnlineNewsPopularity, particulate-matter-ukair-2017, pumadyn32nh,
    mammography, yeast_ml8, sylva_prior, mc1, pc2, Satellite, climate-model-simulation-crashes,
    Asteroid_Dataset, open_payments, Is_fraud.
- **H4 development (`ctr_dev`, 35):** E7's development pool and control set (seen).
- **H4 confirmation (`ctr_conf`, 6):** every fresh dataset with a categorical feature of more than 32
  values: colleges_aaup, medical_charges, musk, rainfall_bangladesh, RelevantImagesDatasetTEST,
  open_payments. (RelevantImagesDatasetTEST is in H1–H3's `dev`, where no CTR arm runs.)

## Metric

TabArena's: 1 − ROC AUC for binary tasks, log-loss for multiclass, RMSE / std for regression.
Log-loss of binary tasks is reported too.

## Procedure

1. **Development:** all arms on `dev` (H1–H3) and `ctr_dev` (H4). Per hypothesis the arm with the
   lowest median relative error over the datasets it changes is the candidate (ties: the simpler one,
   i.e. the lower T, `CAP20K` before `LR_N10K`, the lower K). A hypothesis whose best arm has median
   ≥ 0 on development stops there.
2. **Freeze:** the candidates and the code commit in an amendment, before any confirmation run.
3. **Confirmation:** `D0` and the candidates on `conf` (H1–H3) and `ctr_conf` (H4).
4. **No harm on the 32 held-out datasets** for each confirmed candidate.
5. **TabArena (after the T100 run):** each confirmed candidate alone, then all confirmed candidates
   together, single configuration on the 51 lite tasks against the cached default (`DEF2`).

## Decision rules

A candidate is adopted only if all hold:

1. **Confirmation**, over the datasets the candidate changes: median relative error < 0, one-sided
   Wilcoxon (better) p below the Holm threshold for the hypotheses that reach confirmation (α = 0.05),
   worst dataset ≤ +5%. For H4 (6 datasets, too few for the test): median < 0 and worst ≤ +5%, and
   rule 4 carries the weight.
2. **Time:** total over the confirmation set ≤ 1.5 × `D0`.
3. **Held-out 32:** median ≤ +0.25%, one-sided Wilcoxon (worse) p ≥ 0.05, worst ≤ +5%.
4. **TabArena:** median ≤ +0.25% and one-sided Wilcoxon (worse) p ≥ 0.05 against `DEF2`, and Elo not
   lower than `DEF2` in the table with the GBDT baselines. For H4 additionally: on the TabArena tasks
   with a categorical feature of more than K values, median < 0.

The combined default is adopted if the combination also passes rule 4. Results are reported whatever
the outcome.
