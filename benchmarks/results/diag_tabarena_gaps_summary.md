# Diagnosis of the miter-vs-CatBoost gap on miami_housing, concrete, airfoil (parameters only)

Exploratory, post hoc, not pre-registered. Script: `benchmarks/diag_tabarena_gaps.py`; raw per-fit CSVs: `benchmarks/results/raw/diag_gaps/{airfoil,concrete,miami}.csv` (gitignored). All numbers below are copied from the output of `report` on those CSVs.

## Provenance and protocol
- Hardware Apple M4, 10 cores, 16 GB, `nice -n 10`, at most 2 processes, 1 thread per fit (timings approximate, machine shared).
- Python 3.13.1, miter 0.1.0.dev0 installed from branch m6-lignum at ed7333f (HEAD at the time of the later check: 7f4ee48; I did not verify the installed build byte for byte), lightgbm 4.7.0, catboost 1.2.10, numpy 2.5.5, pandas 3.1.1. TabArena venv used only to export data (`dump`).
- Data and splits: TabArena OpenML tasks (airfoil 363612, concrete 363625, miami 363686), TabArena outer splits repeat 0..2 x fold 0..2 = 9 splits per dataset (the datasets have 30 / 30 / 9 splits in the full protocol). Single model per variant, NOT TabArena's 8-fold bagging and NOT AutoGluon preprocessing; categorical-looking columns are numeric (airfoil attack-angle = its numeric value, miami avno60plus = 0/1). Early stopping on a fixed random 20% of the outer train set (patience 50 for all libraries, max 10000 trees), same for every variant. RMSE on the outer test fold in original units; table = mean and sd over the 9 splits (sd between splits, dominated by split difficulty); the last two columns are the mean paired % difference per split vs `miter base` and vs `cat Plain lr0.05` (negative = better than reference).
- Baselines: miter base = lr 0.05, 31 leaves, min_data_in_leaf 20, lambda_l2 1, m_mult 2. LightGBM base = lr 0.05, 31 leaves, min_child_samples 20, reg_lambda 1. CatBoost default = auto lr (CatBoost's own), otherwise Plain/MVS (T4 in MITER_STATUS.md says Plain is chosen at these n; here `cat lr0.05` and `cat Plain lr0.05` give identical numbers, which confirms it).
- Elapsed: sum of per-fit seconds = airfoil 277 s, concrete 249 s, miami 1185 s (plus restarts and waiting; total wall roughly 35 min).
- Selection caveat: variants are compared on the same outer test folds with no separate selection set, so the best variant of many is optimistically biased (about 40 variants per dataset); differences below ~1% are within the noise of 9 splits and should not be read as wins.
- Not measured: TabArena 8-fold bagging effect, CatBoost tuned+ens itself (only cached TabArena numbers exist), combinations beyond those listed, feature_fraction / bagging_fraction (accepted by the Python API but absent from `src/`, so inert: two variants gave bit-identical RMSE and were dropped), seeds other than one per split, classification.
- Reference (cached TabArena, bagged, from `benchmarks/results/tabarena_m5_summary.md`): miami miter 84027.6 / accurate 83729.9 / CAT def 79760.4 / CAT tuned+ens 78405.1; concrete 4.3631 / 4.3607 / 4.2017 / 4.2018; airfoil 1.4078 / 1.4044 / 1.5783 / 1.3300 (miter default / accurate / CAT default / CAT tuned+ens).

## Findings (details and tables below)
1. **concrete (n=1030): tree shape / leaf size is the lever.** Single-model gap miter base vs CatBoost lr0.05 = +3.79%. miter with 7 leaves: -3.07% vs base, gap to CatBoost +0.57%; leaves4 -2.95%; min_data_in_leaf 40 -2.75%; leaves7+mdl40 -3.50% (gap +0.13%). min_data_in_leaf 5 is worse (+2.1%). lr 0.02 gives only -0.85%, lambda_l2 up to 10 -0.4%, accurate -0.19%. num_leaves 63/127 is identical to 31 (leaf cap is not binding at ~660 training rows). Related prototype note (MITER_STATUS.md T1/T2): tuning concrete with 25 configurations overfit the validation set (-0.97%), so single-config comparisons like this one are the safer reading. LightGBM is much worse than both (+6.2% vs miter base), so this is not specific to oblique splits: the leaf-wise 31-leaf default is too flexible for n~1000. CatBoost's edge is partly its default regularisation (symmetric depth-6 trees, 64 leaves but shared splits): CatBoost Lossguide31 is 0.6% worse than its symmetric default and depth8 is 3.9% worse.
2. **miami_housing (n=13776): partly smaller steps/leaves, the rest is symmetric trees and CatBoost's smoother fit; log-target does not help anyone.** Gap miter base vs CatBoost lr0.05 = +3.68% (TabArena bagged: accurate +4.98% vs CAT default). Best miter variant: lr0.02 + min_data_in_leaf 5 = -2.12% vs base (gap +1.49%); leaves7 -1.48%; lr0.02 -0.89%; mdl5 -0.88%; accurate -0.06% (so projections count is not the lever). CatBoost with leaf-wise growth is 2.3% worse than symmetric (Lossguide31 +1.27% vs Plain default, Depthwise +1.69%), so symmetric trees explain roughly a third of its edge; depth 8/10 are worse (+2.9% / +9.6%). CatBoost l2=1 is better (-0.65%; T4 also found l2=1 helps CatBoost on miami, -1.4%). Ordered boosting is clearly worse here (+5.45%), so it is not the source of CatBoost's edge. Log target: worse for ALL libraries (miter base +2.2%, accurate +2.1%, lr0.02 +2.2% ; LightGBM +1.6%; CatBoost +0.9..1.0%); the loss is similar or larger for miter than for CatBoost, so it is not a miter-specific fix (metric is RMSE in price units, and exp(pred) retransformation bias is not corrected, which I did not test).
3. **airfoil (n=1503): the single-model gap to CatBoost default does not exist; the TabArena gap is only vs tuned+ensembled CatBoost.** miter base 1.5708 vs CatBoost lr0.05 1.6199 (miter -2.8%; consistent with TabArena CAT default 1.5783 vs miter 1.4078 bagged). No parameter variant of miter helps (all within +-0.3% except worse ones: leaves7 +3.0%, mdl5 +2.1%, leaves4 +8.2%); l2=10 -0.23%, lr0.03 -0.26% are noise-level. CatBoost depth8/10 reach 1.5626 / 1.5808, i.e. about miter's level. The remaining gap to CAT tuned+ens (1.33, bagged, 30 splits) is thus not reachable with single-model parameters tried here; it comes from tuning plus ensembling (not measured). LightGBM is far worse (+16%).
4. **lr / number of trees**: lr 0.02-0.03 helps modestly on miami and concrete (-0.9% / -0.8%) at 1.7-2x trees and costs nothing on airfoil, consistent with T4b (lr 0.02 reduced the gap by 0.5-1.2 pp but hurt winning datasets such as sulfur). CatBoost auto-lr (~0.02-0.03 here, 4000-7000 trees) is equal to or slightly better than its lr 0.05 (-0.28%, -0.49%) on miami/concrete and worse on airfoil (+1.9%).
5. **Regularisation lambda_l2** 0.1..10: effect within +-0.4% on all three; not a lever (tested only with 31 leaves except one combination).
6. **Symmetric vs leaf-wise**: see 1-2. LightGBM depth-limited trees (depth 4/6) do not close anything (vs LightGBM base: miami +0.05% / +0.5%, concrete -0.9% / -1.2%, airfoil +10% .. +28% worse).

## Interpretation
Concrete and (partly) miami are small-signal / smooth-target problems where CatBoost's strong default regularisation wins: few effective leaves (concrete), symmetric structure and CatBoost's own sampling (miami). On concrete a trivial leaf-size/leaf-count change recovers almost the whole single-model gap, while it hurts airfoil (needs flexible trees). This is a dataset-dependent trade-off, not a missing feature, and matches the author's T4 decision (no single CatBoost mechanism to transplant); the new information is that the n-dependent complexity default (num_leaves / min_data_in_leaf as a function of n) is the one parameter-level lever that moves concrete a lot.

## Levers in scope of the plan (nothing changed here)
- Both `num_leaves` and `min_data_in_leaf` defaults are [TUNABLE] per the plan table; changing them needs a broad benchmark (many datasets, Wilcoxon, median) and author approval plus an entry in docs/DECISIONS.md. Candidates only: smaller default leaves / larger min_data for small n (n < ~2000), lr 0.02-0.03 for small n. The airfoil result (leaves7 +3.0%, leaves4 +8.2%) shows a global change would hurt; an n-adaptive rule would need its own benchmark.
- Closed / out-of-plan (idea only): symmetric (oblivious) tree mode, Ordered boosting (measured worse anyway), bagging of MVS type.

## Tables (RMSE mean, sd over 9 splits, mean best iteration, mean secs per fit)


### concrete

| variant | n | rmse | sd | trees | secs | vs miter base % | vs cat Plain lr0.05 % |
|---|---|---|---|---|---|---|---|
| miter base | 9 | 4.6993 | 0.4234 | 639.2222 | 0.1278 | 0.0 | 3.793 |
| miter accurate | 9 | 4.6986 | 0.5304 | 571.0 | 0.1633 | -0.1929 | 3.5904 |
| miter m_mult=4 | 9 | 4.6986 | 0.5304 | 571.0 | 0.1633 | -0.1929 | 3.5904 |
| miter lr0.03 | 9 | 4.6784 | 0.4532 | 989.0 | 0.1944 | -0.4891 | 3.3059 |
| miter lr0.02 | 9 | 4.6624 | 0.4579 | 1332.5556 | 0.2589 | -0.8459 | 2.9221 |
| miter leaves7 | 9 | 4.5585 | 0.4742 | 1057.4444 | 0.09 | -3.0691 | 0.5699 |
| miter leaves15 | 9 | 4.6096 | 0.4537 | 790.4444 | 0.1267 | -1.9821 | 1.7257 |
| miter leaves63 | 9 | 4.6993 | 0.4234 | 639.2222 | 0.1278 | 0.0 | 3.793 |
| miter mdl5 | 9 | 4.7985 | 0.4476 | 565.1111 | 0.1533 | 2.0944 | 5.9307 |
| miter mdl40 | 9 | 4.5708 | 0.4319 | 855.3333 | 0.0978 | -2.7509 | 0.9003 |
| miter l2=0.1 | 9 | 4.6884 | 0.4302 | 555.6667 | 0.1111 | -0.2417 | 3.5324 |
| miter l2=3 | 9 | 4.6893 | 0.4436 | 673.6667 | 0.1333 | -0.2441 | 3.5349 |
| miter l2=10 | 9 | 4.6839 | 0.4578 | 841.0 | 0.1589 | -0.3854 | 3.3935 |
| miter lr0.02 mdl5 | 9 | 4.7871 | 0.3957 | 1242.2222 | 0.3411 | 1.9552 | 5.777 |
| lgbm base | 9 | 4.984 | 0.405 | 704.3333 | 0.2078 | 6.2008 | 10.2423 |
| lgbm lr0.02 | 9 | 4.9839 | 0.4005 | 1429.1111 | 0.1933 | 6.2379 | 10.3131 |
| lgbm depth6 leaves63 | 9 | 4.9262 | 0.3342 | 718.5556 | 0.0633 | 5.0439 | 9.0207 |
| lgbm depth4 leaves15 | 9 | 4.9376 | 0.3455 | 968.2222 | 0.0533 | 5.3902 | 9.3734 |
| lgbm mdl5 | 9 | 5.0633 | 0.2685 | 497.5556 | 0.0956 | 8.165 | 12.1876 |
| cat default (auto lr) | 9 | 4.5087 | 0.4216 | 4717.0 | 2.6811 | -4.0435 | -0.4894 |
| cat lr0.05 | 9 | 4.5288 | 0.3886 | 1412.2222 | 0.8222 | -3.5472 | 0.0 |
| cat lr0.02 | 9 | 4.5041 | 0.3997 | 2671.7778 | 1.5289 | -4.1097 | -0.5467 |
| cat Plain lr0.05 | 9 | 4.5288 | 0.3886 | 1412.2222 | 0.8111 | -3.5472 | 0.0 |
| cat Ordered lr0.05 | 9 | 4.7073 | 0.4333 | 1889.6667 | 3.7289 | 0.2906 | 3.9872 |
| cat Lossguide31 lr0.05 | 9 | 4.5561 | 0.4376 | 1241.8889 | 0.5922 | -3.0599 | 0.582 |
| cat Depthwise lr0.05 | 9 | 4.5933 | 0.4183 | 1109.1111 | 1.3656 | -2.2003 | 1.4239 |
| cat l2=1 lr0.05 | 9 | 4.5337 | 0.4307 | 1058.6667 | 0.6256 | -3.4952 | 0.0599 |
| cat depth8 lr0.05 | 9 | 4.7063 | 0.5351 | 1335.3333 | 2.4378 | 0.0891 | 3.7387 |
| cat Plain lr0.05 no-bootstrap | 9 | 4.5266 | 0.4072 | 1224.4444 | 0.7167 | -3.6188 | -0.0629 |
| miter leaves4 | 9 | 4.557 | 0.3899 | 1512.1111 | 0.0744 | -2.9532 | 0.6937 |
| miter leaves7 mdl40 | 9 | 4.5388 | 0.4754 | 1225.5556 | 0.1044 | -3.495 | 0.1288 |
| miter leaves15 mdl40 | 9 | 4.5708 | 0.4319 | 855.3333 | 0.1011 | -2.7509 | 0.9003 |
| miter leaves7 lr0.02 | 9 | 4.5476 | 0.4753 | 1958.0 | 0.1633 | -3.3256 | 0.3307 |
| miter leaves7 l2=10 | 9 | 4.5845 | 0.4358 | 1069.8889 | 0.0911 | -2.446 | 1.2125 |
| miter mdl80 | 9 | 4.7447 | 0.5274 | 2065.4444 | 0.1322 | 0.8196 | 4.5712 |
| lgbm leaves7 mdl40 | 9 | 4.8623 | 0.3919 | 1212.4444 | 0.0678 | 3.5652 | 7.4284 |
| miter leaves127 | 9 | 4.6993 | 0.4234 | 639.2222 | 0.1411 | 0.0 | 3.793 |
| miter leaves255 mdl10 | 9 | 4.8249 | 0.4341 | 459.6667 | 0.1767 | 2.6884 | 6.5603 |
| cat depth10 lr0.05 | 9 | 5.1131 | 0.6003 | 1214.0 | 8.2344 | 8.7965 | 12.7134 |

### airfoil

| variant | n | rmse | sd | trees | secs | vs miter base % | vs cat Plain lr0.05 % |
|---|---|---|---|---|---|---|---|
| miter base | 9 | 1.5708 | 0.0927 | 3094.8889 | 0.9067 | 0.0 | -2.8031 |
| miter accurate | 9 | 1.5721 | 0.1063 | 2255.4444 | 1.0989 | 0.0532 | -2.7281 |
| miter m_mult=4 | 9 | 1.5721 | 0.1063 | 2255.4444 | 1.0311 | 0.0532 | -2.7281 |
| miter lr0.03 | 9 | 1.5673 | 0.105 | 3485.6667 | 1.0222 | -0.2555 | -3.0394 |
| miter lr0.02 | 9 | 1.5746 | 0.1038 | 4116.3333 | 1.22 | 0.2377 | -2.5755 |
| miter leaves7 | 9 | 1.6191 | 0.1318 | 3157.5556 | 0.2767 | 3.0236 | 0.0984 |
| miter leaves15 | 9 | 1.571 | 0.0982 | 2958.3333 | 0.54 | 0.0065 | -2.7879 |
| miter leaves63 | 9 | 1.5806 | 0.095 | 2606.2222 | 0.7989 | 0.6214 | -2.2106 |
| miter mdl5 | 9 | 1.6031 | 0.0923 | 2938.2222 | 0.6978 | 2.0882 | -0.7495 |
| miter mdl40 | 9 | 1.5886 | 0.1008 | 2840.0 | 0.5411 | 1.1566 | -1.6459 |
| miter l2=0.1 | 9 | 1.5706 | 0.0971 | 2369.8889 | 0.6744 | -0.02 | -2.8028 |
| miter l2=3 | 9 | 1.5685 | 0.0946 | 2908.4444 | 0.8356 | -0.1495 | -2.9396 |
| miter l2=10 | 9 | 1.5682 | 0.1156 | 2952.5556 | 0.8256 | -0.2296 | -3.0482 |
| miter lr0.02 mdl5 | 9 | 1.5712 | 0.1054 | 6935.7778 | 1.7667 | 0.0061 | -2.8013 |
| lgbm base | 9 | 1.8246 | 0.1169 | 1468.4444 | 0.3667 | 16.1901 | 12.9038 |
| lgbm lr0.02 | 9 | 1.8637 | 0.103 | 2328.7778 | 0.3811 | 18.7622 | 15.4484 |
| lgbm depth6 leaves63 | 9 | 1.9307 | 0.1106 | 1488.6667 | 0.1267 | 23.0225 | 19.5456 |
| lgbm depth4 leaves15 | 9 | 2.0004 | 0.0992 | 1921.4444 | 0.1067 | 27.6378 | 24.0471 |
| lgbm mdl5 | 9 | 1.7553 | 0.078 | 1219.5556 | 0.1944 | 11.8608 | 8.7898 |
| cat default (auto lr) | 9 | 1.6504 | 0.1474 | 7174.4444 | 1.0167 | 4.9399 | 1.8629 |
| cat lr0.05 | 9 | 1.6199 | 0.1356 | 3029.8889 | 0.4333 | 3.0155 | 0.0 |
| cat lr0.02 | 9 | 1.6401 | 0.1609 | 5881.7778 | 0.8489 | 4.2814 | 1.1818 |
| cat Plain lr0.05 | 9 | 1.6199 | 0.1356 | 3029.8889 | 0.4456 | 3.0155 | 0.0 |
| cat Ordered lr0.05 | 9 | 1.9078 | 0.1278 | 1869.6667 | 0.6933 | 21.4781 | 18.0806 |
| cat Lossguide31 lr0.05 | 9 | 1.7326 | 0.1178 | 1842.2222 | 0.3822 | 10.3359 | 7.1732 |
| cat Depthwise lr0.05 | 9 | 1.6769 | 0.0999 | 2965.2222 | 0.9189 | 6.7812 | 3.7552 |
| cat l2=1 lr0.05 | 9 | 1.6414 | 0.1359 | 2110.1111 | 0.3 | 4.4159 | 1.3538 |
| cat depth8 lr0.05 | 9 | 1.5626 | 0.1228 | 3468.6667 | 1.1156 | -0.5995 | -3.4841 |
| cat Plain lr0.05 no-bootstrap | 9 | 1.6607 | 0.1534 | 2004.6667 | 0.2722 | 5.6043 | 2.5398 |
| miter leaves4 | 9 | 1.6992 | 0.1289 | 3425.7778 | 0.1833 | 8.1564 | 5.1265 |
| miter leaves7 mdl40 | 9 | 1.63 | 0.1026 | 3378.3333 | 0.3056 | 3.7678 | 0.8647 |
| miter leaves15 mdl40 | 9 | 1.5817 | 0.0992 | 3418.3333 | 0.5711 | 0.6951 | -2.1156 |
| miter leaves7 lr0.02 | 9 | 1.5999 | 0.1224 | 7203.3333 | 0.6244 | 1.7885 | -1.0978 |
| miter leaves7 l2=10 | 9 | 1.6007 | 0.1378 | 4597.1111 | 0.3989 | 1.7888 | -1.1178 |
| miter mdl80 | 9 | 1.6914 | 0.1148 | 3820.8889 | 0.3733 | 7.7216 | 4.7294 |
| lgbm leaves7 mdl40 | 9 | 2.0725 | 0.1087 | 2342.6667 | 0.2289 | 32.1505 | 28.538 |
| miter leaves127 | 9 | 1.5806 | 0.095 | 2606.2222 | 0.8956 | 0.6214 | -2.2106 |
| miter leaves255 mdl10 | 9 | 1.6092 | 0.0935 | 1793.4444 | 1.0744 | 2.4826 | -0.3601 |
| cat depth10 lr0.05 | 9 | 1.5808 | 0.1364 | 4738.0 | 4.52 | 0.499 | -2.3711 |

### miami

| variant | n | rmse | sd | trees | secs | vs miter base % | vs cat Plain lr0.05 % |
|---|---|---|---|---|---|---|---|
| miter base | 9 | 87298.6011 | 4449.5019 | 768.6667 | 1.5444 | 0.0 | 3.6834 |
| miter accurate | 9 | 87257.1968 | 4709.8116 | 653.4444 | 2.2233 | -0.0588 | 3.6258 |
| miter m_mult=4 | 9 | 87257.1968 | 4709.8116 | 653.4444 | 2.0622 | -0.0588 | 3.6258 |
| miter lr0.03 | 9 | 86908.8211 | 4551.5843 | 965.7778 | 1.9922 | -0.4507 | 3.2113 |
| miter lr0.02 | 9 | 86515.2834 | 4338.5983 | 1514.3333 | 3.18 | -0.8895 | 2.753 |
| miter leaves7 | 9 | 85974.6382 | 3819.6763 | 1652.0 | 1.2644 | -1.4807 | 2.129 |
| miter leaves15 | 9 | 86842.45 | 3734.8199 | 999.7778 | 1.21 | -0.4777 | 3.1678 |
| miter leaves63 | 9 | 88548.3117 | 4945.7185 | 541.4444 | 2.0844 | 1.4202 | 5.1588 |
| miter mdl5 | 9 | 86532.0754 | 4652.7857 | 693.0 | 1.1244 | -0.8832 | 2.7629 |
| miter mdl40 | 9 | 87920.1143 | 4222.4453 | 785.1111 | 2.1667 | 0.7274 | 4.431 |
| miter l2=0.1 | 9 | 87328.9195 | 4707.263 | 685.5556 | 1.3622 | 0.0278 | 3.714 |
| miter l2=3 | 9 | 87430.8949 | 4311.7392 | 765.3333 | 1.5556 | 0.1655 | 3.846 |
| miter l2=10 | 9 | 87455.5606 | 4395.6224 | 830.6667 | 1.7022 | 0.1854 | 3.8659 |
| miter lr0.02 mdl5 | 9 | 85446.7367 | 4328.5881 | 1609.7778 | 2.6133 | -2.1157 | 1.4879 |
| lgbm base | 9 | 90008.1764 | 4307.0115 | 767.1111 | 0.7222 | 3.1348 | 6.9213 |
| lgbm lr0.02 | 9 | 90032.084 | 4343.828 | 1532.6667 | 1.2822 | 3.1653 | 6.9558 |
| lgbm depth6 leaves63 | 9 | 90496.6871 | 3832.3915 | 652.4444 | 0.3911 | 3.7141 | 7.5146 |
| lgbm depth4 leaves15 | 9 | 90056.0442 | 3537.3061 | 1141.7778 | 0.3778 | 3.226 | 6.9915 |
| lgbm mdl5 | 9 | 89032.6604 | 4636.2835 | 823.6667 | 0.6078 | 1.9894 | 5.7358 |
| cat default (auto lr) | 9 | 83941.2718 | 3401.4153 | 4181.0 | 7.5911 | -3.7865 | -0.283 |
| cat lr0.05 | 9 | 84167.2446 | 2934.7331 | 1539.6667 | 2.8644 | -3.5043 | 0.0 |
| cat lr0.02 | 9 | 84117.5237 | 3200.7833 | 3149.0 | 5.7233 | -3.5639 | -0.065 |
| cat Plain lr0.05 | 9 | 84167.2446 | 2934.7331 | 1539.6667 | 2.8711 | -3.5043 | 0.0 |
| cat Ordered lr0.05 | 9 | 88749.6147 | 3450.3485 | 1510.2222 | 9.5256 | 1.7599 | 5.4535 |
| cat Lossguide31 lr0.05 | 9 | 85223.417 | 3038.7969 | 1050.5556 | 3.1378 | -2.3015 | 1.2664 |
| cat Depthwise lr0.05 | 9 | 85601.9502 | 3528.8121 | 995.0 | 4.6133 | -1.8841 | 1.6897 |
| cat l2=1 lr0.05 | 9 | 83633.5086 | 3582.5501 | 1373.1111 | 2.5511 | -4.1263 | -0.6523 |
| cat depth8 lr0.05 | 9 | 86598.9452 | 3369.7281 | 1274.3333 | 6.5644 | -0.6947 | 2.8924 |
| cat Plain lr0.05 no-bootstrap | 9 | 84392.0218 | 3065.5703 | 1741.8889 | 3.2356 | -3.2463 | 0.2692 |
| miter base | log-y | 9 | 89239.3543 | 3608.3378 | 578.1111 | 1.1167 | 2.3063 | 6.026 |
| miter accurate | log-y | 9 | 89125.5867 | 3962.2579 | 524.6667 | 1.5933 | 2.1538 | 5.8874 |
| miter lr0.02 | log-y | 9 | 88443.3971 | 4174.6233 | 1332.6667 | 2.6011 | 1.3835 | 5.0702 |
| lgbm base | log-y | 9 | 91413.2553 | 3331.8878 | 681.1111 | 0.5556 | 4.8297 | 8.6258 |
| cat default (auto lr) | log-y | 9 | 84830.2933 | 3842.6693 | 3784.2222 | 6.7978 | -2.7542 | 0.7808 |
| cat lr0.05 | log-y | 9 | 84947.5728 | 3382.1557 | 1373.3333 | 2.4967 | -2.5963 | 0.9398 |
| miter leaves4 | 9 | 88803.1494 | 4666.533 | 1526.4444 | 0.8578 | 1.7275 | 5.4645 |
| miter leaves7 mdl40 | 9 | 87146.6109 | 4360.0539 | 1377.6667 | 1.1278 | -0.1532 | 3.5167 |
| miter leaves15 mdl40 | 9 | 87127.8295 | 4083.4878 | 1005.6667 | 1.4467 | -0.1679 | 3.4922 |
| miter leaves7 lr0.02 | 9 | 86435.4676 | 3878.6175 | 2999.7778 | 2.3644 | -0.9459 | 2.6739 |
| miter leaves7 l2=10 | 9 | 86430.0039 | 3843.7074 | 1859.5556 | 1.4278 | -0.9493 | 2.6805 |
| miter mdl80 | 9 | 89529.2518 | 4009.194 | 1019.8889 | 3.47 | 2.604 | 6.3621 |
| lgbm leaves7 mdl40 | 9 | 90085.8104 | 3904.357 | 1569.2222 | 0.5644 | 3.2646 | 7.0273 |
| miter leaves127 | 9 | 89275.4338 | 4951.3705 | 537.7778 | 3.7556 | 2.2491 | 6.0231 |
| miter leaves255 mdl10 | 9 | 90032.7625 | 4979.1451 | 528.1111 | 4.78 | 3.1162 | 6.9245 |
| cat depth10 lr0.05 | 9 | 92266.153 | 4650.9689 | 1087.6667 | 17.6778 | 5.8279 | 9.6231 |
