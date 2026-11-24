# Pre-registration: miter C++ vs histree prototype, binary classification parity

Frozen before any run of `benchmarks/check_m3_clf_parity.py` (SHA-256 recorded in `m3_clf_parity_summary.md`).

**Hypothesis (H):** on the 13 OpenML binary classification datasets, the median relative test log-loss difference
(miter - histree) / histree over datasets satisfies |median| <= 1.0 %.

**Datasets (OpenML suite 337, same 13 as `bench_m3_oblique.py` Part C):** eye_movements 44130, heloc 45026, pol 44122,
bank-marketing 44126, default-of-credit-card-clients 45020, MagicTelescope 44125, house_16H 44123, credit 44089,
california 45028, electricity 44120, jannis 45021, Diabetes130US 45022, MiniBooNE 44128.

**Splits:** OpenML outer split (repeat 0, fold f) for f in 0..4 (reduced to 0..2 if the pilot projects > ~2.5 h wall;
the choice is made on pilot time only, not on results). Inner train/valid: `train_test_split(train_idx, test_size=0.2,
random_state=0)` (not stratified), identical for both implementations. Seed = fold.

**Hyperparameters (both):** learning rate 0.05, 31 leaves, min_data_in_leaf 20, lambda_l2 1.0, min_gain 0, 255 bins,
m = 2*ceil(sqrt(p)), random k in {2,3} features per projection, early stopping 50 rounds on validation log-loss,
max 5000 trees, 1 thread.

**Metric:** test log-loss (predictions clipped to [1e-15, 1-1e-15]). Per dataset: mean over folds. Relative difference
= 100*(miter - histree)/histree, so negative = miter better. Aggregates over datasets: mean, MEDIAN, wins x/13
(miter lower), Wilcoxon signed-rank p (two-sided, on the per-dataset means).

**Primary prototype:** `histree.gb.oblique.ObliqueGBRegressor(objective="binary", selector="R", bin_sample=-3)`
(float R_uni, the configuration of research script b4_run.py; no quantization). Reason: `ObliqueGBInt` is regression-only
(L2 fit loop). **Secondary, exploratory (not part of H):** the prototype's integer kernel `build_tree_int` driven by a
logistic-loss boosting loop written in the check script, with Q16r inputs (quantiles 0.01%/99.99%).

Verdict rule: H supported iff |median| <= 1.0 % for the primary comparison; otherwise reported as not supported.
Everything else (per-dataset outliers, secondary comparison, AUC if reported) is exploratory.
