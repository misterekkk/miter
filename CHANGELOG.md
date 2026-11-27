# Changelog

All notable changes to miter. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **Faster training on large data:** `oblique_sample_rows=65536` is now the default (symmetric
  trees).
  - **How it works:** above 65,536 rows, the oblique directions are chosen on a strided sample of
    the rows. The winning direction's threshold and gain are then computed exactly on all rows, in
    parallel and deterministically.
  - **Speed:** training is about 2.4× faster on 500k rows (100 trees: 19.7 s → 8.3 s on one
    thread, 4.5 s → 2.2 s on 10 threads, Apple M4) and 2.4× faster on covertype.
  - **Quality:** the pre-registered quality checks, run with the much more aggressive 4,096-row
    sample, showed no loss: median 0.00% on 32 held-out OpenML datasets and on 51 TabArena tasks
    (`benchmarks/results/exp_speedups_e2_*`).

- **Faster training on one-hot and sparse columns:** `efb=True` is now the default (symmetric
  trees).
  - **How it works:** numeric features that are never non-default in the same row (e.g. one-hot
    columns) share one axis-histogram pass. Each feature's histogram is recovered exactly, so the
    models are unchanged (bit-identical on all 160 held-out folds and 51 TabArena tasks).
  - **Speed:** covertype (44 of 54 columns one-hot), 100 trees, trains 2.1× faster on one thread
    (22.5 s → 10.5 s). Together with the row sampling above, that is 5.4× faster than 0.1.0
    (57 s → 10.5 s).
- **Faster histograms, models unchanged:** up to four histograms are filled per sweep over the
  rows, the oblique row sample is gathered once per fit, and its refit is vectorized. Saved models
  and predictions are bit-identical to before. 100 trees, one thread: 30% faster on covertype and
  13–19% on 500k × 50 (x86_64 and aarch64 Linux, Apple M4).

### Removed

Options that did nothing, or that failed their pre-registered test and stayed off by default.
Models saved with them by 0.1.0 still load (the options only affected training).

- `robust` (raised for `True`) and `cat_mode` (accepted only `"k1"`).
- `cat_oblique_ts` and `cat_ts_smooth`: out-of-fold target statistics in oblique projections
  (median −0.04%, p = 0.52; `benchmarks/results/exp_cat_oblique_ts_summary.md`). Interactions
  between categories are covered by the CTR splits (`max_ctr_complexity`).
- `oblique_min_distinct` (E1), `oblique_binary_penalty` and `oblique_gain_margin` (E4): no
  effect on held-out quality (`benchmarks/results/exp_speedups_e{1,4}_summary.md`).
- `k_proj` above 3: never tested; `k_proj` is now 0 (2 or 3 at random), 2 or 3, which lignum
  always holds.

## [0.1.0] — 2026-11-27

The first public release.

### Added

- **Gradient boosting with random sparse oblique splits.**
  - Every node tries axis thresholds, categorical splits and
    `m_mult · ceil(sqrt(p))` random 2–3-feature projections `w·x ≥ t`.
  - Projections use int16 weights on 16-bit feature codes with 64 equal-width bins on the node's
    own range, so training and prediction agree bit for bit.
- **Two tree shapes:** symmetric / oblivious (`grow_policy="symmetric"`, `max_depth` 1–8) and
  leaf-wise (`grow_policy="leafwise"`). Oblique and categorical splits work in both.
- **Default: symmetric trees of depth 6** with `min_data_in_leaf=1`, `lambda_l2=3` and `m_mult=4`.
  - **Why:** on 32 OpenML datasets with no overlap with TabArena, this beat the earlier leaf-wise
    default on 27 (median −3.2% error, p < 0.0001). It also beat CatBoost's default on 23 (median
    −1.6%, p = 0.005), in about 0.4× CatBoost's training time.
  - **Leaf-wise defaults:** `min_data_in_leaf`, `lambda_l2` and `m_mult` default to `None` and
    resolve per tree shape, so leaf-wise trees keep 20 / 1.0 / 2.
- **Objectives:** L2 regression, binary log-loss and multiclass softmax, all with sample weights.
- **Inputs:**
  - Native missing values.
  - Native categorical features: LightGBM-style K1 splits, and CatBoost-style online ordered CTRs
    with combinations of up to 3 features (`max_ctr_complexity`, `"auto"` picks it from the data).
- **Quantizers:** linear Q16r (default), `quantile` and per-feature `auto`.
- **Training options:**
  - Early stopping on `eval_set`, `time_limit`.
  - `preset="fast"` / `"accurate"`.
  - `feature_fraction`, `bagging_fraction`, `extra_trees`.
- **Determinism:** the same `random_state` gives a bit-identical model for any number of threads.
- **Prediction through [lignum](https://github.com/misterekkk/lignum) 0.5,** bit-identical to
  miter's own evaluator and faster than LightGBM's own `predict` on the same tree count.
- **Python package:**
  - `MiterRegressor` and `MiterClassifier`, scikit-learn compatible (they pass `check_estimator`).
  - Exact pickling and JSON `save_model` / `load_model`.
  - `miter.tabarena.MiterModel` for AutoGluon and TabArena.
- **Wheels** for Linux (x86_64, aarch64), macOS 15+ (arm64, x86_64) and Windows (x64), Python 3.10+ (abi3 from
  3.12).
- **License:** Apache License 2.0.
