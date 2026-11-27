#pragma once

#include <cstdint>

namespace miter {

/**
 * @brief Tree growth strategy: best-first leaf-wise (default) or level-wise symmetric (oblivious).
 */
enum class GrowPolicy : std::uint8_t {
    Leafwise = 0,
    Symmetric = 1,
};

/**
 * @brief Numeric feature quantization strategy to 16-bit codes.
 */
enum class QuantizerMode : std::uint8_t {
    Linear = 0,
    Quantile = 1,
    Auto = 2,
};

inline constexpr int kCtrComplexityAuto = -1;

/**
 * @brief Training and model hyperparameters for miter.
 *
 * Default values are the research protocol defaults (see ARCHITECTURE.md).
 */
struct Params {
    int n_estimators = 5000;
    double learning_rate = 0.05;
    int num_leaves = 31;
    int max_depth = -1;
    GrowPolicy grow_policy = GrowPolicy::Leafwise;
    int min_data_in_leaf = 20;
    double min_sum_hessian_in_leaf = 1e-3;
    double lambda_l2 = 1.0;
    double min_gain_to_split = 0.0;
    int max_bin = 255;
    int max_cat_to_onehot = 4;
    int min_data_per_group = 100;
    int max_categories = 253;
    double cat_smooth = 10.0;
    double cat_l2 = 10.0;
    int max_cat_threshold = 32;
    int m_mult = 2;
    int k_proj = 0;
    int oblique_max_depth = -1;
    int early_stopping_rounds = 50;
    double time_limit = 0.0;
    int n_jobs = -1;
    std::uint64_t random_state = 0;
    /// Per-tree fraction of features (axis + categorical candidates and oblique pool), in (0, 1].
    double feature_fraction = 1.0;
    /// Per-tree fraction of rows (exact-size sample without replacement), in (0, 1].
    double bagging_fraction = 1.0;
    /// Axis candidates: one random valid threshold per feature instead of the best one.
    bool extra_trees = false;
    /// Numeric features with fewer distinct training values than this are left out of oblique
    /// projections (0 = off; 3 excludes binary features such as one-hot columns).
    int oblique_min_distinct = 0;
    /// Symmetric trees: when a level has more rows than this, oblique directions are chosen on a
    /// deterministic strided sample of this many rows; the winner's threshold and gain are then
    /// computed exactly on all rows (0 = off).
    int oblique_sample_rows = 0;
    /// CatBoost-style native online ordered CTR max combination complexity
    /// (-1 = auto, 0 = off, 1..3).
    int max_ctr_complexity = 0;
    /// Smoothing prior weight for native online ordered CTRs (> 0).
    double ctr_prior_weight = 1.0;
    /// Number of deterministic permutations cycled across trees for native CTRs (>= 1).
    int ctr_n_permutations = 1;
    /// Maximum split borders discretized per native CTR candidate (in [1, 255]).
    int ctr_max_borders = 15;
    /// Sample-size-aware complexity regularization weight for native CTR combinations (>= 0).
    double ctr_model_size_reg = 0.5;
    /// Per-feature 16-bit numeric quantization mode (Linear, Quantile, or Auto).
    QuantizerMode quantizer_mode = QuantizerMode::Linear;
};

} // namespace miter
