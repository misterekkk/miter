#pragma once

#include <cstdint>

namespace miter {

/**
 * @brief Training and model hyperparameters for miter.
 *
 * Default values match the frozen research protocol in docs/IMPLEMENTATION_PLAN.md (Section 2.6).
 */
struct Params {
    int n_estimators = 5000;
    double learning_rate = 0.05;
    int num_leaves = 31;
    int max_depth = -1;
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
};

} // namespace miter
