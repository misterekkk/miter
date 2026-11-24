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
    int m_mult = 2;
    int k_proj = 0;
    int oblique_max_depth = -1;
    int early_stopping_rounds = 50;
    double time_limit = 0.0;
    int n_jobs = -1;
    std::uint64_t random_state = 0;
};

} // namespace miter
