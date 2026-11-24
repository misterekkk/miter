#include "core/booster.hpp"

#include <miter/miter.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/objective.hpp"
#include "core/tree_learner.hpp"
#include "model/evaluator.hpp"

namespace miter {

namespace {

void validate_params(const Params& params) {
    if (params.n_estimators < 1) {
        throw std::invalid_argument("n_estimators must be >= 1");
    }
    if (!std::isfinite(params.learning_rate) || params.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and > 0");
    }
    if (params.num_leaves < 2) {
        throw std::invalid_argument("num_leaves must be >= 2");
    }
    if (params.min_data_in_leaf < 1) {
        throw std::invalid_argument("min_data_in_leaf must be >= 1");
    }
    if (!std::isfinite(params.min_sum_hessian_in_leaf) || params.min_sum_hessian_in_leaf < 0.0) {
        throw std::invalid_argument("min_sum_hessian_in_leaf must be finite and >= 0");
    }
    if (!std::isfinite(params.lambda_l2) || params.lambda_l2 < 0.0) {
        throw std::invalid_argument("lambda_l2 must be finite and >= 0");
    }
    if (!std::isfinite(params.min_gain_to_split) || params.min_gain_to_split < 0.0) {
        throw std::invalid_argument("min_gain_to_split must be finite and >= 0");
    }
    if (params.max_bin < 2 || params.max_bin > 255) {
        throw std::invalid_argument("max_bin must be in [2, 255]");
    }
}

} // namespace

TrainResult fit_axis_booster(const Dataset& ds, Objective objective, std::span<const double> y,
                             std::span<const double> sample_weight, const TransformedData* valid_ds,
                             std::span<const double> y_valid,
                             std::span<const double> valid_sample_weight, const Params& params) {
    validate_params(params);
    if (ds.n_cat_features() > 0U) {
        throw std::invalid_argument(
            "M2 axis booster supports numeric features only (categorical splits come in M4)");
    }
    if (y.size() != ds.n_rows()) {
        throw std::invalid_argument("Target array y length must match dataset n_rows");
    }
    validate_targets_and_weights(objective, y, sample_weight);

    const bool has_valid = (valid_ds != nullptr);
    if (has_valid) {
        if (valid_ds->n_features != ds.n_features() ||
            valid_ds->n_num_features != ds.n_num_features()) {
            throw std::invalid_argument("Validation dataset feature count mismatch");
        }
        if (y_valid.size() != valid_ds->n_rows) {
            throw std::invalid_argument("Validation target y_valid length mismatch");
        }
        validate_targets_and_weights(objective, y_valid, valid_sample_weight);
    }

    const std::size_t n = ds.n_rows();
    const std::size_t p = ds.n_features();

    Model model;
    model.miter_version = std::string(version());
    model.objective = objective;
    model.init_score = compute_init_score(objective, y, sample_weight);
    model.learning_rate = params.learning_rate;
    model.n_features = p;
    model.params = params;
    model.feature_names = ds.feature_names();
    model.feature_kinds.assign(p, 0U);
    model.quantizers.resize(p);
    model.cat_features.resize(p);
    for (std::size_t f = 0; f < p; ++f) {
        const auto& q = ds.quantizer(f);
        model.quantizers[f] = QuantizerInfo{q.lo, q.scale, q.is_constant};
    }

    std::vector<std::size_t> feature_to_num_idx(p);
    for (std::size_t f = 0; f < p; ++f) {
        feature_to_num_idx[f] = f;
    }

    std::vector<double> scores(n, model.init_score);
    std::vector<double> valid_scores(has_valid ? valid_ds->n_rows : 0U, model.init_score);
    std::vector<float> g(n);
    std::vector<float> h(n);

    TreeWorkspace ws;
    ws.prepare(ds, params.num_leaves);

    TrainResult result;
    result.train_loss.reserve(static_cast<std::size_t>(params.n_estimators));
    if (has_valid) {
        result.valid_loss.reserve(static_cast<std::size_t>(params.n_estimators));
    }

    double best_val = std::numeric_limits<double>::infinity();
    std::size_t best_iter = 0U;
    int since_best = 0;
    const auto start_clock = std::chrono::steady_clock::now();

    for (int t = 0; t < params.n_estimators; ++t) {
        if (params.time_limit > 0.0 && t > 0) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start_clock)
                    .count();
            if (elapsed >= params.time_limit) {
                break;
            }
        }

        compute_gradients(objective, scores, y, sample_weight, g.data(), h.data());
        Tree tree = grow_axis_tree(ds, g.data(), h.data(), params, ws, scores.data());
        model.trees.push_back(std::move(tree));

        result.train_loss.push_back(compute_loss(objective, scores, y, sample_weight));

        if (has_valid) {
            Evaluator::add_tree_predictions_from_codes(
                model.trees.back(), valid_ds->codes_row_major.data(), valid_ds->n_rows,
                valid_ds->n_num_features, feature_to_num_idx, valid_scores.data());
            const double vloss =
                compute_loss(objective, valid_scores, y_valid, valid_sample_weight);
            result.valid_loss.push_back(vloss);

            if (vloss < best_val - 1e-12) {
                best_val = vloss;
                best_iter = model.trees.size();
                since_best = 0;
            } else {
                ++since_best;
                if (params.early_stopping_rounds > 0 &&
                    since_best >= params.early_stopping_rounds) {
                    break;
                }
            }
        }
    }

    if (has_valid && best_iter > 0U) {
        model.best_iteration = best_iter;
        model.trees.resize(best_iter);
    } else {
        model.best_iteration = model.trees.size();
    }

    result.model = std::move(model);
    result.final_train_scores = std::move(scores);
    return result;
}

} // namespace miter
