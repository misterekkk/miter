#include "core/booster.hpp"

#include <miter/miter.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/ctr.hpp"
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
    if (params.grow_policy != GrowPolicy::Leafwise && params.grow_policy != GrowPolicy::Symmetric) {
        throw std::invalid_argument("grow_policy must be Leafwise or Symmetric");
    }
    if (params.grow_policy == GrowPolicy::Symmetric) {
        const int sym_depth = (params.max_depth > 0) ? params.max_depth : 6;
        if (sym_depth < 1 || sym_depth > 8) {
            throw std::invalid_argument(
                "For symmetric grow_policy, max_depth must be -1 or in [1, 8]");
        }
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
    if (params.max_cat_to_onehot < 1) {
        throw std::invalid_argument("max_cat_to_onehot must be >= 1");
    }
    if (params.min_data_per_group < 1) {
        throw std::invalid_argument("min_data_per_group must be >= 1");
    }
    if (params.max_categories < 1 || params.max_categories > 253) {
        throw std::invalid_argument("max_categories must be in [1, 253]");
    }
    if (!std::isfinite(params.cat_smooth) || params.cat_smooth < 0.0) {
        throw std::invalid_argument("cat_smooth must be finite and >= 0");
    }
    if (!std::isfinite(params.cat_l2) || params.cat_l2 < 0.0) {
        throw std::invalid_argument("cat_l2 must be finite and >= 0");
    }
    if (params.max_cat_threshold < 1) {
        throw std::invalid_argument("max_cat_threshold must be >= 1");
    }
    if (params.m_mult < 0) {
        throw std::invalid_argument("m_mult must be >= 0");
    }
    if (!(params.feature_fraction > 0.0 && params.feature_fraction <= 1.0)) {
        throw std::invalid_argument("feature_fraction must be in (0, 1]");
    }
    if (!(params.bagging_fraction > 0.0 && params.bagging_fraction <= 1.0)) {
        throw std::invalid_argument("bagging_fraction must be in (0, 1]");
    }
    if (params.k_proj != 0 && (params.k_proj < 2 || params.k_proj > 8)) {
        throw std::invalid_argument("k_proj must be 0 or in [2, 8]");
    }
    if (params.max_ctr_complexity < 0 || params.max_ctr_complexity > 3) {
        throw std::invalid_argument("max_ctr_complexity must be in [0, 3]");
    }
    if (!std::isfinite(params.ctr_prior_weight) || params.ctr_prior_weight <= 0.0) {
        throw std::invalid_argument("ctr_prior_weight must be finite and > 0");
    }
    if (params.ctr_n_permutations < 1) {
        throw std::invalid_argument("ctr_n_permutations must be >= 1");
    }
    if (params.ctr_max_borders < 1 || params.ctr_max_borders > 255) {
        throw std::invalid_argument("ctr_max_borders must be in [1, 255]");
    }
}

double compute_weighted_target_mean(std::span<const double> y,
                                    std::span<const double> sample_weight) noexcept {
    double wy = 0.0;
    double sw = 0.0;
    const bool has_w = !sample_weight.empty();
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double w = has_w ? sample_weight[i] : 1.0;
        wy += w * y[i];
        sw += w;
    }
    return (sw > 0.0) ? (wy / sw) : 0.0;
}

} // namespace

TrainResult fit_booster(const Dataset& ds, Objective objective, std::span<const double> y,
                        std::span<const double> sample_weight, const TransformedData* valid_ds,
                        std::span<const double> y_valid,
                        std::span<const double> valid_sample_weight, const Params& params) {
    validate_params(params);
    if (objective == Objective::Multiclass) {
        const std::uint32_t k_classes = infer_and_validate_multiclass_targets(y, sample_weight, 0U);
        const auto K = static_cast<std::size_t>(k_classes);
        const bool has_valid = (valid_ds != nullptr);
        if (has_valid) {
            if (valid_ds->n_features != ds.n_features() ||
                valid_ds->n_num_features != ds.n_num_features()) {
                throw std::invalid_argument("Validation dataset feature count mismatch");
            }
            if (y_valid.size() != valid_ds->n_rows) {
                throw std::invalid_argument("Validation target y_valid length mismatch");
            }
            infer_and_validate_multiclass_targets(y_valid, valid_sample_weight, k_classes);
        }

        const std::size_t n = ds.n_rows();
        const std::size_t p = ds.n_features();

        Model model;
        model.miter_version = std::string(version());
        model.objective = Objective::Multiclass;
        model.init_score = 0.0;
        model.n_classes = k_classes;
        model.init_scores =
            compute_multiclass_init_scores(y, sample_weight, k_classes, params.n_jobs);
        model.learning_rate = params.learning_rate;
        model.n_features = p;
        model.params = params;
        model.feature_names = ds.feature_names();
        model.feature_kinds.resize(p);
        model.quantizers.resize(p);
        model.cat_features.resize(p);
        std::vector<std::size_t> feature_to_num_idx(p, 0U);

        const auto& kinds = ds.feature_kinds();
        for (std::size_t f = 0; f < p; ++f) {
            model.feature_kinds[f] = (kinds[f] == FeatureKind::Categorical) ? 1U : 0U;
        }
        const auto& num_indices = ds.num_feature_indices();
        for (std::size_t j = 0; j < num_indices.size(); ++j) {
            const std::size_t f = num_indices[j];
            const auto& q = ds.quantizer(j);
            model.quantizers[f] = QuantizerInfo{q.lo, q.scale, q.is_constant};
            feature_to_num_idx[f] = j;
        }
        for (const std::size_t f : ds.cat_feature_indices()) {
            model.cat_features[f].categories = ds.cat_encoder(f).categories;
        }

        std::vector<double> scores(K * n);
        std::vector<double> valid_scores(has_valid ? K * valid_ds->n_rows : 0U);
        for (std::size_t k = 0; k < K; ++k) {
            std::fill(scores.begin() + static_cast<std::ptrdiff_t>(k * n),
                      scores.begin() + static_cast<std::ptrdiff_t>((k + 1U) * n),
                      model.init_scores[k]);
            if (has_valid) {
                std::fill(valid_scores.begin() + static_cast<std::ptrdiff_t>(k * valid_ds->n_rows),
                          valid_scores.begin() +
                              static_cast<std::ptrdiff_t>((k + 1U) * valid_ds->n_rows),
                          model.init_scores[k]);
            }
        }
        std::vector<float> g(K * n);
        std::vector<float> h(K * n);

        TreeWorkspace ws;
        ws.prepare(ds, params);

        const bool ctr_active = (params.max_ctr_complexity > 0 && ds.n_cat_features() > 0U &&
                                 params.grow_policy == GrowPolicy::Symmetric);
        CtrTrainingState ctr_state;
        std::vector<double> mc_class_y;
        std::vector<double> mc_class_prior;
        if (ctr_active) {
            ctr_state.prepare(ds, params);
            mc_class_y.resize(K * n);
            mc_class_prior.assign(K, 0.0);
            const bool has_w = !sample_weight.empty();
            double total_w = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double w = has_w ? sample_weight[i] : 1.0;
                total_w += w;
                const auto yi = static_cast<std::size_t>(y[i]);
                mc_class_y[yi * n + i] = 1.0;
                mc_class_prior[yi] += w;
            }
            if (total_w > 0.0) {
                for (std::size_t k = 0; k < K; ++k) {
                    mc_class_prior[k] /= total_w;
                }
            }
        }

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

            compute_multiclass_gradients(scores.data(), n, k_classes, y, sample_weight, g.data(),
                                         h.data(), params.n_jobs);
            for (std::size_t k = 0; k < K; ++k) {
                CtrGrowContext ctr_ctx;
                if (ctr_active) {
                    ctr_ctx.state = &ctr_state;
                    ctr_ctx.model = &model;
                    ctr_ctx.target_y = mc_class_y.data() + k * n;
                    ctr_ctx.prior = mc_class_prior[k];
                    ctr_ctx.sample_weight = sample_weight.empty() ? nullptr : sample_weight.data();
                    ctr_ctx.target_id = static_cast<std::uint32_t>(k);
                    ctr_ctx.is_unit_target = true;
                }
                Tree tree = grow_tree(ds, g.data() + k * n, h.data() + k * n, params,
                                      static_cast<std::size_t>(t) * K + k, ws,
                                      scores.data() + k * n, ctr_active ? &ctr_ctx : nullptr);
                if (ctr_active) {
                    finalize_pending_ctr_tables(ds, ctr_state, ctr_ctx.target_y, ctr_ctx.prior,
                                                ctr_ctx.sample_weight, params.ctr_prior_weight,
                                                ctr_ctx.target_id, model, ctr_state.scratch);
                }
                model.trees.push_back(std::move(tree));
            }

            result.train_loss.push_back(compute_multiclass_loss(scores.data(), n, k_classes, y,
                                                                sample_weight, params.n_jobs));

            if (has_valid) {
                const std::int16_t* valid_codes =
                    (valid_ds->n_num_features > 0U) ? valid_ds->codes_col_major.data() : nullptr;
                const std::uint32_t* valid_cat_hashes =
                    ctr_active ? valid_ds->cat_hashes_col_major.data() : nullptr;
                const std::size_t tree_base = model.trees.size() - K;
                for (std::size_t k = 0; k < K; ++k) {
                    Evaluator::add_tree_predictions_from_codes(
                        model.trees[tree_base + k], valid_codes, valid_ds->n_rows,
                        valid_ds->n_num_features, feature_to_num_idx,
                        valid_scores.data() + k * valid_ds->n_rows, valid_ds->bins_col_major.data(),
                        params.n_jobs, nullptr, 0U, ctr_active ? &model : nullptr, valid_cat_hashes,
                        ctr_active ? &ws.feature_to_cat_idx : nullptr);
                }
                const double vloss =
                    compute_multiclass_loss(valid_scores.data(), valid_ds->n_rows, k_classes,
                                            y_valid, valid_sample_weight, params.n_jobs);
                result.valid_loss.push_back(vloss);

                if (vloss < best_val - 1e-12) {
                    best_val = vloss;
                    best_iter = model.trees.size() / K;
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
            model.trees.resize(best_iter * K);
        } else {
            model.best_iteration = model.trees.size() / K;
        }

        // Convert class-major scores [k * n + i] -> row-major [i * K + k] to match Evaluator.
        std::vector<double> row_major_scores(K * n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t k = 0; k < K; ++k) {
                row_major_scores[i * K + k] = scores[k * n + i];
            }
        }
        result.model = std::move(model);
        result.final_train_scores = std::move(row_major_scores);
        return result;
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
    model.init_score = compute_init_score(objective, y, sample_weight, params.n_jobs);
    model.learning_rate = params.learning_rate;
    model.n_features = p;
    model.params = params;
    model.feature_names = ds.feature_names();
    model.feature_kinds.resize(p);
    model.quantizers.resize(p);
    model.cat_features.resize(p);
    std::vector<std::size_t> feature_to_num_idx(p, 0U);

    const auto& kinds = ds.feature_kinds();
    for (std::size_t f = 0; f < p; ++f) {
        model.feature_kinds[f] = (kinds[f] == FeatureKind::Categorical) ? 1U : 0U;
    }
    const auto& num_indices = ds.num_feature_indices();
    for (std::size_t j = 0; j < num_indices.size(); ++j) {
        const std::size_t f = num_indices[j];
        const auto& q = ds.quantizer(j);
        model.quantizers[f] = QuantizerInfo{q.lo, q.scale, q.is_constant};
        feature_to_num_idx[f] = j;
    }
    for (const std::size_t f : ds.cat_feature_indices()) {
        model.cat_features[f].categories = ds.cat_encoder(f).categories;
    }

    std::vector<double> scores(n, model.init_score);
    std::vector<double> valid_scores(has_valid ? valid_ds->n_rows : 0U, model.init_score);
    std::vector<float> g(n);
    std::vector<float> h(n);

    TreeWorkspace ws;
    ws.prepare(ds, params);

    const bool ctr_active = (params.max_ctr_complexity > 0 && ds.n_cat_features() > 0U &&
                             params.grow_policy == GrowPolicy::Symmetric);
    CtrTrainingState ctr_state;
    CtrGrowContext ctr_ctx;
    if (ctr_active) {
        ctr_state.prepare(ds, params);
        ctr_ctx.state = &ctr_state;
        ctr_ctx.model = &model;
        ctr_ctx.target_y = y.data();
        ctr_ctx.prior = compute_weighted_target_mean(y, sample_weight);
        ctr_ctx.sample_weight = sample_weight.empty() ? nullptr : sample_weight.data();
        ctr_ctx.target_id = 0U;
        ctr_ctx.is_unit_target = (objective == Objective::Binary);
    }

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

        compute_gradients(objective, scores, y, sample_weight, g.data(), h.data(), params.n_jobs);
        Tree tree = grow_tree(ds, g.data(), h.data(), params, static_cast<std::size_t>(t), ws,
                              scores.data(), ctr_active ? &ctr_ctx : nullptr);
        if (ctr_active) {
            finalize_pending_ctr_tables(ds, ctr_state, ctr_ctx.target_y, ctr_ctx.prior,
                                        ctr_ctx.sample_weight, params.ctr_prior_weight,
                                        ctr_ctx.target_id, model, ctr_state.scratch);
        }
        model.trees.push_back(std::move(tree));

        result.train_loss.push_back(
            compute_loss(objective, scores, y, sample_weight, params.n_jobs));

        if (has_valid) {
            const std::int16_t* valid_codes =
                (valid_ds->n_num_features > 0U) ? valid_ds->codes_col_major.data() : nullptr;
            const std::uint32_t* valid_cat_hashes =
                ctr_active ? valid_ds->cat_hashes_col_major.data() : nullptr;
            Evaluator::add_tree_predictions_from_codes(
                model.trees.back(), valid_codes, valid_ds->n_rows, valid_ds->n_num_features,
                feature_to_num_idx, valid_scores.data(), valid_ds->bins_col_major.data(),
                params.n_jobs, nullptr, 0U, ctr_active ? &model : nullptr, valid_cat_hashes,
                ctr_active ? &ws.feature_to_cat_idx : nullptr);
            const double vloss =
                compute_loss(objective, valid_scores, y_valid, valid_sample_weight, params.n_jobs);
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

TrainResult fit_axis_booster(const Dataset& ds, Objective objective, std::span<const double> y,
                             std::span<const double> sample_weight, const TransformedData* valid_ds,
                             std::span<const double> y_valid,
                             std::span<const double> valid_sample_weight, const Params& params) {
    Params ax = params;
    ax.m_mult = 0;
    return fit_booster(ds, objective, y, sample_weight, valid_ds, y_valid, valid_sample_weight, ax);
}

} // namespace miter
