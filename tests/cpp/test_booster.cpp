#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/booster.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"

namespace {

TEST(BoosterTest, TinyHandComputedRegressionTree) {
    // 4 rows, 1 feature: x = {0, 1, 2, 3}, y = {-1, -1, 1, 1}
    // init_score = 0.0, g = {1, 1, -1, -1}, h = {1, 1, 1, 1}, lambda = 1.0, lr = 0.5
    // Optimal leaves: left {0,1} -> G = 2, H = 2 => val = 0.5 * (-2 / 3) = -1/3
    //                 right {2,3} -> G = -2, H = 2 => val = 0.5 * (2 / 3) = +1/3
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    const std::vector<double> y{-1.0, -1.0, 1.0, 1.0};
    const auto ds = miter::Dataset::fit_numeric(x.data(), 4, 1, false);

    miter::Params params;
    params.n_estimators = 1;
    params.learning_rate = 0.5;
    params.num_leaves = 2;
    params.min_data_in_leaf = 1;
    params.lambda_l2 = 1.0;

    const auto res =
        miter::fit_axis_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, params);
    EXPECT_EQ(res.model.trees.size(), 1U);
    EXPECT_EQ(res.model.best_iteration, 1U);
    EXPECT_DOUBLE_EQ(res.model.init_score, 0.0);
    EXPECT_EQ(res.model.trees[0].n_leaves, 2U);
    EXPECT_EQ(res.model.trees[0].nodes.size(), 3U);

    const auto& root = res.model.trees[0].nodes[0];
    EXPECT_EQ(root.kind, miter::NodeKind::Axis);
    EXPECT_EQ(root.feature, 0U);
    EXPECT_NEAR(res.final_train_scores[0], -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(res.final_train_scores[1], -1.0 / 3.0, 1e-12);
    EXPECT_NEAR(res.final_train_scores[2], 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(res.final_train_scores[3], 1.0 / 3.0, 1e-12);

    std::vector<double> preds(4);
    miter::Evaluator::predict_raw(res.model, x.data(), 4, 1, false, preds.data());
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(preds[i]),
                  std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
    }
}

TEST(BoosterTest, BitwiseExactTrainVsPredictAndJsonRoundTrip) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 rng(2026);
    std::normal_distribution<double> norm(0.0, 1.0);

    const std::size_t n = 120;
    const std::size_t p = 4;
    std::vector<double> x(n * p);
    std::vector<double> y_reg(n);
    std::vector<double> y_bin(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i * p + 0] = norm(rng);
        x[i * p + 1] = 42.0; // constant feature
        x[i * p + 2] = (i % 7 == 0) ? nan : norm(rng) * 3.0;
        x[i * p + 3] = static_cast<double>(i) / 10.0;
        y_reg[i] =
            2.0 * x[i * p + 0] + (std::isnan(x[i * p + 2]) ? -1.0 : x[i * p + 2]) + 0.1 * norm(rng);
        y_bin[i] = (y_reg[i] > 0.0) ? 1.0 : 0.0;
    }
    // Add outliers below lo and above hi
    x[0 * p + 0] = -1000.0;
    x[1 * p + 0] = 1000.0;

    const auto ds = miter::Dataset::fit_numeric(x.data(), n, p, false);

    // Construct an extra matrix with rows placed EXACTLY on axis thresholds + saturated + NaNs
    std::vector<double> x_probe = x;
    for (std::size_t f : {std::size_t{0}, std::size_t{2}, std::size_t{3}}) {
        const auto& binner = ds.binner(f);
        const auto& q = ds.quantizer(f);
        if (!binner.edges.empty() && !q.is_constant) {
            const std::uint16_t edge_u = binner.edges[binner.edges.size() / 2];
            const double exact_val = q.lo + static_cast<double>(edge_u) / q.scale;
            x_probe[(10 + f) * p + f] = exact_val;
        }
    }

    for (const miter::Objective obj : {miter::Objective::Regression, miter::Objective::Binary}) {
        const auto& y = (obj == miter::Objective::Regression) ? y_reg : y_bin;
        miter::Params params;
        params.n_estimators = 15;
        params.learning_rate = 0.1;
        params.num_leaves = 7;
        params.max_depth = 3;
        params.min_data_in_leaf = 5;

        const auto res = miter::fit_axis_booster(ds, obj, y, {}, nullptr, {}, {}, params);

        std::vector<double> pred_train(n);
        miter::Evaluator::predict_raw(res.model, x.data(), n, p, false, pred_train.data());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_train[i]),
                      std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
        }

        // JSON round-trip bitwise exactness on probe matrix
        const std::string json_str = miter::model_to_json(res.model);
        const auto loaded = miter::model_from_json(json_str);
        std::vector<double> pred_orig(n);
        std::vector<double> pred_loaded(n);
        miter::Evaluator::predict_raw(res.model, x_probe.data(), n, p, false, pred_orig.data());
        miter::Evaluator::predict_raw(loaded, x_probe.data(), n, p, false, pred_loaded.data());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_orig[i]),
                      std::bit_cast<std::uint64_t>(pred_loaded[i]));
        }

        // File save/load round-trip
        const auto tmp_path =
            std::filesystem::temp_directory_path() / "miter_test_model_roundtrip.json";
        miter::save_model_json(res.model, tmp_path.string());
        const auto loaded_file = miter::load_model_json(tmp_path.string());
        std::filesystem::remove(tmp_path);
        std::vector<double> pred_file(n);
        miter::Evaluator::predict_raw(loaded_file, x_probe.data(), n, p, false, pred_file.data());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_orig[i]),
                      std::bit_cast<std::uint64_t>(pred_file[i]));
        }
    }
}

TEST(BoosterTest, EarlyStoppingAndSampleWeightsAndJsonValidation) {
    // Noisy validation set so early stopping triggers before n_estimators
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const std::size_t n_tr = 80;
    const std::size_t n_va = 40;
    std::vector<double> x_tr(n_tr * 2);
    std::vector<double> y_tr(n_tr);
    std::vector<double> w_tr(n_tr);
    for (std::size_t i = 0; i < n_tr; ++i) {
        x_tr[i * 2 + 0] = uni(rng);
        x_tr[i * 2 + 1] = uni(rng);
        y_tr[i] = 3.0 * x_tr[i * 2 + 0] + 0.5 * uni(rng);
        w_tr[i] = 1.0 + static_cast<double>(i % 3);
    }
    std::vector<double> x_va(n_va * 2);
    std::vector<double> y_va(n_va);
    for (std::size_t i = 0; i < n_va; ++i) {
        x_va[i * 2 + 0] = uni(rng);
        x_va[i * 2 + 1] = uni(rng);
        y_va[i] = 3.0 * x_va[i * 2 + 0] + 0.5 * uni(rng);
    }

    const auto ds = miter::Dataset::fit_numeric(x_tr.data(), n_tr, 2, false);
    const auto tx_va = ds.transform_numeric(x_va.data(), n_va, 2, false);

    miter::Params params;
    params.n_estimators = 100;
    params.learning_rate = 0.3;
    params.num_leaves = 8;
    params.min_data_in_leaf = 4;
    params.early_stopping_rounds = 5;

    const auto res = miter::fit_axis_booster(ds, miter::Objective::Regression, y_tr, w_tr, &tx_va,
                                             y_va, {}, params);
    EXPECT_LT(res.model.trees.size(), 100U);
    EXPECT_EQ(res.model.trees.size(), res.model.best_iteration);
    EXPECT_GE(res.model.best_iteration, 1U);

    // Corrupted JSON rejected cleanly
    EXPECT_THROW(miter::model_from_json("{invalid"), std::runtime_error);
    EXPECT_THROW(miter::model_from_json(R"({"format_version": 999})"), std::runtime_error);
}

} // namespace
