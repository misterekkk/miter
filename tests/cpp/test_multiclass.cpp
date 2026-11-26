#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/booster.hpp"
#include "core/objective.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"
#include "model/lignum_export.hpp"

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

TEST(MulticlassTest, HandComputedGradientsInitScoresAndLoss) {
    // n = 4, K = 3, labels {0, 1, 2, 0}.
    const std::vector<double> y = {0.0, 1.0, 2.0, 0.0};
    const auto init = miter::compute_multiclass_init_scores(y, {}, 3U, 1);
    ASSERT_EQ(init.size(), 3U);
    // Counts: [2, 1, 1] -> log probs: [log(0.5), log(0.25), log(0.25)], centered by mean.
    const double lp0 = std::log(0.5);
    const double lp1 = std::log(0.25);
    const double mean_lp = (lp0 + lp1 + lp1) / 3.0;
    EXPECT_NEAR(init[0], lp0 - mean_lp, 1e-12);
    EXPECT_NEAR(init[1], lp1 - mean_lp, 1e-12);
    EXPECT_NEAR(init[2], lp1 - mean_lp, 1e-12);

    // Scores class-major [k * n + i]: set row 0 -> [1.0, 0.0, 0.0].
    const std::vector<double> scores = {
        1.0, 0.0, 0.0, 0.0, // class 0 across 4 rows
        0.0, 0.0, 0.0, 0.0, // class 1
        0.0, 0.0, 0.0, 0.0, // class 2
    };
    std::vector<float> g(12U);
    std::vector<float> h(12U);
    miter::compute_multiclass_gradients(scores.data(), 4U, 3U, y, {}, g.data(), h.data(), 1);

    const double e1 = std::exp(1.0);
    const double p0 = e1 / (e1 + 2.0);
    const double p1 = 1.0 / (e1 + 2.0);
    EXPECT_NEAR(g[0U * 4U + 0U], static_cast<float>(p0 - 1.0), 1e-6F);
    EXPECT_NEAR(g[1U * 4U + 0U], static_cast<float>(p1), 1e-6F);
    EXPECT_NEAR(g[2U * 4U + 0U], static_cast<float>(p1), 1e-6F);
    EXPECT_NEAR(h[0U * 4U + 0U], static_cast<float>(2.0 * p0 * (1.0 - p0)), 1e-6F);

    const double expected_loss = ((std::log(e1 + 2.0) - 1.0) + 3.0 * std::log(3.0)) / 4.0;
    EXPECT_NEAR(miter::compute_multiclass_loss(scores.data(), 4U, 3U, y, {}, 1), expected_loss,
                1e-12);
}

TEST(MulticlassTest, TargetValidationRejectsInvalidLabels) {
    const std::vector<double> X = {0.0, 1.0, 2.0, 3.0};
    const auto ds = miter::Dataset::fit_numeric(X.data(), 4U, 1U, false);
    miter::Params prm{};
    prm.early_stopping_rounds = 0;

    const std::vector<double> single_y = {0.0, 0.0, 0.0, 0.0};
    const std::vector<double> neg_y = {-1.0, 0.0, 1.0, 1.0};
    const std::vector<double> frac_y = {0.0, 0.5, 1.0, 2.0};
    // Single distinct class -> throws.
    EXPECT_THROW(
        miter::fit_booster(ds, miter::Objective::Multiclass, single_y, {}, nullptr, {}, {}, prm),
        std::invalid_argument);
    // Negative label -> throws.
    EXPECT_THROW(
        miter::fit_booster(ds, miter::Objective::Multiclass, neg_y, {}, nullptr, {}, {}, prm),
        std::invalid_argument);
    // Non-integer label -> throws.
    EXPECT_THROW(
        miter::fit_booster(ds, miter::Objective::Multiclass, frac_y, {}, nullptr, {}, {}, prm),
        std::invalid_argument);
}

TEST(MulticlassTest, BitwiseExactnessConvergenceAndJsonRoundtripMixed) {
    std::mt19937_64 rng(2026);
    std::normal_distribution<double> norm(0.0, 0.4);
    constexpr std::size_t kN = 450U;
    constexpr std::size_t kPNum = 3U;
    std::vector<double> num_cols(kPNum * kN);
    std::vector<std::int32_t> cat0(kN);
    std::vector<double> y(kN);
    std::vector<double> sw(kN);
    const double centers[3][2] = {{-1.5, -1.0}, {1.5, -1.0}, {0.0, 1.8}};

    for (std::size_t i = 0; i < kN; ++i) {
        const std::size_t cls = i % 3U;
        num_cols[0U * kN + i] = (i % 23U == 0U) ? kNan : (centers[cls][0] + norm(rng));
        num_cols[1U * kN + i] = centers[cls][1] + norm(rng);
        num_cols[2U * kN + i] = 7.0; // constant feature
        cat0[i] = (i % 19U == 0U) ? -1 : static_cast<std::int32_t>(cls);
        y[i] = static_cast<double>(cls);
        sw[i] = 1.0 + 0.5 * static_cast<double>(cls);
    }

    const std::vector<miter::FeatureKind> kinds = {
        miter::FeatureKind::Numeric, miter::FeatureKind::Numeric, miter::FeatureKind::Numeric,
        miter::FeatureKind::Categorical};
    const std::vector<std::string> names = {"x0", "x1", "const_x", "c0"};
    std::vector<miter::NumericColumnView> num_views(kPNum);
    for (std::size_t j = 0; j < kPNum; ++j) {
        num_views[j].f64 = num_cols.data() + j * kN;
        num_views[j].row_stride = 1U;
    }
    std::vector<miter::CategoricalColumnView> cat_views(1U);
    cat_views[0].vocab = {"k0", "k1", "k2"};
    cat_views[0].indices = cat0.data();

    const auto ds = miter::Dataset::fit_mixed(kN, kinds, names, num_views, cat_views);

    for (const auto policy : {miter::GrowPolicy::Leafwise, miter::GrowPolicy::Symmetric}) {
        miter::Params prm{};
        prm.grow_policy = policy;
        prm.num_leaves = 15;
        prm.max_depth = 4;
        prm.min_data_in_leaf = (policy == miter::GrowPolicy::Symmetric) ? 1 : 10;
        prm.lambda_l2 = 1.0;
        prm.m_mult = 2;
        prm.bagging_fraction = 0.85;
        prm.feature_fraction = 0.85;
        prm.learning_rate = 0.15;
        prm.n_estimators = 12;
        prm.early_stopping_rounds = 0;
        prm.random_state = 17U;
        prm.n_jobs = 1;

        const auto res =
            miter::fit_booster(ds, miter::Objective::Multiclass, y, sw, nullptr, {}, {}, prm);
        ASSERT_EQ(res.model.n_classes, 3U);
        ASSERT_EQ(res.model.trees.size(), 36U);
        ASSERT_LT(res.train_loss.back(), res.train_loss.front() * 0.4);
        EXPECT_THROW(miter::export_raw_forest(res.model), std::invalid_argument);

        std::vector<double> eval_raw(kN * 3U);
        miter::Evaluator::predict_raw_mixed(res.model, kN, num_views, cat_views, eval_raw.data());

        const std::string json = miter::model_to_json(res.model);
        const miter::Model loaded = miter::model_from_json(json);
        EXPECT_EQ(miter::model_to_json(loaded), json);

        std::vector<double> loaded_raw(kN * 3U);
        miter::Evaluator::predict_raw_mixed(loaded, kN, num_views, cat_views, loaded_raw.data());

        for (std::size_t idx = 0; idx < kN * 3U; ++idx) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(eval_raw[idx]),
                      std::bit_cast<std::uint64_t>(res.final_train_scores[idx]));
            EXPECT_EQ(std::bit_cast<std::uint64_t>(loaded_raw[idx]),
                      std::bit_cast<std::uint64_t>(eval_raw[idx]));
        }

        std::vector<double> proba(kN * 3U);
        miter::Evaluator::predict_proba_softmax(eval_raw.data(), kN, 3U, proba.data());
        std::size_t correct = 0U;
        for (std::size_t i = 0; i < kN; ++i) {
            const double s = proba[i * 3U + 0U] + proba[i * 3U + 1U] + proba[i * 3U + 2U];
            EXPECT_NEAR(s, 1.0, 1e-12);
            std::size_t best_k = 0U;
            for (std::size_t k = 1U; k < 3U; ++k) {
                if (proba[i * 3U + k] > proba[i * 3U + best_k]) {
                    best_k = k;
                }
            }
            if (best_k == static_cast<std::size_t>(y[i])) {
                ++correct;
            }
        }
        EXPECT_GT(static_cast<double>(correct) / static_cast<double>(kN), 0.90);
    }
}

TEST(MulticlassTest, ThreadDeterminism1Vs4OnLargeDataset) {
    std::mt19937_64 rng(91);
    std::normal_distribution<double> norm(0.0, 1.0);
    constexpr std::size_t kN = 40000U;
    constexpr std::size_t kP = 4U;
    std::vector<double> X(kN * kP);
    std::vector<double> y(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        for (std::size_t j = 0; j < kP; ++j) {
            X[i * kP + j] = (i % 59U == j) ? kNan : norm(rng);
        }
        const double x0 = std::isnan(X[i * kP]) ? 0.0 : X[i * kP];
        const double x1 = std::isnan(X[i * kP + 1U]) ? 0.0 : X[i * kP + 1U];
        const double angle = x0 + 0.5 * x1;
        y[i] = (angle < -0.5) ? 0.0 : ((angle < 0.5) ? 1.0 : ((x1 > 0.0) ? 2.0 : 3.0));
    }
    const auto ds = miter::Dataset::fit_numeric(X.data(), kN, kP, false);

    miter::Params p1{};
    p1.n_estimators = 4;
    p1.early_stopping_rounds = 0;
    p1.m_mult = 2;
    p1.random_state = 5U;
    p1.n_jobs = 1;

    miter::Params p4 = p1;
    p4.n_jobs = 4;

    const auto r1 =
        miter::fit_booster(ds, miter::Objective::Multiclass, y, {}, nullptr, {}, {}, p1);
    const auto r4 =
        miter::fit_booster(ds, miter::Objective::Multiclass, y, {}, nullptr, {}, {}, p4);

    auto m4_norm = r4.model;
    m4_norm.params.n_jobs = 1;
    EXPECT_EQ(miter::model_to_json(r1.model), miter::model_to_json(m4_norm));
    for (std::size_t idx = 0; idx < r1.final_train_scores.size(); ++idx) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(r1.final_train_scores[idx]),
                  std::bit_cast<std::uint64_t>(r4.final_train_scores[idx]));
    }
}

} // namespace
