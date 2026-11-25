#include <lignum/lignum.hpp>

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
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"
#include "model/lignum_bridge.hpp"

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

TEST(SymmetricTreeTest, StructureAndExactHandComputedTinyRegression) {
    // n = 8, p = 2, depth = 2 (4 leaves), no regularization (lam = 0), m_mult = 0 (axis-only).
    // f0 encodes left/right sign (-2 vs +2), f1 encodes inner (-1 vs +1).
    const std::vector<double> X = {
        -2.0, -1.0, // row 0 -> quadrant (L, L): y = 0
        -2.0, -1.0, // row 1 -> (L, L): y = 0
        -2.0, 1.0,  // row 2 -> (L, R): y = 4
        -2.0, 1.0,  // row 3 -> (L, R): y = 4
        2.0,  -1.0, // row 4 -> (R, L): y = 8
        2.0,  -1.0, // row 5 -> (R, L): y = 8
        2.0,  1.0,  // row 6 -> (R, R): y = 12
        2.0,  1.0,  // row 7 -> (R, R): y = 12
    };
    const std::vector<double> y = {0.0, 0.0, 4.0, 4.0, 8.0, 8.0, 12.0, 12.0};
    const auto ds = miter::Dataset::fit_numeric(X.data(), 8U, 2U, false);

    miter::Params prm{};
    prm.grow_policy = miter::GrowPolicy::Symmetric;
    prm.max_depth = 2;
    prm.min_data_in_leaf = 1;
    prm.lambda_l2 = 0.0;
    prm.m_mult = 0;
    prm.learning_rate = 1.0;
    prm.n_estimators = 1;
    prm.early_stopping_rounds = 0;
    prm.n_jobs = 1;

    const auto res =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm);
    ASSERT_EQ(res.model.trees.size(), 1U);
    const miter::Tree& tree = res.model.trees[0];
    ASSERT_EQ(tree.n_leaves, 4U);
    ASSERT_EQ(tree.nodes.size(), 7U);

    // Every level must share identical split params and complete binary child layout.
    EXPECT_EQ(tree.nodes[1].feature, tree.nodes[2].feature);
    EXPECT_EQ(tree.nodes[1].threshold_u16, tree.nodes[2].threshold_u16);
    for (std::size_t u = 0; u < 3U; ++u) {
        EXPECT_EQ(tree.nodes[u].kind, miter::NodeKind::Axis);
        EXPECT_EQ(tree.nodes[u].left_child, 2U * u + 1U);
        EXPECT_EQ(tree.nodes[u].right_child, 2U * u + 2U);
    }

    // With learning_rate = 1.0 and lam = 0, a single depth-2 tree fits exact quadrant means.
    std::vector<double> preds(8);
    miter::Evaluator::predict_raw(res.model, X.data(), 8U, 2U, false, preds.data());
    for (std::size_t i = 0; i < 8U; ++i) {
        EXPECT_NEAR(preds[i], y[i], 1e-12);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(preds[i]),
                  std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
    }
}

TEST(SymmetricTreeTest, EdgeCasesConstantNanTinyAndEmptyLeaves) {
    const std::vector<double> X = {
        1.0, kNan, 0.0, // row 0: const col 0, all-NaN col 1
        1.0, kNan, 1.0, // row 1
        1.0, kNan, 2.0, // row 2
    };
    const std::vector<double> y = {1.0, 2.0, 5.0};
    const auto ds = miter::Dataset::fit_numeric(X.data(), 3U, 3U, false);

    miter::Params prm{};
    prm.grow_policy = miter::GrowPolicy::Symmetric;
    prm.max_depth = 3; // 8 leaves for only 3 rows -> leaves with count 0 must cleanly get 0.0
    prm.min_data_in_leaf = 1;
    prm.lambda_l2 = 1.0;
    prm.n_estimators = 2;
    prm.early_stopping_rounds = 0;
    prm.n_jobs = 1;

    const auto res =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm);
    std::vector<double> preds(3);
    miter::Evaluator::predict_raw(res.model, X.data(), 3U, 3U, false, preds.data());
    for (std::size_t i = 0; i < 3U; ++i) {
        EXPECT_TRUE(std::isfinite(preds[i]));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(preds[i]),
                  std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
    }

    // Invalid max_depth > 8 must throw cleanly.
    prm.max_depth = 9;
    EXPECT_THROW(miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm),
                 std::invalid_argument);
}

TEST(SymmetricTreeTest, BitwiseEqualityTrainEvaluatorLignumAndJson) {
    constexpr std::size_t n = 1200U;
    constexpr std::size_t p = 8U;
    std::mt19937_64 rng(777);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> X(n * p);
    std::vector<double> y_reg(n);
    std::vector<double> y_bin(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X[i * p + j] = (i % 19U == 0U && j == 1U) ? kNan : norm(rng);
        }
        const double f0 = X[i * p + 0U];
        const double f1 = std::isnan(X[i * p + 1U]) ? 0.5 : X[i * p + 1U];
        const double f2 = X[i * p + 2U];
        const double sig = 1.5 * f0 - f1 + 0.7 * f0 * f2;
        y_reg[i] = sig;
        y_bin[i] = (sig > 0.0) ? 1.0 : 0.0;
    }

    const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);

    for (const auto obj : {miter::Objective::Regression, miter::Objective::Binary}) {
        const auto& y = (obj == miter::Objective::Regression) ? y_reg : y_bin;
        for (const int extra_opts : {0, 1}) {
            miter::Params prm{};
            prm.grow_policy = miter::GrowPolicy::Symmetric;
            prm.max_depth = 6;
            prm.min_data_in_leaf = 1;
            prm.lambda_l2 = 3.0;
            prm.m_mult = 2;
            prm.n_estimators = 5;
            prm.early_stopping_rounds = 0;
            prm.n_jobs = 4;
            prm.random_state = 19;
            if (extra_opts == 1) {
                prm.feature_fraction = 0.75;
                prm.bagging_fraction = 0.8;
                prm.extra_trees = true;
            }

            const auto res = miter::fit_booster(ds, obj, y, {}, nullptr, {}, {}, prm);
            std::vector<double> eval_preds(n);
            miter::Evaluator::predict_raw(res.model, X.data(), n, p, false, eval_preds.data());

            const std::string json = miter::model_to_json(res.model);
            const miter::Model reloaded = miter::model_from_json(json);
            std::vector<double> json_preds(n);
            miter::Evaluator::predict_raw(reloaded, X.data(), n, p, false, json_preds.data());

            const lignum::Model lm_knode = miter::to_lignum(res.model, 0, false);
            const lignum::Model lm_binned = miter::to_lignum(res.model, 0, true);
            std::vector<double> lig_knode(n);
            std::vector<double> lig_binned(n);
            lm_knode.predict_raw(X.data(), n, p, lig_knode.data(), 2);
            lm_binned.predict_raw(X.data(), n, p, lig_binned.data(), 2);

            for (std::size_t i = 0; i < n; ++i) {
                const auto ref_bits = std::bit_cast<std::uint64_t>(res.final_train_scores[i]);
                ASSERT_EQ(std::bit_cast<std::uint64_t>(eval_preds[i]), ref_bits);
                ASSERT_EQ(std::bit_cast<std::uint64_t>(json_preds[i]), ref_bits);
                ASSERT_EQ(std::bit_cast<std::uint64_t>(lig_knode[i]), ref_bits);
                ASSERT_EQ(std::bit_cast<std::uint64_t>(lig_binned[i]), ref_bits);
            }
        }
    }

    // k_proj = 4 fallback path exactness between training and Evaluator.
    miter::Params k4_prm{};
    k4_prm.grow_policy = miter::GrowPolicy::Symmetric;
    k4_prm.max_depth = 4;
    k4_prm.min_data_in_leaf = 1;
    k4_prm.k_proj = 4;
    k4_prm.n_estimators = 3;
    k4_prm.early_stopping_rounds = 0;
    k4_prm.n_jobs = 2;
    k4_prm.random_state = 5;
    const auto k4_res =
        miter::fit_booster(ds, miter::Objective::Regression, y_reg, {}, nullptr, {}, {}, k4_prm);
    std::vector<double> k4_eval(n);
    miter::Evaluator::predict_raw(k4_res.model, X.data(), n, p, false, k4_eval.data());
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(std::bit_cast<std::uint64_t>(k4_eval[i]),
                  std::bit_cast<std::uint64_t>(k4_res.final_train_scores[i]));
    }
}

TEST(SymmetricTreeTest, MixedCategoricalExactnessAndThreadDeterminism) {
    // Mixed numeric + categorical dataset with n = 40000 >= 32768 (triggers chunked OpenMP paths)
    // and deterministic model parity across n_jobs in {1, 2, 4, 8}.
    constexpr std::size_t n = 40000U;
    constexpr std::size_t p_num = 6U;
    std::mt19937_64 rng(902);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> num_cols(p_num * n);
    std::vector<std::int32_t> cat0(n);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p_num; ++j) {
            num_cols[j * n + i] = (i % 41U == 0U && j == 0U) ? kNan : norm(rng);
        }
        cat0[i] = (i % 29U == 0U) ? -1 : static_cast<std::int32_t>(i % 7U);
        const double x1 = num_cols[1U * n + i];
        const double x2 = num_cols[2U * n + i];
        const double cat_eff = (cat0[i] >= 0 && cat0[i] < 3) ? 1.2 : -0.6;
        y[i] = 1.5 * x1 - x2 + cat_eff + 0.1 * norm(rng);
    }

    std::vector<miter::FeatureKind> kinds;
    std::vector<std::string> names;
    std::vector<miter::NumericColumnView> num_cols_views;
    for (std::size_t j = 0; j < p_num; ++j) {
        kinds.push_back(miter::FeatureKind::Numeric);
        names.push_back("x" + std::to_string(j));
        miter::NumericColumnView nv;
        nv.f64 = num_cols.data() + j * n;
        nv.row_stride = 1U;
        num_cols_views.push_back(nv);
    }
    kinds.push_back(miter::FeatureKind::Categorical);
    names.push_back("cat");
    std::vector<miter::CategoricalColumnView> cat_views(1);
    cat_views[0].vocab = {"c0", "c1", "c2", "c3", "c4", "c5", "c6"};
    cat_views[0].indices = cat0.data();

    const auto ds = miter::Dataset::fit_mixed(n, kinds, names, num_cols_views, cat_views);

    std::string ref_json;
    std::vector<double> ref_scores;
    for (const int nj : {1, 2, 4, 8}) {
        miter::Params prm{};
        prm.grow_policy = miter::GrowPolicy::Symmetric;
        prm.max_depth = 6;
        prm.min_data_in_leaf = 1;
        prm.lambda_l2 = 3.0;
        prm.m_mult = 2;
        prm.n_estimators = 4;
        prm.early_stopping_rounds = 0;
        prm.n_jobs = nj;
        prm.random_state = 42;

        auto res =
            miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm);
        res.model.params.n_jobs = 0;
        const std::string json = miter::model_to_json(res.model);
        if (nj == 1) {
            ref_json = json;
            ref_scores = res.final_train_scores;
        } else {
            EXPECT_EQ(json, ref_json) << "threads=" << nj;
            ASSERT_EQ(res.final_train_scores.size(), ref_scores.size());
            for (std::size_t i = 0; i < n; ++i) {
                ASSERT_EQ(std::bit_cast<std::uint64_t>(res.final_train_scores[i]),
                          std::bit_cast<std::uint64_t>(ref_scores[i]))
                    << "threads=" << nj << " row=" << i;
            }
        }
    }

    // Single-feature chunk-wave determinism check (n_num_act == 1, m == 0).
    std::vector<double> x_single(num_cols.begin(),
                                 num_cols.begin() + static_cast<std::ptrdiff_t>(n));
    const auto ds_single = miter::Dataset::fit_numeric(x_single.data(), n, 1U, false);
    std::string single_ref;
    for (const int nj : {1, 8}) {
        miter::Params prm{};
        prm.grow_policy = miter::GrowPolicy::Symmetric;
        prm.max_depth = 4;
        prm.min_data_in_leaf = 1;
        prm.m_mult = 0;
        prm.n_estimators = 3;
        prm.early_stopping_rounds = 0;
        prm.n_jobs = nj;
        prm.random_state = 9;
        auto res = miter::fit_booster(ds_single, miter::Objective::Regression, y, {}, nullptr, {},
                                      {}, prm);
        res.model.params.n_jobs = 0;
        const std::string json = miter::model_to_json(res.model);
        if (nj == 1) {
            single_ref = json;
        } else {
            EXPECT_EQ(json, single_ref);
        }
    }
}

} // namespace
