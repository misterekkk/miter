#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/booster.hpp"
#include "core/categorical.hpp"
#include "core/histogram.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"

namespace {

double split_gain_l2(double gl, double hl, double parent_g, double parent_h, double lam) {
    const double gr = parent_g - gl;
    const double hr = parent_h - hl;
    return (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam) -
           (parent_g * parent_g) / (parent_h + lam);
}

TEST(CategoricalTest, BruteForceFisherExactness) {
    for (const int C : {3, 5, 8}) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(C) + 100U);
        std::normal_distribution<double> norm(0.0, 1.0);
        std::uniform_int_distribution<int> cat_dist(0, C - 1);

        for (int trial = 0; trial < 20; ++trial) {
            std::vector<double> effects(static_cast<std::size_t>(C));
            for (int c = 0; c < C; ++c) {
                effects[static_cast<std::size_t>(c)] = norm(rng);
            }

            constexpr std::size_t n = 400U;
            std::vector<std::int32_t> indices(n);
            std::vector<std::string> vocab;
            for (int c = 0; c < C; ++c) {
                vocab.push_back("c" + std::to_string(c));
            }
            std::vector<double> g(n);
            std::vector<double> h(n, 1.0);

            std::vector<miter::HistBin> hist(static_cast<std::size_t>(C) + 2U);
            double parent_g = 0.0;
            double parent_h = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const int c = cat_dist(rng);
                indices[i] = c;
                g[i] = norm(rng) + effects[static_cast<std::size_t>(c)];
                hist[static_cast<std::size_t>(c)].g += g[i];
                hist[static_cast<std::size_t>(c)].h += h[i];
                hist[static_cast<std::size_t>(c)].c += 1;
                parent_g += g[i];
                parent_h += h[i];
            }

            // Use layout mapping matching Dataset categorical bins: num_regular_bins = C + 2.
            miter::FeatureHistLayout layout;
            layout.offsets = {0U};
            layout.num_regular_bins = {static_cast<std::uint16_t>(C + 2)};
            layout.has_nan = {0U};
            layout.is_categorical = {1U};
            layout.active_features = {0U};
            layout.active_cat_features = {0U};
            layout.total_bins = static_cast<std::size_t>(C) + 2U;

            miter::Params params{};
            params.min_data_in_leaf = 1;
            params.min_sum_hessian_in_leaf = 1e-6;
            params.lambda_l2 = 1.0;
            params.min_gain_to_split = 0.0;
            // Force many-vs-many path with 0 extra cat regularization to match Fisher theorem.
            params.max_cat_to_onehot = 1;
            params.min_data_per_group = 1;
            params.cat_smooth = 0.0;
            params.cat_l2 = 0.0;
            params.max_cat_threshold = 255;

            const auto cand = miter::find_best_cat_split(layout, hist.data(), parent_g, parent_h,
                                                         static_cast<std::int32_t>(n), params);
            ASSERT_TRUE(cand.valid);

            double brute_best = -std::numeric_limits<double>::infinity();
            const std::uint32_t mask_limit = (1U << static_cast<unsigned int>(C)) - 1U;
            for (std::uint32_t mask = 1U; mask < mask_limit; ++mask) {
                double gl = 0.0;
                double hl = 0.0;
                for (int c = 0; c < C; ++c) {
                    if ((mask & (1U << static_cast<unsigned int>(c))) != 0U) {
                        gl += hist[static_cast<std::size_t>(c)].g;
                        hl += hist[static_cast<std::size_t>(c)].h;
                    }
                }
                brute_best = std::max(brute_best, split_gain_l2(gl, hl, parent_g, parent_h, 1.0));
            }
            EXPECT_NEAR(cand.gain, brute_best, 1e-10);

            // Bitset realizes the exact reported gain.
            double check_gl = 0.0;
            double check_hl = 0.0;
            std::int32_t check_cl = 0;
            for (int c = 0; c < C; ++c) {
                if (miter::cat_bitset_test(cand.left_bitset, static_cast<std::uint8_t>(c))) {
                    check_gl += hist[static_cast<std::size_t>(c)].g;
                    check_hl += hist[static_cast<std::size_t>(c)].h;
                    check_cl += hist[static_cast<std::size_t>(c)].c;
                }
            }
            EXPECT_GT(check_cl, 0);
            EXPECT_LT(check_cl, static_cast<std::int32_t>(n));
            const double realized = split_gain_l2(check_gl, check_hl, parent_g, parent_h, 1.0);
            EXPECT_NEAR(realized, cand.gain, 1e-10);
        }
    }
}

TEST(CategoricalTest, OneHotRareMergingAndDefaultDirection) {
    // 1) One-hot (n_used == 3 <= max_cat_to_onehot == 4), bin 4 unused in leaf.
    {
        std::vector<miter::HistBin> hist(6);
        hist[0] = {10.0, 2.0, 50};
        hist[1] = {-5.0, 1.0, 40};
        hist[2] = {-5.0, 1.0, 40};
        // hist[3], hist[4], hist[5] have c == 0 (unused)
        miter::FeatureHistLayout layout;
        layout.offsets = {0U};
        layout.num_regular_bins = {6U};
        layout.has_nan = {0U};
        layout.is_categorical = {1U};
        layout.active_features = {0U};
        layout.active_cat_features = {0U};
        layout.total_bins = 6U;

        miter::Params params{};
        params.min_data_in_leaf = 10;
        params.max_cat_to_onehot = 4;

        const auto cand = miter::find_best_cat_split(layout, hist.data(), 0.0, 4.0, 130, params);
        ASSERT_TRUE(cand.valid);
        EXPECT_TRUE(miter::cat_bitset_test(cand.left_bitset, 0U));
        EXPECT_FALSE(miter::cat_bitset_test(cand.left_bitset, 1U));
        EXPECT_FALSE(miter::cat_bitset_test(cand.left_bitset, 2U));
        // hl = 2.0 >= hr = 2.0 -> default_go_left == true -> unused bins (3, 4, 5) go left.
        EXPECT_TRUE(cand.default_go_left);
        EXPECT_TRUE(miter::cat_bitset_test(cand.left_bitset, 3U));
        EXPECT_TRUE(miter::cat_bitset_test(cand.left_bitset, 4U));
    }

    // 2) Rare merging: categories with c < min_data_per_group are grouped together and share
    // direction.
    {
        std::vector<miter::HistBin> hist(8);
        hist[0] = {-20.0, 10.0, 100};
        hist[1] = {12.0, 10.0, 100};
        hist[2] = {2.0, 1.0, 10};
        hist[3] = {2.0, 1.0, 10};
        hist[4] = {4.0, 2.0, 20};
        hist[5] = {0.0, 0.0, 0}; // empty -> default direction

        miter::FeatureHistLayout layout;
        layout.offsets = {0U};
        layout.num_regular_bins = {8U};
        layout.has_nan = {0U};
        layout.is_categorical = {1U};
        layout.active_features = {0U};
        layout.active_cat_features = {0U};
        layout.total_bins = 8U;

        miter::Params params{};
        params.max_cat_to_onehot = 2;
        params.min_data_per_group = 50;
        params.min_data_in_leaf = 10;
        params.cat_l2 = 0.0;
        params.cat_smooth = 0.0;

        const auto cand = miter::find_best_cat_split(layout, hist.data(), 0.0, 24.0, 240, params);
        ASSERT_TRUE(cand.valid);
        const bool g2 = miter::cat_bitset_test(cand.left_bitset, 2U);
        const bool g3 = miter::cat_bitset_test(cand.left_bitset, 3U);
        const bool g4 = miter::cat_bitset_test(cand.left_bitset, 4U);
        EXPECT_EQ(g2, g3);
        EXPECT_EQ(g3, g4);
        EXPECT_EQ(miter::cat_bitset_test(cand.left_bitset, 5U), cand.default_go_left);
    }
}

TEST(CategoricalTest, MixedHistogramSubtractionExactness) {
    constexpr std::size_t n = 300U;
    std::mt19937_64 rng(7);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_int_distribution<std::int32_t> cat_dist(-1, 5); // includes -1 missing

    std::vector<double> x_num(n);
    std::vector<std::int32_t> x_cat(n);
    std::vector<float> g(n);
    std::vector<float> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        x_num[i] = (i % 9U == 0U) ? std::numeric_limits<double>::quiet_NaN() : norm(rng);
        x_cat[i] = cat_dist(rng);
        g[i] = static_cast<float>(norm(rng));
        h[i] = 1.0f;
    }

    const std::vector<std::string> vocab = {"a", "b", "c", "d", "e", "f"};
    miter::DataConfig cfg{};
    cfg.min_data_per_group = 10;
    const auto ds = miter::Dataset::fit_mixed(
        n, {miter::FeatureKind::Numeric, miter::FeatureKind::Categorical}, {"num0", "cat0"},
        {{x_num.data(), nullptr, 1U}}, {{vocab, x_cat.data()}}, cfg);

    const auto layout = miter::FeatureHistLayout::build(ds);
    EXPECT_EQ(layout.active_num_features.size(), 1U);
    EXPECT_EQ(layout.active_cat_features.size(), 1U);

    std::vector<std::uint32_t> rows(n);
    std::iota(rows.begin(), rows.end(), 0U);
    std::vector<float> g_ord(n);
    std::vector<float> h_ord(n);
    std::vector<miter::HistBin> hp(layout.total_bins);
    std::vector<miter::HistBin> hl(layout.total_bins);
    std::vector<miter::HistBin> hr_direct(layout.total_bins);
    std::vector<miter::HistBin> hr_sub(layout.total_bins);

    miter::build_axis_histograms(ds, layout, rows.data(), n, true, g.data(), h.data(), g_ord.data(),
                                 h_ord.data(), hp.data());
    miter::build_axis_histograms(ds, layout, rows.data(), 110U, false, g.data(), h.data(),
                                 g_ord.data(), h_ord.data(), hl.data());
    miter::build_axis_histograms(ds, layout, rows.data() + 110U, n - 110U, false, g.data(),
                                 h.data(), g_ord.data(), h_ord.data(), hr_direct.data());
    miter::subtract_histograms(hr_sub.data(), hp.data(), hl.data(), layout.total_bins);

    for (std::size_t k = 0; k < layout.total_bins; ++k) {
        EXPECT_EQ(hr_sub[k].c, hr_direct[k].c);
        EXPECT_NEAR(hr_sub[k].g, hr_direct[k].g, 1e-11);
        EXPECT_NEAR(hr_sub[k].h, hr_direct[k].h, 1e-11);
    }
}

TEST(CategoricalTest, MixedTrainPredictBitwiseUnseenMissingJsonAndPureCat) {
    constexpr std::size_t n = 480U;
    std::mt19937_64 rng(99);
    std::normal_distribution<double> norm(0.0, 1.0);
    const std::vector<std::string> vocab = {"k0", "k1", "k2", "k3", "k4", "k5", "rare"};

    std::vector<double> x0(n);
    std::vector<double> x1(n);
    std::vector<std::int32_t> cat_idx(n);
    std::vector<double> y_reg(n);
    std::vector<double> y_bin(n);

    for (std::size_t i = 0; i < n; ++i) {
        x0[i] = (i % 15U == 0U) ? std::numeric_limits<double>::quiet_NaN() : norm(rng);
        x1[i] = norm(rng);
        if (i % 20U == 0U) {
            cat_idx[i] = -1; // missing
        } else if (i % 100U == 1U) {
            cat_idx[i] = 6; // rare -> OTHER (< min_data_per_group)
        } else {
            cat_idx[i] = static_cast<std::int32_t>(i % 6U);
        }
        const bool good_cat = (cat_idx[i] == 1 || cat_idx[i] == 4);
        const double sig =
            (good_cat ? 3.0 : -1.0) + (std::isnan(x0[i]) ? 1.5 : 0.7 * (x0[i] + x1[i]));
        y_reg[i] = sig + 0.05 * norm(rng);
        y_bin[i] = (sig > 0.0) ? 1.0 : 0.0;
    }

    miter::DataConfig cfg{};
    cfg.min_data_per_group = 15;
    // Feature order interleaved: [num0, cat0, num1].
    const auto ds = miter::Dataset::fit_mixed(
        n,
        {miter::FeatureKind::Numeric, miter::FeatureKind::Categorical, miter::FeatureKind::Numeric},
        {"num0", "cat0", "num1"}, {{x0.data(), nullptr, 1U}, {x1.data(), nullptr, 1U}},
        {{vocab, cat_idx.data()}}, cfg);

    miter::Params params{};
    params.n_estimators = 12;
    params.learning_rate = 0.1;
    params.num_leaves = 8;
    params.min_data_in_leaf = 10;
    params.min_data_per_group = 15;
    params.max_cat_to_onehot = 2;
    params.m_mult = 2;
    params.early_stopping_rounds = 0;
    params.random_state = 12;

    for (const auto obj : {miter::Objective::Regression, miter::Objective::Binary}) {
        const auto& y = (obj == miter::Objective::Regression) ? y_reg : y_bin;
        const auto res = miter::fit_booster(ds, obj, y, {}, nullptr, {}, {}, params);

        bool saw_cat = false;
        for (const auto& tree : res.model.trees) {
            for (const auto& nd : tree.nodes) {
                if (nd.kind == miter::NodeKind::Categorical) {
                    saw_cat = true;
                    EXPECT_EQ(nd.feature, 1U);
                } else if (nd.kind == miter::NodeKind::Oblique) {
                    const std::uint8_t k = nd.oblique_k;
                    if (k <= 3U) {
                        for (std::uint8_t q = 0; q < k; ++q) {
                            EXPECT_NE(nd.oblique_features[q], 1U);
                        }
                    } else {
                        const auto& ov = tree.oblique_overflow[nd.overflow_offset];
                        for (const std::uint32_t f : ov.features) {
                            EXPECT_NE(f, 1U);
                        }
                    }
                }
            }
        }
        EXPECT_TRUE(saw_cat);

        // Bitwise exact: train scores == predict_raw_mixed == JSON round-trip.
        std::vector<double> pred(n);
        miter::Evaluator::predict_raw_mixed(res.model, n,
                                            {{x0.data(), nullptr, 1U}, {x1.data(), nullptr, 1U}},
                                            {{vocab, cat_idx.data()}}, pred.data());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                      std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
        }

        const std::string json = miter::model_to_json(res.model);
        const miter::Model reloaded = miter::model_from_json(json);
        std::vector<double> pred_reloaded(n);
        miter::Evaluator::predict_raw_mixed(reloaded, n,
                                            {{x0.data(), nullptr, 1U}, {x1.data(), nullptr, 1U}},
                                            {{vocab, cat_idx.data()}}, pred_reloaded.data());
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_reloaded[i]),
                      std::bit_cast<std::uint64_t>(pred[i]));
        }

        // Unseen category at predict time must match training OTHER ("rare") bitwise.
        std::vector<std::string> unseen_vocab = {"unseen_brand_new", "k1"};
        std::vector<std::int32_t> unseen_idx = {0, -1};
        std::vector<double> unseen_num0 = {x0[1], x0[20]};
        std::vector<double> unseen_num1 = {x1[1], x1[20]};
        std::vector<double> pred_unseen(2);
        miter::Evaluator::predict_raw_mixed(
            reloaded, 2U, {{unseen_num0.data(), nullptr, 1U}, {unseen_num1.data(), nullptr, 1U}},
            {{unseen_vocab, unseen_idx.data()}}, pred_unseen.data());
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_unseen[0]),
                  std::bit_cast<std::uint64_t>(pred[1]));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_unseen[1]),
                  std::bit_cast<std::uint64_t>(pred[20]));
    }

    // Pure categorical dataset (0 numeric features).
    const auto ds_cat_only = miter::Dataset::fit_mixed(
        n, {miter::FeatureKind::Categorical}, {"cat0"}, {}, {{vocab, cat_idx.data()}}, cfg);
    const auto res_cat_only = miter::fit_booster(ds_cat_only, miter::Objective::Regression, y_reg,
                                                 {}, nullptr, {}, {}, params);
    std::vector<double> pred_cat_only(n);
    miter::Evaluator::predict_raw_mixed(res_cat_only.model, n, {}, {{vocab, cat_idx.data()}},
                                        pred_cat_only.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_cat_only[i]),
                  std::bit_cast<std::uint64_t>(res_cat_only.final_train_scores[i]));
    }
}

TEST(CategoricalTest, NanVsRestAxisAndObliqueBitwiseExact) {
    // Constant finite value (1.0) vs NaN in feature 0, plus finite + NaN informative features 1, 2.
    constexpr std::size_t n = 200U;
    std::mt19937_64 rng(55);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> X(n * 3U);
    std::vector<double> y(n);

    for (std::size_t i = 0; i < n; ++i) {
        const bool nan0 = (i < 60U);
        const bool nan1 = (i >= 60U && i < 120U);
        X[i * 3U + 0U] = nan0 ? std::numeric_limits<double>::quiet_NaN() : 1.0;
        X[i * 3U + 1U] = nan1 ? std::numeric_limits<double>::quiet_NaN() : norm(rng);
        X[i * 3U + 2U] = norm(rng);
        y[i] = (nan0 ? 5.0 : 0.0) + (nan1 ? -3.0 : 0.5 * X[i * 3U + 2U]);
    }

    const auto ds = miter::Dataset::fit_numeric(X.data(), n, 3U, false);
    miter::Params params{};
    params.n_estimators = 8;
    params.learning_rate = 0.2;
    params.num_leaves = 6;
    params.min_data_in_leaf = 10;
    params.m_mult = 2;
    params.early_stopping_rounds = 0;
    params.random_state = 3;

    const auto res =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, params);
    ASSERT_FALSE(res.model.trees.empty());
    // Root tree must be able to split feature 0 (constant 1.0 vs NaN).
    const auto& root = res.model.trees[0].nodes[0];
    ASSERT_EQ(root.kind, miter::NodeKind::Axis);
    EXPECT_EQ(root.feature, 0U);
    EXPECT_EQ(root.threshold_u16, 65535U);

    std::vector<double> pred(n);
    miter::Evaluator::predict_raw(res.model, X.data(), n, 3U, false, pred.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                  std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
    }
    const auto reloaded = miter::model_from_json(miter::model_to_json(res.model));
    std::vector<double> pred_reloaded(n);
    miter::Evaluator::predict_raw(reloaded, X.data(), n, 3U, false, pred_reloaded.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_reloaded[i]),
                  std::bit_cast<std::uint64_t>(pred[i]));
    }
}

} // namespace
