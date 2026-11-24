#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "core/booster.hpp"
#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "core/rng.hpp"
#include "data/dataset.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"

namespace {

TEST(ObliqueTest, CounterRngDeterminismAndDistinctSampling) {
    miter::CounterRng r1(42, 3, 7, 1, 0);
    miter::CounterRng r2(42, 3, 7, 1, 0);
    miter::CounterRng r_diff(42, 3, 7, 2, 0);

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(r1.next_u64(), r2.next_u64());
    }
    EXPECT_NE(miter::CounterRng(42, 3, 7, 1, 0).next_u64(), r_diff.next_u64());

    miter::CounterRng rs(7, 0, 0, 0, 0);
    std::uint32_t out[5] = {};
    rs.sample_k_distinct(10, 5, out);
    std::set<std::uint32_t> uniq(out, out + 5);
    EXPECT_EQ(uniq.size(), 5U);
    for (const auto idx : out) {
        EXPECT_LT(idx, 10U);
    }
}

TEST(ObliqueTest, WeightQuantizationAndHalfAwayRounding) {
    EXPECT_EQ(miter::round_half_away_i16(0.5), 1);
    EXPECT_EQ(miter::round_half_away_i16(-0.5), -1);
    EXPECT_EQ(miter::round_half_away_i16(2.5), 3);
    EXPECT_EQ(miter::round_half_away_i16(-2.5), -3);
    EXPECT_EQ(miter::round_half_away_i16(0.499999), 0);
    EXPECT_EQ(miter::round_half_away_i16(-0.499999), 0);

    const double w[3] = {1.0, -2.0, 0.5};
    std::int16_t wq[3] = {};
    EXPECT_TRUE(miter::quantize_oblique_weights(w, 3, wq));
    EXPECT_EQ(wq[0], 8192);
    EXPECT_EQ(wq[1], -16384);
    EXPECT_EQ(wq[2], 4096);
}

TEST(ObliqueTest, ExtremeProjectionBoundsNoInt32Overflow) {
    // Max possible |P| for k = 3: 3 * 16384 * 32768 = 1,610,612,736 < 2,147,483,647
    const std::int16_t min_c = -32768;
    const std::int16_t max_c = 32766;
    const std::int16_t min_w = -16384;
    const std::int16_t max_w = 16384;

    for (const std::int16_t c0 : {min_c, max_c}) {
        for (const std::int16_t c1 : {min_c, max_c}) {
            for (const std::int16_t c2 : {min_c, max_c}) {
                for (const std::int16_t w0 : {min_w, max_w}) {
                    for (const std::int16_t w1 : {min_w, max_w}) {
                        for (const std::int16_t w2 : {min_w, max_w}) {
                            const std::int64_t p64 =
                                static_cast<std::int64_t>(w0) * static_cast<std::int64_t>(c0) +
                                static_cast<std::int64_t>(w1) * static_cast<std::int64_t>(c1) +
                                static_cast<std::int64_t>(w2) * static_cast<std::int64_t>(c2);
                            EXPECT_LE(p64, std::numeric_limits<std::int32_t>::max());
                            EXPECT_GE(p64, std::numeric_limits<std::int32_t>::min());
                            const std::int32_t p32 =
                                static_cast<std::int32_t>(w0) * static_cast<std::int32_t>(c0) +
                                static_cast<std::int32_t>(w1) * static_cast<std::int32_t>(c1) +
                                static_cast<std::int32_t>(w2) * static_cast<std::int32_t>(c2);
                            EXPECT_EQ(static_cast<std::int64_t>(p32), p64);
                        }
                    }
                }
            }
        }
    }
}

TEST(ObliqueTest, ExhaustiveBinAndThresholdEquivalence) {
    std::mt19937_64 rng(98765);
    std::uniform_int_distribution<std::int64_t> small_span(1, 200);
    std::uniform_int_distribution<std::int64_t> large_p(-1610612736LL, 1610612736LL);

    std::vector<std::pair<std::int64_t, std::int64_t>> ranges{
        {0, 1}, {-1, 1}, {10, 73}, {10, 74}, {-1610612736LL, 1610612736LL},
    };
    for (int i = 0; i < 250; ++i) {
        const std::int64_t a = large_p(rng);
        ranges.emplace_back(a, a + small_span(rng));
    }
    for (int i = 0; i < 250; ++i) {
        const std::int64_t a = large_p(rng);
        const std::int64_t b = large_p(rng);
        if (a != b) {
            ranges.emplace_back(std::min(a, b), std::max(a, b));
        }
    }

    for (const auto& [pmin, pmax] : ranges) {
        const std::int64_t range = pmax - pmin + 1;
        ASSERT_GE(range, 2);

        for (int b = 0; b < 63; ++b) {
            const std::int64_t thr = miter::oblique_bin_threshold(pmin, range, b);
            EXPECT_GT(thr, pmin);
            EXPECT_LE(thr, pmax + 1);

            // Check exact equivalence bin(P) <= b  <=>  P < thr across [pmin, pmax]
            const std::int64_t probes[] = {pmin, pmax, thr - 1, std::min(pmax, thr),
                                           std::min(pmax, thr + 1)};
            for (const std::int64_t p : probes) {
                const std::uint8_t bin = miter::oblique_bin_index(p, pmin, range);
                EXPECT_LT(bin, 64U);
                EXPECT_EQ(bin <= b, p < thr);
            }
        }
    }
}

TEST(ObliqueTest, DiagonalSplitAndNanDirections) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    const std::size_t n = 300;
    std::vector<double> x(n * 2);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i * 2 + 0] = uni(rng);
        x[i * 2 + 1] = uni(rng);
        y[i] = (x[i * 2 + 0] + x[i * 2 + 1] > 0.0) ? 5.0 : -5.0;
    }
    // Inject NaNs on half-pattern rows belonging to the negative class (nan_go_left testable)
    for (std::size_t i = 0; i < 20; ++i) {
        x[i * 2 + 0] = nan;
        x[i * 2 + 1] = uni(rng);
        y[i] = -5.0;
    }

    const auto ds = miter::Dataset::fit_numeric(x.data(), n, 2, false);
    miter::Params params;
    params.n_estimators = 5;
    params.learning_rate = 0.5;
    params.num_leaves = 4;
    params.min_data_in_leaf = 5;
    params.m_mult = 8; // 8 * ceil(sqrt(2)) = 16 candidates
    params.random_state = 123;

    const auto res =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, params);

    bool saw_oblique = false;
    for (const auto& node : res.model.trees[0].nodes) {
        if (node.kind == miter::NodeKind::Oblique) {
            saw_oblique = true;
        }
    }
    EXPECT_TRUE(saw_oblique);

    std::vector<double> pred(n);
    miter::Evaluator::predict_raw(res.model, x.data(), n, 2, false, pred.data());
    double mse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                  std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
        const double d = pred[i] - y[i];
        mse += d * d;
    }
    mse /= static_cast<double>(n);
    EXPECT_LT(mse, 2.0);
}

TEST(ObliqueTest, BitwiseExactOnSolvedThresholdRowsAndOverflowK4AndJson) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::mt19937 rng(2027);
    std::normal_distribution<double> norm(0.0, 1.0);

    const std::size_t n = 160;
    const std::size_t p = 6;
    std::vector<double> x(n * p);
    std::vector<double> y_reg(n);
    std::vector<double> y_bin(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            x[i * p + j] = (j == 1) ? 9.0 : norm(rng); // feature 1 is constant
        }
        if (i % 13 == 0) {
            x[i * p + 2] = nan;
        }
        const double lin =
            x[i * p + 0] - 0.8 * x[i * p + 3] + 0.5 * x[i * p + 4] + 0.3 * x[i * p + 5];
        y_reg[i] = lin + 0.05 * norm(rng);
        y_bin[i] = (lin > 0.0) ? 1.0 : 0.0;
    }
    x[0 * p + 0] = -50.0;
    x[1 * p + 0] = 50.0;

    const auto ds = miter::Dataset::fit_numeric(x.data(), n, p, false);

    for (const int k_proj : {0, 4}) {
        for (const miter::Objective obj :
             {miter::Objective::Regression, miter::Objective::Binary}) {
            miter::Params params;
            params.n_estimators = 12;
            params.learning_rate = 0.15;
            params.num_leaves = 7;
            params.min_data_in_leaf = 5;
            params.m_mult = 4;
            params.k_proj = k_proj;
            params.random_state = 77;

            const auto& y = (obj == miter::Objective::Regression) ? y_reg : y_bin;
            const auto res = miter::fit_booster(ds, obj, y, {}, nullptr, {}, {}, params);

            // Determinism check: identical JSON bytes on repeated run
            const auto res2 = miter::fit_booster(ds, obj, y, {}, nullptr, {}, {}, params);
            EXPECT_EQ(miter::model_to_json(res.model), miter::model_to_json(res2.model));

            // Train vs predict bitwise exactness
            std::vector<double> pred_train(n);
            miter::Evaluator::predict_raw(res.model, x.data(), n, p, false, pred_train.data());
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_train[i]),
                          std::bit_cast<std::uint64_t>(res.final_train_scores[i]));
            }

            // Construct rows landing EXACTLY on oblique thresholds by solving for one feature
            std::vector<double> x_probe = x;
            for (const auto& tree : res.model.trees) {
                for (const auto& node : tree.nodes) {
                    if (node.kind != miter::NodeKind::Oblique) {
                        continue;
                    }
                    const std::uint8_t k = node.oblique_k;
                    const std::uint32_t* feats =
                        (k <= 3) ? node.oblique_features
                                 : tree.oblique_overflow[node.overflow_offset].features.data();
                    const std::int16_t* weights =
                        (k <= 3) ? node.oblique_weights
                                 : tree.oblique_overflow[node.overflow_offset].weights.data();
                    const std::int64_t thr =
                        (k <= 3) ? static_cast<std::int64_t>(node.threshold_i32)
                                 : tree.oblique_overflow[node.overflow_offset].threshold_i64;

                    // Pick row 10 and solve feature 0's value to sit near/on thr
                    std::int64_t rest = 0;
                    for (std::uint8_t q = 1; q < k; ++q) {
                        const auto& qi = ds.quantizer(feats[q]);
                        const auto c = qi.encode_i16(x_probe[10 * p + feats[q]]);
                        rest +=
                            static_cast<std::int64_t>(weights[q]) * static_cast<std::int64_t>(c);
                    }
                    if (weights[0] != 0) {
                        const std::int64_t target_c = (thr - rest) / weights[0];
                        if (target_c >= -32768 && target_c <= 32766) {
                            const auto u = miter::Q16Quantizer::to_unsigned_code(
                                static_cast<std::int16_t>(target_c));
                            const auto& q0 = ds.quantizer(feats[0]);
                            x_probe[10 * p + feats[0]] = q0.lo + static_cast<double>(u) / q0.scale;
                        }
                    }
                    break;
                }
                break;
            }

            const std::string json = miter::model_to_json(res.model);
            const auto loaded = miter::model_from_json(json);
            std::vector<double> p_orig(n);
            std::vector<double> p_load(n);
            miter::Evaluator::predict_raw(res.model, x_probe.data(), n, p, false, p_orig.data());
            miter::Evaluator::predict_raw(loaded, x_probe.data(), n, p, false, p_load.data());
            for (std::size_t i = 0; i < n; ++i) {
                EXPECT_EQ(std::bit_cast<std::uint64_t>(p_orig[i]),
                          std::bit_cast<std::uint64_t>(p_load[i]));
            }
        }
    }
}

TEST(ObliqueTest, PresetsDepthLimitAndZeroMult) {
    std::mt19937 rng(5);
    std::normal_distribution<double> norm(0.0, 1.0);
    const std::size_t n = 150;
    const std::size_t p = 4;
    std::vector<double> x(n * p);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            x[i * p + j] = norm(rng);
        }
        y[i] = x[i * p + 0] + x[i * p + 1] + 0.1 * norm(rng);
    }
    const auto ds = miter::Dataset::fit_numeric(x.data(), n, p, false);

    // fast preset: oblique_max_depth = 1 -> no oblique splits at depth >= 2
    miter::Params fast_params;
    fast_params.n_estimators = 5;
    fast_params.num_leaves = 16;
    fast_params.min_data_in_leaf = 5;
    fast_params.m_mult = 4;
    fast_params.oblique_max_depth = 1;
    const auto res_fast =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, fast_params);

    for (const auto& tree : res_fast.model.trees) {
        std::vector<std::pair<std::uint32_t, int>> stack{{0U, 0}};
        while (!stack.empty()) {
            const auto [nid, depth] = stack.back();
            stack.pop_back();
            const auto& node = tree.nodes[nid];
            if (node.kind == miter::NodeKind::Leaf) {
                continue;
            }
            if (depth >= 2) {
                EXPECT_NE(node.kind, miter::NodeKind::Oblique);
            }
            stack.emplace_back(node.left_child, depth + 1);
            stack.emplace_back(node.right_child, depth + 1);
        }
    }

    // m_mult = 0 -> zero oblique nodes
    miter::Params ax_params = fast_params;
    ax_params.m_mult = 0;
    ax_params.oblique_max_depth = -1;
    const auto res_ax =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, ax_params);
    for (const auto& tree : res_ax.model.trees) {
        for (const auto& node : tree.nodes) {
            EXPECT_NE(node.kind, miter::NodeKind::Oblique);
        }
    }
}

} // namespace
