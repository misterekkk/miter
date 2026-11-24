#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/booster.hpp"
#include "core/oblique.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"
#include "util/fast_div.hpp"

namespace {

TEST(DeterminismTest, FastDiv64MatchesExactObliqueBin) {
    std::mt19937_64 rng(2026);
    const std::vector<std::int64_t> edge_ranges = {
        1, 2, 3, 63, 64, 65, 255, 256, 65535, 65536, (1LL << 30), 3221225473LL};
    for (const std::int64_t r : edge_ranges) {
        miter::FastDiv64 fd;
        fd.prepare(static_cast<std::uint64_t>(r));
        for (int s = 0; s < 500; ++s) {
            std::uniform_int_distribution<std::int64_t> dist(0, r - 1);
            const std::int64_t delta = (s < 3) ? s : ((s == 3) ? (r - 1) : dist(rng));
            const auto fast_b =
                static_cast<std::uint8_t>(fd.div(static_cast<std::uint64_t>(delta) * 64ULL));
            const std::uint8_t exact_b = miter::oblique_bin_index(delta, 0, r);
            EXPECT_EQ(fast_b, exact_b);
        }
    }

    for (int i = 0; i < 100000; ++i) {
        std::uniform_int_distribution<std::int64_t> pmin_dist(-1610612736LL, 1610612736LL);
        const std::int64_t pmin = pmin_dist(rng);
        std::uniform_int_distribution<std::int64_t> range_dist(1, 3221225473LL);
        const std::int64_t range = std::min(range_dist(rng), 1610612737LL - pmin);
        std::uniform_int_distribution<std::int64_t> p_dist(pmin, pmin + range - 1);
        const std::int64_t p = p_dist(rng);

        miter::FastDiv64 fd;
        fd.prepare(static_cast<std::uint64_t>(range));
        const auto fast_b =
            static_cast<std::uint8_t>(fd.div(static_cast<std::uint64_t>(p - pmin) * 64ULL));
        const std::uint8_t exact_b = miter::oblique_bin_index(p, pmin, range);
        ASSERT_EQ(fast_b, exact_b);
    }
}

TEST(DeterminismTest, BitwiseIdenticalModelsAcrossThreads1_2_4_8) {
    // n = 12000 (> 2 * 4096 chunks) exercises multi-chunk wave reductions and OpenMP scheduling.
    constexpr std::size_t n = 12000U;
    std::mt19937_64 rng(777);
    std::normal_distribution<double> norm(0.0, 1.0);

    // 1) Numeric regression (p = 20, NaNs)
    {
        constexpr std::size_t p = 20U;
        std::vector<double> X(n * p);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                X[i * p + j] = (i % 23U == 0U && j == 3U) ? std::numeric_limits<double>::quiet_NaN()
                                                          : norm(rng);
            }
            const double x3 = std::isnan(X[i * p + 3U]) ? 0.5 : X[i * p + 3U];
            y[i] = X[i * p + 0U] - 0.7 * X[i * p + 1U] + 0.4 * x3 + 0.1 * norm(rng);
        }
        const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);
        const auto val = ds.transform_numeric(X.data(), 1000U, p, false);
        const std::vector<double> yv(y.begin(), y.begin() + 1000);

        std::string ref_json;
        std::vector<double> ref_pred(n);
        for (const int threads : {1, 2, 4, 8}) {
            miter::Params prm{};
            prm.n_estimators = 8;
            prm.num_leaves = 15;
            prm.m_mult = 2;
            prm.early_stopping_rounds = 0;
            prm.n_jobs = threads;
            prm.random_state = 19;

            auto res =
                miter::fit_booster(ds, miter::Objective::Regression, y, {}, &val, yv, {}, prm);
            res.model.params.n_jobs = 0;
            const std::string json = miter::model_to_json(res.model);
            std::vector<double> pred(n);
            miter::Evaluator::predict_raw(res.model, X.data(), n, p, false, pred.data(), 0U,
                                          threads);
            if (threads == 1) {
                ref_json = json;
                ref_pred = pred;
            } else {
                EXPECT_EQ(json, ref_json) << "Mismatch at threads=" << threads;
                for (std::size_t i = 0; i < n; ++i) {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                              std::bit_cast<std::uint64_t>(ref_pred[i]));
                }
            }
        }
    }

    // 2) Mixed numeric + categorical binary classification (p = 12, K1 + oblique + NaNs)
    {
        constexpr std::size_t p_num = 10U;
        std::vector<double> X_num(n * p_num);
        std::vector<std::int32_t> cat0(n);
        std::vector<std::int32_t> cat1(n);
        std::vector<double> y_bin(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < p_num; ++j) {
                X_num[i * p_num + j] = (i % 19U == 0U && j == 0U)
                                           ? std::numeric_limits<double>::quiet_NaN()
                                           : norm(rng);
            }
            cat0[i] = (i % 31U == 0U) ? -1 : static_cast<std::int32_t>(i % 6U);
            cat1[i] = static_cast<std::int32_t>((i * 3U) % 5U);
            const double sig =
                (cat0[i] == 2 ? 1.5 : -0.5) + X_num[i * p_num + 1U] - 0.5 * X_num[i * p_num + 2U];
            y_bin[i] = (sig > 0.0) ? 1.0 : 0.0;
        }
        std::vector<miter::FeatureKind> kinds(12, miter::FeatureKind::Numeric);
        kinds[2] = miter::FeatureKind::Categorical;
        kinds[7] = miter::FeatureKind::Categorical;
        std::vector<miter::NumericColumnView> nviews;
        for (std::size_t j = 0; j < p_num; ++j) {
            nviews.push_back(miter::NumericColumnView{X_num.data() + j, nullptr, p_num});
        }
        const std::vector<std::string> v0 = {"c0", "c1", "c2", "c3", "c4", "c5"};
        const std::vector<std::string> v1 = {"d0", "d1", "d2", "d3", "d4"};
        const std::vector<miter::CategoricalColumnView> cviews = {{v0, cat0.data()},
                                                                  {v1, cat1.data()}};
        miter::DataConfig cfg{};
        cfg.min_data_per_group = 20;
        const auto ds = miter::Dataset::fit_mixed(n, kinds, {}, nviews, cviews, cfg);

        std::string ref_json;
        std::vector<double> ref_pred(n);
        for (const int threads : {1, 2, 4, 8}) {
            miter::Params prm{};
            prm.n_estimators = 6;
            prm.num_leaves = 15;
            prm.min_data_per_group = 20;
            prm.m_mult = 2;
            prm.early_stopping_rounds = 0;
            prm.n_jobs = threads;
            prm.random_state = 23;

            auto res =
                miter::fit_booster(ds, miter::Objective::Binary, y_bin, {}, nullptr, {}, {}, prm);
            res.model.params.n_jobs = 0;
            const std::string json = miter::model_to_json(res.model);
            std::vector<double> pred(n);
            miter::Evaluator::predict_raw_mixed(res.model, n, nviews, cviews, pred.data(), 0U,
                                                threads);
            if (threads == 1) {
                ref_json = json;
                ref_pred = pred;
            } else {
                EXPECT_EQ(json, ref_json) << "Mixed mismatch at threads=" << threads;
                for (std::size_t i = 0; i < n; ++i) {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                              std::bit_cast<std::uint64_t>(ref_pred[i]));
                }
            }
        }
    }

    // 3) High-k (k_proj = 4) oblique regression
    {
        constexpr std::size_t p = 12U;
        std::vector<double> X(n * p);
        std::vector<double> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                X[i * p + j] = norm(rng);
            }
            y[i] = X[i * p + 0U] + X[i * p + 1U] - X[i * p + 2U] + 0.1 * norm(rng);
        }
        const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);

        std::string ref_json;
        for (const int threads : {1, 2, 4, 8}) {
            miter::Params prm{};
            prm.n_estimators = 5;
            prm.num_leaves = 10;
            prm.m_mult = 2;
            prm.k_proj = 4;
            prm.early_stopping_rounds = 0;
            prm.n_jobs = threads;
            prm.random_state = 5;

            auto res =
                miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm);
            res.model.params.n_jobs = 0;
            const std::string json = miter::model_to_json(res.model);
            if (threads == 1) {
                ref_json = json;
            } else {
                EXPECT_EQ(json, ref_json) << "k_proj=4 mismatch at threads=" << threads;
            }
        }
    }
}

TEST(DeterminismTest, LargeLeafParallelPartitionAndChunkedAxisExactness) {
    // n = 40000 >= 32768 and p = 8 < 16 triggers BOTH parallel 2-pass stable row partition
    // and wave-chunked axis histogram reduction.
    constexpr std::size_t n = 40000U;
    constexpr std::size_t p = 8U;
    std::mt19937_64 rng(901);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> X(n * p);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            X[i * p + j] =
                (i % 37U == 0U && j == 1U) ? std::numeric_limits<double>::quiet_NaN() : norm(rng);
        }
        const double sig = 2.0 * X[i * p + 0U] - 1.5 * X[i * p + 2U];
        y[i] = (sig > 0.0) ? 1.0 : 0.0;
    }

    const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);
    miter::Params prm1{};
    prm1.n_estimators = 4;
    prm1.num_leaves = 15;
    prm1.m_mult = 2;
    prm1.early_stopping_rounds = 0;
    prm1.n_jobs = 1;
    prm1.random_state = 42;

    miter::Params prm8 = prm1;
    prm8.n_jobs = 8;

    auto res1 = miter::fit_booster(ds, miter::Objective::Binary, y, {}, nullptr, {}, {}, prm1);
    auto res8 = miter::fit_booster(ds, miter::Objective::Binary, y, {}, nullptr, {}, {}, prm8);
    res1.model.params.n_jobs = 0;
    res8.model.params.n_jobs = 0;

    EXPECT_EQ(miter::model_to_json(res1.model), miter::model_to_json(res8.model));
    ASSERT_EQ(res1.final_train_scores.size(), n);
    ASSERT_EQ(res8.final_train_scores.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(res1.final_train_scores[i]),
                  std::bit_cast<std::uint64_t>(res8.final_train_scores[i]));
    }
}

} // namespace
