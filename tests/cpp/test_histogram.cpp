#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "core/histogram.hpp"
#include "data/dataset.hpp"

namespace {

TEST(HistogramTest, RootBuildAndSubtractionExactness) {
    // 6 rows x 2 features:
    // f0: 0, 1, 2, 3, 4, 5
    // f1: 10, 10, 20, NaN, 20, 10
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> x_rm{
        0.0, 10.0, // row 0
        1.0, 10.0, // row 1
        2.0, 20.0, // row 2
        3.0, nan,  // row 3
        4.0, 20.0, // row 4
        5.0, 10.0, // row 5
    };
    const auto ds = miter::Dataset::fit_numeric(x_rm.data(), 6, 2, false);
    const auto layout = miter::FeatureHistLayout::build(ds);
    EXPECT_EQ(layout.active_features.size(), 2U);
    EXPECT_FALSE(layout.has_nan[0]);
    EXPECT_TRUE(layout.has_nan[1]);

    const std::vector<float> g{1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F};
    const std::vector<float> h{1.0F, 1.5F, 2.0F, 2.5F, 3.0F, 3.5F};

    miter::HistPool pool(3, layout.total_bins);
    std::vector<float> g_ord(6);
    std::vector<float> h_ord(6);
    const std::vector<std::uint32_t> all_rows{0, 1, 2, 3, 4, 5};
    miter::build_axis_histograms(ds, layout, all_rows.data(), 6, true, g.data(), h.data(),
                                 g_ord.data(), h_ord.data(), pool.slot(0));

    // Feature 1 has 2 regular bins (10 -> bin 0, 20 -> bin 1) + NaN bin (bin 2)
    const auto* h1 = pool.slot(0) + layout.offsets[1];
    EXPECT_EQ(h1[0].c, 3); // rows 0, 1, 5
    EXPECT_DOUBLE_EQ(h1[0].g, 1.0 - 2.0 - 6.0);
    EXPECT_DOUBLE_EQ(h1[0].h, 1.0 + 1.5 + 3.5);
    EXPECT_EQ(h1[1].c, 2); // rows 2, 4
    EXPECT_DOUBLE_EQ(h1[1].g, 3.0 + 5.0);
    EXPECT_DOUBLE_EQ(h1[1].h, 2.0 + 3.0);
    EXPECT_EQ(h1[2].c, 1); // row 3 (NaN)
    EXPECT_DOUBLE_EQ(h1[2].g, -4.0);
    EXPECT_DOUBLE_EQ(h1[2].h, 2.5);

    // Build small child {0, 1} and subtract from parent to get {2, 3, 4, 5}, compare vs scratch
    const std::vector<std::uint32_t> small_rows{0, 1};
    const std::vector<std::uint32_t> large_rows{2, 3, 4, 5};
    miter::build_axis_histograms(ds, layout, small_rows.data(), 2, false, g.data(), h.data(),
                                 g_ord.data(), h_ord.data(), pool.slot(1));
    miter::build_axis_histograms(ds, layout, large_rows.data(), 4, false, g.data(), h.data(),
                                 g_ord.data(), h_ord.data(), pool.slot(2));
    miter::subtract_histograms(pool.slot(0), pool.slot(0), pool.slot(1), layout.total_bins);

    for (std::size_t k = 0; k < layout.total_bins; ++k) {
        EXPECT_EQ(pool.slot(0)[k].c, pool.slot(2)[k].c);
        EXPECT_NEAR(pool.slot(0)[k].g, pool.slot(2)[k].g, 1e-12);
        EXPECT_NEAR(pool.slot(0)[k].h, pool.slot(2)[k].h, 1e-12);
    }
}

TEST(HistogramTest, HandComputedAxisSplitAndConstraints) {
    // 4 rows, 1 feature: x = {0, 1, 2, 3}, g = {-2, -2, 2, 2}, h = {1, 1, 1, 1}, lambda = 1
    // Total G = 0, H = 4, parent = 0.
    // Split between {0,1} and {2,3}: G_L = -4, H_L = 2, G_R = 4, H_R = 2
    // Gain = 16/3 + 16/3 - 0 = 32/3
    const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
    const auto ds = miter::Dataset::fit_numeric(x.data(), 4, 1, false);
    const auto layout = miter::FeatureHistLayout::build(ds);
    const std::vector<float> g{-2.0F, -2.0F, 2.0F, 2.0F};
    const std::vector<float> h{1.0F, 1.0F, 1.0F, 1.0F};
    const std::vector<std::uint32_t> rows{0, 1, 2, 3};

    miter::HistPool pool(1, layout.total_bins);
    std::vector<float> g_ord(4);
    std::vector<float> h_ord(4);
    miter::build_axis_histograms(ds, layout, rows.data(), 4, true, g.data(), h.data(), g_ord.data(),
                                 h_ord.data(), pool.slot(0));

    miter::Params params;
    params.min_data_in_leaf = 1;
    params.min_sum_hessian_in_leaf = 1e-3;
    params.lambda_l2 = 1.0;
    params.min_gain_to_split = 0.0;

    const auto split = miter::find_best_axis_split(ds, layout, pool.slot(0), 0.0, 4.0, 4, params);
    EXPECT_TRUE(split.valid);
    EXPECT_EQ(split.feature, 0U);
    EXPECT_EQ(split.bin_idx, 1U);
    EXPECT_EQ(split.threshold_u16, ds.binner(0).edges[1]);
    EXPECT_NEAR(split.gain, 32.0 / 3.0, 1e-12);

    // min_data_in_leaf = 3 rejects any split on 4 rows
    params.min_data_in_leaf = 3;
    EXPECT_FALSE(miter::find_best_axis_split(ds, layout, pool.slot(0), 0.0, 4.0, 4, params).valid);

    // min_sum_hessian_in_leaf = 2.5 rejects split (each side has H = 2.0)
    params.min_data_in_leaf = 1;
    params.min_sum_hessian_in_leaf = 2.5;
    EXPECT_FALSE(miter::find_best_axis_split(ds, layout, pool.slot(0), 0.0, 4.0, 4, params).valid);

    // min_gain_to_split above 32/3 rejects split
    params.min_sum_hessian_in_leaf = 1e-3;
    params.min_gain_to_split = 11.0;
    EXPECT_FALSE(miter::find_best_axis_split(ds, layout, pool.slot(0), 0.0, 4.0, 4, params).valid);
}

TEST(HistogramTest, NanDirectionChosenByGain) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> x{0.0, 1.0, 10.0, 11.0, nan};
    const auto ds = miter::Dataset::fit_numeric(x.data(), 5, 1, false);
    const auto layout = miter::FeatureHistLayout::build(ds);

    miter::Params params;
    params.min_data_in_leaf = 1;
    params.lambda_l2 = 0.0;

    miter::HistPool pool(1, layout.total_bins);
    std::vector<float> g_ord(5);
    std::vector<float> h_ord(5);
    const std::vector<std::uint32_t> rows{0, 1, 2, 3, 4};
    const std::vector<float> h{1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    // Case 1: NaN has g = -5 (same sign as left {0, 1} with g = -5) -> nan_go_left == true
    {
        const std::vector<float> g_left{-5.0F, -5.0F, 5.0F, 5.0F, -5.0F};
        miter::build_axis_histograms(ds, layout, rows.data(), 5, true, g_left.data(), h.data(),
                                     g_ord.data(), h_ord.data(), pool.slot(0));
        const auto split =
            miter::find_best_axis_split(ds, layout, pool.slot(0), -5.0, 5.0, 5, params);
        EXPECT_TRUE(split.valid);
        EXPECT_TRUE(split.nan_go_left);
    }

    // Case 2: NaN has g = +5 (same sign as right {10, 11} with g = +5) -> nan_go_left == false
    {
        const std::vector<float> g_right{-5.0F, -5.0F, 5.0F, 5.0F, 5.0F};
        miter::build_axis_histograms(ds, layout, rows.data(), 5, true, g_right.data(), h.data(),
                                     g_ord.data(), h_ord.data(), pool.slot(0));
        const auto split =
            miter::find_best_axis_split(ds, layout, pool.slot(0), 5.0, 5.0, 5, params);
        EXPECT_TRUE(split.valid);
        EXPECT_FALSE(split.nan_go_left);
    }
}

TEST(HistogramTest, ConstantFeatureOnlyProducesNoSplit) {
    const std::vector<double> x{3.0, 3.0, 3.0, 3.0};
    const auto ds = miter::Dataset::fit_numeric(x.data(), 4, 1, false);
    const auto layout = miter::FeatureHistLayout::build(ds);
    EXPECT_TRUE(layout.active_features.empty());
    EXPECT_EQ(layout.total_bins, 0U);

    miter::Params params;
    params.min_data_in_leaf = 1;
    const auto split = miter::find_best_axis_split(ds, layout, nullptr, 0.0, 4.0, 4, params);
    EXPECT_FALSE(split.valid);
}

} // namespace
