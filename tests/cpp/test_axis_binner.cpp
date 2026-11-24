#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "data/axis_binner.hpp"
#include "data/quantizer.hpp"

namespace {

TEST(AxisBinnerTest, FewerDistinctCodesThanMaxBin) {
    const std::vector<std::uint16_t> codes{50, 10, 50, 100, 10};
    const auto binner = miter::AxisBinner::fit(codes, 255);
    EXPECT_EQ(binner.num_bins, 3);
    EXPECT_EQ(binner.edges, (std::vector<std::uint16_t>{50, 100}));
    EXPECT_FALSE(binner.has_nan);
    EXPECT_EQ(binner.nan_bin(), 3);

    EXPECT_EQ(binner.bin(0), 0);
    EXPECT_EQ(binner.bin(10), 0);
    EXPECT_EQ(binner.bin(49), 0);
    EXPECT_EQ(binner.bin(50), 1);
    EXPECT_EQ(binner.bin(99), 1);
    EXPECT_EQ(binner.bin(100), 2);
    EXPECT_EQ(binner.bin(65534), 2);
    EXPECT_EQ(binner.bin(miter::kQ16NanCode), 3);
}

TEST(AxisBinnerTest, UniformQuantileBinsAndThresholdExactness) {
    std::vector<std::uint16_t> codes(1000);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<std::uint16_t>(i);
    }
    const auto binner = miter::AxisBinner::fit(codes, 10);
    EXPECT_EQ(binner.num_bins, 10);
    EXPECT_EQ(binner.edges,
              (std::vector<std::uint16_t>{100, 200, 300, 400, 500, 600, 700, 800, 900}));

    for (std::size_t b = 0; b < binner.edges.size(); ++b) {
        const std::uint16_t edge = binner.edges[b];
        EXPECT_LE(binner.bin(static_cast<std::uint16_t>(edge - 1)), b);
        EXPECT_EQ(binner.bin(edge), b + 1);
    }
}

TEST(AxisBinnerTest, SkewedDistributionZeroInflated) {
    std::vector<std::uint16_t> codes(900, 0);
    for (std::uint16_t v = 1; v <= 100; ++v) {
        codes.push_back(v);
    }
    const auto binner = miter::AxisBinner::fit(codes, 10);
    EXPECT_LE(binner.num_bins, 10);
    EXPECT_GE(binner.num_bins, 2);
    EXPECT_EQ(binner.edges.front(), 1);
    EXPECT_EQ(binner.bin(0), 0);
    EXPECT_EQ(binner.bin(1), 1);

    // Every bin 0..num_bins-1 must be non-empty on training data
    std::vector<int> counts(binner.num_bins, 0);
    for (const auto u : codes) {
        counts[binner.bin(u)] += 1;
    }
    for (int c : counts) {
        EXPECT_GT(c, 0);
    }
}

TEST(AxisBinnerTest, ExhaustiveBinEdgeInvariant) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<std::uint16_t> dist(0, miter::kQ16MaxCode);
    std::vector<std::uint16_t> codes(5000);
    for (auto& c : codes) {
        c = dist(rng);
    }
    codes[0] = miter::kQ16NanCode;
    const auto binner = miter::AxisBinner::fit(codes, 255);
    EXPECT_TRUE(binner.has_nan);
    EXPECT_LE(binner.num_bins, 255);
    EXPECT_EQ(binner.nan_bin(), binner.num_bins);
    EXPECT_EQ(binner.bin(miter::kQ16NanCode), binner.nan_bin());

    for (std::size_t i = 1; i < binner.edges.size(); ++i) {
        EXPECT_LT(binner.edges[i - 1], binner.edges[i]);
    }

    // Exhaustively check: bin(u) <= b  <=>  u < edges[b] for all u in [0, 65534]
    for (std::uint32_t u = 0; u <= miter::kQ16MaxCode; ++u) {
        const auto u16 = static_cast<std::uint16_t>(u);
        const auto b_u = binner.bin(u16);
        EXPECT_LT(b_u, binner.num_bins);
        for (std::size_t b = 0; b < binner.edges.size(); ++b) {
            EXPECT_EQ(b_u <= b, u16 < binner.edges[b]);
        }
    }
}

TEST(AxisBinnerTest, ConstantAllNanTinyAndInvalid) {
    const std::vector<std::uint16_t> single{42};
    const auto b_single = miter::AxisBinner::fit(single, 255);
    EXPECT_EQ(b_single.num_bins, 1);
    EXPECT_TRUE(b_single.edges.empty());
    EXPECT_EQ(b_single.bin(42), 0);
    EXPECT_EQ(b_single.bin(miter::kQ16NanCode), 1);

    const std::vector<std::uint16_t> all_nan{miter::kQ16NanCode, miter::kQ16NanCode};
    const auto b_nan = miter::AxisBinner::fit(all_nan, 255);
    EXPECT_EQ(b_nan.num_bins, 1);
    EXPECT_TRUE(b_nan.has_nan);
    EXPECT_EQ(b_nan.nan_bin(), 1);
    EXPECT_EQ(b_nan.bin(miter::kQ16NanCode), 1);

    EXPECT_THROW(miter::AxisBinner::fit(single, 1), std::invalid_argument);
    EXPECT_THROW(miter::AxisBinner::fit(single, 256), std::invalid_argument);
    const std::vector<std::uint16_t> empty;
    EXPECT_THROW(miter::AxisBinner::fit(empty, 255), std::invalid_argument);
}

} // namespace
