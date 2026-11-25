#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

#include "data/axis_binner.hpp"
#include "data/quantizer.hpp"

namespace {

// Expected edges below come from the research rule, computed with numpy 2.5:
//   np.unique(np.ceil(np.quantile(u, np.linspace(0, 1, max_bin + 1)[1:-1])))
TEST(AxisBinnerTest, FewDistinctCodesFollowNumpyQuantiles) {
    // sorted {10, 10, 50, 50, 100}, q = .25/.5/.75 -> 10, 50, 50 -> edges {10, 50}.
    // Bin 0 (u < 10) is empty on training data, as in the prototype; 100 shares a bin with 50.
    const std::vector<std::uint16_t> codes{50, 10, 50, 100, 10};
    const auto binner = miter::AxisBinner::fit(codes, 4);
    EXPECT_EQ(binner.num_bins, 3);
    EXPECT_EQ(binner.edges, (std::vector<std::uint16_t>{10, 50}));
    EXPECT_FALSE(binner.has_nan);
    EXPECT_EQ(binner.nan_bin(), 3);

    EXPECT_EQ(binner.bin(0), 0);
    EXPECT_EQ(binner.bin(9), 0);
    EXPECT_EQ(binner.bin(10), 1);
    EXPECT_EQ(binner.bin(49), 1);
    EXPECT_EQ(binner.bin(50), 2);
    EXPECT_EQ(binner.bin(100), 2);
    EXPECT_EQ(binner.bin(65534), 2);
    EXPECT_EQ(binner.bin(miter::kQ16NanCode), 3);
}

TEST(AxisBinnerTest, InterpolatedEdgesBetweenTwoValues) {
    // {0, 100}, q = .25/.5/.75 -> 25, 50, 75 (linear interpolation, not one bin per value).
    const std::vector<std::uint16_t> codes{100, 0};
    const auto binner = miter::AxisBinner::fit(codes, 4);
    EXPECT_EQ(binner.edges, (std::vector<std::uint16_t>{25, 50, 75}));
}

TEST(AxisBinnerTest, FractionalEdgeRoundsUpToIntegerThreshold) {
    // {0, 0, 1}: q = 0.5 -> 0, q = 0.75 -> vi = 1.5 -> 0.5 -> ceil 1; searchsorted(right) on
    // edge 0.5 puts u = 0 left and u = 1 right, exactly like threshold 1 with `u >= t`.
    const std::vector<std::uint16_t> codes{0, 1, 0};
    const auto binner = miter::AxisBinner::fit(codes, 4);
    EXPECT_EQ(binner.edges, (std::vector<std::uint16_t>{0, 1}));
    EXPECT_EQ(binner.bin(0), 1);
    EXPECT_EQ(binner.bin(1), 2);
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
    // 900 zeros + 1..100, max_bin 10: q = .1-.8 -> 0, q = .9 -> vi = 899.1 -> 0.1 -> ceil 1.
    // The whole tail 1..100 is one bin (rare tail values merged, as in the research rule).
    std::vector<std::uint16_t> codes(900, 0);
    for (std::uint16_t v = 1; v <= 100; ++v) {
        codes.push_back(v);
    }
    const auto binner = miter::AxisBinner::fit(codes, 10);
    EXPECT_EQ(binner.edges, (std::vector<std::uint16_t>{0, 1}));
    EXPECT_EQ(binner.num_bins, 3);
    EXPECT_EQ(binner.bin(0), 1);
    EXPECT_EQ(binner.bin(1), 2);
    EXPECT_EQ(binner.bin(100), 2);
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

TEST(AxisBinnerTest, LargeTransformLutMatchesBinExactly) {
    std::mt19937 rng(98765);
    std::uniform_int_distribution<std::uint16_t> dist(0, miter::kQ16MaxCode);
    std::vector<std::uint16_t> codes(10000);
    for (auto& c : codes) {
        c = dist(rng);
    }
    codes[0] = 0;
    codes[1] = miter::kQ16MaxCode;
    codes[2] = miter::kQ16NanCode;
    codes[9999] = miter::kQ16NanCode;

    const auto binner = miter::AxisBinner::fit(codes, 255);
    std::vector<std::uint8_t> out(codes.size());
    binner.transform(codes, out.data());
    for (std::size_t i = 0; i < codes.size(); ++i) {
        EXPECT_EQ(out[i], binner.bin(codes[i]));
    }
}

} // namespace
