#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

#include "data/quantizer.hpp"

namespace {

TEST(QuantizerTest, Type7QuantilesHandComputed) {
    const std::vector<double> single{42.0};
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(single, 0.0), 42.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(single, 0.5), 42.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(single, 1.0), 42.0);

    const std::vector<double> pair{10.0, 20.0};
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(pair, 0.0), 10.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(pair, 0.25), 12.5);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(pair, 0.5), 15.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(pair, 0.75), 17.5);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(pair, 1.0), 20.0);

    const std::vector<double> five{1.0, 2.0, 4.0, 8.0, 16.0};
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(five, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(five, 0.25), 2.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(five, 0.5), 4.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(five, 0.75), 8.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(five, 0.3), 2.0 + 0.2 * (4.0 - 2.0));

    // Exact index on n = 10001: h = 0.0001 * 10000 = 1.0, h_hi = 0.9999 * 10000 = 9999.0
    std::vector<double> seq(10001);
    for (std::size_t i = 0; i < seq.size(); ++i) {
        seq[i] = static_cast<double>(i);
    }
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(seq, 0.0001), 1.0);
    EXPECT_DOUBLE_EQ(miter::quantile_type7_sorted(seq, 0.9999), 9999.0);
}

TEST(QuantizerTest, NormalRangeAndSignedCodes) {
    // 10001 values 0..10000 => lo = 1.0, hi = 9999.0, scale = 65534 / 9998
    std::vector<double> vals(10001);
    for (std::size_t i = 0; i < vals.size(); ++i) {
        vals[i] = static_cast<double>(i);
    }
    const auto q = miter::Q16Quantizer::fit(vals);
    EXPECT_FALSE(q.is_constant);
    EXPECT_DOUBLE_EQ(q.lo, 1.0);
    EXPECT_DOUBLE_EQ(q.scale, 65534.0 / 9998.0);

    EXPECT_EQ(q.encode_u16(1.0), 0);
    EXPECT_EQ(q.encode_i16(1.0), -32768);

    EXPECT_EQ(q.encode_u16(9999.0), 65534);
    EXPECT_EQ(q.encode_i16(9999.0), 32766);

    // Midpoint (lo + hi) / 2 = 5000.0 -> (5000 - 1) * (65534 / 9998) = 32767.0 -> c = -1
    EXPECT_EQ(q.encode_u16(5000.0), 32767);
    EXPECT_EQ(q.encode_i16(5000.0), -1);
}

TEST(QuantizerTest, HalfUpRoundingWithFloor) {
    // Construct quantizer with lo = 0.0, hi = 65534.0 => scale = 1.0 exactly.
    // On n = 2: q(0.0001) = 0.0001 * 65534, so use manually populated Q16Quantizer to test exact .5
    // ties.
    miter::Q16Quantizer q;
    q.lo = 0.0;
    q.scale = 1.0;
    q.is_constant = false;

    EXPECT_EQ(q.encode_u16(0.499999), 0);
    EXPECT_EQ(q.encode_u16(0.5), 1);
    EXPECT_EQ(q.encode_u16(1.499999), 1);
    EXPECT_EQ(q.encode_u16(1.5), 2);
    EXPECT_EQ(q.encode_u16(2.5), 3);
    EXPECT_EQ(q.encode_u16(3.5), 4);
    EXPECT_EQ(q.encode_u16(65533.5), 65534);
}

TEST(QuantizerTest, SaturationAndNan) {
    const std::vector<double> vals{10.0, 20.0, std::numeric_limits<double>::quiet_NaN()};
    const auto q = miter::Q16Quantizer::fit(vals);
    EXPECT_FALSE(q.is_constant);

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(q.encode_u16(-1e30), 0);
    EXPECT_EQ(q.encode_u16(-inf), 0);
    EXPECT_EQ(q.encode_i16(-inf), -32768);

    EXPECT_EQ(q.encode_u16(1e30), 65534);
    EXPECT_EQ(q.encode_u16(inf), 65534);
    EXPECT_EQ(q.encode_i16(inf), 32766);

    EXPECT_EQ(q.encode_u16(nan), miter::kQ16NanCode);
    EXPECT_EQ(q.encode_i16(nan), miter::kQ16SignedNanCode);
}

TEST(QuantizerTest, QuantileFallbackToMinMax) {
    // 20,000 zeros and one 10.0: q(0.0001) == 0.0 and q(0.9999) == 0.0, so fall back to min/max.
    std::vector<double> vals(20001, 0.0);
    vals.back() = 10.0;
    const auto q = miter::Q16Quantizer::fit(vals);
    EXPECT_FALSE(q.is_constant);
    EXPECT_DOUBLE_EQ(q.lo, 0.0);
    EXPECT_DOUBLE_EQ(q.scale, 65534.0 / 10.0);
    EXPECT_EQ(q.encode_u16(0.0), 0);
    EXPECT_EQ(q.encode_u16(10.0), 65534);
}

TEST(QuantizerTest, ConstantAndAllNanAndTinyFeatures) {
    const double nan = std::numeric_limits<double>::quiet_NaN();

    const std::vector<double> single{7.5};
    const auto q_single = miter::Q16Quantizer::fit(single);
    EXPECT_TRUE(q_single.is_constant);
    EXPECT_EQ(q_single.encode_u16(7.5), 0);
    EXPECT_EQ(q_single.encode_u16(nan), miter::kQ16NanCode);

    const std::vector<double> const_with_nan{3.0, nan, 3.0, 3.0};
    const auto q_const = miter::Q16Quantizer::fit(const_with_nan);
    EXPECT_TRUE(q_const.is_constant);
    EXPECT_EQ(q_const.encode_u16(3.0), 0);
    EXPECT_EQ(q_const.encode_u16(99.0), 0);
    EXPECT_EQ(q_const.encode_u16(nan), miter::kQ16NanCode);

    const std::vector<double> all_nan{nan, nan};
    const auto q_nan = miter::Q16Quantizer::fit(all_nan);
    EXPECT_TRUE(q_nan.is_constant);
    EXPECT_EQ(q_nan.encode_u16(nan), miter::kQ16NanCode);
    EXPECT_EQ(q_nan.encode_i16(nan), miter::kQ16SignedNanCode);

    const std::vector<double> pair{1.0, 5.0};
    const auto q_pair = miter::Q16Quantizer::fit(pair);
    EXPECT_FALSE(q_pair.is_constant);
    EXPECT_EQ(q_pair.encode_u16(1.0), 0);
    EXPECT_EQ(q_pair.encode_u16(5.0), 65534);

    const std::vector<double> empty;
    EXPECT_THROW(miter::Q16Quantizer::fit(empty), std::invalid_argument);
}

TEST(QuantizerTest, SignedUnsignedRoundTrip) {
    for (std::uint32_t u = 0; u <= miter::kQ16NanCode; ++u) {
        const auto u16 = static_cast<std::uint16_t>(u);
        const auto i16 = miter::Q16Quantizer::to_signed_code(u16);
        EXPECT_EQ(miter::Q16Quantizer::to_unsigned_code(i16), u16);
    }
}

} // namespace
