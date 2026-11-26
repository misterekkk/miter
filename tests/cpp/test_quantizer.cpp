#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "core/booster.hpp"
#include "data/dataset.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"
#include "model/lignum_export.hpp"

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

TEST(QuantizerTest, QuantileModePreservesHeavyTailBodyAndMonotonicity) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> vals(2000);
    for (std::size_t i = 0; i < 1980U; ++i) {
        vals[i] = 0.001 + 0.099 * (static_cast<double>(i) / 1979.0);
    }
    for (std::size_t i = 1980U; i < 2000U; ++i) {
        vals[i] = 1e9 + 1e7 * static_cast<double>(i - 1980U);
    }

    const auto q_lin = miter::Q16Quantizer::fit(vals, miter::QuantizerMode::Linear);
    const auto q_qnt = miter::Q16Quantizer::fit(vals, miter::QuantizerMode::Quantile);
    const auto q_auto = miter::Q16Quantizer::fit(vals, miter::QuantizerMode::Auto);

    EXPECT_TRUE(q_lin.knots.empty());
    EXPECT_FALSE(q_qnt.knots.empty());
    EXPECT_FALSE(q_auto.knots.empty());

    std::set<std::uint16_t> lin_body;
    std::set<std::uint16_t> qnt_body;
    std::uint16_t prev_u = 0;
    for (std::size_t i = 0; i < 1980U; ++i) {
        lin_body.insert(q_lin.encode_u16(vals[i]));
        const std::uint16_t u = q_qnt.encode_u16(vals[i]);
        qnt_body.insert(u);
        EXPECT_GE(u, prev_u);
        prev_u = u;
    }
    EXPECT_LE(lin_body.size(), 2U);
    EXPECT_GT(qnt_body.size(), 500U);
    EXPECT_GT(*qnt_body.rbegin() - *qnt_body.begin(), 50000U);

    // Check global monotonicity including tail, saturation, and NaN.
    for (std::size_t i = 1; i < vals.size(); ++i) {
        EXPECT_GE(q_qnt.encode_u16(vals[i]), q_qnt.encode_u16(vals[i - 1U]));
    }
    EXPECT_EQ(q_qnt.encode_u16(-inf), 0U);
    EXPECT_EQ(q_qnt.encode_u16(0.0), 0U);
    EXPECT_EQ(q_qnt.encode_u16(inf), miter::kQ16MaxCode);
    EXPECT_EQ(q_qnt.encode_u16(1e15), miter::kQ16MaxCode);
    EXPECT_EQ(q_qnt.encode_u16(nan), miter::kQ16NanCode);
}

TEST(QuantizerTest, AutoModeKeepsCleanFeaturesLinearAndTriggersOnMultimodalSkew) {
    std::mt19937_64 rng(7);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> unif(-5.0, 5.0);

    std::vector<double> gauss(4000), uni(4000), redshift_like(4000);
    for (std::size_t i = 0; i < 4000U; ++i) {
        gauss[i] = norm(rng);
        uni[i] = unif(rng);
        if (i < 1000U) {
            redshift_like[i] = -0.001 + 0.002 * (static_cast<double>(i) / 1000.0);
        } else {
            redshift_like[i] = 0.05 + 6.95 * (static_cast<double>(i - 1000U) / 3000.0);
        }
    }

    const auto qg = miter::Q16Quantizer::fit(gauss, miter::QuantizerMode::Auto);
    const auto qu = miter::Q16Quantizer::fit(uni, miter::QuantizerMode::Auto);
    const auto qr = miter::Q16Quantizer::fit(redshift_like, miter::QuantizerMode::Auto);
    EXPECT_TRUE(qg.knots.empty());
    EXPECT_TRUE(qu.knots.empty());
    EXPECT_FALSE(qr.knots.empty());
}

TEST(QuantizerTest, QuantileBoosterJsonRoundTripAndEvaluatorBitExact) {
    constexpr std::size_t n = 1000U;
    constexpr std::size_t p = 2U;
    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    std::vector<double> X(n * p);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double body = (i % 100U == 99U)
                                ? (1e9 * static_cast<double>(i % 7U + 1U))
                                : (0.001 + 0.099 * static_cast<double>(rng() % 1000ULL) / 1000.0);
        const double clean = norm(rng);
        X[i * p + 0U] = body;
        X[i * p + 1U] = clean;
        const double u_body = (std::min(body, 0.1) - 0.05) * 20.0;
        y[i] = ((u_body + clean > 0.0) ? 1.0 : -1.0) + 0.05 * norm(rng);
    }

    miter::DataConfig dcfg;
    dcfg.quantizer_mode = miter::QuantizerMode::Auto;
    const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false, dcfg, {"heavy", "clean"});
    EXPECT_FALSE(ds.quantizer(0).knots.empty());
    EXPECT_TRUE(ds.quantizer(1).knots.empty());

    // 1. Axis-only (m_mult=0): exports cleanly to raw forest and matches Evaluator bit-for-bit.
    miter::Params pax{};
    pax.n_estimators = 10;
    pax.learning_rate = 0.1;
    pax.m_mult = 0;
    pax.early_stopping_rounds = 0;
    pax.quantizer_mode = miter::QuantizerMode::Auto;
    pax.n_jobs = 1;
    const auto rax =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, pax);
    const miter::RawForest rf = miter::export_raw_forest(rax.model);
    EXPECT_EQ(rf.trees.size(), 10U);

    // 2. Oblique + Quantile: JSON round-trip & 1 vs 4 threads bit-exact, raw export throws.
    miter::Params pob = pax;
    pob.m_mult = 8;
    pob.grow_policy = miter::GrowPolicy::Symmetric;
    pob.max_depth = 3;
    const auto rob1 =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, pob);
    auto pob4 = pob;
    pob4.n_jobs = 4;
    const auto rob4 =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, pob4);

    const std::string j1 = miter::model_to_json(rob1.model);
    auto m4_norm = rob4.model;
    m4_norm.params.n_jobs = 1;
    EXPECT_EQ(j1, miter::model_to_json(m4_norm));

    const miter::Model loaded = miter::model_from_json(j1);
    EXPECT_EQ(miter::model_to_json(loaded), j1);
    EXPECT_THROW(miter::export_raw_forest(rob1.model), std::invalid_argument);

    std::vector<double> pred_orig(n), pred_load(n);
    miter::Evaluator::predict_raw(rob1.model, X.data(), n, p, false, pred_orig.data());
    miter::Evaluator::predict_raw(loaded, X.data(), n, p, false, pred_load.data());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_orig[i]),
                  std::bit_cast<std::uint64_t>(pred_load[i]));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(pred_orig[i]),
                  std::bit_cast<std::uint64_t>(rob1.final_train_scores[i]));
    }
}

} // namespace
