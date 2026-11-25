#include <lignum/lignum.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

#include "core/booster.hpp"
#include "data/dataset.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "model/lignum_bridge.hpp"
#include "model/lignum_export.hpp"

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Reference walk in lignum's normalised semantics on raw doubles (what lignum will implement).
double eval_raw_tree(const miter::RawForest& f, const miter::RawTree& tree, const double* row) {
    std::uint32_t i = 0;
    for (;;) {
        const miter::RawNode& n = tree.nodes[i];
        bool right = false;
        switch (n.kind) {
        case miter::RawNode::Kind::Leaf:
            return n.value;
        case miter::RawNode::Kind::Numeric: {
            const double x = row[n.feature];
            right = std::isnan(x) ? n.default_right : (x >= n.threshold);
            break;
        }
        case miter::RawNode::Kind::Categorical:
            ADD_FAILURE() << "categorical node in a numeric test";
            return 0.0;
        case miter::RawNode::Kind::Oblique: {
            bool nan = false;
            std::int32_t p = 0;
            for (int q = 0; q < 3; ++q) {
                const double x = row[n.features[q]];
                nan = nan || std::isnan(x);
                const miter::QuantizerInfo& qi = f.quantizers[n.features[q]];
                const miter::Q16Quantizer quant{qi.lo, qi.scale, qi.is_constant};
                const std::int32_t c =
                    static_cast<std::int32_t>(quant.encode_u16(x)) - miter::kQ16CodeOffset;
                p += static_cast<std::int32_t>(n.weights[q]) * c;
            }
            right = nan ? n.default_right : (p >= n.oblique_threshold);
            break;
        }
        }
        i = right ? n.right : n.left;
    }
}

double eval_raw(const miter::RawForest& f, const double* row) {
    double s = f.init_score;
    for (const auto& t : f.trees) {
        s += eval_raw_tree(f, t, row);
    }
    return s;
}

TEST(LignumExportTest, AxisRawThresholdIsExactBoundary) {
    std::mt19937_64 rng(11);
    long checks = 0;
    for (int qi = 0; qi < 60; ++qi) {
        std::vector<double> v(500);
        std::lognormal_distribution<double> ln(0.0, 1.0 + (qi % 5));
        std::normal_distribution<double> nd(qi * 1e3, 1e-3 + qi);
        for (auto& x : v) {
            x = (qi % 2 == 1) ? ln(rng) : nd(rng);
        }
        const auto fit = miter::Q16Quantizer::fit(v);
        const miter::QuantizerInfo info{fit.lo, fit.scale, fit.is_constant};
        ASSERT_FALSE(info.is_constant);
        for (int s = 0; s < 60; ++s) {
            const auto t = static_cast<std::uint16_t>(1U + rng() % miter::kQ16MaxCode);
            const double xt = miter::axis_raw_threshold(info, t);
            ASSERT_FALSE(std::isnan(xt));
            ASSERT_GE(fit.encode_u16(xt), t);
            // The previous double is below the threshold: xt is the smallest such value.
            ASSERT_LT(fit.encode_u16(std::nextafter(xt, -kInf)), t);
            for (int d = 0; d < 200; ++d) {
                const double x =
                    v[rng() % v.size()] * (1.0 + 1e-6 * (static_cast<int>(rng() % 2001) - 1000));
                ASSERT_EQ(fit.encode_u16(x) >= t, x >= xt) << "x=" << x << " t=" << t;
                ++checks;
            }
        }
    }
    EXPECT_GT(checks, 0);
    const miter::QuantizerInfo any{0.0, 1.0, false};
    EXPECT_EQ(miter::axis_raw_threshold(any, 0), -kInf);
    EXPECT_THROW(miter::axis_raw_threshold(any, miter::kQ16NanCode), std::invalid_argument);
}

TEST(LignumExportTest, RawForestMatchesEvaluatorBitwise) {
    constexpr std::size_t n = 6000U;
    constexpr std::size_t p = 8U;
    std::mt19937_64 rng(2026);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::lognormal_distribution<double> heavy(0.0, 2.0);
    std::vector<double> X(n * p);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            const bool nan = (j == 2U && i % 17U == 0U) || (j == 5U && i % 29U == 0U);
            X[i * p + j] = nan ? kNan : ((j % 3U == 0U) ? heavy(rng) : norm(rng));
        }
        const double x2 = std::isnan(X[i * p + 2U]) ? 0.0 : X[i * p + 2U];
        y[i] =
            std::log1p(X[i * p + 0U]) + X[i * p + 1U] * x2 + 0.5 * X[i * p + 4U] + 0.1 * norm(rng);
    }
    const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);
    miter::Params prm{};
    prm.n_estimators = 40;
    prm.num_leaves = 31;
    prm.m_mult = 2;
    prm.early_stopping_rounds = 0;
    prm.random_state = 5;
    const auto res =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, prm);
    const miter::Model& model = res.model;
    const miter::RawForest raw = miter::export_raw_forest(model);
    ASSERT_EQ(raw.trees.size(), std::min(model.best_iteration, model.trees.size()));

    std::size_t n_oblique = 0;
    std::size_t n_axis = 0;
    // Test rows: training rows, saturated / infinite values, and rows exactly on every axis
    // threshold and one ulp below it (in every other column: a training row's values).
    std::vector<double> rows(X.begin(), X.begin() + 2000 * p);
    const double specials[] = {kNan, kInf, -kInf, 1e300, -1e300, 0.0, -0.0};
    for (std::size_t s = 0; s < 400; ++s) {
        std::vector<double> r(X.begin() + (s * p), X.begin() + (s + 1) * p);
        r[s % p] = specials[s % 7U];
        rows.insert(rows.end(), r.begin(), r.end());
    }
    for (const auto& t : raw.trees) {
        for (const auto& nd : t.nodes) {
            if (nd.kind == miter::RawNode::Kind::Oblique) {
                ++n_oblique;
            }
            if (nd.kind != miter::RawNode::Kind::Numeric) {
                continue;
            }
            ++n_axis;
            for (const double x : {nd.threshold, std::nextafter(nd.threshold, -kInf)}) {
                std::vector<double> r(X.begin() + (n_axis % n) * p,
                                      X.begin() + (n_axis % n + 1) * p);
                r[nd.feature] = x;
                rows.insert(rows.end(), r.begin(), r.end());
            }
        }
    }
    ASSERT_GT(n_oblique, 0U);
    ASSERT_GT(n_axis, 0U);

    const std::size_t m = rows.size() / p;
    std::vector<double> ref(m);
    miter::Evaluator::predict_raw(model, rows.data(), m, p, false, ref.data());
    for (std::size_t i = 0; i < m; ++i) {
        const double got = eval_raw(raw, &rows[i * p]);
        ASSERT_EQ(std::bit_cast<std::uint64_t>(got), std::bit_cast<std::uint64_t>(ref[i]))
            << "row " << i << ": " << got << " vs " << ref[i];
    }
}

TEST(LignumBridgeTest, LignumPredictionsEqualMiterEvaluatorBitwise) {
    // Regression and binary models with NaN, oblique and axis splits; lignum's KNode kernel
    // (binned falls back for oblique trees) must give miter's raw scores bit for bit.
    constexpr std::size_t n = 5000U;
    constexpr std::size_t p = 6U;
    std::mt19937_64 rng(77);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::lognormal_distribution<double> heavy(0.0, 1.5);
    std::vector<double> X(n * p);
    std::vector<double> y(n);
    std::vector<double> yb(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            const bool nan = (j == 1U && i % 13U == 0U) || (j == 4U && i % 31U == 0U);
            X[i * p + j] = nan ? kNan : ((j % 2U == 0U) ? heavy(rng) : norm(rng));
        }
        const double x1 = std::isnan(X[i * p + 1U]) ? -1.0 : X[i * p + 1U];
        y[i] = std::sin(X[i * p + 0U]) + x1 * X[i * p + 3U] + 0.2 * norm(rng);
        yb[i] = (y[i] > 0.0) ? 1.0 : 0.0;
    }
    std::vector<double> rows(X.begin(), X.begin() + 3000 * p);
    const double specials[] = {kNan, kInf, -kInf, 1e300, -1e300, 0.0};
    for (std::size_t s = 0; s < 300; ++s) {
        std::vector<double> r(X.begin() + s * p, X.begin() + (s + 1) * p);
        r[s % p] = specials[s % 6U];
        rows.insert(rows.end(), r.begin(), r.end());
    }
    const auto ds = miter::Dataset::fit_numeric(X.data(), n, p, false);
    for (const auto obj : {miter::Objective::Regression, miter::Objective::Binary}) {
        miter::Params prm{};
        prm.n_estimators = 30;
        prm.num_leaves = 31;
        prm.m_mult = 2;
        prm.early_stopping_rounds = 0;
        prm.random_state = 3;
        const auto& target = (obj == miter::Objective::Binary) ? yb : y;
        const auto res = miter::fit_booster(ds, obj, target, {}, nullptr, {}, {}, prm);
        // Rows on and one ulp below every axis threshold of this model.
        std::vector<double> test_rows = rows;
        const auto raw = miter::export_raw_forest(res.model);
        std::size_t n_oblique = 0;
        for (const auto& t : raw.trees) {
            for (const auto& nd : t.nodes) {
                n_oblique += nd.kind == miter::RawNode::Kind::Oblique ? 1U : 0U;
                if (nd.kind != miter::RawNode::Kind::Numeric) {
                    continue;
                }
                for (const double x : {nd.threshold, std::nextafter(nd.threshold, -kInf)}) {
                    std::vector<double> r(X.begin(), X.begin() + p);
                    r[nd.feature] = x;
                    test_rows.insert(test_rows.end(), r.begin(), r.end());
                }
            }
        }
        ASSERT_GT(n_oblique, 0U);
        const std::size_t m = test_rows.size() / p;
        std::vector<double> ref(m);
        miter::Evaluator::predict_raw(res.model, test_rows.data(), m, p, false, ref.data());
        for (const bool binned : {false, true}) {
            const lignum::Model lm = miter::to_lignum(res.model, 0, binned);
            EXPECT_EQ(lm.n_oblique, n_oblique);
            if (binned) {
                // The binned kernel takes the ONode path for oblique trees (no silent KNode
                // fallback).
                ASSERT_NE(lm.binned, nullptr);
            } else {
                ASSERT_EQ(lm.binned, nullptr);
            }
            std::vector<double> got(m);
            lm.predict_raw(test_rows.data(), m, p, got.data(), 2);
            for (std::size_t i = 0; i < m; ++i) {
                ASSERT_EQ(std::bit_cast<std::uint64_t>(got[i]),
                          std::bit_cast<std::uint64_t>(ref[i]))
                    << "objective " << static_cast<int>(obj) << " binned " << binned << " row "
                    << i;
            }
            // Format 0.10 round trip (mmap): still miter's raw scores bit for bit.
            const auto path =
                (std::filesystem::temp_directory_path() / "miter_lignum_roundtrip.bin").string();
            lm.save(path);
            const lignum::Model loaded = lignum::load(path, "auto", binned);
            EXPECT_EQ(loaded.source, lignum::model_source::Miter);
            std::vector<double> again(m);
            loaded.predict_raw(test_rows.data(), m, p, again.data(), 2);
            for (std::size_t i = 0; i < m; ++i) {
                ASSERT_EQ(std::bit_cast<std::uint64_t>(again[i]),
                          std::bit_cast<std::uint64_t>(ref[i]))
                    << "after save/load, binned " << binned << " row " << i;
            }
        }
    }
}

} // namespace
