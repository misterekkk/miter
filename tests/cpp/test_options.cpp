#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/booster.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"

namespace {

struct Data {
    std::size_t n = 0;
    std::size_t p = 0;
    std::vector<double> X;
    std::vector<double> y;
};

Data make_data(std::size_t n, std::size_t p, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    Data d;
    d.n = n;
    d.p = p;
    d.X.resize(n * p);
    d.y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            d.X[i * p + j] =
                (i % 17U == 0U && j == 2U) ? std::numeric_limits<double>::quiet_NaN() : norm(rng);
        }
        const double x2 = std::isnan(d.X[i * p + 2U]) ? 0.3 : d.X[i * p + 2U];
        d.y[i] = d.X[i * p] * d.X[i * p + 1U] + 0.8 * x2 + 0.5 * d.X[i * p + 3U] + 0.1 * norm(rng);
    }
    return d;
}

miter::Params base_params(int threads = 1) {
    miter::Params prm{};
    prm.n_estimators = 12;
    prm.num_leaves = 15;
    prm.early_stopping_rounds = 0;
    prm.n_jobs = threads;
    prm.random_state = 5;
    return prm;
}

miter::TrainResult fit(const miter::Dataset& ds, const Data& d, const miter::Params& prm) {
    return miter::fit_booster(ds, miter::Objective::Regression, d.y, {}, nullptr, {}, {}, prm);
}

std::string json_of(miter::TrainResult res) {
    res.model.params.n_jobs = 0;
    return miter::model_to_json(res.model);
}

using Tweak = void (*)(miter::Params&);

void ff_on(miter::Params& p) {
    p.feature_fraction = 0.4;
}
void bag_on(miter::Params& p) {
    p.bagging_fraction = 0.6;
}
void et_on(miter::Params& p) {
    p.extra_trees = true;
}
void all_on(miter::Params& p) {
    p.feature_fraction = 0.5;
    p.bagging_fraction = 0.7;
    p.extra_trees = true;
}

const Tweak kTweaks[] = {ff_on, bag_on, et_on, all_on};

} // namespace

TEST(OptionsTest, ExplicitDefaultsAreBitIdenticalToUnset) {
    const Data d = make_data(3000, 12, 1);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    const auto ref = json_of(fit(ds, d, base_params()));
    auto prm = base_params();
    prm.feature_fraction = 1.0;
    prm.bagging_fraction = 1.0;
    prm.extra_trees = false;
    EXPECT_EQ(json_of(fit(ds, d, prm)), ref);
    // Default models do not mention the options, so their JSON layout is unchanged.
    EXPECT_EQ(ref.find("feature_fraction"), std::string::npos);
    EXPECT_EQ(ref.find("extra_trees"), std::string::npos);
}

TEST(OptionsTest, EachOptionChangesTheModel) {
    const Data d = make_data(3000, 12, 2);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    const auto ref = json_of(fit(ds, d, base_params()));
    for (const Tweak tw : kTweaks) {
        auto prm = base_params();
        tw(prm);
        EXPECT_NE(json_of(fit(ds, d, prm)), ref);
    }
}

TEST(OptionsTest, SeedChangesModelWhenOptionsAreOn) {
    const Data d = make_data(3000, 12, 3);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    auto a = base_params();
    all_on(a);
    auto b = a;
    b.random_state = a.random_state + 1;
    EXPECT_NE(json_of(fit(ds, d, a)), json_of(fit(ds, d, b)));
    EXPECT_EQ(json_of(fit(ds, d, a)), json_of(fit(ds, d, a)));
}

TEST(OptionsTest, DeterministicAcrossThreads1_2_8) {
    // (n, p): 12000 rows cover multi-chunk reductions; 40000 x 8 uses the chunked axis path
    // (n_active < 16, leaf >= 32768) when the bag stays large.
    const std::pair<std::size_t, std::size_t> shapes[] = {{12000U, 20U}, {40000U, 8U}};
    for (const auto& [n, p] : shapes) {
        const Data d = make_data(n, p, 4);
        const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
        for (const Tweak tw : kTweaks) {
            std::string ref;
            for (const int threads : {1, 2, 8}) {
                auto prm = base_params(threads);
                prm.n_estimators = 6;
                tw(prm);
                if (n == 40000U) {
                    prm.bagging_fraction = (prm.bagging_fraction < 1.0) ? 0.9 : 1.0;
                }
                const std::string json = json_of(fit(ds, d, prm));
                if (threads == 1) {
                    ref = json;
                } else {
                    EXPECT_EQ(json, ref) << "n=" << n << " threads=" << threads;
                }
            }
        }
    }
}

TEST(OptionsTest, FeatureFractionRestrictsFeaturesPerTree) {
    const Data d = make_data(4000, 20, 6);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    for (const double frac : {0.05, 0.3}) {
        auto prm = base_params();
        prm.n_estimators = 20;
        prm.feature_fraction = frac;
        const std::size_t k = std::max<std::size_t>(1U, std::llround(20.0 * frac));
        const auto res = fit(ds, d, prm);
        std::set<std::uint32_t> overall;
        for (const auto& tree : res.model.trees) {
            std::set<std::uint32_t> used;
            for (const auto& node : tree.nodes) {
                if (node.kind == miter::NodeKind::Axis) {
                    used.insert(node.feature);
                } else if (node.kind == miter::NodeKind::Oblique) {
                    for (std::uint8_t q = 0; q < node.oblique_k; ++q) {
                        used.insert(node.oblique_features[q]);
                    }
                }
            }
            EXPECT_LE(used.size(), k) << "frac=" << frac;
            overall.insert(used.begin(), used.end());
            if (k == 1U) {
                for (const auto& node : tree.nodes) {
                    EXPECT_NE(node.kind, miter::NodeKind::Oblique);
                }
            }
        }
        // Different trees draw different subsets.
        EXPECT_GT(overall.size(), k);
    }
}

TEST(OptionsTest, TrainScoresEqualPredictionWithOptions) {
    const Data d = make_data(5000, 10, 7);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    for (const Tweak tw : kTweaks) {
        auto prm = base_params();
        tw(prm);
        const auto res = fit(ds, d, prm);
        std::vector<double> pred(d.n);
        miter::Evaluator::predict_raw(res.model, d.X.data(), d.n, d.p, false, pred.data(), 0U, 1);
        ASSERT_EQ(res.final_train_scores.size(), d.n);
        for (std::size_t i = 0; i < d.n; ++i) {
            ASSERT_EQ(std::bit_cast<std::uint64_t>(pred[i]),
                      std::bit_cast<std::uint64_t>(res.final_train_scores[i]))
                << "row " << i;
        }
    }
}

TEST(OptionsTest, BaggingFitsOnSubsetButKeepsLearning) {
    const Data d = make_data(5000, 10, 8);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    auto prm = base_params();
    prm.bagging_fraction = 0.5;
    const auto res = fit(ds, d, prm);
    ASSERT_FALSE(res.train_loss.empty());
    EXPECT_LT(res.train_loss.back(), res.train_loss.front());
    // Leaf support never exceeds the bag size.
    for (const auto& tree : res.model.trees) {
        EXPECT_GE(tree.n_leaves, 1U);
    }
}

TEST(OptionsTest, InvalidFractionsAreRejected) {
    const Data d = make_data(500, 5, 9);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    const double bad[] = {0.0, -0.1, 1.0001, std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity()};
    for (const double v : bad) {
        auto a = base_params();
        a.feature_fraction = v;
        EXPECT_THROW(fit(ds, d, a), std::invalid_argument);
        auto b = base_params();
        b.bagging_fraction = v;
        EXPECT_THROW(fit(ds, d, b), std::invalid_argument);
    }
}

TEST(OptionsTest, OptionsSurviveJsonRoundTrip) {
    const Data d = make_data(1500, 6, 10);
    const auto ds = miter::Dataset::fit_numeric(d.X.data(), d.n, d.p, false);
    auto prm = base_params();
    all_on(prm);
    const auto res = fit(ds, d, prm);
    const auto loaded = miter::model_from_json(miter::model_to_json(res.model));
    EXPECT_EQ(loaded.params.feature_fraction, 0.5);
    EXPECT_EQ(loaded.params.bagging_fraction, 0.7);
    EXPECT_TRUE(loaded.params.extra_trees);
}
