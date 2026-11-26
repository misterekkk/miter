#include <miter/miter.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/booster.hpp"
#include "core/ctr.hpp"
#include "core/objective.hpp"
#include "data/dataset.hpp"
#include "model/evaluator.hpp"
#include "model/json_io.hpp"
#include "model/lignum_export.hpp"

namespace {

TEST(CtrTest, CatHashAndCombineDeterministic) {
    EXPECT_EQ(miter::cat_hash_string("alpha"), miter::cat_hash_string("alpha"));
    EXPECT_NE(miter::cat_hash_string("alpha"), miter::cat_hash_string("beta"));
    const std::string long_s =
        "very_long_categorical_token_exceeding_64_bytes_for_cityhash64_loop_coverage_123456";
    EXPECT_EQ(miter::cat_hash_string(long_s), miter::cat_hash_string(long_s));

    const std::uint64_t ab =
        miter::combine_cat_hash(miter::cat_hash_string("a"), miter::cat_hash_string("b"));
    const std::uint64_t ba =
        miter::combine_cat_hash(miter::cat_hash_string("b"), miter::cat_hash_string("a"));
    EXPECT_NE(ab, ba);
    EXPECT_EQ(ab,
              miter::combine_cat_hash(miter::cat_hash_string("a"), miter::cat_hash_string("b")));
}

TEST(CtrTest, OnlineOrderedCtrExactAndZeroLeakage) {
    const std::vector<std::string> vocab = {"a", "b", "singleton0", "singleton1"};
    const std::vector<std::int32_t> codes = {0, 1, 0, -1, 0, 2, 3, -1, 1, 0};
    const std::vector<double> y = {1.0, 4.0, 2.0, -3.0, 8.0, 0.0, 100.0, 5.0, 0.0, 6.0};
    const std::vector<double> w = {1.0, 2.0, 1.0, 1.0, 1.5, 1.0, 2.0, 1.0, 0.5, 1.5};
    const std::size_t n = codes.size();

    const miter::Dataset ds =
        miter::Dataset::fit_mixed(n, {miter::FeatureKind::Categorical}, {"c0"}, {},
                                  {miter::CategoricalColumnView{vocab, codes.data()}});

    miter::Params prm{};
    prm.grow_policy = miter::GrowPolicy::Symmetric;
    prm.max_ctr_complexity = 1;
    prm.ctr_prior_weight = 3.0;
    prm.ctr_n_permutations = 2;
    prm.random_state = 11U;

    miter::CtrTrainingState state;
    state.prepare(ds, prm);

    std::vector<std::uint64_t> hashes(n);
    std::vector<float> got(n);
    const std::uint32_t feat0 = 0U;
    miter::compute_projection_hashes(ds, std::span<const std::uint32_t>(&feat0, 1U), hashes.data());

    double tot_w = 0.0;
    double tot_wy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        tot_w += w[i];
        tot_wy += w[i] * y[i];
    }
    const double prior = tot_wy / tot_w;

    // 1. TargetMean online pass matches brute-force prefix calculation.
    miter::compute_online_ctr(hashes.data(), n, state.perms[0], y.data(), w.data(), prior,
                              prm.ctr_prior_weight, miter::CtrKind::TargetMean, state.scratch,
                              got.data());

    std::unordered_map<std::uint64_t, double> seen_w;
    std::unordered_map<std::uint64_t, double> seen_wy;
    for (const std::uint32_t idx : state.perms[0]) {
        const std::uint64_t key = hashes[idx];
        const double pw = seen_w[key];
        const double pwy = seen_wy[key];
        const auto expected =
            static_cast<float>((pwy + prm.ctr_prior_weight * prior) / (pw + prm.ctr_prior_weight));
        EXPECT_FLOAT_EQ(got[idx], expected);
        seen_w[key] = pw + w[idx];
        seen_wy[key] = pwy + w[idx] * y[idx];
    }
    // Singletons 'singleton0' (y=0.0) and 'singleton1' (y=100.0) both get exact global prior.
    EXPECT_FLOAT_EQ(got[5], static_cast<float>(prior));
    EXPECT_FLOAT_EQ(got[6], static_cast<float>(prior));

    // 2. Counter pass computes exact LOO log1p(count - 1).
    std::vector<float> cnt_got(n);
    miter::compute_online_ctr(hashes.data(), n, state.perms[0], y.data(), w.data(), prior,
                              prm.ctr_prior_weight, miter::CtrKind::Counter, state.scratch,
                              cnt_got.data());
    EXPECT_FLOAT_EQ(cnt_got[0],
                    static_cast<float>(std::log1p(3.0))); // 'a' appears 4 times -> 3 others
    EXPECT_FLOAT_EQ(cnt_got[5], 0.0f);                    // singleton -> log1p(0)
    EXPECT_FLOAT_EQ(cnt_got[3],
                    static_cast<float>(std::log1p(1.0))); // missing appears 2 times -> 1
}

TEST(CtrTest, DynamicPairDiscoveryRecoversXor) {
    std::mt19937_64 rng(42);
    constexpr std::size_t kTr = 1200U;
    constexpr std::size_t kTe = 400U;
    constexpr std::size_t kTot = kTr + kTe;

    std::vector<std::string> va(6), vb(6), vn(8);
    for (int i = 0; i < 6; ++i) {
        va[i] = "a_" + std::to_string(i);
        vb[i] = "b_" + std::to_string(i);
    }
    for (int i = 0; i < 8; ++i) {
        vn[i] = "n_" + std::to_string(i);
    }

    std::vector<std::int32_t> ca(kTot), cb(kTot), cn(kTot);
    std::vector<double> y(kTot);
    for (std::size_t i = 0; i < kTot; ++i) {
        ca[i] = static_cast<std::int32_t>(rng() % 6ULL);
        cb[i] = static_cast<std::int32_t>(rng() % 6ULL);
        cn[i] = static_cast<std::int32_t>(rng() % 8ULL);
        y[i] = (((ca[i] ^ cb[i]) & 1) != 0) ? 1.0 : 0.0;
    }

    const std::vector<miter::FeatureKind> kinds(3, miter::FeatureKind::Categorical);
    const std::vector<std::string> names = {"ca", "cb", "cn"};
    const std::vector<miter::CategoricalColumnView> tr_views = {
        {va, ca.data()}, {vb, cb.data()}, {vn, cn.data()}};
    const std::vector<miter::CategoricalColumnView> te_views = {
        {va, ca.data() + kTr}, {vb, cb.data() + kTr}, {vn, cn.data() + kTr}};

    miter::DataConfig dcfg;
    dcfg.min_data_per_group = 10;
    const auto ds = miter::Dataset::fit_mixed(kTr, kinds, names, {}, tr_views, dcfg);
    const std::span<const double> y_tr(y.data(), kTr);

    miter::Params p_off{};
    p_off.n_estimators = 12;
    p_off.learning_rate = 0.2;
    p_off.grow_policy = miter::GrowPolicy::Symmetric;
    p_off.max_depth = 2; // shallow enough that individual splits cannot easily memorize 6x6 XOR
    p_off.min_data_in_leaf = 10;
    p_off.min_data_per_group = 10;
    p_off.m_mult = 0;
    p_off.early_stopping_rounds = 0;
    p_off.random_state = 7U;
    p_off.n_jobs = 1;

    miter::Params p_ctr = p_off;
    p_ctr.max_ctr_complexity = 2;
    p_ctr.ctr_prior_weight = 2.0;

    const auto res_off =
        miter::fit_booster(ds, miter::Objective::Binary, y_tr, {}, nullptr, {}, {}, p_off);
    const auto res_ctr =
        miter::fit_booster(ds, miter::Objective::Binary, y_tr, {}, nullptr, {}, {}, p_ctr);

    ASSERT_FALSE(res_ctr.model.ctr_projections.empty());
    bool found_pair_ab = false;
    for (const auto& proj : res_ctr.model.ctr_projections) {
        if (proj.cat_features == std::vector<std::uint32_t>({0U, 1U})) {
            found_pair_ab = true;
        }
    }
    EXPECT_TRUE(found_pair_ab);
    EXPECT_THROW(miter::export_raw_forest(res_ctr.model), std::invalid_argument);

    auto eval_acc = [&](const miter::Model& model) {
        std::vector<double> raw(kTe);
        miter::Evaluator::predict_raw_mixed(model, kTe, {}, te_views, raw.data());
        std::size_t correct = 0U;
        for (std::size_t i = 0; i < kTe; ++i) {
            const int pred = (raw[i] >= 0.0) ? 1 : 0;
            if (pred == static_cast<int>(y[kTr + i])) {
                ++correct;
            }
        }
        return static_cast<double>(correct) / static_cast<double>(kTe);
    };

    const double acc_off = eval_acc(res_off.model);
    const double acc_ctr = eval_acc(res_ctr.model);
    EXPECT_GT(acc_ctr, 0.90);
    EXPECT_GT(acc_ctr, acc_off + 0.15);
}

TEST(CtrTest, HighCardMulticlassBaggingJsonAndThreadDeterminism) {
    std::mt19937_64 rng(99);
    std::normal_distribution<double> norm(0.0, 0.25);
    constexpr std::size_t kTr = 900U;
    constexpr std::size_t kVa = 300U;
    constexpr std::size_t kTot = kTr + kVa;
    constexpr std::size_t kCard = 500U; // > max_categories (253)

    std::vector<std::string> vocab;
    vocab.reserve(kCard + 2U);
    for (std::size_t c = 0; c < kCard; ++c) {
        vocab.push_back("id_" + std::to_string(c));
    }
    const std::size_t kUnseen = kCard;
    vocab.push_back("unseen_token");

    std::vector<double> x_num(kTot);
    std::vector<std::int32_t> c_idx(kTot);
    std::vector<double> y(kTot);
    for (std::size_t i = 0; i < kTot; ++i) {
        x_num[i] = norm(rng);
        const std::size_t cat = rng() % kCard;
        c_idx[i] = (i % 37U == 0U) ? -1 : static_cast<std::int32_t>(cat);
        if (i >= kTr && (i % 19U == 0U)) {
            c_idx[i] = static_cast<std::int32_t>(kUnseen);
        }
        const std::size_t eff_cls = (c_idx[i] >= 0 && static_cast<std::size_t>(c_idx[i]) < kCard)
                                        ? (static_cast<std::size_t>(c_idx[i]) % 3U)
                                        : 0U;
        y[i] = static_cast<double>(eff_cls);
    }

    const std::vector<miter::FeatureKind> kinds = {miter::FeatureKind::Numeric,
                                                   miter::FeatureKind::Categorical};
    const std::vector<std::string> names = {"x0", "high_cat"};
    const auto ds = miter::Dataset::fit_mixed(kTr, kinds, names,
                                              {miter::NumericColumnView{x_num.data(), nullptr, 1U}},
                                              {miter::CategoricalColumnView{vocab, c_idx.data()}});
    const auto va_td =
        ds.transform_mixed(kVa, {miter::NumericColumnView{x_num.data() + kTr, nullptr, 1U}},
                           {miter::CategoricalColumnView{vocab, c_idx.data() + kTr}});

    miter::Params p1{};
    p1.n_estimators = 12;
    p1.learning_rate = 0.15;
    p1.grow_policy = miter::GrowPolicy::Symmetric;
    p1.max_depth = 4;
    p1.min_data_in_leaf = 1;
    p1.bagging_fraction = 0.85; // exercises OOB evaluation with NodeKind::Ctr
    p1.max_ctr_complexity = 2;
    p1.ctr_prior_weight = 2.0;
    p1.early_stopping_rounds = 0;
    p1.random_state = 123U;
    p1.n_jobs = 1;

    miter::Params p4 = p1;
    p4.n_jobs = 4;

    const std::span<const double> y_tr(y.data(), kTr);
    const std::span<const double> y_va(y.data() + kTr, kVa);
    const auto r1 =
        miter::fit_booster(ds, miter::Objective::Multiclass, y_tr, {}, &va_td, y_va, {}, p1);
    const auto r4 =
        miter::fit_booster(ds, miter::Objective::Multiclass, y_tr, {}, &va_td, y_va, {}, p4);

    ASSERT_EQ(r1.model.n_classes, 3U);
    ASSERT_FALSE(r1.model.ctr_tables.empty());
    EXPECT_LT(r1.valid_loss.back(), r1.valid_loss.front() * 0.5);

    const std::string json1 = miter::model_to_json(r1.model);
    auto m4_norm = r4.model;
    m4_norm.params.n_jobs = 1;
    EXPECT_EQ(json1, miter::model_to_json(m4_norm));

    const miter::Model loaded = miter::model_from_json(json1);
    EXPECT_EQ(miter::model_to_json(loaded), json1);

    std::vector<double> va_raw1(kVa * 3U);
    std::vector<double> va_raw_loaded(kVa * 3U);
    miter::Evaluator::predict_raw_mixed(
        r1.model, kVa, {miter::NumericColumnView{x_num.data() + kTr, nullptr, 1U}},
        {miter::CategoricalColumnView{vocab, c_idx.data() + kTr}}, va_raw1.data());
    miter::Evaluator::predict_raw_mixed(
        loaded, kVa, {miter::NumericColumnView{x_num.data() + kTr, nullptr, 1U}},
        {miter::CategoricalColumnView{vocab, c_idx.data() + kTr}}, va_raw_loaded.data());

    for (std::size_t idx = 0; idx < kVa * 3U; ++idx) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(va_raw1[idx]),
                  std::bit_cast<std::uint64_t>(va_raw_loaded[idx]));
    }

    // Check validation loss from fit_booster matches predict_raw_mixed loss identically.
    std::vector<double> class_major(kVa * 3U);
    for (std::size_t i = 0; i < kVa; ++i) {
        for (std::size_t k = 0; k < 3U; ++k) {
            class_major[k * kVa + i] = va_raw1[i * 3U + k];
        }
    }
    const double recomputed_vloss =
        miter::compute_multiclass_loss(class_major.data(), kVa, 3U, y_va, {});
    EXPECT_EQ(std::bit_cast<std::uint64_t>(r1.valid_loss.back()),
              std::bit_cast<std::uint64_t>(recomputed_vloss));
}

TEST(CtrTest, DisabledDefaultBitIdentical) {
    const std::vector<std::string> vocab = {"x", "y", "z"};
    std::vector<std::int32_t> codes(120);
    std::vector<double> x_num(120);
    std::vector<double> y(120);
    for (std::size_t i = 0; i < 120U; ++i) {
        codes[i] = static_cast<std::int32_t>(i % 3U);
        x_num[i] = static_cast<double>(i) * 0.05;
        y[i] = x_num[i] + static_cast<double>(codes[i]);
    }
    const auto ds = miter::Dataset::fit_mixed(
        120U, {miter::FeatureKind::Numeric, miter::FeatureKind::Categorical}, {"x", "c"},
        {miter::NumericColumnView{x_num.data(), nullptr, 1U}},
        {miter::CategoricalColumnView{vocab, codes.data()}});

    miter::Params p_def{};
    p_def.n_estimators = 5;
    p_def.grow_policy = miter::GrowPolicy::Symmetric;
    p_def.max_depth = 3;
    p_def.early_stopping_rounds = 0;

    miter::Params p_explicit = p_def;
    p_explicit.max_ctr_complexity = 0;

    const auto r0 =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, p_def);
    const auto r1 =
        miter::fit_booster(ds, miter::Objective::Regression, y, {}, nullptr, {}, {}, p_explicit);
    EXPECT_TRUE(r0.model.ctr_tables.empty());
    EXPECT_EQ(miter::model_to_json(r0.model), miter::model_to_json(r1.model));
}

} // namespace
