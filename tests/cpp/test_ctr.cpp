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

    // 2. Counter pass computes normalized log1p(count) / log1p(max_count).
    std::vector<float> cnt_got(n);
    miter::compute_online_ctr(hashes.data(), n, state.perms[0], y.data(), w.data(), prior,
                              prm.ctr_prior_weight, miter::CtrKind::Counter, state.scratch,
                              cnt_got.data());
    const double max_ld = std::log1p(4.0);
    EXPECT_FLOAT_EQ(cnt_got[0], 1.0f); // 'a' appears 4 times (max_count)
    EXPECT_FLOAT_EQ(cnt_got[5],
                    static_cast<float>(std::log1p(1.0) / max_ld)); // singleton -> 1/4 norm
    EXPECT_FLOAT_EQ(cnt_got[3],
                    static_cast<float>(std::log1p(2.0) / max_ld)); // missing -> 2/4 norm

    // 3. Multi-prior TargetMean matches three single-prior passes bit-for-bit.
    const std::array<double, 3> unit_priors = {0.0, 0.5, 1.0};
    std::vector<float> multi_got(3U * n);
    const std::size_t n_uniq = miter::compute_online_ctr_multi_prior(
        hashes.data(), n, state.perms[0], y.data(), w.data(), unit_priors, prm.ctr_prior_weight,
        state.scratch, multi_got.data());
    EXPECT_EQ(n_uniq, 5U); // 'a', 'b', missing, 'singleton0', 'singleton1'
    for (std::size_t p_idx = 0U; p_idx < 3U; ++p_idx) {
        std::vector<float> single_got(n);
        miter::compute_online_ctr(hashes.data(), n, state.perms[0], y.data(), w.data(),
                                  unit_priors[p_idx], prm.ctr_prior_weight,
                                  miter::CtrKind::TargetMean, state.scratch, single_got.data());
        for (std::size_t i = 0U; i < n; ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(multi_got[p_idx * n + i]),
                      std::bit_cast<std::uint32_t>(single_got[i]));
        }
    }

    // 4. Uniform O(1) binning matches (bin <= b) == (val < borders[b]) across boundaries.
    std::vector<float> boundary_vals = {0.0f, 1.0f};
    for (int m = 1; m <= 15; ++m) {
        const float edge = static_cast<float>(m) / 16.0f;
        boundary_vals.push_back(std::nextafter(edge, 0.0f));
        boundary_vals.push_back(edge);
        boundary_vals.push_back(std::nextafter(edge, 1.0f));
    }
    std::vector<float> u_borders;
    std::vector<std::uint8_t> u_bins(boundary_vals.size());
    const std::uint16_t nb = miter::build_uniform_unit_borders_and_bins(
        boundary_vals.data(), boundary_vals.size(), 15, u_borders, u_bins.data());
    ASSERT_EQ(nb, 15U);
    for (std::size_t i = 0U; i < boundary_vals.size(); ++i) {
        for (std::uint16_t b = 0U; b < nb; ++b) {
            EXPECT_EQ(u_bins[i] <= b, boundary_vals[i] < u_borders[b]);
        }
    }
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
    EXPECT_FALSE(miter::export_raw_forest(res_ctr.model).trees.empty());

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
        x_num[i] = 0.25 * std::sin(static_cast<double>(i) * 0.7);
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
    p1.n_estimators = 16;
    p1.learning_rate = 0.15;
    p1.grow_policy = miter::GrowPolicy::Symmetric;
    p1.max_depth = 4;
    p1.min_data_in_leaf = 1;
    p1.bagging_fraction = 0.85; // exercises OOB evaluation with NodeKind::Ctr
    p1.max_ctr_complexity = 2;
    p1.ctr_prior_weight = 2.0;
    p1.ctr_n_permutations = 4;
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

TEST(CtrTest, MultiCatPairThreadDeterminismAndEvalBitExact) {
    std::mt19937_64 rng(2026);
    constexpr std::size_t kTr = 800U;
    constexpr std::size_t kTe = 250U;
    constexpr std::size_t kTot = kTr + kTe;

    std::vector<std::string> v0(12), v1(10), v2(3), v3(16);
    for (int i = 0; i < 12; ++i) {
        v0[i] = "c0_" + std::to_string(i);
    }
    for (int i = 0; i < 10; ++i) {
        v1[i] = "c1_" + std::to_string(i);
    }
    for (int i = 0; i < 3; ++i) {
        v2[i] = "c2_" + std::to_string(i);
    }
    for (int i = 0; i < 16; ++i) {
        v3[i] = "c3_" + std::to_string(i);
    }

    std::vector<std::int32_t> c0(kTot), c1(kTot), c2(kTot), c3(kTot);
    std::vector<double> y(kTot);
    for (std::size_t i = 0; i < kTot; ++i) {
        c0[i] = static_cast<std::int32_t>(rng() % 12ULL);
        c1[i] = static_cast<std::int32_t>(rng() % 10ULL);
        c2[i] = static_cast<std::int32_t>(rng() % 3ULL);
        c3[i] = static_cast<std::int32_t>(rng() % 16ULL);
        y[i] = (((c0[i] + c1[i]) % 3 == 0) || (c2[i] == 1 && c3[i] % 2 == 0)) ? 1.0 : 0.0;
    }

    const std::vector<miter::FeatureKind> kinds(4, miter::FeatureKind::Categorical);
    const std::vector<std::string> names = {"c0", "c1", "c2", "c3"};
    const auto ds = miter::Dataset::fit_mixed(
        kTr, kinds, names, {},
        {{v0, c0.data()}, {v1, c1.data()}, {v2, c2.data()}, {v3, c3.data()}});

    miter::Params p1{};
    p1.n_estimators = 15;
    p1.learning_rate = 0.1;
    p1.grow_policy = miter::GrowPolicy::Symmetric;
    p1.max_depth = 4;
    p1.min_data_in_leaf = 1;
    p1.bagging_fraction = 0.85;
    p1.max_ctr_complexity = 2;
    p1.ctr_prior_weight = 1.0;
    p1.early_stopping_rounds = 0;
    p1.random_state = 77U;
    p1.n_jobs = 1;

    miter::Params p4 = p1;
    p4.n_jobs = 4;

    const auto r1 =
        miter::fit_booster(ds, miter::Objective::Binary, std::span<const double>(y.data(), kTr), {},
                           nullptr, {}, {}, p1);
    const auto r4 =
        miter::fit_booster(ds, miter::Objective::Binary, std::span<const double>(y.data(), kTr), {},
                           nullptr, {}, {}, p4);

    auto m4 = r4.model;
    m4.params.n_jobs = 1;
    const std::string j1 = miter::model_to_json(r1.model);
    EXPECT_EQ(j1, miter::model_to_json(m4));
    EXPECT_FALSE(r1.model.ctr_tables.empty());

    std::vector<double> p_tr(kTr);
    miter::Evaluator::predict_raw_mixed(
        r1.model, kTr, {}, {{v0, c0.data()}, {v1, c1.data()}, {v2, c2.data()}, {v3, c3.data()}},
        p_tr.data());
    for (std::size_t i = 0; i < kTr; ++i) {
        EXPECT_TRUE(std::isfinite(p_tr[i]));
    }
}

TEST(CtrTest, ResolveEffectiveCtrComplexity) {
    // 1. No categoricals -> 0 even in auto mode.
    {
        const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
        const auto ds =
            miter::Dataset::fit_mixed(4U, {miter::FeatureKind::Numeric}, {"x"},
                                      {miter::NumericColumnView{x.data(), nullptr, 1U}}, {});
        miter::Params prm{};
        prm.max_ctr_complexity = miter::kCtrComplexityAuto;
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds, {}, prm), 0);
    }

    // 2. Single categorical + numeric.
    {
        constexpr std::size_t kN = 2000U;
        std::vector<double> x(kN, 0.0);
        std::vector<std::int32_t> c(kN, 0);
        for (std::size_t i = 0; i < kN; ++i) {
            x[i] = static_cast<double>(i);
            c[i] = static_cast<std::int32_t>(i % 2U);
        }
        const auto ds = miter::Dataset::fit_mixed(
            kN, {miter::FeatureKind::Numeric, miter::FeatureKind::Categorical}, {"x", "c"},
            {miter::NumericColumnView{x.data(), nullptr, 1U}},
            {miter::CategoricalColumnView{{"a", "b"}, c.data()}});
        miter::Params prm{};
        prm.max_ctr_complexity = miter::kCtrComplexityAuto;
        prm.max_cat_to_onehot = 4;

        // Low-card (card=2 <= 4) -> 0; high-card (card=10) -> 1.
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds, {2U}, prm), 0);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds, {10U}, prm), 1);

        // Explicit complexity bypasses auto.
        prm.max_ctr_complexity = 3;
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds, {2U}, prm), 3);
    }

    // 3. Multiple categoricals scaling with n_rows and p_cat.
    {
        auto make_multi = [](std::size_t n_rows, std::size_t p_cat) {
            std::vector<miter::FeatureKind> kinds(p_cat, miter::FeatureKind::Categorical);
            std::vector<std::string> names;
            std::vector<std::int32_t> codes(n_rows, 0);
            std::vector<miter::CategoricalColumnView> views;
            for (std::size_t k = 0; k < p_cat; ++k) {
                names.push_back("c" + std::to_string(k));
                views.push_back({{"a", "b"}, codes.data()});
            }
            return miter::Dataset::fit_mixed(n_rows, kinds, names, {}, views);
        };
        const auto ds_small = make_multi(500U, 2U);
        const auto ds_mid = make_multi(3000U, 2U);
        const auto ds_large2 = make_multi(8500U, 2U);
        const auto ds_large3 = make_multi(8500U, 3U);

        miter::Params prm{};
        prm.max_ctr_complexity = miter::kCtrComplexityAuto;
        prm.max_cat_to_onehot = 4;

        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_small, {3U, 3U}, prm), 0);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_small, {10U, 3U}, prm), 0);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_small, {20U, 3U}, prm), 1);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_mid, {10U, 10U}, prm), 2);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_large2, {10U, 10U}, prm), 2);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_large3, {10U, 10U, 10U}, prm), 3);

        // 4. Multiclass sample-per-class guard (n < 300 * n_classes requires card > 16).
        const auto ds_mc_2k = make_multi(2000U, 2U);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_mc_2k, {10U, 10U}, prm, 10U), 0);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_mc_2k, {20U, 10U}, prm, 10U), 1);
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_mid, {10U, 10U}, prm, 10U), 2);
        prm.max_ctr_complexity = 1;
        EXPECT_EQ(miter::resolve_effective_ctr_complexity(ds_small, {3U, 3U}, prm, 5U), 1);
    }
}

TEST(CtrTest, SupportGateAndComplexityPenalty) {
    // 1. All-unique categorical column (n_rep == 0 < min_rep) is gated out cleanly.
    constexpr std::size_t kUniqN = 200U;
    std::vector<std::string> uniq_vocab;
    std::vector<std::int32_t> uniq_codes(kUniqN);
    std::vector<double> uniq_y(kUniqN);
    uniq_vocab.reserve(kUniqN);
    for (std::size_t i = 0; i < kUniqN; ++i) {
        uniq_vocab.push_back("u_" + std::to_string(i));
        uniq_codes[i] = static_cast<std::int32_t>(i);
        uniq_y[i] = (i % 2U == 0U) ? 1.0 : 0.0;
    }
    const auto ds_uniq =
        miter::Dataset::fit_mixed(kUniqN, {miter::FeatureKind::Categorical}, {"u"}, {},
                                  {miter::CategoricalColumnView{uniq_vocab, uniq_codes.data()}});
    miter::Params p_gate{};
    p_gate.n_estimators = 4;
    p_gate.grow_policy = miter::GrowPolicy::Symmetric;
    p_gate.max_depth = 2;
    p_gate.max_ctr_complexity = 1;
    p_gate.early_stopping_rounds = 0;
    const auto r_gate =
        miter::fit_booster(ds_uniq, miter::Objective::Binary, uniq_y, {}, nullptr, {}, {}, p_gate);
    EXPECT_TRUE(r_gate.model.ctr_tables.empty());

    // 2. Higher ctr_model_size_reg penalizes pairs/triples more heavily and reduces multi-cat
    // CTR tables on noisy high-cardinality combinations.
    std::mt19937_64 rng(321);
    constexpr std::size_t kN = 500U;
    constexpr std::size_t kCard = 20U;
    std::vector<std::string> v0(kCard), v1(kCard), v2(kCard);
    for (std::size_t c = 0; c < kCard; ++c) {
        v0[c] = "a" + std::to_string(c);
        v1[c] = "b" + std::to_string(c);
        v2[c] = "c" + std::to_string(c);
    }
    std::vector<std::int32_t> c0(kN), c1(kN), c2(kN);
    std::vector<double> y(kN);
    for (std::size_t i = 0; i < kN; ++i) {
        c0[i] = static_cast<std::int32_t>(rng() % kCard);
        c1[i] = static_cast<std::int32_t>(rng() % kCard);
        c2[i] = static_cast<std::int32_t>(rng() % kCard);
        y[i] = (c0[i] % 2 == 0) ? 0.65 : 0.35;
        if (rng() % 3ULL == 0ULL) {
            y[i] = 1.0 - y[i];
        }
    }
    const auto ds = miter::Dataset::fit_mixed(
        kN,
        {miter::FeatureKind::Categorical, miter::FeatureKind::Categorical,
         miter::FeatureKind::Categorical},
        {"c0", "c1", "c2"}, {}, {{v0, c0.data()}, {v1, c1.data()}, {v2, c2.data()}});

    miter::Params p_low{};
    p_low.n_estimators = 12;
    p_low.learning_rate = 0.1;
    p_low.grow_policy = miter::GrowPolicy::Symmetric;
    p_low.max_depth = 4;
    p_low.min_data_in_leaf = 1;
    p_low.m_mult = 0;
    p_low.max_ctr_complexity = 2;
    p_low.ctr_model_size_reg = 0.0;
    p_low.early_stopping_rounds = 0;
    p_low.n_jobs = 1;
    p_low.random_state = 19U;

    miter::Params p_high = p_low;
    p_high.ctr_model_size_reg = 15.0;

    const auto r_low =
        miter::fit_booster(ds, miter::Objective::Binary, y, {}, nullptr, {}, {}, p_low);
    const auto r_high =
        miter::fit_booster(ds, miter::Objective::Binary, y, {}, nullptr, {}, {}, p_high);

    auto count_complex = [](const miter::Model& m) {
        std::size_t cnt = 0U;
        for (const auto& t : m.ctr_tables) {
            if (m.ctr_projections[t.projection].complexity() >= 2U) {
                ++cnt;
            }
        }
        return cnt;
    };
    const std::size_t c_low = count_complex(r_low.model);
    const std::size_t c_high = count_complex(r_high.model);
    EXPECT_GT(c_low, 0U);
    EXPECT_LT(c_high, c_low);
}

TEST(CtrTest, AutoComplexityJsonAndThreadDeterminism) {
    std::mt19937_64 rng(555);
    constexpr std::size_t kTr = 1600U;
    constexpr std::size_t kVa = 400U;
    constexpr std::size_t kTot = kTr + kVa;
    constexpr std::size_t kCard = 8U;

    std::vector<std::string> v0(kCard), v1(kCard), v2(kCard);
    for (std::size_t c = 0; c < kCard; ++c) {
        v0[c] = "a_" + std::to_string(c);
        v1[c] = "b_" + std::to_string(c);
        v2[c] = "c_" + std::to_string(c);
    }
    std::vector<std::int32_t> c0(kTot), c1(kTot), c2(kTot);
    std::vector<double> y(kTot);
    for (std::size_t i = 0; i < kTot; ++i) {
        c0[i] = static_cast<std::int32_t>(rng() % kCard);
        c1[i] = static_cast<std::int32_t>(rng() % kCard);
        c2[i] = static_cast<std::int32_t>(rng() % kCard);
        y[i] = (((c0[i] ^ c1[i]) & 1) != 0) ? 1.0 : 0.0;
    }

    const auto ds = miter::Dataset::fit_mixed(
        kTr,
        {miter::FeatureKind::Categorical, miter::FeatureKind::Categorical,
         miter::FeatureKind::Categorical},
        {"c0", "c1", "c2"}, {}, {{v0, c0.data()}, {v1, c1.data()}, {v2, c2.data()}});
    const auto va_td = ds.transform_mixed(
        kVa, {}, {{v0, c0.data() + kTr}, {v1, c1.data() + kTr}, {v2, c2.data() + kTr}});

    miter::Params p1{};
    p1.n_estimators = 12;
    p1.learning_rate = 0.15;
    p1.grow_policy = miter::GrowPolicy::Symmetric;
    p1.max_depth = 3;
    p1.min_data_in_leaf = 5;
    p1.m_mult = 0;
    p1.max_ctr_complexity = miter::kCtrComplexityAuto;
    p1.early_stopping_rounds = 0;
    p1.random_state = 42U;
    p1.n_jobs = 1;

    miter::Params p4 = p1;
    p4.n_jobs = 4;

    const std::span<const double> y_tr(y.data(), kTr);
    const std::span<const double> y_va(y.data() + kTr, kVa);
    const auto r1 =
        miter::fit_booster(ds, miter::Objective::Binary, y_tr, {}, &va_td, y_va, {}, p1);
    const auto r4 =
        miter::fit_booster(ds, miter::Objective::Binary, y_tr, {}, &va_td, y_va, {}, p4);

    EXPECT_EQ(r1.model.params.max_ctr_complexity, miter::kCtrComplexityAuto);
    EXPECT_FALSE(r1.model.ctr_tables.empty());
    const std::string j1 = miter::model_to_json(r1.model);
    auto m4_norm = r4.model;
    m4_norm.params.n_jobs = 1;
    EXPECT_EQ(j1, miter::model_to_json(m4_norm));

    const miter::Model loaded = miter::model_from_json(j1);
    EXPECT_EQ(loaded.params.max_ctr_complexity, miter::kCtrComplexityAuto);
    EXPECT_EQ(miter::model_to_json(loaded), j1);

    std::vector<double> va_raw1(kVa), va_raw_loaded(kVa);
    miter::Evaluator::predict_raw_mixed(
        r1.model, kVa, {}, {{v0, c0.data() + kTr}, {v1, c1.data() + kTr}, {v2, c2.data() + kTr}},
        va_raw1.data());
    miter::Evaluator::predict_raw_mixed(
        loaded, kVa, {}, {{v0, c0.data() + kTr}, {v1, c1.data() + kTr}, {v2, c2.data() + kTr}},
        va_raw_loaded.data());
    for (std::size_t i = 0; i < kVa; ++i) {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(va_raw1[i]),
                  std::bit_cast<std::uint64_t>(va_raw_loaded[i]));
    }
}

} // namespace
