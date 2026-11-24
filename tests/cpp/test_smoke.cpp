#include <miter/miter.hpp>

#include <gtest/gtest.h>
#include <string>

namespace {

TEST(SmokeTest, VersionNonEmpty) {
    const auto ver = miter::version();
    EXPECT_FALSE(ver.empty());
    EXPECT_TRUE(ver.starts_with("0.1.0"));
}

TEST(SmokeTest, DefaultParamsMatchFrozenSpec) {
    const miter::Params params{};
    EXPECT_EQ(params.n_estimators, 5000);
    EXPECT_DOUBLE_EQ(params.learning_rate, 0.05);
    EXPECT_EQ(params.num_leaves, 31);
    EXPECT_EQ(params.max_depth, -1);
    EXPECT_EQ(params.min_data_in_leaf, 20);
    EXPECT_DOUBLE_EQ(params.min_sum_hessian_in_leaf, 1e-3);
    EXPECT_DOUBLE_EQ(params.lambda_l2, 1.0);
    EXPECT_DOUBLE_EQ(params.min_gain_to_split, 0.0);
    EXPECT_EQ(params.max_bin, 255);
    EXPECT_EQ(params.max_cat_to_onehot, 4);
    EXPECT_EQ(params.min_data_per_group, 100);
    EXPECT_EQ(params.max_categories, 253);
    EXPECT_DOUBLE_EQ(params.cat_smooth, 10.0);
    EXPECT_DOUBLE_EQ(params.cat_l2, 10.0);
    EXPECT_EQ(params.max_cat_threshold, 32);
    EXPECT_EQ(params.m_mult, 2);
    EXPECT_EQ(params.k_proj, 0);
    EXPECT_EQ(params.n_jobs, -1);
    EXPECT_EQ(params.random_state, 0U);
}

TEST(SmokeTest, LignumLinkedAndCallable) {
    EXPECT_TRUE(miter::lignum_linked());
    const auto h1 = miter::lignum_smoke_hash("miter");
    const auto h2 = miter::lignum_smoke_hash("miter");
    const auto h3 = miter::lignum_smoke_hash("other");
    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, h3);
}

TEST(SmokeTest, OpenMPQuery) {
    EXPECT_GE(miter::openmp_max_threads(), 1);
}

} // namespace
