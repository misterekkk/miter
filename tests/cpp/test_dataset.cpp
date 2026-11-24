#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "data/dataset.hpp"

namespace {

TEST(DatasetTest, NumericRowMajorAndColMajorAlignmentAndConsistency) {
    // 4 rows x 3 cols:
    // col 0: 0, 1, 2, 3
    // col 1: constant 5, 5, 5, 5
    // col 2: 10, NaN, 20, 30
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> row_major{
        0.0, 5.0, 10.0, 1.0, 5.0, nan, 2.0, 5.0, 20.0, 3.0, 5.0, 30.0,
    };
    const std::vector<double> col_major{
        0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 5.0, 10.0, nan, 20.0, 30.0,
    };

    const auto ds_rm = miter::Dataset::fit_numeric(row_major.data(), 4, 3, /*col_major=*/false);
    const auto ds_cm = miter::Dataset::fit_numeric(col_major.data(), 4, 3, /*col_major=*/true);

    EXPECT_EQ(ds_rm.n_rows(), 4U);
    EXPECT_EQ(ds_rm.n_features(), 3U);
    EXPECT_EQ(ds_rm.n_num_features(), 3U);
    EXPECT_EQ(ds_rm.n_cat_features(), 0U);

    // 64-byte alignment check
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ds_rm.codes_row_major().data()) % 64U, 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ds_rm.bins_col_major().data()) % 64U, 0U);

    // Exact match between row-major and col-major inputs
    EXPECT_EQ(ds_rm.codes_row_major().size(), 12U);
    EXPECT_EQ(ds_rm.bins_col_major().size(), 12U);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(ds_rm.codes_row_major()[i], ds_cm.codes_row_major()[i]);
        EXPECT_EQ(ds_rm.bins_col_major()[i], ds_cm.bins_col_major()[i]);
    }

    // Constant feature flag & NaN code/bin check
    EXPECT_FALSE(ds_rm.quantizer(0).is_constant);
    EXPECT_TRUE(ds_rm.quantizer(1).is_constant);
    EXPECT_FALSE(ds_rm.quantizer(2).is_constant);
    EXPECT_EQ(ds_rm.codes_row_major()[1 * 3 + 2], miter::kQ16SignedNanCode);
    EXPECT_EQ(ds_rm.bins_col_major()[2 * 4 + 1], ds_rm.binner(2).nan_bin());

    // Transform on training data gives identical codes and bins
    const auto tx = ds_rm.transform_numeric(row_major.data(), 4, 3, /*col_major=*/false);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(tx.codes_row_major[i], ds_rm.codes_row_major()[i]);
        EXPECT_EQ(tx.bins_col_major[i], ds_rm.bins_col_major()[i]);
    }
}

TEST(DatasetTest, Float32InputSupport) {
    const std::vector<float> f32_rm{
        1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F,
    };
    const std::vector<double> f64_rm{
        1.0, 10.0, 2.0, 20.0, 3.0, 30.0,
    };
    const auto ds_f32 = miter::Dataset::fit_numeric(f32_rm.data(), 3, 2, false);
    const auto ds_f64 = miter::Dataset::fit_numeric(f64_rm.data(), 3, 2, false);
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(ds_f32.codes_row_major()[i], ds_f64.codes_row_major()[i]);
        EXPECT_EQ(ds_f32.bins_col_major()[i], ds_f64.bins_col_major()[i]);
    }
}

TEST(DatasetTest, MixedNumericAndCategorical) {
    const std::vector<miter::FeatureKind> kinds{
        miter::FeatureKind::Numeric,
        miter::FeatureKind::Categorical,
        miter::FeatureKind::Numeric,
    };
    const std::vector<std::string> names{"num_a", "cat_b", "num_c"};

    const std::vector<double> col_a{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> col_c{100.0, 200.0, 300.0, 400.0};
    const std::vector<miter::NumericColumnView> num_cols{
        miter::NumericColumnView{col_a.data(), nullptr, 1},
        miter::NumericColumnView{col_c.data(), nullptr, 1},
    };

    const std::vector<std::string> cat_vocab{"red", "blue"};
    const std::vector<std::int32_t> cat_idx{0, 1, 0, -1};
    const std::vector<miter::CategoricalColumnView> cat_cols{
        miter::CategoricalColumnView{cat_vocab, cat_idx.data()},
    };

    miter::DataConfig cfg;
    cfg.min_data_per_group = 1;
    const auto ds = miter::Dataset::fit_mixed(4, kinds, names, num_cols, cat_cols, cfg);

    EXPECT_EQ(ds.n_rows(), 4U);
    EXPECT_EQ(ds.n_features(), 3U);
    EXPECT_EQ(ds.n_num_features(), 2U);
    EXPECT_EQ(ds.n_cat_features(), 1U);
    EXPECT_EQ(ds.feature_names(), names);

    // codes_row_major only contains numeric columns (4 x 2)
    EXPECT_EQ(ds.codes_row_major().size(), 8U);
    // bins_col_major contains all features (4 x 3)
    EXPECT_EQ(ds.bins_col_major().size(), 12U);

    // Categorical column is feature 1: red (count 2 -> 0), blue (count 1 -> 1), OTHER = 2, MISSING
    // = 3
    EXPECT_EQ(ds.bins_col_major()[1 * 4 + 0], 0);
    EXPECT_EQ(ds.bins_col_major()[1 * 4 + 1], 1);
    EXPECT_EQ(ds.bins_col_major()[1 * 4 + 2], 0);
    EXPECT_EQ(ds.bins_col_major()[1 * 4 + 3], 3);

    // Transform with unseen category "green" -> OTHER (2)
    const std::vector<double> t_a{1.0, 4.0};
    const std::vector<double> t_c{100.0, 400.0};
    const std::vector<std::string> t_vocab{"green", "blue"};
    const std::vector<std::int32_t> t_idx{0, 1};
    const auto tx = ds.transform_mixed(2,
                                       {miter::NumericColumnView{t_a.data(), nullptr, 1},
                                        miter::NumericColumnView{t_c.data(), nullptr, 1}},
                                       {miter::CategoricalColumnView{t_vocab, t_idx.data()}});
    EXPECT_EQ(tx.bins_col_major[1 * 2 + 0], 2); // green -> OTHER
    EXPECT_EQ(tx.bins_col_major[1 * 2 + 1], 1); // blue -> 1
}

TEST(DatasetTest, ValidationErrors) {
    const std::vector<double> x{1.0, 2.0};
    EXPECT_THROW(miter::Dataset::fit_numeric(x.data(), 0, 2, false), std::invalid_argument);
    EXPECT_THROW(miter::Dataset::fit_numeric(x.data(), 2, 0, false), std::invalid_argument);

    const auto ds = miter::Dataset::fit_numeric(x.data(), 1, 2, false);
    EXPECT_THROW(ds.transform_numeric(x.data(), 1, 3, false), std::invalid_argument);
    EXPECT_THROW(ds.transform_numeric(x.data(), 0, 2, false), std::invalid_argument);
}

} // namespace
