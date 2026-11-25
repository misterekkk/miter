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
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ds_rm.codes_col_major().data()) % 64U, 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ds_rm.bins_col_major().data()) % 64U, 0U);

    // Exact match between row-major and col-major inputs
    EXPECT_EQ(ds_rm.codes_col_major().size(), 12U);
    EXPECT_EQ(ds_rm.bins_col_major().size(), 12U);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(ds_rm.codes_col_major()[i], ds_cm.codes_col_major()[i]);
        EXPECT_EQ(ds_rm.bins_col_major()[i], ds_cm.bins_col_major()[i]);
    }

    // Constant feature flag & NaN code/bin check (row 1, col 2 in col-major)
    EXPECT_FALSE(ds_rm.quantizer(0).is_constant);
    EXPECT_TRUE(ds_rm.quantizer(1).is_constant);
    EXPECT_FALSE(ds_rm.quantizer(2).is_constant);
    EXPECT_EQ(ds_rm.codes_col_major()[2 * 4 + 1], miter::kQ16SignedNanCode);
    EXPECT_EQ(ds_rm.bins_col_major()[2 * 4 + 1], ds_rm.binner(2).nan_bin());

    // Transform on training data gives identical codes and bins
    const auto tx = ds_rm.transform_numeric(row_major.data(), 4, 3, /*col_major=*/false);
    for (std::size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(tx.codes_col_major[i], ds_rm.codes_col_major()[i]);
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
        EXPECT_EQ(ds_f32.codes_col_major()[i], ds_f64.codes_col_major()[i]);
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

    // codes_col_major only contains numeric columns (4 x 2)
    EXPECT_EQ(ds.codes_col_major().size(), 8U);
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

TEST(DatasetTest, MultiThreadFitAndTransformBitIdenticalToSingleThread) {
    const std::size_t n_rows = 12000;
    const std::size_t n_cols = 8;
    std::vector<double> data(n_rows * n_cols);
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            data[i * n_cols + j] = static_cast<double>((i * 17U + j * 31U) % 997U) * 0.1;
        }
    }
    data[7] = std::numeric_limits<double>::quiet_NaN();

    miter::DataConfig c1;
    c1.n_jobs = 1;
    miter::DataConfig c4;
    c4.n_jobs = 4;

    const auto ds1 = miter::Dataset::fit_numeric(data.data(), n_rows, n_cols, false, c1);
    const auto ds4 = miter::Dataset::fit_numeric(data.data(), n_rows, n_cols, false, c4);
    EXPECT_EQ(ds1.codes_col_major().size(), ds4.codes_col_major().size());
    for (std::size_t i = 0; i < ds1.codes_col_major().size(); ++i) {
        EXPECT_EQ(ds1.codes_col_major()[i], ds4.codes_col_major()[i]);
    }
    for (std::size_t i = 0; i < ds1.bins_col_major().size(); ++i) {
        EXPECT_EQ(ds1.bins_col_major()[i], ds4.bins_col_major()[i]);
    }

    const auto tx1 = ds1.transform_numeric(data.data(), n_rows, n_cols, false, 1);
    const auto tx4 = ds4.transform_numeric(data.data(), n_rows, n_cols, false, 4);
    for (std::size_t i = 0; i < tx1.codes_col_major.size(); ++i) {
        EXPECT_EQ(tx1.codes_col_major[i], tx4.codes_col_major[i]);
    }
    for (std::size_t i = 0; i < tx1.bins_col_major.size(); ++i) {
        EXPECT_EQ(tx1.bins_col_major[i], tx4.bins_col_major[i]);
    }
}

} // namespace
