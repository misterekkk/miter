#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "data/axis_binner.hpp"
#include "data/cat_encoder.hpp"
#include "data/quantizer.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

enum class FeatureKind : std::uint8_t {
    Numeric = 0,
    Categorical = 1,
};

struct DataConfig {
    int max_bin = 255;
    int min_data_per_group = kDefaultMinDataPerGroup;
    int max_categories = kDefaultMaxCategories;
    int n_jobs = 1;
    QuantizerMode quantizer_mode = QuantizerMode::Linear;
};

struct NumericColumnView {
    const double* f64 = nullptr;
    const float* f32 = nullptr;
    std::size_t row_stride = 1;
};

struct CategoricalColumnView {
    std::vector<std::string> vocab;
    const std::int32_t* indices = nullptr; // < 0 means missing
};

struct TransformedData {
    std::size_t n_rows = 0;
    std::size_t n_num_features = 0;
    std::size_t n_features = 0;
    AlignedBuffer<std::int16_t> codes_col_major;
    AlignedBuffer<std::uint8_t> bins_col_major;
    AlignedBuffer<std::uint32_t> cat_hashes_col_major;
};

/**
 * @brief Training dataset holding column-major Q16r signed codes (numeric features) and
 *        column-major uint8 bin indices (all features), plus fitted quantizers/binners/encoders.
 */
class Dataset {
public:
    static Dataset fit_numeric(const double* data, std::size_t n_rows, std::size_t n_cols,
                               bool col_major, const DataConfig& cfg = {},
                               std::vector<std::string> feature_names = {});

    static Dataset fit_numeric(const float* data, std::size_t n_rows, std::size_t n_cols,
                               bool col_major, const DataConfig& cfg = {},
                               std::vector<std::string> feature_names = {});

    static Dataset fit_mixed(std::size_t n_rows, std::vector<FeatureKind> kinds,
                             std::vector<std::string> feature_names,
                             const std::vector<NumericColumnView>& num_cols,
                             const std::vector<CategoricalColumnView>& cat_cols,
                             const DataConfig& cfg = {});

    TransformedData transform_numeric(const double* data, std::size_t n_rows, std::size_t n_cols,
                                      bool col_major, int n_jobs = 1) const;

    TransformedData transform_numeric(const float* data, std::size_t n_rows, std::size_t n_cols,
                                      bool col_major, int n_jobs = 1) const;

    TransformedData transform_mixed(std::size_t n_rows,
                                    const std::vector<NumericColumnView>& num_cols,
                                    const std::vector<CategoricalColumnView>& cat_cols,
                                    int n_jobs = 1) const;

    std::size_t n_rows() const noexcept { return n_rows_; }

    std::size_t n_features() const noexcept { return n_features_; }

    std::size_t n_num_features() const noexcept { return n_num_features_; }

    std::size_t n_cat_features() const noexcept { return n_cat_features_; }

    const DataConfig& config() const noexcept { return config_; }

    const std::vector<FeatureKind>& feature_kinds() const noexcept { return feature_kinds_; }

    const std::vector<std::string>& feature_names() const noexcept { return feature_names_; }

    const std::vector<std::size_t>& num_feature_indices() const noexcept {
        return num_feature_indices_;
    }

    const std::vector<std::size_t>& cat_feature_indices() const noexcept {
        return cat_feature_indices_;
    }

    const Q16Quantizer& quantizer(std::size_t num_idx) const { return quantizers_.at(num_idx); }

    const std::vector<Q16Quantizer>& quantizers() const noexcept { return quantizers_; }

    const AxisBinner& binner(std::size_t feature_idx) const;

    const CatEncoder& cat_encoder(std::size_t feature_idx) const;

    std::uint16_t num_bins_for_feature(std::size_t feature_idx) const;

    std::size_t cat_sub_index(std::size_t feature_idx) const noexcept {
        return feature_to_sub_idx_[feature_idx];
    }

    const AlignedBuffer<std::int16_t>& codes_col_major() const noexcept { return codes_col_major_; }

    const AlignedBuffer<std::uint8_t>& bins_col_major() const noexcept { return bins_col_major_; }

    const AlignedBuffer<std::uint32_t>& cat_hashes_col_major() const noexcept {
        return cat_hashes_col_major_;
    }

private:
    std::size_t n_rows_ = 0;
    std::size_t n_features_ = 0;
    std::size_t n_num_features_ = 0;
    std::size_t n_cat_features_ = 0;
    DataConfig config_;
    std::vector<FeatureKind> feature_kinds_;
    std::vector<std::string> feature_names_;
    std::vector<std::size_t> num_feature_indices_;
    std::vector<std::size_t> cat_feature_indices_;
    std::vector<std::size_t> feature_to_sub_idx_;
    std::vector<Q16Quantizer> quantizers_;
    std::vector<AxisBinner> binners_;
    std::vector<CatEncoder> cat_encoders_;
    AlignedBuffer<std::int16_t> codes_col_major_;
    AlignedBuffer<std::uint8_t> bins_col_major_;
    AlignedBuffer<std::uint32_t> cat_hashes_col_major_;
};

} // namespace miter
