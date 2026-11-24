#include "data/dataset.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace miter {

namespace {

template <typename FloatT>
std::vector<NumericColumnView> make_numeric_views(const FloatT* data, std::size_t n_rows,
                                                  std::size_t n_cols, bool col_major) {
    if (n_rows == 0 || n_cols == 0) {
        throw std::invalid_argument("Dataset requires at least 1 row and 1 feature");
    }
    if (data == nullptr) {
        throw std::invalid_argument("Dataset received null numeric matrix pointer");
    }
    std::vector<NumericColumnView> views;
    views.reserve(n_cols);
    const std::size_t stride = col_major ? 1U : n_cols;
    for (std::size_t j = 0; j < n_cols; ++j) {
        const FloatT* col_ptr = col_major ? (data + j * n_rows) : (data + j);
        NumericColumnView view;
        if constexpr (std::is_same_v<FloatT, double>) {
            view.f64 = col_ptr;
        } else {
            view.f32 = col_ptr;
        }
        view.row_stride = stride;
        views.push_back(view);
    }
    return views;
}

void extract_numeric_column(const NumericColumnView& col, std::size_t n_rows,
                            std::vector<double>& out) {
    if (col.row_stride == 0) {
        throw std::invalid_argument("NumericColumnView requires row_stride >= 1");
    }
    if ((col.f64 == nullptr) == (col.f32 == nullptr)) {
        throw std::invalid_argument(
            "NumericColumnView must provide exactly one of f64 or f32 pointer");
    }
    if (col.f64 != nullptr) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            out[i] = col.f64[i * col.row_stride];
        }
    } else {
        for (std::size_t i = 0; i < n_rows; ++i) {
            out[i] = static_cast<double>(col.f32[i * col.row_stride]);
        }
    }
}

void validate_categorical_views(std::size_t n_rows,
                                const std::vector<CategoricalColumnView>& cat_cols) {
    for (const auto& col : cat_cols) {
        if (n_rows > 0 && col.indices == nullptr) {
            throw std::invalid_argument("CategoricalColumnView received null indices pointer");
        }
    }
}

} // namespace

Dataset Dataset::fit_numeric(const double* data, std::size_t n_rows, std::size_t n_cols,
                             bool col_major, const DataConfig& cfg,
                             std::vector<std::string> feature_names) {
    std::vector<FeatureKind> kinds(n_cols, FeatureKind::Numeric);
    auto views = make_numeric_views(data, n_rows, n_cols, col_major);
    return fit_mixed(n_rows, std::move(kinds), std::move(feature_names), views, {}, cfg);
}

Dataset Dataset::fit_numeric(const float* data, std::size_t n_rows, std::size_t n_cols,
                             bool col_major, const DataConfig& cfg,
                             std::vector<std::string> feature_names) {
    std::vector<FeatureKind> kinds(n_cols, FeatureKind::Numeric);
    auto views = make_numeric_views(data, n_rows, n_cols, col_major);
    return fit_mixed(n_rows, std::move(kinds), std::move(feature_names), views, {}, cfg);
}

Dataset Dataset::fit_mixed(std::size_t n_rows, std::vector<FeatureKind> kinds,
                           std::vector<std::string> feature_names,
                           const std::vector<NumericColumnView>& num_cols,
                           const std::vector<CategoricalColumnView>& cat_cols,
                           const DataConfig& cfg) {
    if (n_rows == 0) {
        throw std::invalid_argument("Dataset::fit requires at least 1 row");
    }
    if (kinds.empty()) {
        throw std::invalid_argument("Dataset::fit requires at least 1 feature");
    }
    if (cfg.max_bin < 2 || cfg.max_bin > 255) {
        throw std::invalid_argument("Dataset::fit requires max_bin in [2, 255]");
    }

    Dataset ds;
    ds.n_rows_ = n_rows;
    ds.n_features_ = kinds.size();
    ds.config_ = cfg;
    ds.feature_kinds_ = std::move(kinds);

    if (feature_names.empty()) {
        ds.feature_names_.reserve(ds.n_features_);
        for (std::size_t f = 0; f < ds.n_features_; ++f) {
            ds.feature_names_.push_back("f" + std::to_string(f));
        }
    } else {
        if (feature_names.size() != ds.n_features_) {
            throw std::invalid_argument("feature_names size must match number of features");
        }
        ds.feature_names_ = std::move(feature_names);
    }

    ds.feature_to_sub_idx_.resize(ds.n_features_);
    for (std::size_t f = 0; f < ds.n_features_; ++f) {
        if (ds.feature_kinds_[f] == FeatureKind::Numeric) {
            ds.feature_to_sub_idx_[f] = ds.num_feature_indices_.size();
            ds.num_feature_indices_.push_back(f);
        } else {
            ds.feature_to_sub_idx_[f] = ds.cat_feature_indices_.size();
            ds.cat_feature_indices_.push_back(f);
        }
    }
    ds.n_num_features_ = ds.num_feature_indices_.size();
    ds.n_cat_features_ = ds.cat_feature_indices_.size();

    if (num_cols.size() != ds.n_num_features_) {
        throw std::invalid_argument("num_cols count does not match numeric features in kinds");
    }
    if (cat_cols.size() != ds.n_cat_features_) {
        throw std::invalid_argument("cat_cols count does not match categorical features in kinds");
    }
    validate_categorical_views(n_rows, cat_cols);

    ds.quantizers_.resize(ds.n_num_features_);
    ds.binners_.resize(ds.n_num_features_);
    ds.cat_encoders_.resize(ds.n_cat_features_);
    ds.codes_row_major_.resize(n_rows * ds.n_num_features_);
    ds.bins_col_major_.resize(n_rows * ds.n_features_);

    std::vector<double> col_f64(n_rows);
    std::vector<std::uint16_t> u_col(n_rows);
    std::int16_t* codes_ptr = ds.codes_row_major_.data();
    std::uint8_t* bins_ptr = ds.bins_col_major_.data();

    for (std::size_t j = 0; j < ds.n_num_features_; ++j) {
        extract_numeric_column(num_cols[j], n_rows, col_f64);
        ds.quantizers_[j] = Q16Quantizer::fit(col_f64);
        const Q16Quantizer& q = ds.quantizers_[j];
        for (std::size_t i = 0; i < n_rows; ++i) {
            const std::uint16_t u = q.encode_u16(col_f64[i]);
            u_col[i] = u;
            codes_ptr[i * ds.n_num_features_ + j] = Q16Quantizer::to_signed_code(u);
        }
        ds.binners_[j] = AxisBinner::fit(u_col, cfg.max_bin);
        const std::size_t f = ds.num_feature_indices_[j];
        ds.binners_[j].transform(u_col, bins_ptr + f * n_rows);
    }

    for (std::size_t c = 0; c < ds.n_cat_features_; ++c) {
        ds.cat_encoders_[c] = CatEncoder::fit_indexed(
            cat_cols[c].vocab, std::span<const std::int32_t>(cat_cols[c].indices, n_rows),
            cfg.min_data_per_group, cfg.max_categories);
        const std::size_t f = ds.cat_feature_indices_[c];
        ds.cat_encoders_[c].transform_indexed(
            cat_cols[c].vocab, std::span<const std::int32_t>(cat_cols[c].indices, n_rows),
            bins_ptr + f * n_rows);
    }

    return ds;
}

TransformedData Dataset::transform_numeric(const double* data, std::size_t n_rows,
                                           std::size_t n_cols, bool col_major) const {
    if (n_cat_features_ != 0 || n_cols != n_num_features_) {
        throw std::invalid_argument("Dataset::transform_numeric feature count/kind mismatch");
    }
    auto views = make_numeric_views(data, n_rows, n_cols, col_major);
    return transform_mixed(n_rows, views, {});
}

TransformedData Dataset::transform_numeric(const float* data, std::size_t n_rows,
                                           std::size_t n_cols, bool col_major) const {
    if (n_cat_features_ != 0 || n_cols != n_num_features_) {
        throw std::invalid_argument("Dataset::transform_numeric feature count/kind mismatch");
    }
    auto views = make_numeric_views(data, n_rows, n_cols, col_major);
    return transform_mixed(n_rows, views, {});
}

TransformedData Dataset::transform_mixed(std::size_t n_rows,
                                         const std::vector<NumericColumnView>& num_cols,
                                         const std::vector<CategoricalColumnView>& cat_cols) const {
    if (n_rows == 0) {
        throw std::invalid_argument("Dataset::transform requires at least 1 row");
    }
    if (num_cols.size() != n_num_features_) {
        throw std::invalid_argument("Dataset::transform numeric feature count mismatch");
    }
    if (cat_cols.size() != n_cat_features_) {
        throw std::invalid_argument("Dataset::transform categorical feature count mismatch");
    }
    validate_categorical_views(n_rows, cat_cols);

    TransformedData out;
    out.n_rows = n_rows;
    out.n_num_features = n_num_features_;
    out.n_features = n_features_;
    out.codes_row_major.resize(n_rows * n_num_features_);
    out.bins_col_major.resize(n_rows * n_features_);

    std::vector<double> col_f64(n_rows);
    std::vector<std::uint16_t> u_col(n_rows);
    std::int16_t* codes_ptr = out.codes_row_major.data();
    std::uint8_t* bins_ptr = out.bins_col_major.data();

    for (std::size_t j = 0; j < n_num_features_; ++j) {
        extract_numeric_column(num_cols[j], n_rows, col_f64);
        const Q16Quantizer& q = quantizers_[j];
        for (std::size_t i = 0; i < n_rows; ++i) {
            const std::uint16_t u = q.encode_u16(col_f64[i]);
            u_col[i] = u;
            codes_ptr[i * n_num_features_ + j] = Q16Quantizer::to_signed_code(u);
        }
        const std::size_t f = num_feature_indices_[j];
        binners_[j].transform(u_col, bins_ptr + f * n_rows);
    }

    for (std::size_t c = 0; c < n_cat_features_; ++c) {
        const std::size_t f = cat_feature_indices_[c];
        cat_encoders_[c].transform_indexed(
            cat_cols[c].vocab, std::span<const std::int32_t>(cat_cols[c].indices, n_rows),
            bins_ptr + f * n_rows);
    }

    return out;
}

const AxisBinner& Dataset::binner(std::size_t feature_idx) const {
    if (feature_idx >= n_features_ || feature_kinds_[feature_idx] != FeatureKind::Numeric) {
        throw std::out_of_range("Dataset::binner feature index is not a valid numeric feature");
    }
    return binners_[feature_to_sub_idx_[feature_idx]];
}

const CatEncoder& Dataset::cat_encoder(std::size_t feature_idx) const {
    if (feature_idx >= n_features_ || feature_kinds_[feature_idx] != FeatureKind::Categorical) {
        throw std::out_of_range(
            "Dataset::cat_encoder feature index is not a valid categorical feature");
    }
    return cat_encoders_[feature_to_sub_idx_[feature_idx]];
}

std::uint16_t Dataset::num_bins_for_feature(std::size_t feature_idx) const {
    if (feature_idx >= n_features_) {
        throw std::out_of_range("Dataset::num_bins_for_feature index out of range");
    }
    const std::size_t sub_idx = feature_to_sub_idx_[feature_idx];
    if (feature_kinds_[feature_idx] == FeatureKind::Numeric) {
        return binners_[sub_idx].total_bins();
    }
    return cat_encoders_[sub_idx].num_codes();
}

} // namespace miter
