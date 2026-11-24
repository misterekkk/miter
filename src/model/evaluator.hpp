#pragma once

#include <miter/model.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "data/dataset.hpp"

namespace miter {

class Evaluator {
public:
    static void predict_raw(const Model& model, const double* data, std::size_t n_rows,
                            std::size_t n_cols, bool col_major, double* out,
                            std::size_t num_iteration = 0, int n_jobs = 1);

    static void predict_raw(const Model& model, const float* data, std::size_t n_rows,
                            std::size_t n_cols, bool col_major, double* out,
                            std::size_t num_iteration = 0, int n_jobs = 1);

    static void predict_raw_views(const Model& model, std::size_t n_rows,
                                  const std::vector<NumericColumnView>& num_cols, double* out,
                                  std::size_t num_iteration = 0, int n_jobs = 1);

    static void predict_raw_mixed(const Model& model, std::size_t n_rows,
                                  const std::vector<NumericColumnView>& num_cols,
                                  const std::vector<CategoricalColumnView>& cat_cols, double* out,
                                  std::size_t num_iteration = 0, int n_jobs = 1);

    static void predict_proba_from_raw(const double* raw, std::size_t n_rows, double* proba_2col);

    static void add_tree_predictions_from_codes(
        const Tree& tree, const std::int16_t* codes_row_major, std::size_t n_rows,
        std::size_t n_num_features, const std::vector<std::size_t>& feature_to_num_idx,
        double* scores, const std::uint8_t* bins_col_major = nullptr, int n_jobs = 1) noexcept;
};

} // namespace miter
