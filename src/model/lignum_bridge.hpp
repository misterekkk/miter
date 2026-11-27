#pragma once

#include <miter/model.hpp>

#include <lignum/model.hpp>

#include <cstddef>
#include <vector>

#include "data/dataset.hpp"

namespace miter {

/**
 * @brief Builds a lignum model from a miter model (ARCHITECTURE.md, "Prediction").
 *
 * Numeric columns are passed to lignum as raw values; categorical columns are passed as
 * miter's category codes (0..K-1, OTHER = K, MISSING = K + 1), with 32-bit string hashes appended
 * at `n_features + sub_idx` when native CTR tables are present.
 * Raw scores equal Evaluator::predict_raw bit for bit.
 */
lignum::Model to_lignum(const Model& model, std::size_t num_iteration = 0, bool binned = true);

/**
 * @brief Raw scores of `lm` (built by to_lignum from `model`) for miter's column views.
 *
 * Numeric columns: raw values (float32 widened exactly); categorical columns: miter's codes,
 * assigned exactly as Evaluator::predict_raw_mixed does (unknown category -> OTHER, missing or
 * out-of-range index -> MISSING). `num_cols` / `cat_cols` follow the model's numeric /
 * categorical feature order. Equal to Evaluator::predict_raw_mixed / predict_raw_views bit for bit.
 * A C-contiguous float64 matrix of a numeric model is passed to lignum without a copy.
 */
void lignum_predict_raw(const lignum::Model& lm, const Model& model, std::size_t n_rows,
                        const std::vector<NumericColumnView>& num_cols,
                        const std::vector<CategoricalColumnView>& cat_cols, double* out,
                        int n_jobs);

} // namespace miter
