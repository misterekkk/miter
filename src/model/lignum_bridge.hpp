#pragma once

#include <miter/model.hpp>

#include <lignum/model.hpp>

#include <cstddef>

namespace miter {

/**
 * @brief Builds a lignum model from a miter model (docs/M6_DESIGN.md).
 *
 * Numeric columns are passed to lignum as raw values; categorical columns must be passed as
 * miter's category codes (0..K-1, OTHER = K, MISSING = K + 1), which miter's Python layer computes.
 * Raw scores equal Evaluator::predict_raw bit for bit.
 */
lignum::Model to_lignum(const Model& model, std::size_t num_iteration = 0, bool binned = true);

} // namespace miter
