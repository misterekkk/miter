#pragma once

#include <miter/model.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace miter {

/**
 * @brief A miter tree in lignum's normalised split semantics (docs/M6_DESIGN.md §1).
 *
 * Numeric:     go right iff isnan(x) ? default_right : x >= threshold   (raw double input)
 * Categorical: go right iff the row's miter category code is NOT in `cat_left_codes`
 * Oblique:     go right iff any of the k inputs is NaN ? default_right : P >= oblique_threshold,
 *              P = sum_q weights[q] * (u(x_{features[q]}) - 32768) in int32, u = Q16r code.
 *
 * Axis splits need no quantization: miter's `u >= t` is exactly `x >= axis_raw_threshold(q, t)`.
 */
struct RawNode {
    enum class Kind : std::uint8_t { Leaf = 0, Numeric = 1, Categorical = 2, Oblique = 3 };
    Kind kind = Kind::Leaf;
    bool default_right = false;
    std::uint32_t left = 0;
    std::uint32_t right = 0;
    std::uint32_t feature = 0;                 // Numeric / Categorical: input column
    double threshold = 0.0;                    // Numeric
    double value = 0.0;                        // Leaf (already scaled by the learning rate)
    std::vector<std::uint16_t> cat_left_codes; // Categorical: sorted codes going left
    std::uint8_t k = 0;                        // Oblique: 2 or 3
    std::uint32_t features[3] = {0, 0, 0};     // Oblique: input columns
    std::int16_t weights[3] = {0, 0, 0};
    std::int32_t oblique_threshold = 0;
};

struct RawTree {
    std::vector<RawNode> nodes; // node 0 is the root
};

struct RawForest {
    double init_score = 0.0;
    Objective objective = Objective::Regression;
    std::vector<QuantizerInfo> quantizers; // per input column (used by oblique splits)
    std::vector<RawTree> trees;            // model order, truncated to the iteration count
};

/**
 * @brief Smallest double x (never NaN) with encode_u16(x) >= t under quantizer `q`.
 *
 * Exact by bisection over the ordered doubles; encode is monotone non-decreasing in x.
 * t == 0 gives -infinity (every non-NaN value goes right).
 */
double axis_raw_threshold(const QuantizerInfo& q, std::uint16_t t);

/**
 * @brief Converts the first `num_iteration` trees (0 = best_iteration, or all) of a miter model.
 * @throws std::invalid_argument for oblique splits with k > 3 (not exported).
 */
RawForest export_raw_forest(const Model& model, std::size_t num_iteration = 0);

} // namespace miter
