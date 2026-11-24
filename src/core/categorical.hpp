#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstdint>
#include <limits>

#include "core/histogram.hpp"

namespace miter {

struct CatSplitCandidate {
    double gain = -std::numeric_limits<double>::infinity();
    std::uint32_t feature = 0;
    CatBitset left_bitset;
    bool default_go_left = true;
    bool valid = false;
};

/**
 * @brief Find the best K1 categorical split across all active categorical features.
 *
 * Uses zero heap allocations inside the hot loop (at most 255 codes per categorical feature).
 */
CatSplitCandidate find_best_cat_split(const FeatureHistLayout& layout, const HistBin* hist,
                                      double parent_g, double parent_h, std::int32_t parent_c,
                                      const Params& params) noexcept;

} // namespace miter
