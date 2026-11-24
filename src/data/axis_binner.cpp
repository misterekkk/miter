#include "data/axis_binner.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "data/quantizer.hpp"

namespace miter {

AxisBinner AxisBinner::fit(std::span<const std::uint16_t> codes, int max_bin) {
    if (codes.empty()) {
        throw std::invalid_argument("AxisBinner::fit requires at least 1 row");
    }
    if (max_bin < 2 || max_bin > 255) {
        throw std::invalid_argument("AxisBinner::fit requires max_bin in [2, 255]");
    }

    std::vector<std::uint32_t> counts(static_cast<std::size_t>(kQ16MaxCode) + 1U, 0U);
    std::size_t nan_count = 0;
    for (const std::uint16_t c : codes) {
        if (c == kQ16NanCode) {
            ++nan_count;
        } else {
            ++counts[c];
        }
    }

    std::vector<std::uint16_t> distinct_codes;
    std::vector<std::uint32_t> distinct_counts;
    std::uint64_t n_valid = 0;
    for (std::uint32_t u = 0; u <= kQ16MaxCode; ++u) {
        if (counts[u] > 0U) {
            distinct_codes.push_back(static_cast<std::uint16_t>(u));
            distinct_counts.push_back(counts[u]);
            n_valid += counts[u];
        }
    }

    AxisBinner binner;
    binner.has_nan = (nan_count > 0);

    const std::size_t n_distinct = distinct_codes.size();
    if (n_distinct <= 1U) {
        binner.num_bins = 1;
        return binner;
    }

    if (n_distinct <= static_cast<std::size_t>(max_bin)) {
        binner.edges.reserve(n_distinct - 1U);
        for (std::size_t i = 1; i < n_distinct; ++i) {
            binner.edges.push_back(distinct_codes[i]);
        }
        binner.num_bins = static_cast<std::uint16_t>(n_distinct);
        return binner;
    }

    binner.edges.reserve(static_cast<std::size_t>(max_bin) - 1U);
    std::uint64_t rem_count = n_valid;
    auto rem_bins = static_cast<std::uint64_t>(max_bin);
    std::uint64_t cur_bin_count = 0;
    for (std::size_t i = 0; i + 1U < n_distinct && rem_bins > 1U; ++i) {
        cur_bin_count += distinct_counts[i];
        rem_count -= distinct_counts[i];
        // Exact integer comparison for cur_bin_count >= (cur_bin_count + rem_count) / rem_bins
        if (cur_bin_count * rem_bins >= (cur_bin_count + rem_count)) {
            binner.edges.push_back(distinct_codes[i + 1U]);
            cur_bin_count = 0;
            --rem_bins;
        }
    }
    binner.num_bins = static_cast<std::uint16_t>(binner.edges.size() + 1U);
    return binner;
}

std::uint8_t AxisBinner::bin(std::uint16_t u) const noexcept {
    if (u == kQ16NanCode) {
        return nan_bin();
    }
    const auto it = std::upper_bound(edges.begin(), edges.end(), u);
    return static_cast<std::uint8_t>(it - edges.begin());
}

void AxisBinner::transform(std::span<const std::uint16_t> codes, std::uint8_t* out) const noexcept {
    for (std::size_t i = 0; i < codes.size(); ++i) {
        out[i] = bin(codes[i]);
    }
}

} // namespace miter
