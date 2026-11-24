#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace miter {

/**
 * @brief Axis histogram binner operating on Q16r unsigned 16-bit codes.
 *
 * Research rule (prototype `histree` `_fit_binning`, T7): thresholds are
 * `unique(ceil(np.quantile(u, linspace(0, 1, max_bin + 1)[1:-1])))` over the non-NaN codes, with
 * numpy's linear quantile reproduced bit for bit. Rare tail values are therefore merged, not given
 * their own bins. Stores strictly increasing uint16 thresholds `edges` (at most `max_bin - 1`). A
 * non-NaN code `u` maps to `bin(u) = upper_bound(edges, u)` in `[0, num_bins - 1]`, so row goes
 * right iff `u >= edges[b]` (`bin(u) <= b <=> u < edges[b]`). NaN (`kQ16NanCode = 65535`) always
 * maps to `nan_bin() = num_bins` (`<= 255`).
 */
struct AxisBinner {
    std::vector<std::uint16_t> edges;
    std::uint16_t num_bins = 1;
    bool has_nan = false;

    std::uint8_t nan_bin() const noexcept { return static_cast<std::uint8_t>(num_bins); }

    std::uint16_t total_bins() const noexcept { return static_cast<std::uint16_t>(num_bins + 1U); }

    static AxisBinner fit(std::span<const std::uint16_t> codes, int max_bin = 255);

    std::uint8_t bin(std::uint16_t u) const noexcept;

    void transform(std::span<const std::uint16_t> codes, std::uint8_t* out) const noexcept;
};

} // namespace miter
