#include "data/axis_binner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "data/quantizer.hpp"

namespace miter {

namespace {

/**
 * np.quantile(sorted, q, method="linear") bit for bit (numpy 2.x `_quantile`):
 *   virtual index vi = (n - 1) * q (`_QuantileMethods["linear"]`), prev = floor(vi) (clamped to the
 * ends), gamma = vi - prev, and the two-sided `_lerp` (a + d*g for g < 0.5, b - d*(1 - g)
 * otherwise). Requires -ffp-contract=off (set in CMake).
 */
template <typename SortedAt>
double quantile_numpy_linear(std::size_t n, double q, const SortedAt& sorted_at) {
    const double vi = static_cast<double>(n - 1U) * q;
    if (vi >= static_cast<double>(n - 1U)) {
        return sorted_at(n - 1U);
    }
    if (vi < 0.0) {
        return sorted_at(0U);
    }
    const double prev = std::floor(vi);
    const auto prev_idx = static_cast<std::size_t>(prev);
    const double a = sorted_at(prev_idx);
    const double b = sorted_at(prev_idx + 1U);
    const double gamma = vi - prev;
    const double d = b - a;
    return (gamma >= 0.5) ? (b - d * (1.0 - gamma)) : (a + d * gamma);
}

} // namespace

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

    std::size_t n_valid = 0;
    std::size_t n_distinct = 0;
    for (std::uint32_t u = 0; u <= kQ16MaxCode; ++u) {
        n_valid += counts[u];
        n_distinct += (counts[u] > 0U) ? 1U : 0U;
    }

    AxisBinner binner;
    binner.has_nan = (nan_count > 0);
    if (n_distinct <= 1U) {
        binner.num_bins = 1;
        return binner;
    }

    // Sorted non-NaN codes as (cumulative end index, code) runs, for order statistics.
    std::vector<std::size_t> run_end;
    std::vector<std::uint16_t> run_code;
    run_end.reserve(n_distinct);
    run_code.reserve(n_distinct);
    std::size_t cum = 0;
    for (std::uint32_t u = 0; u <= kQ16MaxCode; ++u) {
        if (counts[u] > 0U) {
            cum += counts[u];
            run_end.push_back(cum);
            run_code.push_back(static_cast<std::uint16_t>(u));
        }
    }
    const auto sorted_at = [&](std::size_t idx) -> double {
        const auto it = std::upper_bound(run_end.begin(), run_end.end(), idx);
        return static_cast<double>(run_code[static_cast<std::size_t>(it - run_end.begin())]);
    };

    // Research rule (prototype `_fit_binning`): edges = np.unique(np.quantile(u, qs)) with
    // qs = np.linspace(0, 1, max_bin + 1)[1:-1]; bin(u) = searchsorted(edges, u, side="right").
    // For integer codes, `u >= e` <=> `u >= ceil(e)`, so the integer thresholds are
    // unique(ceil(e)); distinct float edges with the same ceil only bound bins that contain no
    // integer code, so the partition of every possible input is identical to the prototype's.
    const double step = 1.0 / static_cast<double>(max_bin); // np.linspace: i * (delta / div)
    binner.edges.reserve(static_cast<std::size_t>(max_bin) - 1U);
    for (int i = 1; i < max_bin; ++i) {
        const double q = static_cast<double>(i) * step;
        const double e = quantile_numpy_linear(n_valid, q, sorted_at);
        const auto t = static_cast<std::uint16_t>(std::ceil(e));
        if (binner.edges.empty() || t > binner.edges.back()) {
            binner.edges.push_back(t);
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
