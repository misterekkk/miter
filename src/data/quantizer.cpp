#include "data/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace miter {

namespace {

template <typename FloatT>
Q16Quantizer fit_strided_impl(const FloatT* data, std::size_t n, std::size_t stride) {
    if (n == 0) {
        throw std::invalid_argument("Q16Quantizer::fit requires at least 1 row");
    }
    if (data == nullptr) {
        throw std::invalid_argument("Q16Quantizer::fit received null data pointer");
    }
    if (stride == 0) {
        throw std::invalid_argument("Q16Quantizer::fit requires stride >= 1");
    }

    std::vector<double> vals;
    vals.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double v = static_cast<double>(data[i * stride]);
        if (std::isfinite(v)) {
            vals.push_back(v);
        }
    }

    if (vals.empty()) {
        return Q16Quantizer{0.0, 0.0, true};
    }

    std::sort(vals.begin(), vals.end());

    double lo = quantile_type7_sorted(vals, kQ16QuantileLo);
    double hi = quantile_type7_sorted(vals, kQ16QuantileHi);
    if (hi == lo) {
        lo = vals.front();
        hi = vals.back();
    }

    if (hi <= lo) {
        return Q16Quantizer{lo, 0.0, true};
    }

    const double span = hi - lo;
    const double scale = static_cast<double>(kQ16MaxCode) / span;
    if (!std::isfinite(scale) || scale <= 0.0) {
        return Q16Quantizer{lo, 0.0, true};
    }

    return Q16Quantizer{lo, scale, false};
}

} // namespace

double quantile_type7_sorted(std::span<const double> sorted, double q) {
    if (sorted.empty()) {
        throw std::invalid_argument("quantile_type7_sorted requires a non-empty span");
    }
    if (!(q >= 0.0 && q <= 1.0)) {
        throw std::invalid_argument("quantile_type7_sorted requires q in [0, 1]");
    }
    if (sorted.size() == 1) {
        return sorted[0];
    }

    const double pos = q * static_cast<double>(sorted.size() - 1);
    const auto lo_idx = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi_idx = std::min(lo_idx + 1, sorted.size() - 1);
    const double frac = pos - static_cast<double>(lo_idx);
    const double a = sorted[lo_idx];
    const double b = sorted[hi_idx];
    if (a == b || frac == 0.0) {
        return a;
    }
    return a + frac * (b - a);
}

Q16Quantizer Q16Quantizer::fit(std::span<const double> values) {
    return fit_strided_impl(values.data(), values.size(), 1);
}

Q16Quantizer Q16Quantizer::fit_strided(const double* data, std::size_t n, std::size_t stride) {
    return fit_strided_impl(data, n, stride);
}

Q16Quantizer Q16Quantizer::fit_strided(const float* data, std::size_t n, std::size_t stride) {
    return fit_strided_impl(data, n, stride);
}

std::uint16_t Q16Quantizer::encode_u16(double x) const noexcept {
    if (std::isnan(x)) {
        return kQ16NanCode;
    }
    if (is_constant) {
        return 0;
    }

    const double v = (x - lo) * scale;
    if (v <= 0.0) {
        return 0;
    }
    const auto max_code_d = static_cast<double>(kQ16MaxCode);
    if (v >= max_code_d) {
        return kQ16MaxCode;
    }

    // Use floor(v + 0.5) explicitly per frozen spec (independent of FP rounding mode).
    const double rounded = std::floor(v + 0.5);
    if (rounded <= 0.0) {
        return 0;
    }
    if (rounded >= max_code_d) {
        return kQ16MaxCode;
    }
    return static_cast<std::uint16_t>(rounded);
}

} // namespace miter
