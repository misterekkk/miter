#pragma once

#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace miter {

inline constexpr std::uint16_t kQ16MaxCode = 65534;
inline constexpr std::uint16_t kQ16NanCode = 65535;
inline constexpr std::int32_t kQ16CodeOffset = 32768;
inline constexpr std::int16_t kQ16SignedNanCode = 32767;
inline constexpr double kQ16QuantileLo = 0.0001;
inline constexpr double kQ16QuantileHi = 0.9999;
inline constexpr std::size_t kQ16QuantileIntervals = 256U;
inline constexpr std::size_t kQ16AutoMinSliceSpan = 32U;

/**
 * @brief Exact type-7 sample quantile (linear interpolation) on a non-empty sorted span.
 */
double quantile_type7_sorted(std::span<const double> sorted, double q);

/**
 * @brief Q16r input quantizer (ARCHITECTURE.md, "Input quantization").
 *
 * Maps non-NaN numeric values to 16-bit codes u in [0, 65534] via:
 *   u = clamp(floor((double(x) - lo) * scale + 0.5), 0, 65534)
 * and NaN to 65535. Centred signed arithmetic codes are c = u - 32768 (NaN = 32767).
 */
struct Q16Quantizer {
    double lo = 0.0;
    double scale = 0.0;
    bool is_constant = true;
    std::vector<double> knots;

    static Q16Quantizer fit(std::span<const double> values,
                            QuantizerMode mode = QuantizerMode::Linear);
    static Q16Quantizer fit_strided(const double* data, std::size_t n, std::size_t stride,
                                    QuantizerMode mode = QuantizerMode::Linear);
    static Q16Quantizer fit_strided(const float* data, std::size_t n, std::size_t stride,
                                    QuantizerMode mode = QuantizerMode::Linear);

    std::uint16_t encode_u16(double x) const noexcept;

    static constexpr std::int16_t to_signed_code(std::uint16_t u) noexcept {
        return static_cast<std::int16_t>(static_cast<std::int32_t>(u) - kQ16CodeOffset);
    }

    static constexpr std::uint16_t to_unsigned_code(std::int16_t c) noexcept {
        return static_cast<std::uint16_t>(static_cast<std::int32_t>(c) + kQ16CodeOffset);
    }

    std::int16_t encode_i16(double x) const noexcept { return to_signed_code(encode_u16(x)); }
};

} // namespace miter
