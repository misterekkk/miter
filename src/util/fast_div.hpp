#pragma once

#include <bit>
#include <cstdint>

namespace miter {

/**
 * @brief Exact unsigned integer division num / divisor for 1 <= divisor <= 2^34 and num < 2^38.
 *
 * Replaces 64-bit hardware division in hot oblique binning loops while remaining
 * bitwise identical to `num / divisor`.
 */
struct FastDiv64 {
    std::uint64_t divisor = 1U;
#if defined(__SIZEOF_INT128__)
    std::uint64_t magic = 0U;
    std::uint8_t shift = 0U;
#endif

    void prepare(std::uint64_t d) noexcept {
        divisor = (d == 0U) ? 1U : d;
#if defined(__SIZEOF_INT128__)
        if (divisor <= 1U) {
            magic = 0U;
            shift = 0U;
            return;
        }
        const int l = std::bit_width(divisor - 1U);
        const unsigned __int128 num =
            (static_cast<unsigned __int128>(1U) << static_cast<unsigned int>(64 + l)) +
            static_cast<unsigned __int128>(divisor - 1U);
        const unsigned __int128 q_wide = num / static_cast<unsigned __int128>(divisor);
        magic = static_cast<std::uint64_t>(q_wide);
        shift = static_cast<std::uint8_t>(l);
#endif
    }

    std::uint64_t div(std::uint64_t num) const noexcept {
#if defined(__SIZEOF_INT128__)
        if (divisor <= 1U) {
            return num;
        }
        const auto t1 = static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(num) * static_cast<unsigned __int128>(magic)) >> 64U);
        return (num + t1) >> shift;
#else
        return num / divisor;
#endif
    }
};

} // namespace miter
