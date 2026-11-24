#pragma once

#include <bit>
#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#define MITER_FASTDIV_HAS_UMULH 1
#endif

namespace miter {

namespace detail {

/// High 64 bits of the 128-bit product a * b, using only 64-bit arithmetic (32-bit limbs).
constexpr std::uint64_t mulhi64_portable(std::uint64_t a, std::uint64_t b) noexcept {
    const std::uint64_t a_lo = a & 0xFFFFFFFFU;
    const std::uint64_t a_hi = a >> 32U;
    const std::uint64_t b_lo = b & 0xFFFFFFFFU;
    const std::uint64_t b_hi = b >> 32U;
    const std::uint64_t lo_lo = a_lo * b_lo;
    const std::uint64_t hi_lo = a_hi * b_lo;
    const std::uint64_t lo_hi = a_lo * b_hi;
    const std::uint64_t hi_hi = a_hi * b_hi;
    // Middle column: cannot overflow (each term < 2^32 after the shifts/masks, sum < 3 * 2^32).
    const std::uint64_t cross = (lo_lo >> 32U) + (hi_lo & 0xFFFFFFFFU) + lo_hi;
    return hi_hi + (hi_lo >> 32U) + (cross >> 32U);
}

/// High 64 bits of a * b using the fastest exact primitive available on this compiler.
inline std::uint64_t mulhi64(std::uint64_t a, std::uint64_t b) noexcept {
#if defined(__SIZEOF_INT128__)
    return static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b)) >> 64U);
#elif defined(MITER_FASTDIV_HAS_UMULH)
    return __umulh(a, b);
#else
    return mulhi64_portable(a, b);
#endif
}

/**
 * floor((hi * 2^64) / d) for hi < d, by restoring binary long division (64 steps).
 *
 * The quotient fits in 64 bits because hi < d. Only used in FastDiv64::prepare (once per
 * candidate per node), never in the per-row loop.
 */
constexpr std::uint64_t div128_hi_by_64_portable(std::uint64_t hi, std::uint64_t d) noexcept {
    std::uint64_t rem = hi;
    std::uint64_t q = 0U;
    for (int i = 0; i < 64; ++i) {
        const bool carry = (rem >> 63U) != 0U;
        rem <<= 1U;
        q <<= 1U;
        if (carry || rem >= d) {
            rem -= d; // wraps correctly when carry is set: true remainder < d.
            q |= 1U;
        }
    }
    return q;
}

} // namespace detail

/**
 * @brief Exact unsigned integer division num / divisor for 1 <= divisor <= 2^34 and num < 2^38.
 *
 * Replaces 64-bit hardware division in hot oblique binning loops while remaining
 * bitwise identical to `num / divisor`.
 *
 * Round-up multiplier (Granlund–Montgomery): with l = ceil(log2 d),
 *   magic = ceil(2^(64+l) / d) - 2^64 = ceil(2^64 * (2^l - d) / d)
 *   q     = (num + mulhi(num, magic)) >> l.
 * `num + mulhi` cannot overflow because num < 2^38.
 *
 * `Portable = true` forces the int128-free path (64-bit limbs only) on every compiler, so the
 * MSVC-style arithmetic is unit-tested on Linux/macOS too. Production code uses `FastDiv64`,
 * which picks `unsigned __int128` (GCC/Clang), `__umulh` (MSVC x64/arm64), or the portable path.
 */
template <bool Portable> struct BasicFastDiv64 {
    std::uint64_t divisor = 1U;
    std::uint64_t magic = 0U;
    std::uint8_t shift = 0U;

    void prepare(std::uint64_t d) noexcept {
        divisor = (d == 0U) ? 1U : d;
        if (divisor <= 1U) {
            magic = 0U;
            shift = 0U;
            return;
        }
        const int l = std::bit_width(divisor - 1U);
        shift = static_cast<std::uint8_t>(l);
        // 2^l - d < d because d > 2^(l-1); l <= 35 so 2^l fits in 64 bits.
        const std::uint64_t excess = (std::uint64_t{1} << static_cast<unsigned int>(l)) - divisor;
        // ceil(excess * 2^64 / d) = floor((excess * 2^64 + d - 1) / d). excess * 2^64 is never a
        // multiple of d unless excess == 0 (d a power of two), so ceil = floor + (excess != 0).
#if defined(__SIZEOF_INT128__)
        if constexpr (!Portable) {
            const unsigned __int128 num = (static_cast<unsigned __int128>(excess) << 64U) +
                                          static_cast<unsigned __int128>(divisor - 1U);
            magic = static_cast<std::uint64_t>(num / static_cast<unsigned __int128>(divisor));
            return;
        }
#endif
        magic = detail::div128_hi_by_64_portable(excess, divisor) + (excess != 0U ? 1U : 0U);
    }

    std::uint64_t div(std::uint64_t num) const noexcept {
        if (divisor <= 1U) {
            return num;
        }
        std::uint64_t t1 = 0U;
        if constexpr (Portable) {
            t1 = detail::mulhi64_portable(num, magic);
        } else {
            t1 = detail::mulhi64(num, magic);
        }
        return (num + t1) >> shift;
    }
};

using FastDiv64 = BasicFastDiv64<false>;
using FastDiv64Portable = BasicFastDiv64<true>;

} // namespace miter
