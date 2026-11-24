#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace miter {

/**
 * @brief Counter-based deterministic SplitMix64 RNG keyed by (seed, tree, node, cand, purpose).
 *
 * Holds only local per-instance state so parallel threads never share RNG state.
 */
class CounterRng {
public:
    CounterRng(std::uint64_t seed, std::uint64_t tree_index, std::uint64_t node_id,
               std::uint64_t candidate_index, std::uint64_t purpose) noexcept {
        std::uint64_t s = mix64(seed ^ 0x243f6a8885a308d3ULL);
        s = mix64(s ^ tree_index);
        s = mix64(s ^ node_id);
        s = mix64(s ^ candidate_index);
        s = mix64(s ^ purpose);
        state_ = (s == 0ULL) ? 0x9e3779b97f4a7c15ULL : s;
    }

    std::uint64_t next_u64() noexcept {
        state_ += 0x9e3779b97f4a7c15ULL;
        return mix64(state_);
    }

    // Strictly in (0, 1) so std::log never sees 0.0.
    double next_uniform01() noexcept {
        const std::uint64_t u53 = next_u64() >> 11U;
        return (static_cast<double>(u53) + 0.5) * 0x1.0p-53;
    }

    double next_normal() noexcept {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        const double u1 = next_uniform01();
        const double u2 = next_uniform01();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = kTwoPi * u2;
        spare_ = r * std::sin(theta);
        has_spare_ = true;
        return r * std::cos(theta);
    }

    void sample_k_distinct(std::size_t n, std::size_t k, std::uint32_t* out) noexcept {
        if (k >= n) {
            for (std::size_t i = 0; i < n; ++i) {
                out[i] = static_cast<std::uint32_t>(i);
            }
            return;
        }
        for (std::size_t i = 0; i < k; ++i) {
            for (;;) {
                const auto pick = static_cast<std::uint32_t>(next_u64() % n);
                bool dup = false;
                for (std::size_t prev = 0; prev < i; ++prev) {
                    if (out[prev] == pick) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    out[i] = pick;
                    break;
                }
            }
        }
    }

private:
    static constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
        z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31U);
    }

    std::uint64_t state_ = 0;
    double spare_ = 0.0;
    bool has_spare_ = false;
};

} // namespace miter
