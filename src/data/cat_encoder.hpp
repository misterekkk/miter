#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace miter {

inline constexpr int kDefaultMinDataPerGroup = 100;
inline constexpr int kDefaultMaxCategories = 253;

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view sv) const noexcept {
        return std::hash<std::string_view>{}(sv);
    }
};

/**
 * @brief K1 categorical encoder (docs/IMPLEMENTATION_PLAN.md Section 2.5).
 *
 * Ranks categories with frequency >= min_data_per_group descending (ties broken by ascending name)
 * up to max_categories (default 253), assigning codes 0..K-1.
 * Code K = OTHER (rare or unseen categories), code K + 1 = MISSING (NaN/null).
 */
struct CatEncoder {
    std::vector<std::string> categories;
    std::unordered_map<std::string, std::uint8_t, TransparentStringHash, std::equal_to<>> lookup;

    std::uint8_t other_code() const noexcept {
        return static_cast<std::uint8_t>(categories.size());
    }

    std::uint8_t missing_code() const noexcept {
        return static_cast<std::uint8_t>(categories.size() + 1U);
    }

    std::uint16_t num_codes() const noexcept {
        return static_cast<std::uint16_t>(categories.size() + 2U);
    }

    static CatEncoder fit_indexed(std::span<const std::string> vocab,
                                  std::span<const std::int32_t> row_indices,
                                  int min_data_per_group = kDefaultMinDataPerGroup,
                                  int max_categories = kDefaultMaxCategories);

    static CatEncoder fit_strings(std::span<const std::string_view> values,
                                  std::span<const std::uint8_t> is_missing,
                                  int min_data_per_group = kDefaultMinDataPerGroup,
                                  int max_categories = kDefaultMaxCategories);

    std::uint8_t encode(std::string_view key, bool missing = false) const noexcept;

    void transform_indexed(std::span<const std::string> vocab,
                           std::span<const std::int32_t> row_indices, std::uint8_t* out) const;

    void transform_strings(std::span<const std::string_view> values,
                           std::span<const std::uint8_t> is_missing, std::uint8_t* out) const;
};

} // namespace miter
