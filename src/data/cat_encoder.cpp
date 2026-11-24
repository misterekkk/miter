#include "data/cat_encoder.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace miter {

namespace {

void validate_cat_params(int min_data_per_group, int max_categories) {
    if (min_data_per_group < 1) {
        throw std::invalid_argument("CatEncoder requires min_data_per_group >= 1");
    }
    if (max_categories < 1 || max_categories > kDefaultMaxCategories) {
        throw std::invalid_argument("CatEncoder requires max_categories in [1, 253]");
    }
}

CatEncoder build_encoder_from_counts(std::vector<std::pair<std::string, std::size_t>>& entries,
                                     int min_data_per_group, int max_categories) {
    const auto min_count = static_cast<std::size_t>(min_data_per_group);
    std::erase_if(entries, [min_count](const auto& item) { return item.second < min_count; });

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    });

    if (entries.size() > static_cast<std::size_t>(max_categories)) {
        entries.resize(static_cast<std::size_t>(max_categories));
    }

    CatEncoder enc;
    enc.categories.reserve(entries.size());
    enc.lookup.reserve(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto code = static_cast<std::uint8_t>(i);
        enc.categories.push_back(std::move(entries[i].first));
        enc.lookup.emplace(enc.categories.back(), code);
    }
    return enc;
}

} // namespace

CatEncoder CatEncoder::fit_indexed(std::span<const std::string> vocab,
                                   std::span<const std::int32_t> row_indices,
                                   int min_data_per_group, int max_categories) {
    validate_cat_params(min_data_per_group, max_categories);

    std::vector<std::size_t> counts(vocab.size(), 0);
    for (const std::int32_t idx : row_indices) {
        if (idx < 0) {
            continue;
        }
        if (static_cast<std::size_t>(idx) >= vocab.size()) {
            throw std::out_of_range("CatEncoder::fit_indexed category index out of range");
        }
        ++counts[static_cast<std::size_t>(idx)];
    }

    std::vector<std::pair<std::string, std::size_t>> entries;
    entries.reserve(vocab.size());
    for (std::size_t i = 0; i < vocab.size(); ++i) {
        if (counts[i] > 0) {
            entries.emplace_back(vocab[i], counts[i]);
        }
    }
    return build_encoder_from_counts(entries, min_data_per_group, max_categories);
}

CatEncoder CatEncoder::fit_strings(std::span<const std::string_view> values,
                                   std::span<const std::uint8_t> is_missing, int min_data_per_group,
                                   int max_categories) {
    validate_cat_params(min_data_per_group, max_categories);
    if (values.size() != is_missing.size()) {
        throw std::invalid_argument("CatEncoder::fit_strings values/is_missing length mismatch");
    }

    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> counts;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (is_missing[i] != 0) {
            continue;
        }
        if (const auto it = counts.find(values[i]); it != counts.end()) {
            ++it->second;
        } else {
            counts.emplace(std::string(values[i]), 1U);
        }
    }

    std::vector<std::pair<std::string, std::size_t>> entries;
    entries.reserve(counts.size());
    for (auto& kv : counts) {
        entries.emplace_back(kv.first, kv.second);
    }
    return build_encoder_from_counts(entries, min_data_per_group, max_categories);
}

std::uint8_t CatEncoder::encode(std::string_view key, bool missing) const noexcept {
    if (missing) {
        return missing_code();
    }
    if (const auto it = lookup.find(key); it != lookup.end()) {
        return it->second;
    }
    return other_code();
}

void CatEncoder::transform_indexed(std::span<const std::string> vocab,
                                   std::span<const std::int32_t> row_indices,
                                   std::uint8_t* out) const {
    if (!row_indices.empty() && out == nullptr) {
        throw std::invalid_argument("CatEncoder::transform_indexed received null output pointer");
    }

    std::vector<std::uint8_t> remap(vocab.size(), other_code());
    for (std::size_t i = 0; i < vocab.size(); ++i) {
        if (const auto it = lookup.find(vocab[i]); it != lookup.end()) {
            remap[i] = it->second;
        }
    }

    const std::uint8_t miss = missing_code();
    for (std::size_t i = 0; i < row_indices.size(); ++i) {
        const std::int32_t idx = row_indices[i];
        if (idx < 0) {
            out[i] = miss;
        } else {
            if (static_cast<std::size_t>(idx) >= vocab.size()) {
                throw std::out_of_range(
                    "CatEncoder::transform_indexed category index out of range");
            }
            out[i] = remap[static_cast<std::size_t>(idx)];
        }
    }
}

void CatEncoder::transform_strings(std::span<const std::string_view> values,
                                   std::span<const std::uint8_t> is_missing,
                                   std::uint8_t* out) const {
    if (values.size() != is_missing.size()) {
        throw std::invalid_argument(
            "CatEncoder::transform_strings values/is_missing length mismatch");
    }
    if (!values.empty() && out == nullptr) {
        throw std::invalid_argument("CatEncoder::transform_strings received null output pointer");
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = encode(values[i], is_missing[i] != 0);
    }
}

} // namespace miter
