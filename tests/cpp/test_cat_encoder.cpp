#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "data/cat_encoder.hpp"

namespace {

TEST(CatEncoderTest, FrequencyRankingAndTieBreaking) {
    const std::vector<std::string> vocab{"beta", "alpha", "gamma", "rare"};
    // alpha: 3, gamma: 3, beta: 4, rare: 1
    std::vector<std::int32_t> indices{
        0, 0, 0, 0, // beta x4
        2, 2, 2,    // gamma x3
        1, 1, 1,    // alpha x3
        3,          // rare x1
        -1          // missing
    };

    const auto enc = miter::CatEncoder::fit_indexed(vocab, indices, /*min_data_per_group=*/2,
                                                    /*max_categories=*/253);
    // Ranking: beta (4) -> 0, alpha (3, "alpha" < "gamma") -> 1, gamma (3) -> 2
    EXPECT_EQ(enc.categories, (std::vector<std::string>{"beta", "alpha", "gamma"}));
    EXPECT_EQ(enc.other_code(), 3);
    EXPECT_EQ(enc.missing_code(), 4);
    EXPECT_EQ(enc.num_codes(), 5);

    std::vector<std::uint8_t> out(indices.size());
    enc.transform_indexed(vocab, indices, out.data());
    EXPECT_EQ(out[0], 0);  // beta
    EXPECT_EQ(out[4], 2);  // gamma
    EXPECT_EQ(out[7], 1);  // alpha
    EXPECT_EQ(out[10], 3); // rare -> OTHER
    EXPECT_EQ(out[11], 4); // missing -> MISSING

    // Unseen vocabulary at transform time -> OTHER
    const std::vector<std::string> test_vocab{"unseen", "alpha"};
    const std::vector<std::int32_t> test_indices{0, 1, -1};
    std::vector<std::uint8_t> test_out(3);
    enc.transform_indexed(test_vocab, test_indices, test_out.data());
    EXPECT_EQ(test_out, (std::vector<std::uint8_t>{3, 1, 4}));
}

TEST(CatEncoderTest, MinDataPerGroupAndMaxCategoriesBoundary) {
    const std::vector<std::string> vocab{"c0", "c1", "c2", "c3"};
    std::vector<std::int32_t> indices;
    // c0: 5, c1: 4, c2: 2, c3: 1 (with min_data_per_group=2, max_categories=2 => keep c0, c1)
    indices.insert(indices.end(), 5, 0);
    indices.insert(indices.end(), 4, 1);
    indices.insert(indices.end(), 2, 2);
    indices.insert(indices.end(), 1, 3);

    const auto enc = miter::CatEncoder::fit_indexed(vocab, indices, /*min_data_per_group=*/2,
                                                    /*max_categories=*/2);
    EXPECT_EQ(enc.categories, (std::vector<std::string>{"c0", "c1"}));
    EXPECT_EQ(enc.other_code(), 2);
    EXPECT_EQ(enc.missing_code(), 3);
    EXPECT_EQ(enc.encode("c0"), 0);
    EXPECT_EQ(enc.encode("c1"), 1);
    EXPECT_EQ(enc.encode("c2"), 2); // truncated by max_categories -> OTHER
    EXPECT_EQ(enc.encode("c3"), 2); // below min_data_per_group -> OTHER
    EXPECT_EQ(enc.encode("", /*missing=*/true), 3);
}

TEST(CatEncoderTest, StringAndIndexedParityAndAllMissing) {
    const std::vector<std::string> vocab{"x", "y"};
    const std::vector<std::int32_t> indices{0, 1, 0, -1, 1};
    const std::vector<std::string_view> strs{"x", "y", "x", "", "y"};
    const std::vector<std::uint8_t> is_missing{0, 0, 0, 1, 0};

    const auto enc_idx = miter::CatEncoder::fit_indexed(vocab, indices, 1, 253);
    const auto enc_str = miter::CatEncoder::fit_strings(strs, is_missing, 1, 253);
    EXPECT_EQ(enc_idx.categories, enc_str.categories);

    std::vector<std::uint8_t> out_idx(indices.size());
    std::vector<std::uint8_t> out_str(strs.size());
    enc_idx.transform_indexed(vocab, indices, out_idx.data());
    enc_str.transform_strings(strs, is_missing, out_str.data());
    EXPECT_EQ(out_idx, out_str);

    // All missing -> K = 0, OTHER = 0, MISSING = 1
    const std::vector<std::int32_t> all_miss{-1, -1, -1};
    const auto enc_miss = miter::CatEncoder::fit_indexed(vocab, all_miss, 1, 253);
    EXPECT_TRUE(enc_miss.categories.empty());
    EXPECT_EQ(enc_miss.other_code(), 0);
    EXPECT_EQ(enc_miss.missing_code(), 1);
    EXPECT_EQ(enc_miss.num_codes(), 2);
}

TEST(CatEncoderTest, ValidationErrors) {
    const std::vector<std::string> vocab{"a"};
    const std::vector<std::int32_t> indices{0};
    EXPECT_THROW(miter::CatEncoder::fit_indexed(vocab, indices, 0, 253), std::invalid_argument);
    EXPECT_THROW(miter::CatEncoder::fit_indexed(vocab, indices, 1, 0), std::invalid_argument);
    EXPECT_THROW(miter::CatEncoder::fit_indexed(vocab, indices, 1, 254), std::invalid_argument);

    const std::vector<std::int32_t> bad_idx{1};
    EXPECT_THROW(miter::CatEncoder::fit_indexed(vocab, bad_idx, 1, 253), std::out_of_range);
}

} // namespace
