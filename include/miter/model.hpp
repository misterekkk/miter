#pragma once

#include <miter/params.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace miter {

enum class Objective : std::uint8_t {
    Regression = 0,
    Binary = 1,
    Multiclass = 2,
};

enum class NodeKind : std::uint8_t {
    Leaf = 0,
    Axis = 1,
    Oblique = 2,
    Categorical = 3,
    Ctr = 4,
};

struct Node {
    NodeKind kind = NodeKind::Leaf;
    bool nan_go_left = true;
    std::uint8_t oblique_k = 0;
    std::uint32_t left_child = 0;
    std::uint32_t right_child = 0;
    std::uint32_t feature = 0;
    std::uint16_t threshold_u16 = 0; // Axis: right iff u >= threshold_u16 (left iff u < threshold)
    std::int32_t threshold_i32 = 0;  // Oblique (k <= 3): left iff P < threshold_i32
    float ctr_threshold = 0.0f;      // Ctr: left iff ctr_val < ctr_threshold
    std::uint32_t oblique_features[3] = {0, 0, 0};
    std::int16_t oblique_weights[3] = {0, 0, 0};
    std::uint32_t overflow_offset = 0;   // Index into Tree::oblique_overflow when oblique_k > 3
    std::uint32_t cat_bitset_offset = 0; // Index into Tree::cat_bitsets when kind == Categorical
    double leaf_value = 0.0;             // Scaled by learning_rate
    double split_gain = 0.0;
};

enum class CtrKind : std::uint8_t {
    TargetMean = 0,
    Counter = 1,
};

struct CtrProjection {
    std::vector<std::uint32_t> cat_features;
};

struct CtrTable {
    std::uint32_t projection = 0;
    CtrKind kind = CtrKind::TargetMean;
    float default_value = 0.0f;
    std::vector<std::uint64_t> keys;
    std::vector<std::uint32_t> slots; // 0 = empty, else entry + 1
    std::vector<float> values;
    std::uint64_t mask = 0;
};

struct CatBitset {
    std::array<std::uint64_t, 4> words = {0ULL, 0ULL, 0ULL, 0ULL};
};

inline void cat_bitset_set(CatBitset& bs, std::uint8_t code) noexcept {
    bs.words[static_cast<std::size_t>(code) >> 6U] |=
        (std::uint64_t{1} << (static_cast<unsigned int>(code) & 63U));
}

inline bool cat_bitset_test(const CatBitset& bs, std::uint8_t code) noexcept {
    return ((bs.words[static_cast<std::size_t>(code) >> 6U] >>
             (static_cast<unsigned int>(code) & 63U)) &
            1ULL) != 0ULL;
}

struct ObliqueOverflow {
    std::vector<std::uint32_t> features;
    std::vector<std::int16_t> weights;
    std::int64_t threshold_i64 = 0;
};

struct Tree {
    std::vector<Node> nodes;
    std::vector<ObliqueOverflow> oblique_overflow;
    std::vector<CatBitset> cat_bitsets;
    std::uint32_t n_leaves = 1;
};

struct QuantizerInfo {
    double lo = 0.0;
    double scale = 0.0;
    bool is_constant = true;
    std::vector<double> knots;
};

struct CatFeatureInfo {
    std::vector<std::string> categories;
};

struct Model {
    std::string miter_version;
    Objective objective = Objective::Regression;
    double init_score = 0.0;
    std::uint32_t n_classes = 1;
    std::vector<double> init_scores;
    double learning_rate = 0.05;
    std::size_t n_features = 0;
    std::size_t best_iteration = 0;
    Params params;
    std::vector<std::string> feature_names;
    std::vector<std::uint8_t> feature_kinds; // 0 = Numeric, 1 = Categorical
    std::vector<QuantizerInfo> quantizers;
    std::vector<CatFeatureInfo> cat_features;
    std::vector<CtrProjection> ctr_projections;
    std::vector<CtrTable> ctr_tables;
    std::vector<Tree> trees;

    std::vector<std::uint64_t> feature_importances_split() const;
    std::vector<double> feature_importances_gain() const;
};

} // namespace miter
