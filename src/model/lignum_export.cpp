#include "model/lignum_export.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "data/quantizer.hpp"

namespace miter {

namespace {

// Order-preserving bijection between non-NaN doubles and signed 64-bit keys
// (-0.0 and +0.0 map to neighbouring keys; both encode identically).
std::int64_t double_key(double d) noexcept {
    const auto i = std::bit_cast<std::int64_t>(d);
    return (i < 0) ? (std::numeric_limits<std::int64_t>::min() - i) : i;
}

double key_double(std::int64_t k) noexcept {
    const std::int64_t i = (k < 0) ? (std::numeric_limits<std::int64_t>::min() - k) : k;
    return std::bit_cast<double>(i);
}

} // namespace

double axis_raw_threshold(const QuantizerInfo& q, std::uint16_t t) {
    if (t == 0U) {
        return -std::numeric_limits<double>::infinity();
    }
    if (t > kQ16MaxCode) {
        throw std::invalid_argument("axis_raw_threshold: threshold above the largest code");
    }
    const Q16Quantizer quant{q.lo, q.scale, q.is_constant, q.knots};
    // Invariant: encode(key_double(lo)) < t <= encode(key_double(hi)).
    std::uint64_t lo =
        static_cast<std::uint64_t>(double_key(-std::numeric_limits<double>::infinity()));
    std::uint64_t hi =
        static_cast<std::uint64_t>(double_key(std::numeric_limits<double>::infinity()));
    if (quant.encode_u16(key_double(static_cast<std::int64_t>(hi))) < t) {
        throw std::invalid_argument(
            "axis_raw_threshold: threshold not reachable (constant feature)");
    }
    if (quant.encode_u16(key_double(static_cast<std::int64_t>(lo))) >= t) {
        return -std::numeric_limits<double>::infinity();
    }
    // Unsigned arithmetic: the key range spans the whole int64 range.
    while (hi - lo > 1U) {
        const std::uint64_t mid = lo + (hi - lo) / 2U;
        if (quant.encode_u16(key_double(static_cast<std::int64_t>(mid))) >= t) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return key_double(static_cast<std::int64_t>(hi));
}

RawForest export_raw_forest(const Model& model, std::size_t num_iteration) {
    const std::uint32_t K =
        (model.objective == Objective::Multiclass && model.n_classes > 1U) ? model.n_classes : 1U;
    if (K > 1U && model.init_scores.size() != K) {
        throw std::invalid_argument("export_raw_forest: multiclass init_scores size mismatch");
    }
    RawForest out;
    out.init_score = model.init_score;
    out.n_classes = K;
    out.init_scores = model.init_scores;
    out.objective = model.objective;
    out.quantizers = model.quantizers;
    const std::size_t max_rounds_avail = model.trees.size() / K;
    const std::size_t n_rounds = (num_iteration == 0U)
                                     ? std::min(model.best_iteration, max_rounds_avail)
                                     : std::min(num_iteration, max_rounds_avail);
    const std::size_t n_trees = n_rounds * K;
    out.trees.resize(n_trees);
    if (K > 1U) {
        out.tree_outputs.resize(n_trees);
        for (std::size_t t = 0; t < n_trees; ++t) {
            out.tree_outputs[t] = static_cast<std::uint16_t>(t % K);
        }
    }
    for (std::size_t t = 0; t < n_trees; ++t) {
        const Tree& tree = model.trees[t];
        RawTree& rt = out.trees[t];
        rt.nodes.resize(tree.nodes.size());
        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const Node& n = tree.nodes[i];
            RawNode& r = rt.nodes[i];
            r.left = n.left_child;
            r.right = n.right_child;
            r.default_right = !n.nan_go_left;
            switch (n.kind) {
            case NodeKind::Leaf:
                r.kind = RawNode::Kind::Leaf;
                r.value = n.leaf_value;
                break;
            case NodeKind::Axis:
                r.kind = RawNode::Kind::Numeric;
                r.feature = n.feature;
                if (n.threshold_u16 == kQ16NanCode) {
                    // NaN-vs-rest split: every non-NaN value goes left. No double x satisfies
                    // encode(x) >= 65535, so swap the children: threshold -inf sends every
                    // non-NaN value right (to the old left child), NaN takes the flipped default.
                    r.threshold = -std::numeric_limits<double>::infinity();
                    r.left = n.right_child;
                    r.right = n.left_child;
                    r.default_right = n.nan_go_left;
                } else {
                    r.threshold =
                        axis_raw_threshold(model.quantizers.at(n.feature), n.threshold_u16);
                }
                break;
            case NodeKind::Categorical: {
                r.kind = RawNode::Kind::Categorical;
                r.feature = n.feature;
                const CatBitset& bs = tree.cat_bitsets.at(n.cat_bitset_offset);
                for (unsigned c = 0; c < 256U; ++c) {
                    if (cat_bitset_test(bs, static_cast<std::uint8_t>(c))) {
                        r.cat_left_codes.push_back(static_cast<std::uint16_t>(c));
                    }
                }
                // Missing values arrive as the MISSING code, never as NaN.
                r.default_right = false;
                break;
            }
            case NodeKind::Oblique:
                if (n.oblique_k > 3U) {
                    throw std::invalid_argument(
                        "export_raw_forest: oblique split with k = " + std::to_string(n.oblique_k) +
                        " (only k <= 3 is exported)");
                }
                r.kind = RawNode::Kind::Oblique;
                r.k = n.oblique_k;
                for (int q = 0; q < 3; ++q) {
                    r.features[q] = n.oblique_features[q];
                    r.weights[q] = (q < n.oblique_k) ? n.oblique_weights[q] : std::int16_t{0};
                }
                if (r.k == 2U) {
                    r.features[2] =
                        r.features[0]; // weight 0: contributes nothing, NaN check is a duplicate
                }
                r.oblique_threshold = n.threshold_i32;
                break;
            case NodeKind::Ctr:
                r.kind = RawNode::Kind::Ctr;
                r.feature = n.feature;
                r.threshold = static_cast<double>(n.ctr_threshold);
                r.default_right = false;
                break;
            }
        }
    }
    return out;
}

} // namespace miter
