#include "model/lignum_bridge.hpp"

#include <lignum/build.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "core/ctr.hpp"
#include "model/lignum_export.hpp"

namespace miter {

lignum::Model to_lignum(const Model& model, std::size_t num_iteration, bool binned) {
    const RawForest raw = export_raw_forest(model, num_iteration);
    const std::size_t p = model.n_features;
    const bool has_ctr = !model.ctr_tables.empty();
    std::size_t n_cat = 0;
    std::vector<std::size_t> feature_to_cat_idx;
    if (has_ctr) {
        if (model.feature_kinds.size() != p) {
            throw std::invalid_argument("to_lignum: feature_kinds size mismatch");
        }
        feature_to_cat_idx.assign(p, 0U);
        for (std::size_t j = 0; j < p; ++j) {
            if (model.feature_kinds[j] != 0U) {
                feature_to_cat_idx[j] = n_cat++;
            }
        }
    }

    lignum::ir::Forest f;
    if (raw.n_classes > 1U) {
        f.base_scores = raw.init_scores;
        f.tree_outputs = raw.tree_outputs;
        f.transform = lignum::transform::Softmax;
        f.objective = "multiclass";
    } else {
        f.base_score = raw.init_score;
        f.transform = (raw.objective == Objective::Binary) ? lignum::transform::Sigmoid
                                                           : lignum::transform::Identity;
        f.objective = (raw.objective == Objective::Binary) ? "binary" : "regression";
    }
    f.n_features = static_cast<std::uint32_t>(p + n_cat);
    f.feature_names = model.feature_names;
    if (has_ctr && !f.feature_names.empty()) {
        f.feature_names.reserve(p + n_cat);
        for (std::size_t sub = 0; sub < n_cat; ++sub) {
            f.feature_names.push_back("__ctr_hash_" + std::to_string(sub));
        }
    }
    f.source = lignum::model_source::Miter;
    bool any_oblique = false;
    f.trees.resize(raw.trees.size());
    for (std::size_t t = 0; t < raw.trees.size(); ++t) {
        auto& nodes = f.trees[t].nodes;
        nodes.resize(raw.trees[t].nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const RawNode& r = raw.trees[t].nodes[i];
            lignum::ir::Node& n = nodes[i];
            n.default_right = r.default_right;
            n.left = r.left;
            n.right = r.right;
            n.feature = r.feature;
            switch (r.kind) {
            case RawNode::Kind::Leaf:
                n.kind = lignum::ir::Node::Kind::Leaf;
                n.value = r.value;
                break;
            case RawNode::Kind::Numeric:
                n.kind = lignum::ir::Node::Kind::Numeric;
                n.value = r.threshold;
                break;
            case RawNode::Kind::Categorical:
                n.kind = lignum::ir::Node::Kind::Categorical;
                n.categories.assign(r.cat_left_codes.begin(), r.cat_left_codes.end());
                break;
            case RawNode::Kind::Oblique:
                any_oblique = true;
                n.kind = lignum::ir::Node::Kind::Oblique;
                n.oblique.k = r.k;
                for (int q = 0; q < 3; ++q) {
                    n.oblique.features[q] = r.features[q];
                    n.oblique.weights[q] = r.weights[q];
                }
                n.oblique.threshold = r.oblique_threshold;
                break;
            case RawNode::Kind::Ctr:
                n.kind = lignum::ir::Node::Kind::Ctr;
                n.value = r.threshold;
                break;
            }
        }
    }
    if (any_oblique) {
        const std::size_t total_q = has_ctr ? (p + n_cat) : raw.quantizers.size();
        f.quantizers.resize(total_q);
        f.quantizer_constant.resize(total_q, 1U);
        f.quantizer_knots.resize(total_q);
        for (std::size_t j = 0; j < raw.quantizers.size(); ++j) {
            f.quantizers[j] =
                lignum::FeatureQuantizer{raw.quantizers[j].lo, raw.quantizers[j].scale};
            f.quantizer_constant[j] = raw.quantizers[j].is_constant ? 1U : 0U;
            f.quantizer_knots[j] = raw.quantizers[j].knots;
        }
    }
    if (has_ctr) {
        f.ctr_tables.resize(model.ctr_tables.size());
        for (std::size_t ti = 0; ti < model.ctr_tables.size(); ++ti) {
            const CtrTable& src = model.ctr_tables[ti];
            if (src.projection >= model.ctr_projections.size()) {
                throw std::invalid_argument("to_lignum: CTR projection index out of range");
            }
            const CtrProjection& proj = model.ctr_projections[src.projection];
            if (proj.cat_features.empty() || proj.cat_features.size() > 3U) {
                throw std::invalid_argument("to_lignum: CTR projection must have 1..3 columns");
            }
            if (src.mask > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("to_lignum: CTR mask overflows uint32");
            }
            lignum::ir::CtrTable& dst = f.ctr_tables[ti];
            dst.columns.resize(proj.cat_features.size());
            for (std::size_t q = 0; q < proj.cat_features.size(); ++q) {
                const std::uint32_t col = proj.cat_features[q];
                if (col >= p || model.feature_kinds[col] == 0U) {
                    throw std::invalid_argument(
                        "to_lignum: CTR references non-categorical feature");
                }
                dst.columns[q] = static_cast<std::uint32_t>(p + feature_to_cat_idx[col]);
            }
            dst.default_value = src.default_value;
            dst.mask = static_cast<std::uint32_t>(src.mask);
            dst.keys = src.keys;
            dst.slots = src.slots;
            dst.values = src.values;
        }
    }
    return lignum::build_model(f, binned);
}

void lignum_predict_raw(const lignum::Model& lm, const Model& model, std::size_t n_rows,
                        const std::vector<NumericColumnView>& num_cols,
                        const std::vector<CategoricalColumnView>& cat_cols, double* out,
                        int n_jobs) {
    const std::size_t p = model.n_features;
    if (n_rows == 0U) {
        return;
    }
    if (out == nullptr) {
        throw std::invalid_argument("lignum_predict_raw received null output pointer");
    }
    if (model.feature_kinds.size() != p || model.cat_features.size() != p) {
        throw std::invalid_argument("Model feature metadata size mismatch");
    }
    const std::size_t k_out = (model.objective == Objective::Multiclass && model.n_classes > 1U)
                                  ? static_cast<std::size_t>(model.n_classes)
                                  : 1U;
    const bool has_ctr = !model.ctr_tables.empty();
    struct Col {
        bool is_cat = false;
        std::size_t sub = 0;
        std::vector<std::uint8_t> vocab_to_code;  // categorical
        std::vector<std::uint32_t> vocab_to_hash; // 32-bit string hashes for CTR
        std::uint8_t missing = 1U;
    };
    std::vector<Col> cols(p);
    std::size_t n_num = 0;
    std::size_t n_cat = 0;
    for (std::size_t f = 0; f < p; ++f) {
        if (model.feature_kinds[f] == 0U) {
            cols[f].sub = n_num++;
            continue;
        }
        cols[f].is_cat = true;
        cols[f].sub = n_cat++;
    }
    if (num_cols.size() != n_num || cat_cols.size() != n_cat) {
        throw std::invalid_argument("Feature count mismatch in predict");
    }
    const std::size_t stride = has_ctr ? (p + n_cat) : p;
    if (lm.n_features != stride) {
        throw std::invalid_argument("lignum model feature count mismatch");
    }
    for (std::size_t f = 0; f < p; ++f) {
        if (!cols[f].is_cat) {
            continue;
        }
        const auto& cats = model.cat_features[f].categories;
        const auto other = static_cast<std::uint8_t>(cats.size());
        cols[f].missing = static_cast<std::uint8_t>(cats.size() + 1U);
        std::unordered_map<std::string_view, std::uint8_t> map;
        map.reserve(cats.size());
        for (std::size_t k = 0; k < cats.size(); ++k) {
            map.emplace(std::string_view(cats[k]), static_cast<std::uint8_t>(k));
        }
        const auto& col = cat_cols[cols[f].sub];
        if (col.indices == nullptr) {
            throw std::invalid_argument("CategoricalColumnView received null indices pointer");
        }
        cols[f].vocab_to_code.assign(col.vocab.size(), other);
        if (has_ctr) {
            cols[f].vocab_to_hash.resize(col.vocab.size());
        }
        for (std::size_t v = 0; v < col.vocab.size(); ++v) {
            if (const auto it = map.find(std::string_view(col.vocab[v])); it != map.end()) {
                cols[f].vocab_to_code[v] = it->second;
            }
            if (has_ctr) {
                cols[f].vocab_to_hash[v] = cat_hash_string(col.vocab[v]);
            }
        }
    }

    // Fast path: a C-contiguous float64 matrix of a numeric model goes to lignum as it is.
    if (n_cat == 0U && !has_ctr) {
        bool contiguous = true;
        for (std::size_t j = 0; j < p && contiguous; ++j) {
            contiguous = num_cols[j].f64 != nullptr && num_cols[j].f64 == num_cols[0].f64 + j &&
                         (n_rows == 1U || num_cols[j].row_stride == p);
        }
        if (contiguous) {
            lm.predict_raw(num_cols[0].f64, n_rows, p, out, n_jobs);
            return;
        }
    }

    // Otherwise row-major blocks: numeric values as doubles, categorical codes (and appended
    // 32-bit string hashes when the model uses native CTR tables).
    constexpr std::size_t kBlock = 65536;
    std::vector<double> buf(std::min(kBlock, n_rows) * stride);
    for (std::size_t base = 0; base < n_rows; base += kBlock) {
        const std::size_t blk = std::min(kBlock, n_rows - base);
        for (std::size_t f = 0; f < p; ++f) {
            const Col& c = cols[f];
            if (!c.is_cat) {
                const NumericColumnView& v = num_cols[c.sub];
                for (std::size_t r = 0; r < blk; ++r) {
                    const std::size_t i = (base + r) * v.row_stride;
                    buf[r * stride + f] =
                        (v.f64 != nullptr) ? v.f64[i] : static_cast<double>(v.f32[i]);
                }
            } else {
                const CategoricalColumnView& v = cat_cols[c.sub];
                const std::size_t vocab_sz = c.vocab_to_code.size();
                if (has_ctr) {
                    for (std::size_t r = 0; r < blk; ++r) {
                        const std::int32_t idx = v.indices[base + r];
                        const bool valid = idx >= 0 && static_cast<std::size_t>(idx) < vocab_sz;
                        const std::uint8_t code =
                            valid ? c.vocab_to_code[static_cast<std::size_t>(idx)] : c.missing;
                        const std::uint32_t h = valid
                                                    ? c.vocab_to_hash[static_cast<std::size_t>(idx)]
                                                    : kMissingCatHash;
                        buf[r * stride + f] = static_cast<double>(code);
                        buf[r * stride + p + c.sub] = static_cast<double>(h);
                    }
                } else {
                    for (std::size_t r = 0; r < blk; ++r) {
                        const std::int32_t idx = v.indices[base + r];
                        const std::uint8_t code =
                            (idx < 0 || static_cast<std::size_t>(idx) >= vocab_sz)
                                ? c.missing
                                : c.vocab_to_code[static_cast<std::size_t>(idx)];
                        buf[r * stride + f] = static_cast<double>(code);
                    }
                }
            }
        }
        lm.predict_raw(buf.data(), blk, stride, out + base * k_out, n_jobs);
    }
}

} // namespace miter
