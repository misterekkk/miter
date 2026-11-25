#include "model/lignum_bridge.hpp"

#include <lignum/build.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "model/lignum_export.hpp"

namespace miter {

lignum::Model to_lignum(const Model& model, std::size_t num_iteration, bool binned) {
    const RawForest raw = export_raw_forest(model, num_iteration);
    lignum::ir::Forest f;
    f.base_score = raw.init_score;
    f.transform = (raw.objective == Objective::Binary) ? lignum::transform::Sigmoid
                                                       : lignum::transform::Identity;
    f.objective = (raw.objective == Objective::Binary) ? "binary" : "regression";
    f.n_features = static_cast<std::uint32_t>(model.n_features);
    f.feature_names = model.feature_names;
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
            }
        }
    }
    if (any_oblique) {
        f.quantizers.resize(raw.quantizers.size());
        f.quantizer_constant.resize(raw.quantizers.size());
        for (std::size_t j = 0; j < raw.quantizers.size(); ++j) {
            f.quantizers[j] =
                lignum::FeatureQuantizer{raw.quantizers[j].lo, raw.quantizers[j].scale};
            f.quantizer_constant[j] = raw.quantizers[j].is_constant ? 1U : 0U;
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
    struct Col {
        bool is_cat = false;
        std::size_t sub = 0;
        std::vector<std::uint8_t> vocab_to_code; // categorical
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
        for (std::size_t v = 0; v < col.vocab.size(); ++v) {
            if (const auto it = map.find(std::string_view(col.vocab[v])); it != map.end()) {
                cols[f].vocab_to_code[v] = it->second;
            }
        }
    }

    // Fast path: a C-contiguous float64 matrix of a numeric model goes to lignum as it is.
    if (n_cat == 0U) {
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

    // Otherwise row-major blocks: numeric values as doubles, categorical columns as codes.
    constexpr std::size_t kBlock = 65536;
    std::vector<double> buf(std::min(kBlock, n_rows) * p);
    for (std::size_t base = 0; base < n_rows; base += kBlock) {
        const std::size_t blk = std::min(kBlock, n_rows - base);
        for (std::size_t f = 0; f < p; ++f) {
            const Col& c = cols[f];
            if (!c.is_cat) {
                const NumericColumnView& v = num_cols[c.sub];
                for (std::size_t r = 0; r < blk; ++r) {
                    const std::size_t i = (base + r) * v.row_stride;
                    buf[r * p + f] = (v.f64 != nullptr) ? v.f64[i] : static_cast<double>(v.f32[i]);
                }
            } else {
                const CategoricalColumnView& v = cat_cols[c.sub];
                const std::size_t vocab_sz = c.vocab_to_code.size();
                for (std::size_t r = 0; r < blk; ++r) {
                    const std::int32_t idx = v.indices[base + r];
                    const std::uint8_t code = (idx < 0 || static_cast<std::size_t>(idx) >= vocab_sz)
                                                  ? c.missing
                                                  : c.vocab_to_code[static_cast<std::size_t>(idx)];
                    buf[r * p + f] = static_cast<double>(code);
                }
            }
        }
        lm.predict_raw(buf.data(), blk, p, out + base, n_jobs);
    }
}

} // namespace miter
