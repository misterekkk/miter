#include "model/lignum_bridge.hpp"

#include <lignum/build.hpp>

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

} // namespace miter
