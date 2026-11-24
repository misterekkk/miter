#include "model/evaluator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/objective.hpp"
#include "data/quantizer.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

constexpr std::size_t kEvalBlockRows = 256;

template <typename FloatT>
std::vector<NumericColumnView> make_eval_views(const FloatT* data, std::size_t n_rows,
                                               std::size_t n_cols, bool col_major) {
    if (n_rows == 0 || n_cols == 0) {
        return {};
    }
    if (data == nullptr) {
        throw std::invalid_argument("Evaluator received null input pointer");
    }
    std::vector<NumericColumnView> views;
    views.reserve(n_cols);
    const std::size_t stride = col_major ? 1U : n_cols;
    for (std::size_t j = 0; j < n_cols; ++j) {
        const FloatT* col_ptr = col_major ? (data + j * n_rows) : (data + j);
        NumericColumnView v;
        if constexpr (std::is_same_v<FloatT, double>) {
            v.f64 = col_ptr;
        } else {
            v.f32 = col_ptr;
        }
        v.row_stride = stride;
        views.push_back(v);
    }
    return views;
}

double eval_tree_u16(const Tree& tree, const std::uint16_t* row_codes) noexcept {
    std::uint32_t nid = 0U;
    for (;;) {
        const Node& node = tree.nodes[nid];
        if (node.kind == NodeKind::Leaf) {
            return node.leaf_value;
        }
        bool go_left = false;
        if (node.kind == NodeKind::Axis) {
            const std::uint16_t u = row_codes[node.feature];
            go_left = (u == kQ16NanCode) ? node.nan_go_left : (u < node.threshold_u16);
        } else if (node.kind == NodeKind::Categorical) {
            const auto code = static_cast<std::uint8_t>(row_codes[node.feature]);
            go_left = cat_bitset_test(tree.cat_bitsets[node.cat_bitset_offset], code);
        } else {
            const std::uint8_t k = node.oblique_k;
            if (k <= 3U) {
                const std::uint16_t u0 = row_codes[node.oblique_features[0]];
                const std::uint16_t u1 = row_codes[node.oblique_features[1]];
                const std::uint16_t u2 =
                    (k == 3U) ? row_codes[node.oblique_features[2]] : std::uint16_t{0};
                if (u0 == kQ16NanCode || u1 == kQ16NanCode || (k == 3U && u2 == kQ16NanCode)) {
                    go_left = node.nan_go_left;
                } else {
                    const std::int32_t c0 = Q16Quantizer::to_signed_code(u0);
                    const std::int32_t c1 = Q16Quantizer::to_signed_code(u1);
                    const std::int32_t c2 =
                        (k == 3U) ? static_cast<std::int32_t>(Q16Quantizer::to_signed_code(u2)) : 0;
                    const std::int32_t p32 =
                        static_cast<std::int32_t>(node.oblique_weights[0]) * c0 +
                        static_cast<std::int32_t>(node.oblique_weights[1]) * c1 +
                        static_cast<std::int32_t>(node.oblique_weights[2]) * c2;
                    go_left = (p32 < node.threshold_i32);
                }
            } else {
                const auto& ov = tree.oblique_overflow[node.overflow_offset];
                std::int64_t acc = 0;
                bool missing = false;
                for (std::uint8_t q = 0; q < k; ++q) {
                    const std::uint16_t u = row_codes[ov.features[q]];
                    if (u == kQ16NanCode) {
                        missing = true;
                        break;
                    }
                    acc += static_cast<std::int64_t>(ov.weights[q]) *
                           static_cast<std::int64_t>(Q16Quantizer::to_signed_code(u));
                }
                go_left = missing ? node.nan_go_left : (acc < ov.threshold_i64);
            }
        }
        nid = go_left ? node.left_child : node.right_child;
    }
}

} // namespace

std::vector<std::uint64_t> Model::feature_importances_split() const {
    std::vector<std::uint64_t> counts(n_features, 0U);
    const std::size_t n_trees = std::min(best_iteration, trees.size());
    for (std::size_t t = 0; t < n_trees; ++t) {
        const auto& tree = trees[t];
        for (const Node& node : tree.nodes) {
            if (node.kind == NodeKind::Axis || node.kind == NodeKind::Categorical) {
                if (node.feature < n_features) {
                    counts[node.feature] += 1U;
                }
            } else if (node.kind == NodeKind::Oblique) {
                const std::uint8_t k = node.oblique_k;
                if (k <= 3U) {
                    for (std::uint8_t q = 0; q < k; ++q) {
                        if (node.oblique_features[q] < n_features) {
                            counts[node.oblique_features[q]] += 1U;
                        }
                    }
                } else {
                    const auto& ov = tree.oblique_overflow[node.overflow_offset];
                    for (const std::uint32_t f : ov.features) {
                        if (f < n_features) {
                            counts[f] += 1U;
                        }
                    }
                }
            }
        }
    }
    return counts;
}

std::vector<double> Model::feature_importances_gain() const {
    std::vector<double> gains(n_features, 0.0);
    const std::size_t n_trees = std::min(best_iteration, trees.size());
    for (std::size_t t = 0; t < n_trees; ++t) {
        const auto& tree = trees[t];
        for (const Node& node : tree.nodes) {
            if (node.kind == NodeKind::Axis || node.kind == NodeKind::Categorical) {
                if (node.feature < n_features) {
                    gains[node.feature] += node.split_gain;
                }
            } else if (node.kind == NodeKind::Oblique) {
                const std::uint8_t k = node.oblique_k;
                if (k <= 3U) {
                    for (std::uint8_t q = 0; q < k; ++q) {
                        if (node.oblique_features[q] < n_features) {
                            gains[node.oblique_features[q]] += node.split_gain;
                        }
                    }
                } else {
                    const auto& ov = tree.oblique_overflow[node.overflow_offset];
                    for (const std::uint32_t f : ov.features) {
                        if (f < n_features) {
                            gains[f] += node.split_gain;
                        }
                    }
                }
            }
        }
    }
    return gains;
}

void Evaluator::predict_raw(const Model& model, const double* data, std::size_t n_rows,
                            std::size_t n_cols, bool col_major, double* out,
                            std::size_t num_iteration, int n_jobs) {
    if (n_rows == 0U) {
        return;
    }
    if (n_cols != model.n_features) {
        throw std::invalid_argument("Evaluator::predict_raw feature count mismatch");
    }
    auto views = make_eval_views(data, n_rows, n_cols, col_major);
    predict_raw_views(model, n_rows, views, out, num_iteration, n_jobs);
}

void Evaluator::predict_raw(const Model& model, const float* data, std::size_t n_rows,
                            std::size_t n_cols, bool col_major, double* out,
                            std::size_t num_iteration, int n_jobs) {
    if (n_rows == 0U) {
        return;
    }
    if (n_cols != model.n_features) {
        throw std::invalid_argument("Evaluator::predict_raw feature count mismatch");
    }
    auto views = make_eval_views(data, n_rows, n_cols, col_major);
    predict_raw_views(model, n_rows, views, out, num_iteration, n_jobs);
}

void Evaluator::predict_raw_views(const Model& model, std::size_t n_rows,
                                  const std::vector<NumericColumnView>& num_cols, double* out,
                                  std::size_t num_iteration, int n_jobs) {
    if (n_rows == 0U) {
        return;
    }
    if (out == nullptr) {
        throw std::invalid_argument("Evaluator received null output pointer");
    }
    const std::size_t p = model.n_features;
    if (num_cols.size() != p || model.quantizers.size() != p || model.feature_kinds.size() != p) {
        throw std::invalid_argument("Evaluator feature count mismatch");
    }
    for (std::size_t f = 0; f < p; ++f) {
        if (model.feature_kinds[f] != 0U) {
            throw std::invalid_argument(
                "Model contains categorical features; use predict_raw_mixed instead");
        }
    }

    const std::size_t max_trees = (num_iteration == 0U)
                                      ? std::min(model.best_iteration, model.trees.size())
                                      : std::min(num_iteration, model.trees.size());

    std::vector<Q16Quantizer> quantizers(p);
    for (std::size_t f = 0; f < p; ++f) {
        quantizers[f] = Q16Quantizer{model.quantizers[f].lo, model.quantizers[f].scale,
                                     model.quantizers[f].is_constant};
    }

    const std::size_t n_blocks = (n_rows + kEvalBlockRows - 1U) / kEvalBlockRows;
    const int n_threads = resolve_num_threads(n_jobs);

    auto eval_block = [&](std::size_t blk_idx, std::uint16_t* block_codes) noexcept {
        const std::size_t base = blk_idx * kEvalBlockRows;
        const std::size_t blk = std::min(kEvalBlockRows, n_rows - base);

        for (std::size_t f = 0; f < p; ++f) {
            const NumericColumnView& col = num_cols[f];
            const Q16Quantizer& q = quantizers[f];
            if (col.f64 != nullptr) {
                for (std::size_t r = 0; r < blk; ++r) {
                    const double x = col.f64[(base + r) * col.row_stride];
                    block_codes[r * p + f] = q.encode_u16(x);
                }
            } else {
                for (std::size_t r = 0; r < blk; ++r) {
                    const double x = static_cast<double>(col.f32[(base + r) * col.row_stride]);
                    block_codes[r * p + f] = q.encode_u16(x);
                }
            }
        }

        for (std::size_t r = 0; r < blk; ++r) {
            double score = model.init_score;
            const std::uint16_t* row_codes = block_codes + r * p;
            for (std::size_t t = 0; t < max_trees; ++t) {
                score += eval_tree_u16(model.trees[t], row_codes);
            }
            out[base + r] = score;
        }
    };

    if (n_threads > 1 && n_rows >= 4096U && n_blocks >= 2U) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
        {
            std::vector<std::uint16_t> block_codes(kEvalBlockRows * p);
#pragma omp for schedule(static)
            for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(n_blocks); ++bi) {
                eval_block(static_cast<std::size_t>(bi), block_codes.data());
            }
        }
#endif
    } else {
        std::vector<std::uint16_t> block_codes(kEvalBlockRows * p);
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            eval_block(bi, block_codes.data());
        }
    }
}

void Evaluator::predict_raw_mixed(const Model& model, std::size_t n_rows,
                                  const std::vector<NumericColumnView>& num_cols,
                                  const std::vector<CategoricalColumnView>& cat_cols, double* out,
                                  std::size_t num_iteration, int n_jobs) {
    if (n_rows == 0U) {
        return;
    }
    if (out == nullptr) {
        throw std::invalid_argument("Evaluator received null output pointer");
    }
    const std::size_t p = model.n_features;
    if (model.feature_kinds.size() != p || model.quantizers.size() != p ||
        model.cat_features.size() != p) {
        throw std::invalid_argument("Model feature metadata size mismatch");
    }

    std::size_t expected_num = 0U;
    std::size_t expected_cat = 0U;
    for (std::size_t f = 0; f < p; ++f) {
        if (model.feature_kinds[f] == 0U) {
            ++expected_num;
        } else {
            ++expected_cat;
        }
    }
    if (num_cols.size() != expected_num || cat_cols.size() != expected_cat) {
        throw std::invalid_argument("Evaluator mixed feature count mismatch");
    }

    struct MixedColMeta {
        bool is_cat = false;
        std::size_t sub_idx = 0U;
        Q16Quantizer quant;
        std::uint8_t other_code = 0U;
        std::uint8_t missing_code = 1U;
        std::vector<std::uint8_t> vocab_to_code;
    };

    std::vector<MixedColMeta> meta(p);
    std::size_t n_idx = 0U;
    std::size_t c_idx = 0U;
    for (std::size_t f = 0; f < p; ++f) {
        if (model.feature_kinds[f] == 0U) {
            meta[f].is_cat = false;
            meta[f].sub_idx = n_idx++;
            meta[f].quant = Q16Quantizer{model.quantizers[f].lo, model.quantizers[f].scale,
                                         model.quantizers[f].is_constant};
        } else {
            const auto& cats = model.cat_features[f].categories;
            const auto other = static_cast<std::uint8_t>(cats.size());
            const auto missing = static_cast<std::uint8_t>(cats.size() + 1U);
            std::unordered_map<std::string_view, std::uint8_t> map;
            map.reserve(cats.size());
            for (std::size_t k = 0; k < cats.size(); ++k) {
                map.emplace(std::string_view(cats[k]), static_cast<std::uint8_t>(k));
            }
            const auto& col = cat_cols[c_idx];
            if (col.indices == nullptr) {
                throw std::invalid_argument("CategoricalColumnView received null indices pointer");
            }
            meta[f].is_cat = true;
            meta[f].sub_idx = c_idx++;
            meta[f].other_code = other;
            meta[f].missing_code = missing;
            meta[f].vocab_to_code.resize(col.vocab.size(), other);
            for (std::size_t v = 0; v < col.vocab.size(); ++v) {
                if (const auto it = map.find(std::string_view(col.vocab[v])); it != map.end()) {
                    meta[f].vocab_to_code[v] = it->second;
                }
            }
        }
    }

    const std::size_t max_trees = (num_iteration == 0U)
                                      ? std::min(model.best_iteration, model.trees.size())
                                      : std::min(num_iteration, model.trees.size());

    const std::size_t n_blocks = (n_rows + kEvalBlockRows - 1U) / kEvalBlockRows;
    const int n_threads = resolve_num_threads(n_jobs);

    auto eval_block = [&](std::size_t blk_idx, std::uint16_t* block_codes) noexcept {
        const std::size_t base = blk_idx * kEvalBlockRows;
        const std::size_t blk = std::min(kEvalBlockRows, n_rows - base);

        for (std::size_t f = 0; f < p; ++f) {
            const MixedColMeta& m = meta[f];
            if (!m.is_cat) {
                const NumericColumnView& col = num_cols[m.sub_idx];
                if (col.f64 != nullptr) {
                    for (std::size_t r = 0; r < blk; ++r) {
                        const double x = col.f64[(base + r) * col.row_stride];
                        block_codes[r * p + f] = m.quant.encode_u16(x);
                    }
                } else {
                    for (std::size_t r = 0; r < blk; ++r) {
                        const double x = static_cast<double>(col.f32[(base + r) * col.row_stride]);
                        block_codes[r * p + f] = m.quant.encode_u16(x);
                    }
                }
            } else {
                const CategoricalColumnView& col = cat_cols[m.sub_idx];
                const std::size_t vocab_sz = m.vocab_to_code.size();
                for (std::size_t r = 0; r < blk; ++r) {
                    const std::int32_t idx = col.indices[base + r];
                    const std::uint8_t code = (idx < 0 || static_cast<std::size_t>(idx) >= vocab_sz)
                                                  ? m.missing_code
                                                  : m.vocab_to_code[static_cast<std::size_t>(idx)];
                    block_codes[r * p + f] = static_cast<std::uint16_t>(code);
                }
            }
        }

        for (std::size_t r = 0; r < blk; ++r) {
            double score = model.init_score;
            const std::uint16_t* row_codes = block_codes + r * p;
            for (std::size_t t = 0; t < max_trees; ++t) {
                score += eval_tree_u16(model.trees[t], row_codes);
            }
            out[base + r] = score;
        }
    };

    if (n_threads > 1 && n_rows >= 4096U && n_blocks >= 2U) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
        {
            std::vector<std::uint16_t> block_codes(kEvalBlockRows * p);
#pragma omp for schedule(static)
            for (std::ptrdiff_t bi = 0; bi < static_cast<std::ptrdiff_t>(n_blocks); ++bi) {
                eval_block(static_cast<std::size_t>(bi), block_codes.data());
            }
        }
#endif
    } else {
        std::vector<std::uint16_t> block_codes(kEvalBlockRows * p);
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
            eval_block(bi, block_codes.data());
        }
    }
}

void Evaluator::predict_proba_from_raw(const double* raw, std::size_t n_rows, double* proba_2col) {
    for (std::size_t i = 0; i < n_rows; ++i) {
        const double p1 = sigmoid(raw[i]);
        proba_2col[2U * i] = 1.0 - p1;
        proba_2col[2U * i + 1U] = p1;
    }
}

void Evaluator::add_tree_predictions_from_codes(const Tree& tree,
                                                const std::int16_t* codes_row_major,
                                                std::size_t n_rows, std::size_t n_num_features,
                                                const std::vector<std::size_t>& feature_to_num_idx,
                                                double* scores, const std::uint8_t* bins_col_major,
                                                int n_jobs) noexcept {
    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_rows >= 4096U);

    auto score_row = [&](std::size_t i) noexcept {
        const std::int16_t* row = (n_num_features > 0U && codes_row_major != nullptr)
                                      ? (codes_row_major + i * n_num_features)
                                      : nullptr;
        std::uint32_t nid = 0U;
        for (;;) {
            const Node& node = tree.nodes[nid];
            if (node.kind == NodeKind::Leaf) {
                scores[i] += node.leaf_value;
                break;
            }
            bool go_left = false;
            if (node.kind == NodeKind::Axis) {
                if (row == nullptr) {
                    break;
                }
                const std::size_t num_idx = feature_to_num_idx[node.feature];
                const std::uint16_t u = Q16Quantizer::to_unsigned_code(row[num_idx]);
                go_left = (u == kQ16NanCode) ? node.nan_go_left : (u < node.threshold_u16);
            } else if (node.kind == NodeKind::Categorical) {
                if (bins_col_major == nullptr) {
                    break;
                }
                const std::uint8_t code =
                    bins_col_major[static_cast<std::size_t>(node.feature) * n_rows + i];
                go_left = cat_bitset_test(tree.cat_bitsets[node.cat_bitset_offset], code);
            } else {
                if (row == nullptr) {
                    break;
                }
                const std::uint8_t k = node.oblique_k;
                if (k <= 3U) {
                    const std::int16_t c0 = row[feature_to_num_idx[node.oblique_features[0]]];
                    const std::int16_t c1 = row[feature_to_num_idx[node.oblique_features[1]]];
                    const std::int16_t c2 = (k == 3U)
                                                ? row[feature_to_num_idx[node.oblique_features[2]]]
                                                : static_cast<std::int16_t>(0);
                    if (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode ||
                        (k == 3U && c2 == kQ16SignedNanCode)) {
                        go_left = node.nan_go_left;
                    } else {
                        const std::int32_t p32 =
                            static_cast<std::int32_t>(node.oblique_weights[0]) * c0 +
                            static_cast<std::int32_t>(node.oblique_weights[1]) * c1 +
                            static_cast<std::int32_t>(node.oblique_weights[2]) * c2;
                        go_left = (p32 < node.threshold_i32);
                    }
                } else {
                    const auto& ov = tree.oblique_overflow[node.overflow_offset];
                    std::int64_t acc = 0;
                    bool missing = false;
                    for (std::uint8_t q = 0; q < k; ++q) {
                        const std::int16_t c = row[feature_to_num_idx[ov.features[q]]];
                        if (c == kQ16SignedNanCode) {
                            missing = true;
                            break;
                        }
                        acc +=
                            static_cast<std::int64_t>(ov.weights[q]) * static_cast<std::int64_t>(c);
                    }
                    go_left = missing ? node.nan_go_left : (acc < ov.threshold_i64);
                }
            }
            nid = go_left ? node.left_child : node.right_child;
        }
    };

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_rows); ++i) {
            score_row(static_cast<std::size_t>(i));
        }
    } else {
        for (std::size_t i = 0; i < n_rows; ++i) {
            score_row(i);
        }
    }
}

} // namespace miter
