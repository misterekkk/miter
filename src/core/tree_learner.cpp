#include "core/tree_learner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "core/categorical.hpp"
#include "data/quantizer.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

struct SplitChoice {
    NodeKind kind = NodeKind::Leaf;
    double gain = -std::numeric_limits<double>::infinity();
    bool valid = false;
    AxisSplitCandidate axis;
    CatSplitCandidate cat;
    ObliqueSplitCandidate oblique;
};

struct LeafState {
    std::uint32_t node_id = 0;
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    int depth = 0;
    std::size_t hist_slot = 0;
    double g_sum = 0.0;
    double h_sum = 0.0;
    SplitChoice split;
};

void sum_leaf_stats(const std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                    const float* g, const float* h, int n_jobs, TreeWorkspace& ws, double& g_sum,
                    double& h_sum) noexcept {
    const std::size_t n_leaf = static_cast<std::size_t>(end - start);
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    if (n_chunks <= 1U) {
        double gs = 0.0;
        double hs = 0.0;
        for (std::uint32_t t = start; t < end; ++t) {
            const std::uint32_t r = rows[t];
            gs += static_cast<double>(g[r]);
            hs += static_cast<double>(h[r]);
        }
        g_sum = gs;
        h_sum = hs;
        return;
    }

    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_leaf >= 32768U);
    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t t0 = static_cast<std::size_t>(start) + cu * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, static_cast<std::size_t>(end));
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                const std::uint32_t r = rows[t];
                gs += static_cast<double>(g[r]);
                hs += static_cast<double>(h[r]);
            }
            ws.chunk_g[cu] = gs;
            ws.chunk_h[cu] = hs;
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t t0 = static_cast<std::size_t>(start) + c * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, static_cast<std::size_t>(end));
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                const std::uint32_t r = rows[t];
                gs += static_cast<double>(g[r]);
                hs += static_cast<double>(h[r]);
            }
            ws.chunk_g[c] = gs;
            ws.chunk_h[c] = hs;
        }
    }

    double gs = 0.0;
    double hs = 0.0;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        gs += ws.chunk_g[c];
        hs += ws.chunk_h[c];
    }
    g_sum = gs;
    h_sum = hs;
}

void gather_leaf_gh(const std::uint32_t* rows, std::size_t n_leaf, const float* g, const float* h,
                    int n_jobs, float* g_ordered, float* h_ordered) noexcept {
    const int n_threads = resolve_num_threads(n_jobs);
    if (n_threads > 1 && n_leaf >= 16384U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n_leaf); ++t) {
            const std::uint32_t r = rows[static_cast<std::size_t>(t)];
            g_ordered[t] = g[r];
            h_ordered[t] = h[r];
        }
        return;
    }
    for (std::size_t t = 0; t < n_leaf; ++t) {
        const std::uint32_t r = rows[t];
        g_ordered[t] = g[r];
        h_ordered[t] = h[r];
    }
}

template <typename Pred>
std::uint32_t stable_partition_leaf_rows(std::uint32_t* rows, std::uint32_t start,
                                         std::uint32_t end, int n_jobs, TreeWorkspace& ws,
                                         Pred pred) noexcept {
    const std::size_t n_leaf = static_cast<std::size_t>(end - start);
    std::uint32_t* tmp_left = ws.tmp_left.data();
    std::uint32_t* tmp_right = ws.tmp_right.data();
    const int n_threads = resolve_num_threads(n_jobs);

    if (n_threads <= 1 || n_leaf < 32768U) {
        std::uint32_t nl = 0U;
        std::uint32_t nr = 0U;
        for (std::uint32_t t = start; t < end; ++t) {
            const std::uint32_t r = rows[t];
            if (pred(r)) {
                tmp_left[nl++] = r;
            } else {
                tmp_right[nr++] = r;
            }
        }
        if (nl > 0U) {
            std::copy_n(tmp_left, nl, rows + start);
        }
        if (nr > 0U) {
            std::copy_n(tmp_right, nr, rows + start + nl);
        }
        return nl;
    }

    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    std::uint8_t* flags = ws.split_flags.data();

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
        const auto cu = static_cast<std::size_t>(c);
        const std::size_t r0 = cu * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        std::uint32_t nl_c = 0U;
        for (std::size_t t = r0; t < r1; ++t) {
            const bool go_left = pred(rows[start + t]);
            flags[t] = go_left ? 1U : 0U;
            nl_c += go_left ? 1U : 0U;
        }
        ws.chunk_left_counts[cu] = nl_c;
    }

    std::uint32_t nl_total = 0U;
    std::uint32_t nr_total = 0U;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const std::size_t r0 = c * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        const auto len_c = static_cast<std::uint32_t>(r1 - r0);
        const std::uint32_t nl_c = ws.chunk_left_counts[c];
        ws.chunk_left_offs[c] = nl_total;
        ws.chunk_right_offs[c] = nr_total;
        nl_total += nl_c;
        nr_total += (len_c - nl_c);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
    for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
        const auto cu = static_cast<std::size_t>(c);
        const std::size_t r0 = cu * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        std::uint32_t lo = ws.chunk_left_offs[cu];
        std::uint32_t ro = ws.chunk_right_offs[cu];
        for (std::size_t t = r0; t < r1; ++t) {
            const std::uint32_t r = rows[start + t];
            if (flags[t] != 0U) {
                tmp_left[lo++] = r;
            } else {
                tmp_right[ro++] = r;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for schedule(static) nowait
        for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(nl_total); ++t) {
            rows[static_cast<std::size_t>(start) + static_cast<std::size_t>(t)] = tmp_left[t];
        }
#pragma omp for schedule(static)
        for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(nr_total); ++t) {
            rows[static_cast<std::size_t>(start) + nl_total + static_cast<std::size_t>(t)] =
                tmp_right[t];
        }
    }
#else
    if (nl_total > 0U) {
        std::copy_n(tmp_left, nl_total, rows + start);
    }
    if (nr_total > 0U) {
        std::copy_n(tmp_right, nr_total, rows + start + nl_total);
    }
#endif
    return nl_total;
}

std::uint32_t partition_leaf_rows_axis(std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                                       const std::uint8_t* col, const AxisSplitCandidate& split,
                                       std::uint8_t nan_bin, bool has_nan, int n_jobs,
                                       TreeWorkspace& ws) noexcept {
    const std::uint8_t thr_bin = split.bin_idx;
    if (!has_nan) {
        return stable_partition_leaf_rows(
            rows, start, end, n_jobs, ws,
            [col, thr_bin](std::uint32_t r) noexcept { return col[r] <= thr_bin; });
    }
    const bool nan_left = split.nan_go_left;
    return stable_partition_leaf_rows(rows, start, end, n_jobs, ws,
                                      [col, thr_bin, nan_bin, nan_left](std::uint32_t r) noexcept {
                                          const std::uint8_t b = col[r];
                                          return (b == nan_bin) ? nan_left : (b <= thr_bin);
                                      });
}

std::uint32_t partition_leaf_rows_cat(std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                                      const std::uint8_t* col, const CatBitset& bitset, int n_jobs,
                                      TreeWorkspace& ws) noexcept {
    return stable_partition_leaf_rows(
        rows, start, end, n_jobs, ws,
        [col, &bitset](std::uint32_t r) noexcept { return cat_bitset_test(bitset, col[r]); });
}

std::uint32_t partition_leaf_rows_oblique(std::uint32_t* rows, std::uint32_t start,
                                          std::uint32_t end, const std::int16_t* codes,
                                          std::size_t p_num, const ObliqueSplitCandidate& split,
                                          int n_jobs, TreeWorkspace& ws) noexcept {
    const std::uint8_t k = split.k;
    const bool nan_left = split.nan_go_left;

    if (k <= 3U) {
        const auto thr = static_cast<std::int32_t>(split.threshold);
        const std::uint32_t f0 = split.num_indices[0];
        const std::uint32_t f1 = split.num_indices[1];
        const std::uint32_t f2 = (k == 3U) ? split.num_indices[2] : f0;
        const std::int32_t w0 = split.weights[0];
        const std::int32_t w1 = split.weights[1];
        const std::int32_t w2 = (k == 3U) ? split.weights[2] : 0;

        return stable_partition_leaf_rows(
            rows, start, end, n_jobs, ws,
            [codes, p_num, thr, f0, f1, f2, w0, w1, w2, nan_left](std::uint32_t r) noexcept {
                const std::int16_t* row = codes + static_cast<std::size_t>(r) * p_num;
                const std::int16_t c0 = row[f0];
                const std::int16_t c1 = row[f1];
                const std::int16_t c2 = row[f2];
                if (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode || c2 == kQ16SignedNanCode) {
                    return nan_left;
                }
                const std::int32_t p32 = w0 * c0 + w1 * c1 + w2 * c2;
                return p32 < thr;
            });
    }

    const std::int64_t thr = split.threshold;
    return stable_partition_leaf_rows(
        rows, start, end, n_jobs, ws,
        [codes, p_num, &split, k, thr, nan_left](std::uint32_t r) noexcept {
            const std::int16_t* row = codes + static_cast<std::size_t>(r) * p_num;
            std::int64_t acc = 0;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c = row[split.num_indices[q]];
                if (c == kQ16SignedNanCode) {
                    return nan_left;
                }
                acc += static_cast<std::int64_t>(split.weights[q]) * static_cast<std::int64_t>(c);
            }
            return acc < thr;
        });
}

SplitChoice evaluate_leaf_splits(const Dataset& ds, LeafState& leaf, const float* g_seq,
                                 const float* h_seq, std::size_t tree_index, const Params& params,
                                 TreeWorkspace& ws) noexcept {
    SplitChoice choice;
    const auto n_leaf = static_cast<std::size_t>(leaf.end - leaf.start);
    const auto parent_c = static_cast<std::int32_t>(n_leaf);
    if (parent_c < 2 * params.min_data_in_leaf) {
        return choice;
    }

    if (!ws.layout.active_num_features.empty()) {
        choice.axis = find_best_axis_split(ds, ws.layout, ws.pool.slot(leaf.hist_slot), leaf.g_sum,
                                           leaf.h_sum, parent_c, params);
        if (choice.axis.valid) {
            choice.kind = NodeKind::Axis;
            choice.gain = choice.axis.gain;
            choice.valid = true;
        }
    }

    if (!ws.layout.active_cat_features.empty()) {
        choice.cat = find_best_cat_split(ws.layout, ws.pool.slot(leaf.hist_slot), leaf.g_sum,
                                         leaf.h_sum, parent_c, params);
        // Fixed tie-breaking order: axis < categorical < oblique.
        if (choice.cat.valid && (!choice.valid || choice.cat.gain > choice.gain)) {
            choice.kind = NodeKind::Categorical;
            choice.gain = choice.cat.gain;
            choice.valid = true;
        }
    }

    const bool allow_oblique = (params.m_mult > 0) && (params.oblique_max_depth < 0 ||
                                                       leaf.depth <= params.oblique_max_depth);
    if (allow_oblique && ws.oblique.max_m > 0U) {
        choice.oblique = find_best_oblique_split(ds, ws.row_indices.data() + leaf.start, n_leaf,
                                                 g_seq, h_seq, leaf.g_sum, leaf.h_sum, parent_c,
                                                 tree_index, leaf.node_id, params, ws.oblique);
        if (choice.oblique.valid && (!choice.valid || choice.oblique.gain > choice.gain)) {
            choice.kind = NodeKind::Oblique;
            choice.gain = choice.oblique.gain;
            choice.valid = true;
        }
    }

    return choice;
}

} // namespace

void TreeWorkspace::prepare(const Dataset& ds, const Params& params) {
    layout = FeatureHistLayout::build(ds);
    const std::size_t slots = static_cast<std::size_t>(std::max(params.num_leaves, 1)) + 1U;
    pool.reset(slots, layout.total_bins);
    oblique.prepare(ds, params);
    const std::size_t n = ds.n_rows();
    const std::size_t max_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;
    if (layout.active_features.size() < 16U && n >= 32768U) {
        axis_wave_hists.resize(kWaveChunks * layout.total_bins);
    } else {
        axis_wave_hists.resize(0U);
    }
    row_indices.resize(n);
    tmp_left.resize(n);
    tmp_right.resize(n);
    split_flags.resize(n);
    chunk_g.resize(max_chunks);
    chunk_h.resize(max_chunks);
    chunk_left_counts.resize(max_chunks);
    chunk_left_offs.resize(max_chunks);
    chunk_right_offs.resize(max_chunks);
    g_ordered.resize(n);
    h_ordered.resize(n);
}

Tree grow_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
               std::size_t tree_index, TreeWorkspace& ws, double* train_scores) {
    const std::size_t n = ds.n_rows();
    const auto max_leaves = static_cast<std::size_t>(std::max(params.num_leaves, 1));
    std::iota(ws.row_indices.begin(), ws.row_indices.end(), 0U);

    Tree tree;
    tree.nodes.reserve(2U * max_leaves);
    tree.nodes.push_back(Node{});

    std::vector<LeafState> leaves;
    leaves.reserve(max_leaves);

    LeafState root;
    root.node_id = 0U;
    root.start = 0U;
    root.end = static_cast<std::uint32_t>(n);
    root.depth = 0;
    root.hist_slot = 0U;
    sum_leaf_stats(ws.row_indices.data(), root.start, root.end, g, h, params.n_jobs, ws, root.g_sum,
                   root.h_sum);

    const bool root_can_split = (max_leaves > 1U) &&
                                (params.max_depth < 0 || root.depth < params.max_depth) &&
                                (static_cast<std::int32_t>(n) >= 2 * params.min_data_in_leaf);
    if (root_can_split) {
        if (ws.layout.total_bins > 0U) {
            build_axis_histograms(ds, ws.layout, ws.row_indices.data(), n, true, g, h,
                                  ws.g_ordered.data(), ws.h_ordered.data(), ws.pool.slot(0),
                                  params.n_jobs, ws.axis_wave_hists.data());
        }
        root.split = evaluate_leaf_splits(ds, root, g, h, tree_index, params, ws);
    }
    leaves.push_back(root);

    std::size_t next_slot = 1U;
    const std::uint8_t* bins_base = ds.bins_col_major().data();
    const std::size_t p_num = ds.n_num_features();
    const std::int16_t* codes_base = (p_num > 0U) ? ds.codes_row_major().data() : nullptr;

    while (leaves.size() < max_leaves) {
        std::size_t best_leaf_idx = leaves.size();
        double best_gain = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < leaves.size(); ++i) {
            if (leaves[i].split.valid && leaves[i].split.gain > best_gain) {
                best_gain = leaves[i].split.gain;
                best_leaf_idx = i;
            }
        }
        if (best_leaf_idx == leaves.size()) {
            break;
        }

        const LeafState parent = leaves[best_leaf_idx];
        const auto& split = parent.split;

        std::uint32_t nl = 0U;
        if (split.kind == NodeKind::Axis) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(split.axis.feature) * n;
            const auto& binner = ds.binner(split.axis.feature);
            nl = partition_leaf_rows_axis(ws.row_indices.data(), parent.start, parent.end, col,
                                          split.axis, binner.nan_bin(), binner.has_nan,
                                          params.n_jobs, ws);
        } else if (split.kind == NodeKind::Categorical) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(split.cat.feature) * n;
            nl = partition_leaf_rows_cat(ws.row_indices.data(), parent.start, parent.end, col,
                                         split.cat.left_bitset, params.n_jobs, ws);
        } else {
            nl = partition_leaf_rows_oblique(ws.row_indices.data(), parent.start, parent.end,
                                             codes_base, p_num, split.oblique, params.n_jobs, ws);
        }

        const std::uint32_t mid = parent.start + nl;
        const std::uint32_t nr = parent.end - mid;

        const auto left_node_id = static_cast<std::uint32_t>(tree.nodes.size());
        const auto right_node_id = left_node_id + 1U;
        tree.nodes.push_back(Node{});
        tree.nodes.push_back(Node{});

        Node& pnode = tree.nodes[parent.node_id];
        pnode.kind = split.kind;
        pnode.left_child = left_node_id;
        pnode.right_child = right_node_id;
        pnode.split_gain = split.gain;

        if (split.kind == NodeKind::Axis) {
            pnode.nan_go_left = split.axis.nan_go_left;
            pnode.feature = split.axis.feature;
            pnode.threshold_u16 = split.axis.threshold_u16;
        } else if (split.kind == NodeKind::Categorical) {
            pnode.nan_go_left = split.cat.default_go_left;
            pnode.feature = split.cat.feature;
            pnode.cat_bitset_offset = static_cast<std::uint32_t>(tree.cat_bitsets.size());
            tree.cat_bitsets.push_back(split.cat.left_bitset);
        } else {
            const auto& ob = split.oblique;
            pnode.nan_go_left = ob.nan_go_left;
            pnode.oblique_k = ob.k;
            if (ob.k <= 3U) {
                for (std::uint8_t q = 0; q < ob.k; ++q) {
                    pnode.oblique_features[q] = ob.features[q];
                    pnode.oblique_weights[q] = ob.weights[q];
                }
                pnode.threshold_i32 = static_cast<std::int32_t>(ob.threshold);
            } else {
                pnode.overflow_offset = static_cast<std::uint32_t>(tree.oblique_overflow.size());
                ObliqueOverflow ov;
                ov.features.assign(ob.features, ob.features + ob.k);
                ov.weights.assign(ob.weights, ob.weights + ob.k);
                ov.threshold_i64 = ob.threshold;
                tree.oblique_overflow.push_back(std::move(ov));
            }
        }

        LeafState left_leaf;
        left_leaf.node_id = left_node_id;
        left_leaf.start = parent.start;
        left_leaf.end = mid;
        left_leaf.depth = parent.depth + 1;

        LeafState right_leaf;
        right_leaf.node_id = right_node_id;
        right_leaf.start = mid;
        right_leaf.end = parent.end;
        right_leaf.depth = parent.depth + 1;

        if (nl <= nr) {
            sum_leaf_stats(ws.row_indices.data(), left_leaf.start, left_leaf.end, g, h,
                           params.n_jobs, ws, left_leaf.g_sum, left_leaf.h_sum);
            right_leaf.g_sum = parent.g_sum - left_leaf.g_sum;
            right_leaf.h_sum = parent.h_sum - left_leaf.h_sum;
        } else {
            sum_leaf_stats(ws.row_indices.data(), right_leaf.start, right_leaf.end, g, h,
                           params.n_jobs, ws, right_leaf.g_sum, right_leaf.h_sum);
            left_leaf.g_sum = parent.g_sum - right_leaf.g_sum;
            left_leaf.h_sum = parent.h_sum - right_leaf.h_sum;
        }

        const bool will_continue = (leaves.size() + 1U < max_leaves) &&
                                   (params.max_depth < 0 || left_leaf.depth < params.max_depth);
        if (will_continue) {
            const std::size_t small_slot = next_slot++;
            const std::size_t large_slot = parent.hist_slot;

            if (nl <= nr) {
                left_leaf.hist_slot = small_slot;
                right_leaf.hist_slot = large_slot;
                if (ws.layout.total_bins > 0U) {
                    build_axis_histograms(ds, ws.layout, ws.row_indices.data() + left_leaf.start,
                                          nl, false, g, h, ws.g_ordered.data(), ws.h_ordered.data(),
                                          ws.pool.slot(small_slot), params.n_jobs,
                                          ws.axis_wave_hists.data());
                    subtract_histograms(ws.pool.slot(large_slot), ws.pool.slot(parent.hist_slot),
                                        ws.pool.slot(small_slot), ws.layout.total_bins);
                } else {
                    gather_leaf_gh(ws.row_indices.data() + left_leaf.start, nl, g, h, params.n_jobs,
                                   ws.g_ordered.data(), ws.h_ordered.data());
                }
                left_leaf.split = evaluate_leaf_splits(ds, left_leaf, ws.g_ordered.data(),
                                                       ws.h_ordered.data(), tree_index, params, ws);

                const bool right_needs_oblique =
                    (params.m_mult > 0) &&
                    (params.oblique_max_depth < 0 ||
                     right_leaf.depth <= params.oblique_max_depth) &&
                    ws.oblique.max_m > 0U &&
                    (static_cast<std::int32_t>(nr) >= 2 * params.min_data_in_leaf);
                if (right_needs_oblique) {
                    gather_leaf_gh(ws.row_indices.data() + right_leaf.start, nr, g, h,
                                   params.n_jobs, ws.g_ordered.data(), ws.h_ordered.data());
                }
                right_leaf.split =
                    evaluate_leaf_splits(ds, right_leaf, ws.g_ordered.data(), ws.h_ordered.data(),
                                         tree_index, params, ws);
            } else {
                right_leaf.hist_slot = small_slot;
                left_leaf.hist_slot = large_slot;
                if (ws.layout.total_bins > 0U) {
                    build_axis_histograms(ds, ws.layout, ws.row_indices.data() + right_leaf.start,
                                          nr, false, g, h, ws.g_ordered.data(), ws.h_ordered.data(),
                                          ws.pool.slot(small_slot), params.n_jobs,
                                          ws.axis_wave_hists.data());
                    subtract_histograms(ws.pool.slot(large_slot), ws.pool.slot(parent.hist_slot),
                                        ws.pool.slot(small_slot), ws.layout.total_bins);
                } else {
                    gather_leaf_gh(ws.row_indices.data() + right_leaf.start, nr, g, h,
                                   params.n_jobs, ws.g_ordered.data(), ws.h_ordered.data());
                }
                right_leaf.split =
                    evaluate_leaf_splits(ds, right_leaf, ws.g_ordered.data(), ws.h_ordered.data(),
                                         tree_index, params, ws);

                const bool left_needs_oblique =
                    (params.m_mult > 0) &&
                    (params.oblique_max_depth < 0 || left_leaf.depth <= params.oblique_max_depth) &&
                    ws.oblique.max_m > 0U &&
                    (static_cast<std::int32_t>(nl) >= 2 * params.min_data_in_leaf);
                if (left_needs_oblique) {
                    gather_leaf_gh(ws.row_indices.data() + left_leaf.start, nl, g, h, params.n_jobs,
                                   ws.g_ordered.data(), ws.h_ordered.data());
                }
                left_leaf.split = evaluate_leaf_splits(ds, left_leaf, ws.g_ordered.data(),
                                                       ws.h_ordered.data(), tree_index, params, ws);
            }
        }

        leaves[best_leaf_idx] = left_leaf;
        leaves.push_back(right_leaf);
    }

    tree.n_leaves = static_cast<std::uint32_t>(leaves.size());
    const double lr = params.learning_rate;
    const double lam = params.lambda_l2;
    const int n_threads = resolve_num_threads(params.n_jobs);

    for (const auto& leaf : leaves) {
        double gs = 0.0;
        double hs = 0.0;
        sum_leaf_stats(ws.row_indices.data(), leaf.start, leaf.end, g, h, params.n_jobs, ws, gs,
                       hs);
        const double val = lr * (-gs / (hs + lam));
        tree.nodes[leaf.node_id].kind = NodeKind::Leaf;
        tree.nodes[leaf.node_id].leaf_value = val;
        if (train_scores != nullptr) {
            const std::uint32_t* rows_ptr = ws.row_indices.data();
            if (n_threads > 1 && (leaf.end - leaf.start) >= 32768U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
                for (std::ptrdiff_t t = static_cast<std::ptrdiff_t>(leaf.start);
                     t < static_cast<std::ptrdiff_t>(leaf.end); ++t) {
                    train_scores[rows_ptr[t]] += val;
                }
            } else {
                for (std::uint32_t t = leaf.start; t < leaf.end; ++t) {
                    train_scores[rows_ptr[t]] += val;
                }
            }
        }
    }

    return tree;
}

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores) {
    Params ax = params;
    ax.m_mult = 0;
    return grow_tree(ds, g, h, ax, 0U, ws, train_scores);
}

} // namespace miter
