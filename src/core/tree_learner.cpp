#include "core/tree_learner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace miter {

namespace {

struct LeafState {
    std::uint32_t node_id = 0;
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    int depth = 0;
    std::size_t hist_slot = 0;
    double g_sum = 0.0;
    double h_sum = 0.0;
    AxisSplitCandidate split;
};

void sum_leaf_stats(const std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                    const float* g, const float* h, double& g_sum, double& h_sum) noexcept {
    double gs = 0.0;
    double hs = 0.0;
    for (std::uint32_t t = start; t < end; ++t) {
        const std::uint32_t r = rows[t];
        gs += static_cast<double>(g[r]);
        hs += static_cast<double>(h[r]);
    }
    g_sum = gs;
    h_sum = hs;
}

std::uint32_t partition_leaf_rows(std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                                  const std::uint8_t* col, const AxisSplitCandidate& split,
                                  std::uint8_t nan_bin, bool has_nan, std::uint32_t* tmp_left,
                                  std::uint32_t* tmp_right) noexcept {
    std::uint32_t nl = 0;
    std::uint32_t nr = 0;
    const std::uint8_t thr_bin = split.bin_idx;

    if (!has_nan) {
        for (std::uint32_t t = start; t < end; ++t) {
            const std::uint32_t r = rows[t];
            if (col[r] <= thr_bin) {
                tmp_left[nl++] = r;
            } else {
                tmp_right[nr++] = r;
            }
        }
    } else {
        const bool nan_left = split.nan_go_left;
        for (std::uint32_t t = start; t < end; ++t) {
            const std::uint32_t r = rows[t];
            const std::uint8_t b = col[r];
            const bool go_left = (b == nan_bin) ? nan_left : (b <= thr_bin);
            if (go_left) {
                tmp_left[nl++] = r;
            } else {
                tmp_right[nr++] = r;
            }
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

} // namespace

void TreeWorkspace::prepare(const Dataset& ds, int num_leaves) {
    layout = FeatureHistLayout::build(ds);
    const std::size_t slots = static_cast<std::size_t>(std::max(num_leaves, 1)) + 1U;
    pool.reset(slots, layout.total_bins);
    const std::size_t n = ds.n_rows();
    row_indices.resize(n);
    tmp_left.resize(n);
    tmp_right.resize(n);
    g_ordered.resize(n);
    h_ordered.resize(n);
}

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores) {
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

    {
        double gs = 0.0;
        double hs = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            gs += static_cast<double>(g[i]);
            hs += static_cast<double>(h[i]);
        }
        root.g_sum = gs;
        root.h_sum = hs;
    }

    const bool root_can_split = (max_leaves > 1U) &&
                                (params.max_depth < 0 || root.depth < params.max_depth) &&
                                (static_cast<std::int32_t>(n) >= 2 * params.min_data_in_leaf);
    if (root_can_split && ws.layout.total_bins > 0U) {
        build_axis_histograms(ds, ws.layout, ws.row_indices.data(), n, true, g, h,
                              ws.g_ordered.data(), ws.h_ordered.data(), ws.pool.slot(0));
        root.split = find_best_axis_split(ds, ws.layout, ws.pool.slot(0), root.g_sum, root.h_sum,
                                          static_cast<std::int32_t>(n), params);
    }
    leaves.push_back(root);

    std::size_t next_slot = 1U;
    const std::uint8_t* bins_base = ds.bins_col_major().data();

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
        const std::uint8_t* col = bins_base + static_cast<std::size_t>(split.feature) * n;
        const auto& binner = ds.binner(split.feature);

        const std::uint32_t nl = partition_leaf_rows(
            ws.row_indices.data(), parent.start, parent.end, col, split, binner.nan_bin(),
            binner.has_nan, ws.tmp_left.data(), ws.tmp_right.data());
        const std::uint32_t mid = parent.start + nl;
        const std::uint32_t nr = parent.end - mid;

        const auto left_node_id = static_cast<std::uint32_t>(tree.nodes.size());
        const auto right_node_id = left_node_id + 1U;
        tree.nodes.push_back(Node{});
        tree.nodes.push_back(Node{});

        Node& pnode = tree.nodes[parent.node_id];
        pnode.kind = NodeKind::Axis;
        pnode.nan_go_left = split.nan_go_left;
        pnode.left_child = left_node_id;
        pnode.right_child = right_node_id;
        pnode.feature = split.feature;
        pnode.threshold_u16 = split.threshold_u16;
        pnode.split_gain = split.gain;

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

        // Compute exact G, H on the smaller child and subtract from parent to match row order.
        if (nl <= nr) {
            sum_leaf_stats(ws.row_indices.data(), left_leaf.start, left_leaf.end, g, h,
                           left_leaf.g_sum, left_leaf.h_sum);
            right_leaf.g_sum = parent.g_sum - left_leaf.g_sum;
            right_leaf.h_sum = parent.h_sum - left_leaf.h_sum;
        } else {
            sum_leaf_stats(ws.row_indices.data(), right_leaf.start, right_leaf.end, g, h,
                           right_leaf.g_sum, right_leaf.h_sum);
            left_leaf.g_sum = parent.g_sum - right_leaf.g_sum;
            left_leaf.h_sum = parent.h_sum - right_leaf.h_sum;
        }

        const bool will_continue = (leaves.size() + 1U < max_leaves) &&
                                   (params.max_depth < 0 || left_leaf.depth < params.max_depth);
        if (will_continue && ws.layout.total_bins > 0U) {
            const std::size_t small_slot = next_slot++;
            const std::size_t large_slot = parent.hist_slot;
            if (nl <= nr) {
                left_leaf.hist_slot = small_slot;
                right_leaf.hist_slot = large_slot;
                build_axis_histograms(ds, ws.layout, ws.row_indices.data() + left_leaf.start, nl,
                                      false, g, h, ws.g_ordered.data(), ws.h_ordered.data(),
                                      ws.pool.slot(small_slot));
                subtract_histograms(ws.pool.slot(large_slot), ws.pool.slot(parent.hist_slot),
                                    ws.pool.slot(small_slot), ws.layout.total_bins);
            } else {
                right_leaf.hist_slot = small_slot;
                left_leaf.hist_slot = large_slot;
                build_axis_histograms(ds, ws.layout, ws.row_indices.data() + right_leaf.start, nr,
                                      false, g, h, ws.g_ordered.data(), ws.h_ordered.data(),
                                      ws.pool.slot(small_slot));
                subtract_histograms(ws.pool.slot(large_slot), ws.pool.slot(parent.hist_slot),
                                    ws.pool.slot(small_slot), ws.layout.total_bins);
            }

            if (static_cast<std::int32_t>(nl) >= 2 * params.min_data_in_leaf) {
                left_leaf.split = find_best_axis_split(
                    ds, ws.layout, ws.pool.slot(left_leaf.hist_slot), left_leaf.g_sum,
                    left_leaf.h_sum, static_cast<std::int32_t>(nl), params);
            }
            if (static_cast<std::int32_t>(nr) >= 2 * params.min_data_in_leaf) {
                right_leaf.split = find_best_axis_split(
                    ds, ws.layout, ws.pool.slot(right_leaf.hist_slot), right_leaf.g_sum,
                    right_leaf.h_sum, static_cast<std::int32_t>(nr), params);
            }
        }

        leaves[best_leaf_idx] = left_leaf;
        leaves.push_back(right_leaf);
    }

    tree.n_leaves = static_cast<std::uint32_t>(leaves.size());
    const double lr = params.learning_rate;
    const double lam = params.lambda_l2;

    for (const auto& leaf : leaves) {
        double gs = 0.0;
        double hs = 0.0;
        sum_leaf_stats(ws.row_indices.data(), leaf.start, leaf.end, g, h, gs, hs);
        const double val = lr * (-gs / (hs + lam));
        tree.nodes[leaf.node_id].kind = NodeKind::Leaf;
        tree.nodes[leaf.node_id].leaf_value = val;
        if (train_scores != nullptr) {
            for (std::uint32_t t = leaf.start; t < leaf.end; ++t) {
                train_scores[ws.row_indices[t]] += val;
            }
        }
    }

    return tree;
}

} // namespace miter
