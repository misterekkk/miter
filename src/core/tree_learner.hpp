#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "core/symmetric_tree.hpp"
#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

struct TreeWorkspace {
    FeatureHistLayout layout;
    HistPool pool;
    ObliqueScratch oblique;
    SymmetricWorkspace sym;
    AlignedBuffer<HistBin> axis_wave_hists;
    std::vector<std::uint32_t> row_indices;
    std::vector<std::uint32_t> tmp_left;
    std::vector<std::uint32_t> tmp_right;
    std::vector<std::uint8_t> split_flags;
    std::vector<double> chunk_g;
    std::vector<double> chunk_h;
    std::vector<std::uint32_t> chunk_left_counts;
    std::vector<std::uint32_t> chunk_left_offs;
    std::vector<std::uint32_t> chunk_right_offs;
    std::vector<float> g_ordered;
    std::vector<float> h_ordered;

    // Leaf-contiguous working buffers for large datasets (bagging_fraction == 1.0): eliminates
    // scattered row_indices gathers across tree depth while preserving exact row ordering.
    AlignedBuffer<float> work_g;
    AlignedBuffer<float> work_h;
    AlignedBuffer<float> work_f32_tmp_l;
    AlignedBuffer<float> work_f32_tmp_r;
    AlignedBuffer<std::int16_t> work_codes;
    AlignedBuffer<std::int16_t> work_col_tmp;
    std::size_t work_code_threads = 1;

    // feature_fraction: full active lists (the layout's lists are narrowed per tree).
    std::vector<std::uint32_t> base_active_features;
    std::vector<std::uint32_t> base_active_num_features;
    std::vector<std::uint32_t> base_active_cat_features;
    std::vector<std::uint8_t> feature_selected;
    std::vector<std::uint32_t> feature_perm;
    // bagging_fraction: rows outside the bag of the current tree, and feature -> sub-indices.
    std::vector<std::uint32_t> oob_rows;
    std::vector<std::size_t> feature_to_num_idx;
    std::vector<std::size_t> feature_to_cat_idx;

    void prepare(const Dataset& ds, const Params& params);
};

Tree grow_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
               std::size_t tree_index, TreeWorkspace& ws, double* train_scores,
               const CtrGrowContext* ctr_ctx = nullptr);

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores);

} // namespace miter
