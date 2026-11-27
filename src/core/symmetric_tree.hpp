#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/ctr.hpp"
#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

struct TreeWorkspace;

struct CtrGrowContext {
    CtrTrainingState* state = nullptr;
    Model* model = nullptr;
    const double* target_y = nullptr;
    double prior = 0.0;
    const double* sample_weight = nullptr;
    std::uint32_t target_id = 0;
    bool is_unit_target = false;
};

struct SymmetricWorkspace {
    std::size_t max_depth = 0;
    std::size_t max_level_leaves = 0; // 1 << (max_depth - 1)
    std::size_t max_prev_leaves = 0;  // max_level_leaves / 2
    std::size_t prev_axis_total_bins = 0;
    bool use_axis_sub = false;
    std::size_t max_threads = 1;
    std::size_t max_chunks = 0;

    AlignedBuffer<std::uint8_t> leaf_ids;
    AlignedBuffer<HistBin> axis_thread_hists;
    AlignedBuffer<HistBin> oblique_thread_hists;
    AlignedBuffer<HistBin> wave_hists;
    std::vector<std::size_t> prev_axis_offsets;
    AlignedBuffer<HistBin> prev_axis_hists;
    AlignedBuffer<std::uint32_t> small_rows;
    AlignedBuffer<std::uint8_t> small_pairs;
    AlignedBuffer<float> small_g;
    AlignedBuffer<float> small_h;
    std::vector<std::size_t> chunk_small_counts;
    std::vector<std::size_t> chunk_small_offs;
    std::vector<double> chunk_leaf_g;
    std::vector<double> chunk_leaf_h;
    std::vector<std::int32_t> chunk_leaf_c;
    // oblique_sample_rows: the strided sample and the exact refit of the winning direction.
    std::vector<std::uint32_t> samp_rows;
    std::vector<float> samp_g;
    std::vector<float> samp_h;
    std::vector<std::uint8_t> samp_leaf;
    std::vector<HistBin> samp_hist;
    std::vector<std::int64_t> samp_proj;
    std::vector<HistBin> multi_hists;      // single-threaded grouped oblique pass2
    std::vector<HistBin> axis_multi_hists; // grouped axis histograms, 4 per thread
    std::vector<std::int32_t> samp_proj32; // k <= 3 refit on contiguous rows
    std::vector<std::int64_t> samp_chunk_min;
    std::vector<std::int64_t> samp_chunk_max;
    std::vector<std::int16_t> samp_codes; // sampled rows' codes, column-major (no bagging only)
    std::size_t samp_codes_stride = 0;    // row stride samp_codes was built for; 0 = none
    // efb: bundles of mutually exclusive numeric features. A bundle column holds 0 where every
    // member is at its default (most frequent) bin, else 1 + member offset + the member's bin slot.
    struct EfbBundle {
        std::vector<std::uint32_t> members;
        std::uint16_t n_bins = 0;
        std::size_t prev_offset = 0;
    };
    std::vector<EfbBundle> efb_bundles;
    std::vector<std::int32_t> efb_bundle_of;    // per feature: bundle index or -1
    std::vector<std::uint16_t> efb_offset_of;   // per feature: first bundle bin of its slots
    std::vector<std::uint8_t> efb_default_code; // per feature: its default bin code
    AlignedBuffer<std::uint8_t> efb_cols;       // n_bundles x n rows
    AlignedBuffer<HistBin> efb_hists;           // n_bundles x max_level_leaves x 256

    void prepare(const Dataset& ds, const Params& params);
};

Tree grow_symmetric_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                         std::size_t tree_index, TreeWorkspace& ws, std::size_t n_root,
                         bool bagging, double* train_scores,
                         const CtrGrowContext* ctr_ctx = nullptr);

} // namespace miter
