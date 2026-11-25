#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

struct TreeWorkspace;

struct SymmetricWorkspace {
    std::size_t max_depth = 0;
    std::size_t max_level_leaves = 0; // 1 << (max_depth - 1)
    std::size_t max_threads = 1;
    std::size_t max_chunks = 0;

    AlignedBuffer<std::uint8_t> leaf_ids;
    AlignedBuffer<HistBin> axis_thread_hists;
    AlignedBuffer<HistBin> oblique_thread_hists;
    AlignedBuffer<HistBin> wave_hists;
    std::vector<double> chunk_leaf_g;
    std::vector<double> chunk_leaf_h;
    std::vector<std::int32_t> chunk_leaf_c;

    void prepare(const Dataset& ds, const Params& params);
};

Tree grow_symmetric_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                         std::size_t tree_index, TreeWorkspace& ws, std::size_t n_root,
                         bool bagging, double* train_scores);

} // namespace miter
