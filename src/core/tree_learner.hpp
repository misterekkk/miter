#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/histogram.hpp"
#include "data/dataset.hpp"

namespace miter {

struct TreeWorkspace {
    FeatureHistLayout layout;
    HistPool pool;
    std::vector<std::uint32_t> row_indices;
    std::vector<std::uint32_t> tmp_left;
    std::vector<std::uint32_t> tmp_right;
    std::vector<float> g_ordered;
    std::vector<float> h_ordered;

    void prepare(const Dataset& ds, int num_leaves);
};

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores);

} // namespace miter
