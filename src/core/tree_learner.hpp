#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "data/dataset.hpp"

namespace miter {

struct TreeWorkspace {
    FeatureHistLayout layout;
    HistPool pool;
    ObliqueScratch oblique;
    std::vector<std::uint32_t> row_indices;
    std::vector<std::uint32_t> tmp_left;
    std::vector<std::uint32_t> tmp_right;
    std::vector<float> g_ordered;
    std::vector<float> h_ordered;

    void prepare(const Dataset& ds, const Params& params);
};

Tree grow_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
               std::size_t tree_index, TreeWorkspace& ws, double* train_scores);

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores);

} // namespace miter
