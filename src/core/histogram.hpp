#pragma once

#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

struct HistBin {
    double g = 0.0;
    double h = 0.0;
    std::int32_t c = 0;
};

struct FeatureHistLayout {
    std::vector<std::size_t> offsets;
    std::vector<std::uint16_t> num_regular_bins;
    std::vector<std::uint8_t> has_nan;
    std::vector<std::uint8_t> is_categorical;
    std::vector<std::uint32_t> active_features;
    std::vector<std::uint32_t> active_num_features;
    std::vector<std::uint32_t> active_cat_features;
    std::size_t total_bins = 0;

    static FeatureHistLayout build(const Dataset& ds);
};

class HistPool {
public:
    HistPool() = default;
    HistPool(std::size_t num_slots, std::size_t bins_per_slot);

    void reset(std::size_t num_slots, std::size_t bins_per_slot);

    HistBin* slot(std::size_t idx) noexcept { return buffer_.data() + idx * bins_per_slot_; }

    const HistBin* slot(std::size_t idx) const noexcept {
        return buffer_.data() + idx * bins_per_slot_;
    }

    std::size_t bins_per_slot() const noexcept { return bins_per_slot_; }

private:
    AlignedBuffer<HistBin> buffer_;
    std::size_t bins_per_slot_ = 0;
};

struct AxisSplitCandidate {
    double gain = -std::numeric_limits<double>::infinity();
    std::uint32_t feature = 0;
    std::uint8_t bin_idx = 0;
    std::uint16_t threshold_u16 = 0;
    bool nan_go_left = true;
    bool valid = false;
};

void build_axis_histograms(const Dataset& ds, const FeatureHistLayout& layout,
                           const std::uint32_t* leaf_rows, std::size_t n_leaf,
                           bool is_contiguous_root, const float* g, const float* h,
                           float* g_ordered, float* h_ordered, HistBin* hist_out, int n_jobs = 1,
                           HistBin* wave_hist_buf = nullptr) noexcept;

void subtract_histograms(HistBin* dst_large, const HistBin* parent, const HistBin* small,
                         std::size_t total_bins) noexcept;

AxisSplitCandidate find_best_axis_split(const Dataset& ds, const FeatureHistLayout& layout,
                                        const HistBin* hist, double parent_g, double parent_h,
                                        std::int32_t parent_c, const Params& params) noexcept;

} // namespace miter
