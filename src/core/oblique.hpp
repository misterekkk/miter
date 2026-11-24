#pragma once

#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/histogram.hpp"
#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"

namespace miter {

inline constexpr int kObliqueBins = 64;
inline constexpr int kObliqueWeightScale = 16384; // 2^14
inline constexpr int kObliqueMaxK = 8;
inline constexpr std::size_t kObliqueScratchLimitInts = 2'000'000U; // ~8 MB

std::int16_t round_half_away_i16(double v) noexcept;
bool quantize_oblique_weights(const double* w, int k, std::int16_t* wq) noexcept;

inline std::uint8_t oblique_bin_index(std::int64_t p, std::int64_t pmin,
                                      std::int64_t range) noexcept {
    return static_cast<std::uint8_t>(((p - pmin) * static_cast<std::int64_t>(kObliqueBins)) /
                                     range);
}

inline std::int64_t oblique_bin_threshold(std::int64_t pmin, std::int64_t range, int b) noexcept {
    return pmin + ((static_cast<std::int64_t>(b + 1) * range +
                    static_cast<std::int64_t>(kObliqueBins - 1)) /
                   static_cast<std::int64_t>(kObliqueBins));
}

struct ObliqueSplitCandidate {
    double gain = -std::numeric_limits<double>::infinity();
    std::uint8_t k = 0;
    std::uint32_t features[kObliqueMaxK] = {};
    std::uint32_t num_indices[kObliqueMaxK] = {};
    std::int16_t weights[kObliqueMaxK] = {};
    std::int64_t threshold = 0;
    bool nan_go_left = true;
    bool valid = false;
};

struct ObliqueScratch {
    std::vector<std::uint32_t> eligible_features;
    std::vector<std::uint32_t> eligible_num_indices;
    std::vector<std::uint8_t> num_has_nan;
    std::size_t max_m = 0;

    AlignedBuffer<std::int32_t> proj_buf;
    AlignedBuffer<HistBin> hists;

    std::vector<std::uint8_t> cand_k;
    std::vector<std::uint32_t> cand_num_idx;
    std::vector<std::uint32_t> cand_feat;
    std::vector<std::int16_t> cand_w;
    std::vector<std::uint8_t> cand_has_nan;

    std::vector<std::int32_t> cand_pmin;
    std::vector<std::int32_t> cand_pmax;
    std::vector<std::int32_t> cand_valid_cnt;
    std::vector<double> cand_g_nan;
    std::vector<double> cand_h_nan;
    std::vector<std::int32_t> cand_c_nan;
    std::vector<std::uint32_t> active_cands;
    std::vector<std::int64_t> active_ranges;

    void prepare(const Dataset& ds, const Params& params);
};

ObliqueSplitCandidate find_best_oblique_split(const Dataset& ds, const std::uint32_t* leaf_rows,
                                              std::size_t n_leaf, const float* g_seq,
                                              const float* h_seq, double parent_g, double parent_h,
                                              std::int32_t parent_c, std::size_t tree_index,
                                              std::uint32_t node_id, const Params& params,
                                              ObliqueScratch& scratch) noexcept;

} // namespace miter
