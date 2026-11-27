#pragma once

#include <miter/params.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/histogram.hpp"
#include "data/dataset.hpp"
#include "util/aligned_buffer.hpp"
#include "util/fast_div.hpp"

namespace miter {

inline constexpr int kObliqueBins = 64;
inline constexpr int kObliqueWeightScale = 16384; // 2^14
inline constexpr int kObliqueMaxK = 8;
inline constexpr std::size_t kObliqueScratchLimitInts = 8'000'000U; // ~32 MB
inline constexpr std::size_t kObliqueMicroBlock = 256U;

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

struct ObliqueCandMeta {
    std::uint32_t fn[3] = {};
    std::int16_t w[3] = {};
    std::uint16_t uf[3] = {};
    std::uint8_t k = 0;
    std::uint8_t has_nan = 0;
};

struct ObliqueScratch {
    std::vector<std::uint32_t> eligible_features;
    std::vector<std::uint32_t> eligible_num_indices;
    std::vector<std::uint8_t> num_has_nan;
    std::size_t max_m = 0;
    // Candidates drawn per node in the current tree (== max_m unless feature_fraction < 1).
    std::size_t cur_m = 0;
    // Full eligible pool (set once in prepare); eligible_* above is the per-tree view.
    std::vector<std::uint32_t> all_eligible_features;
    std::vector<std::uint32_t> all_eligible_num_indices;
    std::size_t max_chunks = 0;
    std::size_t max_used_feats = 0;
    std::size_t max_threads = 1;

    AlignedBuffer<std::int32_t> proj_buf;
    AlignedBuffer<HistBin> hists;
    AlignedBuffer<HistBin> wave_hists;
    AlignedBuffer<std::int16_t> thread_feat_buf;
    std::vector<const std::int16_t*> thread_col_ptrs;

    std::vector<ObliqueCandMeta> cand_meta;
    std::vector<std::uint8_t> cand_k;
    std::vector<std::uint32_t> cand_num_idx;
    std::vector<std::uint32_t> cand_feat;
    std::vector<std::int16_t> cand_w;
    std::vector<std::uint8_t> cand_has_nan;
    std::vector<std::uint32_t> used_num_feats;
    std::vector<std::int32_t> num_to_used;

    std::vector<std::int32_t> cand_pmin;
    std::vector<std::int32_t> cand_pmax;
    std::vector<std::int32_t> cand_valid_cnt;
    std::vector<double> cand_g_nan;
    std::vector<double> cand_h_nan;
    std::vector<std::int32_t> cand_c_nan;

    std::vector<std::int32_t> chunk_pmin;
    std::vector<std::int32_t> chunk_pmax;
    std::vector<std::int32_t> chunk_valid_cnt;
    std::vector<double> chunk_g_nan;
    std::vector<double> chunk_h_nan;
    std::vector<std::int32_t> chunk_c_nan;

    std::vector<std::uint32_t> active_cands;
    std::vector<std::int64_t> active_ranges;
    std::vector<FastDiv64> active_divs;
    std::vector<ObliqueSplitCandidate> i64_candidates;

    void prepare(const Dataset& ds, const Params& params);

    /// Restrict the projection pool to features with selected[feature] != 0 and recompute cur_m
    /// (m_mult * ceil(sqrt(p_eligible))), as the plan's p_eligible "after feature_fraction".
    void restrict_pool(const std::vector<std::uint8_t>& selected, const Params& params);
};

ObliqueSplitCandidate find_best_oblique_split(const Dataset& ds, const std::int16_t* codes_base,
                                              std::size_t leaf_code_offset, bool is_contiguous_leaf,
                                              const std::uint32_t* leaf_rows, std::size_t n_leaf,
                                              const float* g_seq, const float* h_seq,
                                              double parent_g, double parent_h,
                                              std::int32_t parent_c, std::size_t tree_index,
                                              std::uint32_t node_id, const Params& params,
                                              ObliqueScratch& scratch) noexcept;

} // namespace miter
