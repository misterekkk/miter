// Internal to the symmetric (oblivious) tree learner: shared types and the functions the
// split searches in symmetric_{axis,ctr,efb,oblique}.cpp export to symmetric_tree.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/ctr.hpp"
#include "core/histogram.hpp"
#include "core/oblique.hpp"
#include "core/symmetric_tree.hpp"
#include "core/tree_learner.hpp"
#include "data/dataset.hpp"

namespace miter::sym_detail {

constexpr std::size_t kSymMaxLeaves = 256U; // max_depth <= 8 => 2^8 = 256 leaves
constexpr std::int32_t kMissingProjSentinel = std::numeric_limits<std::int32_t>::min();
constexpr std::uint8_t kMissingBinSentinel = static_cast<std::uint8_t>(kObliqueBins);
constexpr std::uint64_t kPurposeExtraTrees = 3U;

struct SymSplitChoice {
    NodeKind kind = NodeKind::Leaf;
    double gain = -std::numeric_limits<double>::infinity();
    bool valid = false;
    AxisSplitCandidate axis;
    CatSplitCandidate cat;
    ObliqueSplitCandidate oblique;
    CtrSplitCandidate ctr;
    std::uint32_t ctr_id = 0;
};

inline bool eval_multi_leaf_gain(const double* gl_arr, const double* hl_arr,
                                 const std::int32_t* cl_arr, const double* leaf_g,
                                 const double* leaf_h, const std::int32_t* leaf_c,
                                 const double* parent_score, std::size_t n_leaves, double lam,
                                 double min_h, std::int32_t min_c, double min_gain,
                                 double& total_gain_out) noexcept {
    double total_gain = 0.0;
    bool any_split = false;
    for (std::size_t l = 0; l < n_leaves; ++l) {
        const std::int32_t cl = cl_arr[l];
        const std::int32_t cr = leaf_c[l] - cl;
        if (cl <= 0 || cr <= 0) {
            continue;
        }
        if (cl < min_c || cr < min_c) {
            return false;
        }
        const double hl = hl_arr[l];
        const double hr = leaf_h[l] - hl;
        if (lam == 0.0 && (hl < min_h || hr < min_h)) {
            return false;
        }
        const double gl = gl_arr[l];
        const double gr = leaf_g[l] - gl;
        total_gain += (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam) - parent_score[l];
        any_split = true;
    }
    if (!any_split || !(total_gain > min_gain)) {
        return false;
    }
    total_gain_out = total_gain;
    return true;
}

void add_unique_ctr_proj(std::vector<CtrProjection>& used, const CtrProjection& proj);

CtrSplitCandidate
find_sym_ctr_split(const Dataset& ds, const std::vector<std::uint32_t>& active_cat,
                   const std::vector<CtrProjection>& used_projs, const std::uint8_t* leaf_ids,
                   const std::uint32_t* rows, std::size_t n_root, bool is_contiguous,
                   std::size_t n_leaves, const float* g_seq, const float* h_seq,
                   const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
                   const double* parent_score, double lam, double min_h, std::int32_t min_c,
                   double min_gain, const Params& params, std::size_t tree_index, std::size_t depth,
                   int n_threads, SymmetricWorkspace& sym, const CtrGrowContext& ctr_ctx);

AxisSplitCandidate scan_sym_axis_feature(const Dataset& ds, std::uint32_t f, std::uint16_t nb,
                                         bool has_nan, std::size_t n_leaves, const HistBin* fh,
                                         const double* leaf_g, const double* leaf_h,
                                         const std::int32_t* leaf_c, const double* parent_score,
                                         double total_h, double lam, double min_h,
                                         std::int32_t min_c, double min_gain, const Params& params,
                                         std::size_t tree_index, int level) noexcept;

CatSplitCandidate scan_sym_cat_feature(std::uint32_t f, std::uint16_t nb, std::size_t n_leaves,
                                       const HistBin* fh, const double* leaf_g,
                                       const double* leaf_h, const std::int32_t* leaf_c,
                                       double total_h, double lam, double min_h, std::int32_t min_c,
                                       double min_gain, const Params& params) noexcept;

std::size_t collect_small_pair_rows(const std::uint8_t* leaf_ids, const std::uint32_t* rows,
                                    std::size_t n_root, bool is_contiguous, const float* g_seq,
                                    const float* h_seq, const std::int32_t* leaf_c,
                                    std::size_t n_leaves, int n_threads, SymmetricWorkspace& sym,
                                    std::uint8_t* small_bit_out) noexcept;

void build_sym_feature_hist(const std::uint8_t* col, const std::uint32_t* rows, std::size_t n_root,
                            bool is_contiguous, const std::uint8_t* leaf_ids, const float* g_seq,
                            const float* h_seq, std::size_t d, std::size_t depth_limit,
                            std::size_t n_leaves, std::size_t stride, bool allow_wave,
                            int n_threads, std::size_t n_small, const std::uint8_t* small_bit,
                            SymmetricWorkspace& sym, HistBin* hist_f, HistBin* prev_f) noexcept;

ObliqueSplitCandidate find_sym_oblique_split(
    const Dataset& ds, const std::uint32_t* leaf_rows, std::size_t n_root, bool is_contiguous,
    const std::uint8_t* leaf_ids, std::size_t n_leaves, const float* g_seq, const float* h_seq,
    const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
    const double* parent_score, std::size_t tree_index, int level, const Params& params,
    ObliqueScratch& scratch, SymmetricWorkspace& sym, const std::int16_t* codes_override = nullptr,
    std::size_t n_rows_override = 0U) noexcept;

ObliqueSplitCandidate find_sym_oblique_split_sampled(
    const Dataset& ds, const std::uint32_t* rows, std::size_t n_root, bool is_contiguous,
    const std::uint8_t* leaf_ids, std::size_t n_leaves, const float* g_seq, const float* h_seq,
    const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
    const double* parent_score, std::size_t tree_index, int level, const Params& params,
    ObliqueScratch& scratch, SymmetricWorkspace& sym);

void build_efb_bundles(const Dataset& ds, const Params& params, SymmetricWorkspace& sym,
                       std::size_t& total_bins);

void expand_efb_member(const SymmetricWorkspace& sym, std::uint32_t f, std::size_t stride,
                       std::size_t n_leaves, const double* leaf_g, const double* leaf_h,
                       const std::int32_t* leaf_c, HistBin* hist_f) noexcept;

// build_sym_feature_hist for 1..4 features at once (no chunk waves), bit-identical to calling it
// per feature with allow_wave = false.
void build_sym_feature_hist_group(std::size_t n_feat, const std::uint8_t* const* cols,
                                  const std::size_t* strides, HistBin* const* hist_fs,
                                  HistBin* const* prev_fs, const std::uint32_t* rows,
                                  std::size_t n_root, bool is_contiguous,
                                  const std::uint8_t* leaf_ids, const float* g_seq,
                                  const float* h_seq, std::size_t d, std::size_t depth_limit,
                                  std::size_t n_leaves, std::size_t n_small,
                                  const std::uint8_t* small_bit, SymmetricWorkspace& sym) noexcept;

} // namespace miter::sym_detail
