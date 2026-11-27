#include "core/symmetric_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/categorical.hpp"
#include "core/rng.hpp"
#include "core/symmetric_internal.hpp"
#include "core/tree_learner.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "util/fast_div.hpp"
#include "util/threading.hpp"

namespace miter {

using sym_detail::add_unique_ctr_proj;
using sym_detail::build_efb_bundles;
using sym_detail::build_sym_feature_hist;
using sym_detail::build_sym_feature_hist_group;
using sym_detail::collect_small_pair_rows;
using sym_detail::expand_efb_member;
using sym_detail::find_sym_ctr_split;
using sym_detail::find_sym_oblique_split;
using sym_detail::find_sym_oblique_split_sampled;
using sym_detail::kPurposeExtraTrees;
using sym_detail::kSymMaxLeaves;
using sym_detail::scan_sym_axis_feature;
using sym_detail::scan_sym_cat_feature;
using sym_detail::SymSplitChoice;

namespace {

template <typename Pred>
void advance_sym_leaves(const SymSplitChoice& /*unused*/, std::uint8_t* leaf_ids,
                        std::size_t n_root, std::size_t new_n_leaves, int n_jobs,
                        SymmetricWorkspace& sym, const float* g_seq, const float* h_seq,
                        double* leaf_g, double* leaf_h, std::int32_t* leaf_c, Pred pred) noexcept {
    const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
    const int n_threads = resolve_num_threads(n_jobs);
    if (n_threads > 1 && n_chunks > 1U && n_root >= 32768U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            double* cg = sym.chunk_leaf_g.data() + cu * new_n_leaves;
            double* ch = sym.chunk_leaf_h.data() + cu * new_n_leaves;
            std::int32_t* cc = sym.chunk_leaf_c.data() + cu * new_n_leaves;
            std::memset(static_cast<void*>(cg), 0, new_n_leaves * sizeof(double));
            std::memset(static_cast<void*>(ch), 0, new_n_leaves * sizeof(double));
            std::memset(static_cast<void*>(cc), 0, new_n_leaves * sizeof(std::int32_t));
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t new_l =
                    static_cast<std::uint8_t>((leaf_ids[t] << 1U) | (pred(t) ? 0U : 1U));
                leaf_ids[t] = new_l;
                cg[new_l] += static_cast<double>(g_seq[t]);
                ch[new_l] += static_cast<double>(h_seq[t]);
                cc[new_l] += 1;
            }
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t r0 = c * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            double* cg = sym.chunk_leaf_g.data() + c * new_n_leaves;
            double* ch = sym.chunk_leaf_h.data() + c * new_n_leaves;
            std::int32_t* cc = sym.chunk_leaf_c.data() + c * new_n_leaves;
            std::memset(static_cast<void*>(cg), 0, new_n_leaves * sizeof(double));
            std::memset(static_cast<void*>(ch), 0, new_n_leaves * sizeof(double));
            std::memset(static_cast<void*>(cc), 0, new_n_leaves * sizeof(std::int32_t));
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t new_l =
                    static_cast<std::uint8_t>((leaf_ids[t] << 1U) | (pred(t) ? 0U : 1U));
                leaf_ids[t] = new_l;
                cg[new_l] += static_cast<double>(g_seq[t]);
                ch[new_l] += static_cast<double>(h_seq[t]);
                cc[new_l] += 1;
            }
        }
    }

    std::memset(static_cast<void*>(leaf_g), 0, new_n_leaves * sizeof(double));
    std::memset(static_cast<void*>(leaf_h), 0, new_n_leaves * sizeof(double));
    std::memset(static_cast<void*>(leaf_c), 0, new_n_leaves * sizeof(std::int32_t));
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const double* cg = sym.chunk_leaf_g.data() + c * new_n_leaves;
        const double* ch = sym.chunk_leaf_h.data() + c * new_n_leaves;
        const std::int32_t* cc = sym.chunk_leaf_c.data() + c * new_n_leaves;
        for (std::size_t l = 0; l < new_n_leaves; ++l) {
            leaf_g[l] += cg[l];
            leaf_h[l] += ch[l];
            leaf_c[l] += cc[l];
        }
    }
}

} // namespace

void SymmetricWorkspace::prepare(const Dataset& ds, const Params& params) {
    const std::size_t n = ds.n_rows();
    samp_codes.clear();
    samp_codes_stride = 0U;
    const int d = (params.max_depth > 0) ? params.max_depth : 6;
    max_depth = static_cast<std::size_t>(std::clamp(d, 1, 8));
    max_level_leaves = std::size_t{1} << (max_depth - 1U);
    max_prev_leaves = max_level_leaves / 2U;
    max_threads = static_cast<std::size_t>(std::max(resolve_num_threads(params.n_jobs), 1));
    max_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;

    leaf_ids.resize(n);
    axis_thread_hists.resize(max_threads * max_level_leaves * 256U);
    oblique_thread_hists.resize(max_threads * max_level_leaves *
                                (static_cast<std::size_t>(kObliqueBins) + 1U));
    // Grouped histograms: 4 per thread (oblique pass2 and axis features).
    multi_hists.resize(max_threads * 4U * max_level_leaves *
                       (static_cast<std::size_t>(kObliqueBins) + 1U));
    axis_multi_hists.resize(max_threads * 4U * max_level_leaves * 256U);
    const std::size_t wave_slots = (max_chunks > 1U) ? std::min(max_chunks, kWaveChunks) : 0U;
    wave_hists.resize(wave_slots * max_level_leaves * 256U);

    const std::size_t p_feat = ds.n_features();
    prev_axis_offsets.assign(p_feat, 0U);
    std::size_t total_bins = 0U;
    for (std::size_t f = 0U; f < p_feat; ++f) {
        prev_axis_offsets[f] = total_bins;
        if (ds.feature_kinds()[f] == FeatureKind::Categorical) {
            const std::uint16_t nc = ds.cat_encoder(f).num_codes();
            if (nc >= 2U) {
                total_bins += static_cast<std::size_t>(nc);
            }
        } else {
            const auto& binner = ds.binner(f);
            if (binner.num_bins >= 2U || binner.has_nan) {
                total_bins +=
                    static_cast<std::size_t>(binner.num_bins) + (binner.has_nan ? 1U : 0U);
            }
        }
    }
    prev_axis_total_bins = total_bins;
    build_efb_bundles(ds, params, *this, total_bins);
    prev_axis_total_bins = total_bins;
    use_axis_sub = (max_prev_leaves > 0U) && (prev_axis_total_bins > 0U) &&
                   (prev_axis_total_bins * max_prev_leaves <= 2'000'000U);
    if (use_axis_sub) {
        prev_axis_hists.resize(prev_axis_total_bins * max_prev_leaves);
        const std::size_t max_small = n / 2U + 1U;
        small_rows.resize(max_small);
        small_pairs.resize(max_small);
        small_g.resize(max_small);
        small_h.resize(max_small);
        chunk_small_counts.resize(max_chunks);
        chunk_small_offs.resize(max_chunks + 1U);
    } else {
        prev_axis_hists.resize(0);
        small_rows.resize(0);
        small_pairs.resize(0);
        small_g.resize(0);
        small_h.resize(0);
        chunk_small_counts.clear();
        chunk_small_offs.clear();
    }

    const std::size_t final_leaves = std::size_t{1} << max_depth;
    chunk_leaf_g.resize(max_chunks * final_leaves);
    chunk_leaf_h.resize(max_chunks * final_leaves);
    chunk_leaf_c.resize(max_chunks * final_leaves);
}

Tree grow_symmetric_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                         std::size_t tree_index, TreeWorkspace& ws, std::size_t n_root,
                         bool bagging, double* train_scores, const CtrGrowContext* ctr_ctx) {
    const std::size_t n = ds.n_rows();
    SymmetricWorkspace& sym = ws.sym;
    const std::size_t depth_limit = sym.max_depth;
    const std::uint32_t* rows = ws.row_indices.data();
    const bool is_contiguous = !bagging;

    const float* g_seq = g;
    const float* h_seq = h;
    if (bagging) {
        for (std::size_t t = 0; t < n_root; ++t) {
            const std::uint32_t r = rows[t];
            ws.g_ordered[t] = g[r];
            ws.h_ordered[t] = h[r];
        }
        g_seq = ws.g_ordered.data();
        h_seq = ws.h_ordered.data();
    }

    std::uint8_t* leaf_ids = sym.leaf_ids.data();
    std::memset(static_cast<void*>(leaf_ids), 0, n_root * sizeof(std::uint8_t));

    double leaf_g[kSymMaxLeaves] = {};
    double leaf_h[kSymMaxLeaves] = {};
    std::int32_t leaf_c[kSymMaxLeaves] = {};
    {
        const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
        const int n_threads =
            std::min(resolve_num_threads(params.n_jobs), static_cast<int>(sym.max_threads));
        if (n_threads > 1 && n_chunks > 1U && n_root >= 32768U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                const auto cu = static_cast<std::size_t>(c);
                const std::size_t r0 = cu * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
                double gs = 0.0;
                double hs = 0.0;
                for (std::size_t t = r0; t < r1; ++t) {
                    gs += static_cast<double>(g_seq[t]);
                    hs += static_cast<double>(h_seq[t]);
                }
                ws.chunk_g[cu] = gs;
                ws.chunk_h[cu] = hs;
            }
        } else {
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t r0 = c * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
                double gs = 0.0;
                double hs = 0.0;
                for (std::size_t t = r0; t < r1; ++t) {
                    gs += static_cast<double>(g_seq[t]);
                    hs += static_cast<double>(h_seq[t]);
                }
                ws.chunk_g[c] = gs;
                ws.chunk_h[c] = hs;
            }
        }
        double gs = 0.0;
        double hs = 0.0;
        for (std::size_t c = 0; c < n_chunks; ++c) {
            gs += ws.chunk_g[c];
            hs += ws.chunk_h[c];
        }
        leaf_g[0] = gs;
        leaf_h[0] = hs;
        leaf_c[0] = static_cast<std::int32_t>(n_root);
    }

    const std::uint8_t* bins_base = ds.bins_col_major().data();
    const std::size_t p_num = ds.n_num_features();
    const std::int16_t* codes_base = (p_num > 0U) ? ds.codes_col_major().data() : nullptr;

    std::array<SymSplitChoice, 8> level_splits{};
    std::vector<CtrProjection> used_projs;
    const bool allow_ctr =
        (ctr_ctx != nullptr && ctr_ctx->state != nullptr && ctr_ctx->model != nullptr &&
         ctr_ctx->state->effective_max_complexity > 0 && ds.n_cat_features() > 0U);
    std::size_t actual_depth = 0U;
    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const int n_threads =
        std::min(resolve_num_threads(params.n_jobs), static_cast<int>(sym.max_threads));

    for (std::size_t d = 0U; d < depth_limit; ++d) {
        if (n_root < 2U) {
            break;
        }
        const std::size_t n_leaves = std::size_t{1} << d;
        double total_h = 0.0;
        double parent_score[kSymMaxLeaves] = {};
        for (std::size_t l = 0; l < n_leaves; ++l) {
            total_h += leaf_h[l];
            if (leaf_c[l] > 0 && (leaf_h[l] + lam) > 0.0) {
                parent_score[l] = (leaf_g[l] * leaf_g[l]) / (leaf_h[l] + lam);
            }
        }

        SymSplitChoice choice;
        const auto& active_num = ws.layout.active_num_features;
        const std::size_t n_num_act = active_num.size();
        const auto& active_cat = ws.layout.active_cat_features;
        const std::size_t n_cat_act = active_cat.size();
        std::uint8_t small_bit[kSymMaxLeaves / 2U] = {};
        std::size_t n_small = 0U;
        if (d >= 1U && sym.use_axis_sub && (n_num_act + n_cat_act > 0U)) {
            n_small = collect_small_pair_rows(leaf_ids, rows, n_root, is_contiguous, g_seq, h_seq,
                                              leaf_c, n_leaves, n_threads, sym, small_bit);
        }

        // 1. Axis-aligned numeric splits (bundled features first: one pass per bundle).
        if (n_num_act > 0U && !sym.efb_bundles.empty()) {
            for (std::size_t bi = 0U; bi < sym.efb_bundles.size(); ++bi) {
                const auto& bun = sym.efb_bundles[bi];
                bool any_active = false;
                for (const std::uint32_t f : bun.members) {
                    any_active =
                        any_active || std::binary_search(active_num.begin(), active_num.end(), f);
                }
                if (!any_active) {
                    continue;
                }
                HistBin* prev_b =
                    sym.use_axis_sub
                        ? (sym.prev_axis_hists.data() + bun.prev_offset * sym.max_prev_leaves)
                        : nullptr;
                build_sym_feature_hist(sym.efb_cols.data() + bi * n, rows, n_root, is_contiguous,
                                       leaf_ids, g_seq, h_seq, d, depth_limit, n_leaves, bun.n_bins,
                                       true, n_threads, n_small, small_bit, sym,
                                       sym.efb_hists.data() + bi * sym.max_level_leaves * 256U,
                                       prev_b);
            }
        }
        if (n_num_act == 1U) {
            HistBin* hist_f = sym.axis_thread_hists.data();
            const bool allow_wave = (n_num_act == 1U);
            for (const std::uint32_t f : active_num) {
                const std::uint16_t nb = ws.layout.num_regular_bins[f];
                const bool has_nan = ws.layout.has_nan[f] != 0U;
                const std::size_t stride = static_cast<std::size_t>(nb) + (has_nan ? 1U : 0U);
                HistBin* prev_f = sym.use_axis_sub
                                      ? (sym.prev_axis_hists.data() +
                                         sym.prev_axis_offsets[f] * sym.max_prev_leaves)
                                      : nullptr;
                const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n;
                if (!sym.efb_bundle_of.empty() && sym.efb_bundle_of[f] >= 0) {
                    expand_efb_member(sym, f, stride, n_leaves, leaf_g, leaf_h, leaf_c, hist_f);
                } else {
                    build_sym_feature_hist(col, rows, n_root, is_contiguous, leaf_ids, g_seq, h_seq,
                                           d, depth_limit, n_leaves, stride, allow_wave, n_threads,
                                           n_small, small_bit, sym, hist_f, prev_f);
                }
                const AxisSplitCandidate c = scan_sym_axis_feature(
                    ds, f, nb, has_nan, n_leaves, hist_f, leaf_g, leaf_h, leaf_c, parent_score,
                    total_h, lam, min_h, min_c, min_gain, params, tree_index, static_cast<int>(d));
                if (c.valid && c.gain > choice.axis.gain) {
                    choice.axis = c;
                }
            }
        } else if (n_num_act > 1U) {
            // Up to kAxisGroup consecutive regular features share one sweep over the rows
            // (build_sym_feature_hist_group, identical sums); an EFB member expands from its
            // bundle on its own. Candidates are compared in feature order, as before.
            constexpr std::size_t kAxisGroup = 4U;
            const auto is_efb = [&sym](std::uint32_t f) {
                return !sym.efb_bundle_of.empty() && sym.efb_bundle_of[f] >= 0;
            };
            const bool par = n_threads > 1 && n_root * n_num_act >= 20000U;
            // In parallel, smaller groups when there are too few features to balance the threads.
            const auto nt = static_cast<std::size_t>(n_threads);
            const std::size_t grp = !par                     ? kAxisGroup
                                    : (n_num_act >= 8U * nt) ? 4U
                                    : (n_num_act >= 4U * nt) ? 2U
                                                             : 1U;
            std::vector<std::pair<std::size_t, std::size_t>> groups;
            for (std::size_t i = 0; i < n_num_act;) {
                std::size_t len = 1U;
                if (!is_efb(active_num[i])) {
                    while (len < grp && i + len < n_num_act && !is_efb(active_num[i + len])) {
                        ++len;
                    }
                }
                groups.emplace_back(i, len);
                i += len;
            }
            const std::size_t hist_cap = sym.max_level_leaves * 256U;
            std::vector<AxisSplitCandidate> cand_res(n_num_act);
            const auto run_group = [&](std::size_t gidx, HistBin* bufs) noexcept {
                const std::size_t i0 = groups[gidx].first;
                const std::size_t len = groups[gidx].second;
                std::size_t strides[kAxisGroup] = {};
                const std::uint8_t* cols[kAxisGroup] = {};
                HistBin* hist_fs[kAxisGroup] = {};
                HistBin* prev_fs[kAxisGroup] = {};
                for (std::size_t gi = 0; gi < len; ++gi) {
                    const std::uint32_t f = active_num[i0 + gi];
                    const bool has_nan = ws.layout.has_nan[f] != 0U;
                    strides[gi] = static_cast<std::size_t>(ws.layout.num_regular_bins[f]) +
                                  (has_nan ? 1U : 0U);
                    cols[gi] = bins_base + static_cast<std::size_t>(f) * n;
                    hist_fs[gi] = bufs + gi * hist_cap;
                    prev_fs[gi] = sym.use_axis_sub
                                      ? (sym.prev_axis_hists.data() +
                                         sym.prev_axis_offsets[f] * sym.max_prev_leaves)
                                      : nullptr;
                }
                if (is_efb(active_num[i0])) {
                    expand_efb_member(sym, active_num[i0], strides[0], n_leaves, leaf_g, leaf_h,
                                      leaf_c, hist_fs[0]);
                } else {
                    build_sym_feature_hist_group(len, cols, strides, hist_fs, prev_fs, rows, n_root,
                                                 is_contiguous, leaf_ids, g_seq, h_seq, d,
                                                 depth_limit, n_leaves, n_small, small_bit, sym);
                }
                for (std::size_t gi = 0; gi < len; ++gi) {
                    const std::uint32_t f = active_num[i0 + gi];
                    cand_res[i0 + gi] = scan_sym_axis_feature(
                        ds, f, ws.layout.num_regular_bins[f], ws.layout.has_nan[f] != 0U, n_leaves,
                        hist_fs[gi], leaf_g, leaf_h, leaf_c, parent_score, total_h, lam, min_h,
                        min_c, min_gain, params, tree_index, static_cast<int>(d));
                }
            };
            if (par) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                for (std::ptrdiff_t gs = 0; gs < static_cast<std::ptrdiff_t>(groups.size()); ++gs) {
#ifdef _OPENMP
                    const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                    const std::size_t tid = 0U;
#endif
                    run_group(static_cast<std::size_t>(gs),
                              sym.axis_multi_hists.data() + tid * kAxisGroup * hist_cap);
                }
            } else {
                for (std::size_t gidx = 0; gidx < groups.size(); ++gidx) {
                    run_group(gidx, sym.axis_multi_hists.data());
                }
            }
            for (const AxisSplitCandidate& c : cand_res) {
                if (c.valid && c.gain > choice.axis.gain) {
                    choice.axis = c;
                }
            }
        }
        if (n_num_act > 0U) {
            if (choice.axis.valid) {
                choice.kind = NodeKind::Axis;
                choice.gain = choice.axis.gain;
                choice.valid = true;
            }
        }

        // 2. Categorical K1 splits.
        if (n_cat_act > 0U) {
            if (n_threads > 1 && n_cat_act >= 2U && n_root * n_cat_act >= 20000U) {
                std::vector<CatSplitCandidate> cand_res(n_cat_act);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                for (std::ptrdiff_t idx_s = 0; idx_s < static_cast<std::ptrdiff_t>(n_cat_act);
                     ++idx_s) {
                    const auto idx = static_cast<std::size_t>(idx_s);
                    const std::uint32_t f = active_cat[idx];
                    const std::uint16_t nb = ws.layout.num_regular_bins[f];
#ifdef _OPENMP
                    const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                    const std::size_t tid = 0U;
#endif
                    HistBin* hist_f =
                        sym.axis_thread_hists.data() + tid * sym.max_level_leaves * 256U;
                    HistBin* prev_f = sym.use_axis_sub
                                          ? (sym.prev_axis_hists.data() +
                                             sym.prev_axis_offsets[f] * sym.max_prev_leaves)
                                          : nullptr;
                    const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n;
                    build_sym_feature_hist(col, rows, n_root, is_contiguous, leaf_ids, g_seq, h_seq,
                                           d, depth_limit, n_leaves, nb, false, 1, n_small,
                                           small_bit, sym, hist_f, prev_f);
                    cand_res[idx] =
                        scan_sym_cat_feature(f, nb, n_leaves, hist_f, leaf_g, leaf_h, leaf_c,
                                             total_h, lam, min_h, min_c, min_gain, params);
                }
                for (const CatSplitCandidate& c : cand_res) {
                    if (c.valid && c.gain > choice.cat.gain) {
                        choice.cat = c;
                    }
                }
            } else {
                HistBin* hist_f = sym.axis_thread_hists.data();
                const bool allow_wave = (n_cat_act == 1U);
                for (const std::uint32_t f : active_cat) {
                    const std::uint16_t nb = ws.layout.num_regular_bins[f];
                    HistBin* prev_f = sym.use_axis_sub
                                          ? (sym.prev_axis_hists.data() +
                                             sym.prev_axis_offsets[f] * sym.max_prev_leaves)
                                          : nullptr;
                    const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n;
                    build_sym_feature_hist(col, rows, n_root, is_contiguous, leaf_ids, g_seq, h_seq,
                                           d, depth_limit, n_leaves, nb, allow_wave, n_threads,
                                           n_small, small_bit, sym, hist_f, prev_f);
                    const CatSplitCandidate c =
                        scan_sym_cat_feature(f, nb, n_leaves, hist_f, leaf_g, leaf_h, leaf_c,
                                             total_h, lam, min_h, min_c, min_gain, params);
                    if (c.valid && c.gain > choice.cat.gain) {
                        choice.cat = c;
                    }
                }
            }
            if (choice.cat.valid && (!choice.valid || choice.cat.gain > choice.gain)) {
                choice.kind = NodeKind::Categorical;
                choice.gain = choice.cat.gain;
                choice.valid = true;
            }
        }

        // 3. Oblique R_uni splits.
        const bool allow_oblique =
            (params.m_mult > 0) &&
            (params.oblique_max_depth < 0 || static_cast<int>(d) <= params.oblique_max_depth) &&
            (ws.oblique.cur_m > 0U);
        if (allow_oblique) {
            const bool sampled = params.oblique_sample_rows > 0 &&
                                 n_root > static_cast<std::size_t>(params.oblique_sample_rows);
            choice.oblique =
                sampled
                    ? find_sym_oblique_split_sampled(ds, rows, n_root, is_contiguous, leaf_ids,
                                                     n_leaves, g_seq, h_seq, leaf_g, leaf_h, leaf_c,
                                                     parent_score, tree_index, static_cast<int>(d),
                                                     params, ws.oblique, sym)
                    : find_sym_oblique_split(ds, rows, n_root, is_contiguous, leaf_ids, n_leaves,
                                             g_seq, h_seq, leaf_g, leaf_h, leaf_c, parent_score,
                                             tree_index, static_cast<int>(d), params, ws.oblique,
                                             sym);
            if (choice.oblique.valid && (!choice.valid || choice.oblique.gain > choice.gain)) {
                choice.kind = NodeKind::Oblique;
                choice.gain = choice.oblique.gain;
                choice.valid = true;
            }
        }

        // 4. Online ordered CTR splits (singles & dynamic category/numeric combinations).
        if (allow_ctr && n_cat_act > 0U) {
            choice.ctr = find_sym_ctr_split(ds, active_cat, used_projs, leaf_ids, rows, n_root,
                                            is_contiguous, n_leaves, g_seq, h_seq, leaf_g, leaf_h,
                                            leaf_c, parent_score, lam, min_h, min_c, min_gain,
                                            params, tree_index, d, n_threads, sym, *ctr_ctx);
            if (choice.ctr.valid && (!choice.valid || choice.ctr.gain > choice.gain)) {
                choice.kind = NodeKind::Ctr;
                choice.gain = choice.ctr.gain;
                choice.valid = true;
            }
        }

        if (!choice.valid) {
            break;
        }
        if (choice.kind == NodeKind::Categorical) {
            add_unique_ctr_proj(used_projs, CtrProjection{{choice.cat.feature}});
        } else if (choice.kind == NodeKind::Ctr) {
            choice.ctr_id = ctr_ctx->state->register_ctr(choice.ctr.proj, choice.ctr.kind,
                                                         ctr_ctx->target_id, choice.ctr.prior_id,
                                                         choice.ctr.prior_value, *ctr_ctx->model);
            add_unique_ctr_proj(used_projs, choice.ctr.proj);
        }
        level_splits[d] = choice;
        actual_depth = d + 1U;

        const std::size_t next_leaves = n_leaves * 2U;
        if (choice.kind == NodeKind::Axis) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(choice.axis.feature) * n;
            const auto& binner = ds.binner(choice.axis.feature);
            const std::uint8_t thr_bin = choice.axis.bin_idx;
            if (!binner.has_nan) {
                advance_sym_leaves(choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq,
                                   h_seq, leaf_g, leaf_h, leaf_c,
                                   [col, rows, is_contiguous, thr_bin](std::size_t t) noexcept {
                                       return col[is_contiguous ? t : rows[t]] <= thr_bin;
                                   });
            } else {
                const std::uint8_t nan_bin = binner.nan_bin();
                const bool nan_left = choice.axis.nan_go_left;
                advance_sym_leaves(
                    choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq, h_seq, leaf_g,
                    leaf_h, leaf_c,
                    [col, rows, is_contiguous, thr_bin, nan_bin, nan_left](std::size_t t) noexcept {
                        const std::uint8_t b = col[is_contiguous ? t : rows[t]];
                        return (b == nan_bin) ? nan_left : (b <= thr_bin);
                    });
            }
        } else if (choice.kind == NodeKind::Categorical) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(choice.cat.feature) * n;
            const CatBitset bs = choice.cat.left_bitset;
            advance_sym_leaves(choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq,
                               h_seq, leaf_g, leaf_h, leaf_c,
                               [col, rows, is_contiguous, &bs](std::size_t t) noexcept {
                                   return cat_bitset_test(bs, col[is_contiguous ? t : rows[t]]);
                               });
        } else if (choice.kind == NodeKind::Ctr) {
            const std::uint8_t* win_bins = ctr_ctx->state->win_ctr_bins.data();
            const std::uint8_t win_b = ctr_ctx->state->win_ctr_bin_idx;
            advance_sym_leaves(choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq,
                               h_seq, leaf_g, leaf_h, leaf_c,
                               [win_bins, rows, is_contiguous, win_b](std::size_t t) noexcept {
                                   return win_bins[is_contiguous ? t : rows[t]] <= win_b;
                               });
        } else {
            const auto& ob = choice.oblique;
            if (ob.k <= 3U) {
                const std::int16_t* c0 =
                    codes_base + static_cast<std::size_t>(ob.num_indices[0]) * n;
                const std::int16_t* c1 =
                    codes_base + static_cast<std::size_t>(ob.num_indices[1]) * n;
                const std::int16_t* c2 =
                    (ob.k == 3U) ? (codes_base + static_cast<std::size_t>(ob.num_indices[2]) * n)
                                 : c0;
                const std::int32_t w0 = ob.weights[0];
                const std::int32_t w1 = ob.weights[1];
                const std::int32_t w2 = (ob.k == 3U) ? ob.weights[2] : 0;
                const auto thr = static_cast<std::int32_t>(ob.threshold);
                const bool nan_left = ob.nan_go_left;
                advance_sym_leaves(choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq,
                                   h_seq, leaf_g, leaf_h, leaf_c,
                                   [c0, c1, c2, rows, is_contiguous, w0, w1, w2, thr,
                                    nan_left](std::size_t t) noexcept {
                                       const std::size_t idx = is_contiguous ? t : rows[t];
                                       const std::int16_t x0 = c0[idx];
                                       const std::int16_t x1 = c1[idx];
                                       const std::int16_t x2 = c2[idx];
                                       if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode ||
                                           x2 == kQ16SignedNanCode) {
                                           return nan_left;
                                       }
                                       return (w0 * x0 + w1 * x1 + w2 * x2) < thr;
                                   });
            } else {
                const std::int64_t thr = ob.threshold;
                const bool nan_left = ob.nan_go_left;
                advance_sym_leaves(
                    choice, leaf_ids, n_root, next_leaves, params.n_jobs, sym, g_seq, h_seq, leaf_g,
                    leaf_h, leaf_c,
                    [codes_base, n, rows, is_contiguous, &ob, thr,
                     nan_left](std::size_t t) noexcept {
                        const std::size_t idx = is_contiguous ? t : rows[t];
                        std::int64_t acc = 0;
                        for (std::uint8_t q = 0; q < ob.k; ++q) {
                            const std::int16_t x =
                                codes_base[static_cast<std::size_t>(ob.num_indices[q]) * n + idx];
                            if (x == kQ16SignedNanCode) {
                                return nan_left;
                            }
                            acc += static_cast<std::int64_t>(ob.weights[q]) *
                                   static_cast<std::int64_t>(x);
                        }
                        return acc < thr;
                    });
            }
        }
    }

    // Materialize complete binary tree of depth actual_depth.
    const std::size_t n_final_leaves = std::size_t{1} << actual_depth;
    Tree tree;
    tree.n_leaves = static_cast<std::uint32_t>(n_final_leaves);
    tree.nodes.resize(2U * n_final_leaves - 1U);

    for (std::size_t d = 0U; d < actual_depth; ++d) {
        const SymSplitChoice& sp = level_splits[d];
        std::uint32_t shared_offset = 0U;
        if (sp.kind == NodeKind::Categorical) {
            shared_offset = static_cast<std::uint32_t>(tree.cat_bitsets.size());
            tree.cat_bitsets.push_back(sp.cat.left_bitset);
        } else if (sp.kind == NodeKind::Oblique && sp.oblique.k > 3U) {
            shared_offset = static_cast<std::uint32_t>(tree.oblique_overflow.size());
            ObliqueOverflow ov;
            ov.features.assign(sp.oblique.features, sp.oblique.features + sp.oblique.k);
            ov.weights.assign(sp.oblique.weights, sp.oblique.weights + sp.oblique.k);
            ov.threshold_i64 = sp.oblique.threshold;
            tree.oblique_overflow.push_back(std::move(ov));
        }

        const std::size_t level_start = (std::size_t{1} << d) - 1U;
        const std::size_t level_cnt = std::size_t{1} << d;
        for (std::size_t k = 0U; k < level_cnt; ++k) {
            const std::size_t u = level_start + k;
            Node& nd = tree.nodes[u];
            nd.kind = sp.kind;
            nd.left_child = static_cast<std::uint32_t>(2U * u + 1U);
            nd.right_child = static_cast<std::uint32_t>(2U * u + 2U);
            nd.split_gain = (k == 0U) ? sp.gain : 0.0;
            if (sp.kind == NodeKind::Axis) {
                nd.nan_go_left = sp.axis.nan_go_left;
                nd.feature = sp.axis.feature;
                nd.threshold_u16 = sp.axis.threshold_u16;
            } else if (sp.kind == NodeKind::Categorical) {
                nd.nan_go_left = sp.cat.default_go_left;
                nd.feature = sp.cat.feature;
                nd.cat_bitset_offset = shared_offset;
            } else if (sp.kind == NodeKind::Ctr) {
                nd.feature = sp.ctr_id;
                nd.ctr_threshold = sp.ctr.threshold;
            } else {
                const auto& ob = sp.oblique;
                nd.nan_go_left = ob.nan_go_left;
                nd.oblique_k = ob.k;
                if (ob.k <= 3U) {
                    for (std::uint8_t q = 0; q < ob.k; ++q) {
                        nd.oblique_features[q] = ob.features[q];
                        nd.oblique_weights[q] = ob.weights[q];
                    }
                    nd.threshold_i32 = static_cast<std::int32_t>(ob.threshold);
                } else {
                    nd.overflow_offset = shared_offset;
                }
            }
        }
    }

    double leaf_vals[kSymMaxLeaves] = {};
    const double lr = params.learning_rate;
    for (std::size_t l = 0; l < n_final_leaves; ++l) {
        const double denom = leaf_h[l] + lam;
        const double val = (leaf_c[l] > 0 && denom > 0.0) ? (lr * (-leaf_g[l] / denom)) : 0.0;
        leaf_vals[l] = val;
        Node& ln = tree.nodes[(n_final_leaves - 1U) + l];
        ln.kind = NodeKind::Leaf;
        ln.leaf_value = val;
    }

    if (train_scores != nullptr) {
        if (!bagging) {
            if (n_threads > 1 && n_root >= 32768U) {
                const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                    const std::size_t t0 = static_cast<std::size_t>(c) * kDetChunkRows;
                    const std::size_t t1 = std::min(t0 + kDetChunkRows, n_root);
                    for (std::size_t t = t0; t < t1; ++t) {
                        train_scores[t] += leaf_vals[leaf_ids[t]];
                    }
                }
            } else {
                for (std::size_t t = 0; t < n_root; ++t) {
                    train_scores[t] += leaf_vals[leaf_ids[t]];
                }
            }
        } else {
            for (std::size_t t = 0; t < n_root; ++t) {
                train_scores[rows[t]] += leaf_vals[leaf_ids[t]];
            }
            if (!ws.oob_rows.empty()) {
                if (allow_ctr) {
                    finalize_pending_ctr_tables(ds, *ctr_ctx->state, ctr_ctx->target_y,
                                                ctr_ctx->prior, ctr_ctx->sample_weight,
                                                params.ctr_prior_weight, ctr_ctx->target_id,
                                                *ctr_ctx->model, ctr_ctx->state->scratch);
                }
                Evaluator::add_tree_predictions_from_codes(
                    tree, codes_base, n, p_num, ws.feature_to_num_idx, train_scores, bins_base,
                    params.n_jobs, ws.oob_rows.data(), ws.oob_rows.size(),
                    allow_ctr ? ctr_ctx->model : nullptr,
                    allow_ctr ? ds.cat_hashes_col_major().data() : nullptr,
                    allow_ctr ? &ws.feature_to_cat_idx : nullptr);
            }
        }
    }

    return tree;
}

} // namespace miter
