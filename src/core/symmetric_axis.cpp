// Symmetric trees: axis-aligned and categorical histograms and split scans.

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

namespace miter::sym_detail {

namespace {

struct CatGroup {
    double g = 0.0;
    double h = 0.0;
    double score = 0.0;
    std::int32_t c = 0;
    std::int16_t bin = -1;
};

void build_axis_hist_feat(const std::uint8_t* col, const std::uint32_t* leaf_rows, std::size_t r0,
                          std::size_t r1, bool is_contiguous, const std::uint8_t* leaf_ids,
                          std::size_t stride, const float* g_seq, const float* h_seq,
                          HistBin* hist_out) noexcept {
    if (is_contiguous) {
        std::size_t t = r0;
        for (; t + 4U <= r1; t += 4U) {
            const std::size_t idx0 = static_cast<std::size_t>(leaf_ids[t]) * stride + col[t];
            const std::size_t idx1 =
                static_cast<std::size_t>(leaf_ids[t + 1U]) * stride + col[t + 1U];
            const std::size_t idx2 =
                static_cast<std::size_t>(leaf_ids[t + 2U]) * stride + col[t + 2U];
            const std::size_t idx3 =
                static_cast<std::size_t>(leaf_ids[t + 3U]) * stride + col[t + 3U];
            hist_out[idx0].g += static_cast<double>(g_seq[t]);
            hist_out[idx0].h += static_cast<double>(h_seq[t]);
            hist_out[idx0].c += 1;
            hist_out[idx1].g += static_cast<double>(g_seq[t + 1U]);
            hist_out[idx1].h += static_cast<double>(h_seq[t + 1U]);
            hist_out[idx1].c += 1;
            hist_out[idx2].g += static_cast<double>(g_seq[t + 2U]);
            hist_out[idx2].h += static_cast<double>(h_seq[t + 2U]);
            hist_out[idx2].c += 1;
            hist_out[idx3].g += static_cast<double>(g_seq[t + 3U]);
            hist_out[idx3].h += static_cast<double>(h_seq[t + 3U]);
            hist_out[idx3].c += 1;
        }
        for (; t < r1; ++t) {
            const std::size_t idx = static_cast<std::size_t>(leaf_ids[t]) * stride + col[t];
            hist_out[idx].g += static_cast<double>(g_seq[t]);
            hist_out[idx].h += static_cast<double>(h_seq[t]);
            hist_out[idx].c += 1;
        }
    } else {
        std::size_t t = r0;
        for (; t + 4U <= r1; t += 4U) {
            const std::size_t idx0 =
                static_cast<std::size_t>(leaf_ids[t]) * stride + col[leaf_rows[t]];
            const std::size_t idx1 =
                static_cast<std::size_t>(leaf_ids[t + 1U]) * stride + col[leaf_rows[t + 1U]];
            const std::size_t idx2 =
                static_cast<std::size_t>(leaf_ids[t + 2U]) * stride + col[leaf_rows[t + 2U]];
            const std::size_t idx3 =
                static_cast<std::size_t>(leaf_ids[t + 3U]) * stride + col[leaf_rows[t + 3U]];
            hist_out[idx0].g += static_cast<double>(g_seq[t]);
            hist_out[idx0].h += static_cast<double>(h_seq[t]);
            hist_out[idx0].c += 1;
            hist_out[idx1].g += static_cast<double>(g_seq[t + 1U]);
            hist_out[idx1].h += static_cast<double>(h_seq[t + 1U]);
            hist_out[idx1].c += 1;
            hist_out[idx2].g += static_cast<double>(g_seq[t + 2U]);
            hist_out[idx2].h += static_cast<double>(h_seq[t + 2U]);
            hist_out[idx2].c += 1;
            hist_out[idx3].g += static_cast<double>(g_seq[t + 3U]);
            hist_out[idx3].h += static_cast<double>(h_seq[t + 3U]);
            hist_out[idx3].c += 1;
        }
        for (; t < r1; ++t) {
            const std::size_t idx =
                static_cast<std::size_t>(leaf_ids[t]) * stride + col[leaf_rows[t]];
            hist_out[idx].g += static_cast<double>(g_seq[t]);
            hist_out[idx].h += static_cast<double>(h_seq[t]);
            hist_out[idx].c += 1;
        }
    }
}

void build_axis_hist_small_pairs(const std::uint8_t* col, const std::uint32_t* small_rows,
                                 const std::uint8_t* small_pairs, const float* small_g,
                                 const float* small_h, std::size_t s0, std::size_t s1,
                                 std::size_t stride, HistBin* small_hist_out) noexcept {
    std::size_t s = s0;
    for (; s + 4U <= s1; s += 4U) {
        const std::size_t idx0 =
            static_cast<std::size_t>(small_pairs[s]) * stride + col[small_rows[s]];
        const std::size_t idx1 =
            static_cast<std::size_t>(small_pairs[s + 1U]) * stride + col[small_rows[s + 1U]];
        const std::size_t idx2 =
            static_cast<std::size_t>(small_pairs[s + 2U]) * stride + col[small_rows[s + 2U]];
        const std::size_t idx3 =
            static_cast<std::size_t>(small_pairs[s + 3U]) * stride + col[small_rows[s + 3U]];
        small_hist_out[idx0].g += static_cast<double>(small_g[s]);
        small_hist_out[idx0].h += static_cast<double>(small_h[s]);
        small_hist_out[idx0].c += 1;
        small_hist_out[idx1].g += static_cast<double>(small_g[s + 1U]);
        small_hist_out[idx1].h += static_cast<double>(small_h[s + 1U]);
        small_hist_out[idx1].c += 1;
        small_hist_out[idx2].g += static_cast<double>(small_g[s + 2U]);
        small_hist_out[idx2].h += static_cast<double>(small_h[s + 2U]);
        small_hist_out[idx2].c += 1;
        small_hist_out[idx3].g += static_cast<double>(small_g[s + 3U]);
        small_hist_out[idx3].h += static_cast<double>(small_h[s + 3U]);
        small_hist_out[idx3].c += 1;
    }
    for (; s < s1; ++s) {
        const std::size_t idx =
            static_cast<std::size_t>(small_pairs[s]) * stride + col[small_rows[s]];
        small_hist_out[idx].g += static_cast<double>(small_g[s]);
        small_hist_out[idx].h += static_cast<double>(small_h[s]);
        small_hist_out[idx].c += 1;
    }
}

// G features' histograms in one sweep over the rows (or the small-child rows): each histogram is
// filled in the same row order as build_axis_hist_feat / build_axis_hist_small_pairs, so the sums
// are identical; the G independent read-modify-write chains overlap.
template <std::size_t G>
void build_axis_hist_multi(const std::uint8_t* const* cols, const std::uint32_t* leaf_rows,
                           std::size_t n_root, bool is_contiguous, const std::uint8_t* leaf_ids,
                           const std::size_t* strides, const float* g_seq, const float* h_seq,
                           HistBin* const* hists) noexcept {
    for (std::size_t t = 0; t < n_root; ++t) {
        const std::size_t r = is_contiguous ? t : leaf_rows[t];
        const std::size_t l = leaf_ids[t];
        const double gv = static_cast<double>(g_seq[t]);
        const double hv = static_cast<double>(h_seq[t]);
        for (std::size_t gi = 0; gi < G; ++gi) {
            HistBin& dst = hists[gi][l * strides[gi] + cols[gi][r]];
            dst.g += gv;
            dst.h += hv;
            dst.c += 1;
        }
    }
}

template <std::size_t G>
void build_axis_hist_small_pairs_multi(const std::uint8_t* const* cols,
                                       const std::uint32_t* small_rows,
                                       const std::uint8_t* small_pairs, const float* small_g,
                                       const float* small_h, std::size_t n_small,
                                       const std::size_t* strides, HistBin* const* hists) noexcept {
    for (std::size_t s = 0; s < n_small; ++s) {
        const std::size_t r = small_rows[s];
        const std::size_t pr = small_pairs[s];
        const double gv = static_cast<double>(small_g[s]);
        const double hv = static_cast<double>(small_h[s]);
        for (std::size_t gi = 0; gi < G; ++gi) {
            HistBin& dst = hists[gi][pr * strides[gi] + cols[gi][r]];
            dst.g += gv;
            dst.h += hv;
            dst.c += 1;
        }
    }
}

void reconstruct_sym_hist_from_small(HistBin* hist_f, const HistBin* prev_f,
                                     const std::uint8_t* small_bit, std::size_t n_pairs,
                                     std::size_t stride) noexcept {
    // Reconstruct right-to-left so in-place writes to 2*p / 2*p+1 never clobber unprocessed pairs.
    for (std::size_t p = n_pairs; p > 0U; --p) {
        const std::size_t pp = p - 1U;
        const std::size_t sb = small_bit[pp];
        const std::size_t small_l = 2U * pp + sb;
        const std::size_t large_l = 2U * pp + (1U - sb);
        const HistBin* src_small = hist_f + pp * stride;
        const HistBin* src_prev = prev_f + pp * stride;
        HistBin* dst_small = hist_f + small_l * stride;
        HistBin* dst_large = hist_f + large_l * stride;
        for (std::size_t b = 0U; b < stride; ++b) {
            const HistBin sh = src_small[b];
            const HistBin ph = src_prev[b];
            dst_small[b] = sh;
            dst_large[b] = HistBin{ph.g - sh.g, ph.h - sh.h, ph.c - sh.c};
        }
    }
}

} // namespace

AxisSplitCandidate scan_sym_axis_feature(const Dataset& ds, std::uint32_t f, std::uint16_t nb,
                                         bool has_nan, std::size_t n_leaves, const HistBin* fh,
                                         const double* leaf_g, const double* leaf_h,
                                         const std::int32_t* leaf_c, const double* parent_score,
                                         double total_h, double lam, double min_h,
                                         std::int32_t min_c, double min_gain, const Params& params,
                                         std::size_t tree_index, int level) noexcept {
    AxisSplitCandidate cand;
    if (nb < 2U && !has_nan) {
        return cand;
    }
    const std::size_t stride = static_cast<std::size_t>(nb) + (has_nan ? 1U : 0U);
    const auto& edges = ds.binner(f).edges;

    std::int32_t total_c_nan = 0;
    double total_h_nan = 0.0;
    if (has_nan) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            total_c_nan += fh[l * stride + nb].c;
            total_h_nan += fh[l * stride + nb].h;
        }
    }
    const bool active_nan = has_nan && (total_c_nan > 0);
    if (nb < 2U && !active_nan) {
        return cand;
    }

    double gl[kSymMaxLeaves] = {};
    double hl[kSymMaxLeaves] = {};
    std::int32_t cl[kSymMaxLeaves] = {};
    double gl_with_nan[kSymMaxLeaves] = {};
    double hl_with_nan[kSymMaxLeaves] = {};
    std::int32_t cl_with_nan[kSymMaxLeaves] = {};

    if (params.extra_trees) {
        struct Pos {
            double gain;
            std::uint16_t bin;
            bool nan_left;
            bool final_nan;
        };
        std::array<Pos, 257> pos{};
        std::size_t nv = 0U;
        double cum_hl = 0.0;

        for (std::uint16_t b = 0; b + 1U < nb; ++b) {
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& hb = fh[l * stride + b];
                gl[l] += hb.g;
                hl[l] += hb.h;
                cl[l] += hb.c;
                cum_hl += hb.h;
            }
            if (!active_nan) {
                double g_val = 0.0;
                if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                         lam, min_h, min_c, min_gain, g_val)) {
                    pos[nv++] = Pos{g_val, b, (cum_hl >= (total_h - cum_hl)), false};
                }
            } else {
                for (std::size_t l = 0; l < n_leaves; ++l) {
                    const HistBin& nb_nan = fh[l * stride + nb];
                    gl_with_nan[l] = gl[l] + nb_nan.g;
                    hl_with_nan[l] = hl[l] + nb_nan.h;
                    cl_with_nan[l] = cl[l] + nb_nan.c;
                }
                double gain_a = 0.0;
                double gain_b = 0.0;
                const bool ok_a = eval_multi_leaf_gain(
                    gl_with_nan, hl_with_nan, cl_with_nan, leaf_g, leaf_h, leaf_c, parent_score,
                    n_leaves, lam, min_h, min_c, min_gain, gain_a);
                const bool ok_b =
                    eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                         lam, min_h, min_c, min_gain, gain_b);
                if (ok_a && (!ok_b || gain_a >= gain_b)) {
                    pos[nv++] = Pos{gain_a, b, true, false};
                } else if (ok_b) {
                    pos[nv++] = Pos{gain_b, b, false, false};
                }
            }
        }
        if (active_nan) {
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& hb = fh[l * stride + (nb - 1U)];
                gl[l] += hb.g;
                hl[l] += hb.h;
                cl[l] += hb.c;
            }
            double g_val = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                     lam, min_h, min_c, min_gain, g_val)) {
                pos[nv++] = Pos{g_val, static_cast<std::uint16_t>(nb - 1U), false, true};
            }
        }
        if (nv == 0U) {
            return cand;
        }
        CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index),
                       static_cast<std::uint64_t>(level), static_cast<std::uint64_t>(f),
                       kPurposeExtraTrees);
        const Pos& chosen = pos[static_cast<std::size_t>(rng.next_u64() % nv)];
        cand.gain = chosen.gain;
        cand.feature = f;
        cand.bin_idx = static_cast<std::uint8_t>(chosen.bin);
        cand.threshold_u16 = chosen.final_nan ? kQ16NanCode : edges[chosen.bin];
        cand.nan_go_left = chosen.nan_left;
        cand.valid = true;
        return cand;
    }

    double cum_hl = 0.0;
    for (std::uint16_t b = 0; b + 1U < nb; ++b) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            const HistBin& hb = fh[l * stride + b];
            gl[l] += hb.g;
            hl[l] += hb.h;
            cl[l] += hb.c;
            cum_hl += hb.h;
        }
        if (!active_nan) {
            double gain = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                     lam, min_h, min_c, min_gain, gain) &&
                gain > cand.gain) {
                cand.gain = gain;
                cand.feature = f;
                cand.bin_idx = static_cast<std::uint8_t>(b);
                cand.threshold_u16 = edges[b];
                cand.nan_go_left = (cum_hl >= (total_h - cum_hl));
                cand.valid = true;
            }
        } else {
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& nb_nan = fh[l * stride + nb];
                gl_with_nan[l] = gl[l] + nb_nan.g;
                hl_with_nan[l] = hl[l] + nb_nan.h;
                cl_with_nan[l] = cl[l] + nb_nan.c;
            }
            double gain_a = 0.0;
            if (eval_multi_leaf_gain(gl_with_nan, hl_with_nan, cl_with_nan, leaf_g, leaf_h, leaf_c,
                                     parent_score, n_leaves, lam, min_h, min_c, min_gain, gain_a) &&
                gain_a > cand.gain) {
                cand.gain = gain_a;
                cand.feature = f;
                cand.bin_idx = static_cast<std::uint8_t>(b);
                cand.threshold_u16 = edges[b];
                cand.nan_go_left = true;
                cand.valid = true;
            }
            double gain_b = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                     lam, min_h, min_c, min_gain, gain_b) &&
                gain_b > cand.gain) {
                cand.gain = gain_b;
                cand.feature = f;
                cand.bin_idx = static_cast<std::uint8_t>(b);
                cand.threshold_u16 = edges[b];
                cand.nan_go_left = false;
                cand.valid = true;
            }
        }
    }

    if (active_nan) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            const HistBin& hb = fh[l * stride + (nb - 1U)];
            gl[l] += hb.g;
            hl[l] += hb.h;
            cl[l] += hb.c;
        }
        double gain = 0.0;
        if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves, lam,
                                 min_h, min_c, min_gain, gain) &&
            gain > cand.gain) {
            cand.gain = gain;
            cand.feature = f;
            cand.bin_idx = static_cast<std::uint8_t>(nb - 1U);
            cand.threshold_u16 = kQ16NanCode;
            cand.nan_go_left = false;
            cand.valid = true;
        }
    }
    (void)total_h_nan;
    return cand;
}

CatSplitCandidate scan_sym_cat_feature(std::uint32_t f, std::uint16_t nb, std::size_t n_leaves,
                                       const HistBin* fh, const double* leaf_g,
                                       const double* leaf_h, const std::int32_t* leaf_c,
                                       double total_h, double lam, double min_h, std::int32_t min_c,
                                       double min_gain, const Params& params) noexcept {
    CatSplitCandidate cand;
    double tot_g[256] = {};
    double tot_h[256] = {};
    std::int32_t tot_c[256] = {};
    int n_used = 0;
    for (std::uint16_t b = 0; b < nb; ++b) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            const HistBin& hb = fh[l * nb + b];
            tot_g[b] += hb.g;
            tot_h[b] += hb.h;
            tot_c[b] += hb.c;
        }
        if (tot_c[b] > 0) {
            ++n_used;
        }
    }
    if (n_used < 2) {
        return cand;
    }

    if (n_used <= params.max_cat_to_onehot) {
        double pscore[kSymMaxLeaves] = {};
        for (std::size_t l = 0; l < n_leaves; ++l) {
            pscore[l] = (leaf_c[l] > 0 && (leaf_h[l] + lam) > 0.0)
                            ? (leaf_g[l] * leaf_g[l]) / (leaf_h[l] + lam)
                            : 0.0;
        }
        double gl[kSymMaxLeaves];
        double hl[kSymMaxLeaves];
        std::int32_t cl[kSymMaxLeaves];
        std::int32_t best_b = -1;
        double best_hl = 0.0;
        for (std::uint16_t b = 0; b < nb; ++b) {
            if (tot_c[b] <= 0) {
                continue;
            }
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& hb = fh[l * nb + b];
                gl[l] = hb.g;
                hl[l] = hb.h;
                cl[l] = hb.c;
            }
            double gain = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, pscore, n_leaves, lam,
                                     min_h, min_c, min_gain, gain) &&
                gain > cand.gain) {
                cand.gain = gain;
                best_b = static_cast<std::int32_t>(b);
                best_hl = tot_h[b];
            }
        }
        if (best_b >= 0) {
            cand.feature = f;
            cand.default_go_left = (best_hl >= (total_h - best_hl));
            cand.left_bitset = CatBitset{};
            for (std::uint16_t b = 0; b < nb; ++b) {
                if (static_cast<std::int32_t>(b) == best_b ||
                    (tot_c[b] <= 0 && cand.default_go_left)) {
                    cat_bitset_set(cand.left_bitset, static_cast<std::uint8_t>(b));
                }
            }
            cand.valid = true;
        }
        return cand;
    }

    CatGroup groups[256];
    int n_groups = 0;
    std::int32_t rare_c = 0;
    double rare_g = 0.0;
    double rare_h = 0.0;
    const std::int32_t min_grp = std::max(params.min_data_per_group, 1);

    for (std::uint16_t b = 0; b < nb; ++b) {
        if (tot_c[b] <= 0) {
            continue;
        }
        if (tot_c[b] >= min_grp) {
            groups[n_groups++] =
                CatGroup{tot_g[b], tot_h[b], 0.0, tot_c[b], static_cast<std::int16_t>(b)};
        } else {
            rare_g += tot_g[b];
            rare_h += tot_h[b];
            rare_c += tot_c[b];
        }
    }
    if (rare_c > 0) {
        groups[n_groups++] = CatGroup{rare_g, rare_h, 0.0, rare_c, -1};
    }
    if (n_groups < 2) {
        return cand;
    }

    const double smooth = params.cat_smooth;
    for (int q = 0; q < n_groups; ++q) {
        const double denom = groups[q].h + smooth;
        groups[q].score = (denom > 0.0) ? (groups[q].g / denom) : 0.0;
    }
    std::sort(groups, groups + n_groups, [](const CatGroup& a, const CatGroup& b) {
        if (a.score != b.score) {
            return a.score < b.score;
        }
        return a.bin < b.bin;
    });

    const double lam2 = lam + params.cat_l2;
    double pscore2[kSymMaxLeaves] = {};
    for (std::size_t l = 0; l < n_leaves; ++l) {
        pscore2[l] = (leaf_c[l] > 0 && (leaf_h[l] + lam2) > 0.0)
                         ? (leaf_g[l] * leaf_g[l]) / (leaf_h[l] + lam2)
                         : 0.0;
    }

    const int lim = std::min(std::max(params.max_cat_threshold, 1), n_groups - 1);
    int best_dir = -1;
    int best_t = 0;
    double best_hl = 0.0;

    for (int dir = 0; dir < 2; ++dir) {
        double gl[kSymMaxLeaves] = {};
        double hl[kSymMaxLeaves] = {};
        std::int32_t cl[kSymMaxLeaves] = {};
        double cum_hl = 0.0;
        for (int t = 1; t <= lim; ++t) {
            const int q = (dir == 0) ? (t - 1) : (n_groups - t);
            const std::int16_t bin = groups[q].bin;
            if (bin >= 0) {
                for (std::size_t l = 0; l < n_leaves; ++l) {
                    const HistBin& hb = fh[l * nb + static_cast<std::size_t>(bin)];
                    gl[l] += hb.g;
                    hl[l] += hb.h;
                    cl[l] += hb.c;
                }
                cum_hl += groups[q].h;
            } else {
                for (std::uint16_t b = 0; b < nb; ++b) {
                    if (tot_c[b] > 0 && tot_c[b] < min_grp) {
                        for (std::size_t l = 0; l < n_leaves; ++l) {
                            const HistBin& hb = fh[l * nb + b];
                            gl[l] += hb.g;
                            hl[l] += hb.h;
                            cl[l] += hb.c;
                        }
                    }
                }
                cum_hl += rare_h;
            }
            double gain = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, pscore2, n_leaves, lam2,
                                     min_h, min_c, min_gain, gain) &&
                gain > cand.gain) {
                cand.gain = gain;
                best_dir = dir;
                best_t = t;
                best_hl = cum_hl;
            }
        }
    }

    if (best_dir >= 0) {
        cand.feature = f;
        cand.default_go_left = (best_hl >= (total_h - best_hl));
        cand.left_bitset = CatBitset{};
        bool rare_left = false;
        for (int t = 1; t <= best_t; ++t) {
            const int q = (best_dir == 0) ? (t - 1) : (n_groups - t);
            if (groups[q].bin < 0) {
                rare_left = true;
            } else {
                cat_bitset_set(cand.left_bitset, static_cast<std::uint8_t>(groups[q].bin));
            }
        }
        for (std::uint16_t b = 0; b < nb; ++b) {
            if ((tot_c[b] <= 0 && cand.default_go_left) ||
                (tot_c[b] > 0 && tot_c[b] < min_grp && rare_left)) {
                cat_bitset_set(cand.left_bitset, static_cast<std::uint8_t>(b));
            }
        }
        cand.valid = true;
    }
    return cand;
}

std::size_t collect_small_pair_rows(const std::uint8_t* leaf_ids, const std::uint32_t* rows,
                                    std::size_t n_root, bool is_contiguous, const float* g_seq,
                                    const float* h_seq, const std::int32_t* leaf_c,
                                    std::size_t n_leaves, int n_threads, SymmetricWorkspace& sym,
                                    std::uint8_t* small_bit_out) noexcept {
    const std::size_t n_pairs = n_leaves / 2U;
    for (std::size_t p = 0U; p < n_pairs; ++p) {
        small_bit_out[p] = (leaf_c[2U * p] <= leaf_c[2U * p + 1U]) ? 0U : 1U;
    }
    const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
    const bool use_omp = (n_threads > 1) && (n_chunks > 1U) && (n_root >= 32768U);
    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            std::size_t cnt = 0U;
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t l = leaf_ids[t];
                if ((l & 1U) == small_bit_out[l >> 1U]) {
                    ++cnt;
                }
            }
            sym.chunk_small_counts[cu] = cnt;
        }
    } else {
        for (std::size_t c = 0U; c < n_chunks; ++c) {
            const std::size_t r0 = c * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            std::size_t cnt = 0U;
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t l = leaf_ids[t];
                if ((l & 1U) == small_bit_out[l >> 1U]) {
                    ++cnt;
                }
            }
            sym.chunk_small_counts[c] = cnt;
        }
    }

    std::size_t n_small = 0U;
    sym.chunk_small_offs[0] = 0U;
    for (std::size_t c = 0U; c < n_chunks; ++c) {
        n_small += sym.chunk_small_counts[c];
        sym.chunk_small_offs[c + 1U] = n_small;
    }
    if (n_small == 0U) {
        return 0U;
    }

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            std::size_t pos = sym.chunk_small_offs[cu];
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t l = leaf_ids[t];
                if ((l & 1U) == small_bit_out[l >> 1U]) {
                    sym.small_rows[pos] = is_contiguous ? static_cast<std::uint32_t>(t) : rows[t];
                    sym.small_pairs[pos] = static_cast<std::uint8_t>(l >> 1U);
                    sym.small_g[pos] = g_seq[t];
                    sym.small_h[pos] = h_seq[t];
                    ++pos;
                }
            }
        }
    } else {
        for (std::size_t c = 0U; c < n_chunks; ++c) {
            const std::size_t r0 = c * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            std::size_t pos = sym.chunk_small_offs[c];
            for (std::size_t t = r0; t < r1; ++t) {
                const std::uint8_t l = leaf_ids[t];
                if ((l & 1U) == small_bit_out[l >> 1U]) {
                    sym.small_rows[pos] = is_contiguous ? static_cast<std::uint32_t>(t) : rows[t];
                    sym.small_pairs[pos] = static_cast<std::uint8_t>(l >> 1U);
                    sym.small_g[pos] = g_seq[t];
                    sym.small_h[pos] = h_seq[t];
                    ++pos;
                }
            }
        }
    }
    return n_small;
}

void build_sym_feature_hist(const std::uint8_t* col, const std::uint32_t* rows, std::size_t n_root,
                            bool is_contiguous, const std::uint8_t* leaf_ids, const float* g_seq,
                            const float* h_seq, std::size_t d, std::size_t depth_limit,
                            std::size_t n_leaves, std::size_t stride, bool allow_wave,
                            int n_threads, std::size_t n_small, const std::uint8_t* small_bit,
                            SymmetricWorkspace& sym, HistBin* hist_f, HistBin* prev_f) noexcept {
    const bool use_sub = sym.use_axis_sub && (d >= 1U);
    if (!use_sub) {
        const std::size_t h_elems = n_leaves * stride;
        std::memset(static_cast<void*>(hist_f), 0, h_elems * sizeof(HistBin));
        const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
        const bool use_wave = allow_wave && (n_chunks > 1U) && (n_root >= 32768U);
        if (!use_wave) {
            build_axis_hist_feat(col, rows, 0U, n_root, is_contiguous, leaf_ids, stride, g_seq,
                                 h_seq, hist_f);
        } else {
            for (std::size_t w_base = 0; w_base < n_chunks; w_base += kWaveChunks) {
                const std::size_t w_len = std::min(kWaveChunks, n_chunks - w_base);
                if (n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                    for (std::ptrdiff_t w = 0; w < static_cast<std::ptrdiff_t>(w_len); ++w) {
                        const auto wu = static_cast<std::size_t>(w);
                        const std::size_t c = w_base + wu;
                        const std::size_t r0 = c * kDetChunkRows;
                        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
                        HistBin* wh = sym.wave_hists.data() + wu * h_elems;
                        std::memset(static_cast<void*>(wh), 0, h_elems * sizeof(HistBin));
                        build_axis_hist_feat(col, rows, r0, r1, is_contiguous, leaf_ids, stride,
                                             g_seq, h_seq, wh);
                    }
                } else {
                    for (std::size_t wu = 0; wu < w_len; ++wu) {
                        const std::size_t c = w_base + wu;
                        const std::size_t r0 = c * kDetChunkRows;
                        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
                        HistBin* wh = sym.wave_hists.data() + wu * h_elems;
                        std::memset(static_cast<void*>(wh), 0, h_elems * sizeof(HistBin));
                        build_axis_hist_feat(col, rows, r0, r1, is_contiguous, leaf_ids, stride,
                                             g_seq, h_seq, wh);
                    }
                }
                for (std::size_t w = 0; w < w_len; ++w) {
                    const HistBin* wh = sym.wave_hists.data() + w * h_elems;
                    for (std::size_t b = 0; b < h_elems; ++b) {
                        hist_f[b].g += wh[b].g;
                        hist_f[b].h += wh[b].h;
                        hist_f[b].c += wh[b].c;
                    }
                }
            }
        }
    } else {
        const std::size_t n_pairs = n_leaves / 2U;
        const std::size_t small_elems = n_pairs * stride;
        std::memset(static_cast<void*>(hist_f), 0, small_elems * sizeof(HistBin));
        const std::uint32_t* s_rows = sym.small_rows.data();
        const std::uint8_t* s_pairs = sym.small_pairs.data();
        const float* sg = sym.small_g.data();
        const float* sh = sym.small_h.data();
        const std::size_t n_small_chunks = (n_small + kDetChunkRows - 1U) / kDetChunkRows;
        const bool use_wave = allow_wave && (n_small_chunks > 1U) && (n_small >= 32768U);
        if (!use_wave) {
            build_axis_hist_small_pairs(col, s_rows, s_pairs, sg, sh, 0U, n_small, stride, hist_f);
        } else {
            for (std::size_t w_base = 0; w_base < n_small_chunks; w_base += kWaveChunks) {
                const std::size_t w_len = std::min(kWaveChunks, n_small_chunks - w_base);
                if (n_threads > 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                    for (std::ptrdiff_t w = 0; w < static_cast<std::ptrdiff_t>(w_len); ++w) {
                        const auto wu = static_cast<std::size_t>(w);
                        const std::size_t c = w_base + wu;
                        const std::size_t s0 = c * kDetChunkRows;
                        const std::size_t s1 = std::min(s0 + kDetChunkRows, n_small);
                        HistBin* wh = sym.wave_hists.data() + wu * small_elems;
                        std::memset(static_cast<void*>(wh), 0, small_elems * sizeof(HistBin));
                        build_axis_hist_small_pairs(col, s_rows, s_pairs, sg, sh, s0, s1, stride,
                                                    wh);
                    }
                } else {
                    for (std::size_t wu = 0; wu < w_len; ++wu) {
                        const std::size_t c = w_base + wu;
                        const std::size_t s0 = c * kDetChunkRows;
                        const std::size_t s1 = std::min(s0 + kDetChunkRows, n_small);
                        HistBin* wh = sym.wave_hists.data() + wu * small_elems;
                        std::memset(static_cast<void*>(wh), 0, small_elems * sizeof(HistBin));
                        build_axis_hist_small_pairs(col, s_rows, s_pairs, sg, sh, s0, s1, stride,
                                                    wh);
                    }
                }
                for (std::size_t w = 0; w < w_len; ++w) {
                    const HistBin* wh = sym.wave_hists.data() + w * small_elems;
                    for (std::size_t b = 0; b < small_elems; ++b) {
                        hist_f[b].g += wh[b].g;
                        hist_f[b].h += wh[b].h;
                        hist_f[b].c += wh[b].c;
                    }
                }
            }
        }
        reconstruct_sym_hist_from_small(hist_f, prev_f, small_bit, n_pairs, stride);
    }

    if (sym.use_axis_sub && (d + 1U < depth_limit)) {
        std::memcpy(static_cast<void*>(prev_f), static_cast<const void*>(hist_f),
                    n_leaves * stride * sizeof(HistBin));
    }
}

void build_sym_feature_hist_group(std::size_t n_feat, const std::uint8_t* const* cols,
                                  const std::size_t* strides, HistBin* const* hist_fs,
                                  HistBin* const* prev_fs, const std::uint32_t* rows,
                                  std::size_t n_root, bool is_contiguous,
                                  const std::uint8_t* leaf_ids, const float* g_seq,
                                  const float* h_seq, std::size_t d, std::size_t depth_limit,
                                  std::size_t n_leaves, std::size_t n_small,
                                  const std::uint8_t* small_bit, SymmetricWorkspace& sym) noexcept {
    const bool use_sub = sym.use_axis_sub && (d >= 1U);
    const std::size_t n_pairs = n_leaves / 2U;
    for (std::size_t gi = 0; gi < n_feat; ++gi) {
        const std::size_t elems = (use_sub ? n_pairs : n_leaves) * strides[gi];
        std::memset(static_cast<void*>(hist_fs[gi]), 0, elems * sizeof(HistBin));
    }
    if (!use_sub) {
        switch (n_feat) {
        case 4U:
            build_axis_hist_multi<4U>(cols, rows, n_root, is_contiguous, leaf_ids, strides, g_seq,
                                      h_seq, hist_fs);
            break;
        case 3U:
            build_axis_hist_multi<3U>(cols, rows, n_root, is_contiguous, leaf_ids, strides, g_seq,
                                      h_seq, hist_fs);
            break;
        case 2U:
            build_axis_hist_multi<2U>(cols, rows, n_root, is_contiguous, leaf_ids, strides, g_seq,
                                      h_seq, hist_fs);
            break;
        default:
            build_axis_hist_multi<1U>(cols, rows, n_root, is_contiguous, leaf_ids, strides, g_seq,
                                      h_seq, hist_fs);
            break;
        }
    } else {
        const std::uint32_t* s_rows = sym.small_rows.data();
        const std::uint8_t* s_pairs = sym.small_pairs.data();
        const float* sg = sym.small_g.data();
        const float* sh = sym.small_h.data();
        switch (n_feat) {
        case 4U:
            build_axis_hist_small_pairs_multi<4U>(cols, s_rows, s_pairs, sg, sh, n_small, strides,
                                                  hist_fs);
            break;
        case 3U:
            build_axis_hist_small_pairs_multi<3U>(cols, s_rows, s_pairs, sg, sh, n_small, strides,
                                                  hist_fs);
            break;
        case 2U:
            build_axis_hist_small_pairs_multi<2U>(cols, s_rows, s_pairs, sg, sh, n_small, strides,
                                                  hist_fs);
            break;
        default:
            build_axis_hist_small_pairs_multi<1U>(cols, s_rows, s_pairs, sg, sh, n_small, strides,
                                                  hist_fs);
            break;
        }
        for (std::size_t gi = 0; gi < n_feat; ++gi) {
            reconstruct_sym_hist_from_small(hist_fs[gi], prev_fs[gi], small_bit, n_pairs,
                                            strides[gi]);
        }
    }
    if (sym.use_axis_sub && (d + 1U < depth_limit)) {
        for (std::size_t gi = 0; gi < n_feat; ++gi) {
            std::memcpy(static_cast<void*>(prev_fs[gi]), static_cast<const void*>(hist_fs[gi]),
                        n_leaves * strides[gi] * sizeof(HistBin));
        }
    }
}

} // namespace miter::sym_detail
