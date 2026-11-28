// Symmetric trees: oblique (projection) split search, full and row-sampled.

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

void gather_sym_block_features(const std::int16_t* codes, std::size_t n_rows,
                               const std::uint32_t* leaf_rows, bool is_contiguous, std::size_t mb0,
                               std::size_t len, std::size_t n_used, const std::uint32_t* used_feats,
                               std::int16_t* feat_tile, const std::int16_t** col_ptrs) noexcept {
    if (is_contiguous) {
        for (std::size_t u = 0; u < n_used; ++u) {
            col_ptrs[u] = codes + static_cast<std::size_t>(used_feats[u]) * n_rows + mb0;
        }
        return;
    }
    const std::uint32_t* lr = leaf_rows + mb0;
    for (std::size_t u = 0; u < n_used; ++u) {
        const std::int16_t* src = codes + static_cast<std::size_t>(used_feats[u]) * n_rows;
        std::int16_t* dst = feat_tile + u * kObliqueMicroBlock;
        col_ptrs[u] = dst;
        for (std::size_t i = 0; i < len; ++i) {
            dst[i] = src[lr[i]];
        }
    }
}

void sym_oblique_pass1_chunk(const std::int16_t* codes, std::size_t n_rows,
                             const std::uint32_t* leaf_rows, std::size_t n_root, bool is_contiguous,
                             std::size_t r0, std::size_t r1, std::size_t m,
                             const ObliqueCandMeta* metas, std::size_t n_used,
                             const std::uint32_t* used_feats, bool any_has_nan, bool store_proj,
                             std::int32_t* proj_buf, std::int16_t* feat_tile,
                             const std::int16_t** col_ptrs, std::int32_t* out_pmin,
                             std::int32_t* out_pmax, std::int32_t* out_valid) noexcept {
    for (std::size_t j = 0; j < m; ++j) {
        out_pmin[j] = std::numeric_limits<std::int32_t>::max();
        out_pmax[j] = std::numeric_limits<std::int32_t>::min();
        out_valid[j] = 0;
    }

    for (std::size_t mb0 = r0; mb0 < r1; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, r1 - mb0);
        gather_sym_block_features(codes, n_rows, leaf_rows, is_contiguous, mb0, len, n_used,
                                  used_feats, feat_tile, col_ptrs);
        for (std::size_t j = 0; j < m; ++j) {
            const ObliqueCandMeta& cm = metas[j];
            if (cm.k < 2U) {
                continue;
            }
            const std::int16_t* c0 = col_ptrs[cm.uf[0]];
            const std::int16_t* c1 = col_ptrs[cm.uf[1]];
            const std::int32_t w0 = cm.w[0];
            const std::int32_t w1 = cm.w[1];
            std::int32_t lmin = out_pmin[j];
            std::int32_t lmax = out_pmax[j];
            std::int32_t* p_dst = store_proj ? (proj_buf + j * n_root + mb0) : nullptr;

            if (!any_has_nan || cm.has_nan == 0U) {
                if (cm.k == 2U) {
                    if (store_proj) {
                        for (std::size_t i = 0; i < len; ++i) {
                            const std::int32_t p32 = w0 * c0[i] + w1 * c1[i];
                            p_dst[i] = p32;
                            lmin = (p32 < lmin) ? p32 : lmin;
                            lmax = (p32 > lmax) ? p32 : lmax;
                        }
                    } else {
                        for (std::size_t i = 0; i < len; ++i) {
                            const std::int32_t p32 = w0 * c0[i] + w1 * c1[i];
                            lmin = (p32 < lmin) ? p32 : lmin;
                            lmax = (p32 > lmax) ? p32 : lmax;
                        }
                    }
                } else {
                    const std::int16_t* c2 = col_ptrs[cm.uf[2]];
                    const std::int32_t w2 = cm.w[2];
                    if (store_proj) {
                        for (std::size_t i = 0; i < len; ++i) {
                            const std::int32_t p32 = w0 * c0[i] + w1 * c1[i] + w2 * c2[i];
                            p_dst[i] = p32;
                            lmin = (p32 < lmin) ? p32 : lmin;
                            lmax = (p32 > lmax) ? p32 : lmax;
                        }
                    } else {
                        for (std::size_t i = 0; i < len; ++i) {
                            const std::int32_t p32 = w0 * c0[i] + w1 * c1[i] + w2 * c2[i];
                            lmin = (p32 < lmin) ? p32 : lmin;
                            lmax = (p32 > lmax) ? p32 : lmax;
                        }
                    }
                }
                out_pmin[j] = lmin;
                out_pmax[j] = lmax;
                out_valid[j] += static_cast<std::int32_t>(len);
            } else {
                const std::int16_t* c2 = col_ptrs[cm.uf[2]];
                const std::int32_t w2 = cm.w[2];
                std::int32_t vc = out_valid[j];
                for (std::size_t i = 0; i < len; ++i) {
                    const std::int16_t x0 = c0[i];
                    const std::int16_t x1 = c1[i];
                    const std::int16_t x2 = c2[i];
                    if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode ||
                        x2 == kQ16SignedNanCode) {
                        if (store_proj) {
                            p_dst[i] = kMissingProjSentinel;
                        }
                    } else {
                        const std::int32_t p32 = w0 * x0 + w1 * x1 + w2 * x2;
                        if (store_proj) {
                            p_dst[i] = p32;
                        }
                        lmin = (p32 < lmin) ? p32 : lmin;
                        lmax = (p32 > lmax) ? p32 : lmax;
                        ++vc;
                    }
                }
                out_pmin[j] = lmin;
                out_pmax[j] = lmax;
                out_valid[j] = vc;
            }
        }
    }
}

// Bins of one candidate's projections for rows [mb0, mb0 + len): from the stored projections
// when store_proj, else computed from the codes (kMissingBinSentinel where a code is missing).
void sym_oblique_pass2_bins(const std::int16_t* codes, std::size_t n_rows,
                            const std::uint32_t* leaf_rows, bool is_contiguous, std::size_t mb0,
                            std::size_t len, const ObliqueCandMeta& cm, std::int32_t pmin,
                            const FastDiv64& div, bool store_proj, const std::int32_t* p_col,
                            std::uint8_t* bins) noexcept {
    const std::int64_t pmin64 = pmin;
    if (store_proj) {
        const std::int32_t* pm = p_col + mb0;
        if (cm.has_nan == 0U) {
            for (std::size_t i = 0; i < len; ++i) {
                const auto num =
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(pm[i]) - pmin64) *
                    static_cast<std::uint64_t>(kObliqueBins);
                bins[i] = static_cast<std::uint8_t>(div.div(num));
            }
        } else {
            for (std::size_t i = 0; i < len; ++i) {
                const std::int32_t p32 = pm[i];
                if (p32 == kMissingProjSentinel) {
                    bins[i] = kMissingBinSentinel;
                } else {
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
            }
        }
        return;
    }

    const std::int16_t* col0 = codes + static_cast<std::size_t>(cm.fn[0]) * n_rows;
    const std::int16_t* col1 = codes + static_cast<std::size_t>(cm.fn[1]) * n_rows;
    const std::int16_t* col2 = codes + static_cast<std::size_t>(cm.fn[2]) * n_rows;
    const std::int32_t w0 = cm.w[0];
    const std::int32_t w1 = cm.w[1];
    const std::int32_t w2 = cm.w[2];
    if (cm.has_nan == 0U) {
        if (cm.k == 2U) {
            if (is_contiguous) {
                for (std::size_t i = 0; i < len; ++i) {
                    const std::size_t t = mb0 + i;
                    const std::int32_t p32 = w0 * col0[t] + w1 * col1[t];
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
            } else {
                for (std::size_t i = 0; i < len; ++i) {
                    const std::uint32_t r = leaf_rows[mb0 + i];
                    const std::int32_t p32 = w0 * col0[r] + w1 * col1[r];
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
            }
        } else {
            if (is_contiguous) {
                for (std::size_t i = 0; i < len; ++i) {
                    const std::size_t t = mb0 + i;
                    const std::int32_t p32 = w0 * col0[t] + w1 * col1[t] + w2 * col2[t];
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
            } else {
                for (std::size_t i = 0; i < len; ++i) {
                    const std::uint32_t r = leaf_rows[mb0 + i];
                    const std::int32_t p32 = w0 * col0[r] + w1 * col1[r] + w2 * col2[r];
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
            }
        }
    } else {
        for (std::size_t i = 0; i < len; ++i) {
            const std::size_t idx = is_contiguous ? (mb0 + i) : leaf_rows[mb0 + i];
            const std::int16_t x0 = col0[idx];
            const std::int16_t x1 = col1[idx];
            const std::int16_t x2 = col2[idx];
            if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode || x2 == kQ16SignedNanCode) {
                bins[i] = kMissingBinSentinel;
            } else {
                const std::int32_t p32 = w0 * x0 + w1 * x1 + w2 * x2;
                const auto num =
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin64) *
                    static_cast<std::uint64_t>(kObliqueBins);
                bins[i] = static_cast<std::uint8_t>(div.div(num));
            }
        }
    }
}

void sym_oblique_pass2_cand(const std::int16_t* codes, std::size_t n_rows,
                            const std::uint32_t* leaf_rows, std::size_t n_root, bool is_contiguous,
                            std::size_t r0, std::size_t r1, const ObliqueCandMeta& cm,
                            std::int32_t pmin, FastDiv64 div, bool store_proj,
                            const std::int32_t* p_col, const std::uint8_t* leaf_ids,
                            const float* g_seq, const float* h_seq, HistBin* hist_out) noexcept {
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    std::uint8_t bins[kObliqueMicroBlock];
    for (std::size_t mb0 = r0; mb0 < r1; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, r1 - mb0);
        sym_oblique_pass2_bins(codes, n_rows, leaf_rows, is_contiguous, mb0, len, cm, pmin, div,
                               store_proj, p_col, bins);
        std::size_t i = 0U;
        for (; i + 4U <= len; i += 4U) {
            const std::size_t t = mb0 + i;
            const std::size_t idx0 = static_cast<std::size_t>(leaf_ids[t]) * kStride + bins[i];
            const std::size_t idx1 =
                static_cast<std::size_t>(leaf_ids[t + 1U]) * kStride + bins[i + 1U];
            const std::size_t idx2 =
                static_cast<std::size_t>(leaf_ids[t + 2U]) * kStride + bins[i + 2U];
            const std::size_t idx3 =
                static_cast<std::size_t>(leaf_ids[t + 3U]) * kStride + bins[i + 3U];
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
        for (; i < len; ++i) {
            const std::size_t t = mb0 + i;
            HistBin& dst = hist_out[static_cast<std::size_t>(leaf_ids[t]) * kStride + bins[i]];
            dst.g += static_cast<double>(g_seq[t]);
            dst.h += static_cast<double>(h_seq[t]);
            dst.c += 1;
        }
    }
    (void)n_root;
}

// pass2 for G candidates in one sweep over the rows: G independent histograms, each filled in row
// order exactly as sym_oblique_pass2_cand would, so the results are identical; the independent
// read-modify-write chains overlap instead of stalling on repeated bins.
template <std::size_t G>
void sym_oblique_pass2_multi(const std::int16_t* codes, std::size_t n_rows,
                             const std::uint32_t* leaf_rows, std::size_t n_root, bool is_contiguous,
                             const ObliqueCandMeta* const* cms, const std::int32_t* pmins,
                             const FastDiv64* divs, bool store_proj,
                             const std::int32_t* const* p_cols, const std::uint8_t* leaf_ids,
                             const float* g_seq, const float* h_seq,
                             HistBin* const* hists) noexcept {
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    std::uint8_t bins[G][kObliqueMicroBlock];
    for (std::size_t mb0 = 0; mb0 < n_root; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, n_root - mb0);
        for (std::size_t gi = 0; gi < G; ++gi) {
            sym_oblique_pass2_bins(codes, n_rows, leaf_rows, is_contiguous, mb0, len, *cms[gi],
                                   pmins[gi], divs[gi], store_proj, p_cols[gi], bins[gi]);
        }
        for (std::size_t i = 0; i < len; ++i) {
            const std::size_t t = mb0 + i;
            const std::size_t base = static_cast<std::size_t>(leaf_ids[t]) * kStride;
            const double gv = static_cast<double>(g_seq[t]);
            const double hv = static_cast<double>(h_seq[t]);
            for (std::size_t gi = 0; gi < G; ++gi) {
                HistBin& dst = hists[gi][base + bins[gi][i]];
                dst.g += gv;
                dst.h += hv;
                dst.c += 1;
            }
        }
    }
}

void scan_sym_oblique_hist(const HistBin* hist, std::size_t n_leaves, const double* leaf_g,
                           const double* leaf_h, const std::int32_t* leaf_c,
                           const double* parent_score, double lam, double min_h, std::int32_t min_c,
                           double min_gain, std::int64_t pmin, std::int64_t range, bool check_nan,
                           double& best_gain, std::int64_t& best_thr, bool& best_nan_left,
                           bool& found) noexcept {
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    double gl[kSymMaxLeaves] = {};
    double hl[kSymMaxLeaves] = {};
    std::int32_t cl[kSymMaxLeaves] = {};
    double gl_nan[kSymMaxLeaves] = {};
    double hl_nan[kSymMaxLeaves] = {};
    std::int32_t cl_nan[kSymMaxLeaves] = {};
    double total_h_all = 0.0;
    for (std::size_t l = 0; l < n_leaves; ++l) {
        total_h_all += leaf_h[l];
    }

    double cum_hl = 0.0;
    for (int b = 0; b < kObliqueBins - 1; ++b) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            const HistBin& hb = hist[l * kStride + static_cast<std::size_t>(b)];
            gl[l] += hb.g;
            hl[l] += hb.h;
            cl[l] += hb.c;
            cum_hl += hb.h;
        }
        if (!check_nan) {
            double gain = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                     lam, min_h, min_c, min_gain, gain) &&
                gain > best_gain) {
                best_gain = gain;
                best_thr = oblique_bin_threshold(pmin, range, b);
                best_nan_left = (cum_hl >= (total_h_all - cum_hl));
                found = true;
            }
        } else {
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& hb = hist[l * kStride + kObliqueBins];
                gl_nan[l] = gl[l] + hb.g;
                hl_nan[l] = hl[l] + hb.h;
                cl_nan[l] = cl[l] + hb.c;
            }
            double gain_a = 0.0;
            if (eval_multi_leaf_gain(gl_nan, hl_nan, cl_nan, leaf_g, leaf_h, leaf_c, parent_score,
                                     n_leaves, lam, min_h, min_c, min_gain, gain_a) &&
                gain_a > best_gain) {
                best_gain = gain_a;
                best_thr = oblique_bin_threshold(pmin, range, b);
                best_nan_left = true;
                found = true;
            }
            double gain_b = 0.0;
            if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves,
                                     lam, min_h, min_c, min_gain, gain_b) &&
                gain_b > best_gain) {
                best_gain = gain_b;
                best_thr = oblique_bin_threshold(pmin, range, b);
                best_nan_left = false;
                found = true;
            }
        }
    }

    if (check_nan) {
        for (std::size_t l = 0; l < n_leaves; ++l) {
            const HistBin& hb = hist[l * kStride + static_cast<std::size_t>(kObliqueBins - 1)];
            gl[l] += hb.g;
            hl[l] += hb.h;
            cl[l] += hb.c;
        }
        double gain = 0.0;
        if (eval_multi_leaf_gain(gl, hl, cl, leaf_g, leaf_h, leaf_c, parent_score, n_leaves, lam,
                                 min_h, min_c, min_gain, gain) &&
            gain > best_gain) {
            best_gain = gain;
            best_thr = pmin + range;
            best_nan_left = false;
            found = true;
        }
    }
}

// Projections w0*x0 + w1*x1 + w2*x2 of rows [c0, c1) into out (INT32_MIN where a code is missing),
// with their min and max over the valid rows (INT32_MAX / INT32_MIN when none). A separate function
// so the loop's min/max reductions are recognized and vectorized.
void project3_i32(const std::int16_t* x0, const std::int16_t* x1, const std::int16_t* x2,
                  std::int32_t w0, std::int32_t w1, std::int32_t w2, std::size_t c0, std::size_t c1,
                  std::int32_t* out, std::int32_t& lo_out, std::int32_t& hi_out) noexcept {
    constexpr std::int32_t kLo = std::numeric_limits<std::int32_t>::max();
    constexpr std::int32_t kHi = std::numeric_limits<std::int32_t>::min();
    std::int32_t lo = kLo;
    std::int32_t hi = kHi;
    for (std::size_t i = c0; i < c1; ++i) {
        const std::int16_t a = x0[i];
        const std::int16_t b = x1[i];
        const std::int16_t c = x2[i];
        const bool missing =
            (a == kQ16SignedNanCode) | (b == kQ16SignedNanCode) | (c == kQ16SignedNanCode);
        const std::int32_t p = w0 * a + w1 * b + w2 * c;
        out[i] = missing ? kHi : p;
        lo = std::min(lo, missing ? kLo : p);
        hi = std::max(hi, missing ? kHi : p);
    }
    lo_out = lo;
    hi_out = hi;
}

} // namespace

ObliqueSplitCandidate
find_sym_oblique_split(const Dataset& ds, const std::uint32_t* leaf_rows, std::size_t n_root,
                       bool is_contiguous, const std::uint8_t* leaf_ids, std::size_t n_leaves,
                       const float* g_seq, const float* h_seq, const double* leaf_g,
                       const double* leaf_h, const std::int32_t* leaf_c, const double* parent_score,
                       std::size_t tree_index, int level, const Params& params,
                       ObliqueScratch& scratch, SymmetricWorkspace& sym,
                       const std::int16_t* codes_override, std::size_t n_rows_override) noexcept {
    ObliqueSplitCandidate best;
    const std::size_t p_elig = scratch.eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U || scratch.cur_m == 0U || n_root < 2U) {
        return best;
    }

    const std::size_t m = scratch.cur_m;
    // codes_override: a column-major code matrix of exactly the rows searched (the sampled path's
    // materialized sample), so the search reads contiguous columns instead of gathering by row.
    const std::size_t n_rows = (codes_override != nullptr) ? n_rows_override : ds.n_rows();
    const std::int16_t* codes =
        (codes_override != nullptr) ? codes_override : ds.codes_col_major().data();

    std::uint32_t picked[kObliqueMaxK] = {};
    double raw_w[kObliqueMaxK] = {};
    std::int16_t wq[kObliqueMaxK] = {};
    bool all_k_le3 = true;
    bool any_has_nan = false;
    std::size_t n_used = 0U;

    for (std::size_t j = 0; j < m; ++j) {
        CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index),
                       static_cast<std::uint64_t>(level), static_cast<std::uint64_t>(j), 0ULL);
        int k =
            (params.k_proj > 0) ? params.k_proj : static_cast<int>(2U + (rng.next_u64() & 1ULL));
        k = std::min(k, static_cast<int>(p_elig));
        k = std::min(k, kObliqueMaxK);
        if (k < 2) {
            scratch.cand_k[j] = 0U;
            scratch.cand_meta[j].k = 0U;
            continue;
        }
        if (k > 3) {
            all_k_le3 = false;
        }
        rng.sample_k_distinct(p_elig, static_cast<std::size_t>(k), picked);
        for (int q = 0; q < k; ++q) {
            raw_w[q] = rng.next_normal();
        }
        if (!quantize_oblique_weights(raw_w, k, wq)) {
            scratch.cand_k[j] = 0U;
            scratch.cand_meta[j].k = 0U;
            continue;
        }
        scratch.cand_k[j] = static_cast<std::uint8_t>(k);
        std::uint8_t has_nan = 0U;
        for (int q = 0; q < k; ++q) {
            const std::uint32_t elig_idx = picked[q];
            const std::uint32_t num_idx = scratch.eligible_num_indices[elig_idx];
            scratch.cand_num_idx[j * kObliqueMaxK + q] = num_idx;
            scratch.cand_feat[j * kObliqueMaxK + q] = scratch.eligible_features[elig_idx];
            scratch.cand_w[j * kObliqueMaxK + q] = wq[q];
            has_nan |= scratch.num_has_nan[num_idx];
        }
        scratch.cand_has_nan[j] = has_nan;
        if (has_nan != 0U) {
            any_has_nan = true;
        }
        if (k <= 3) {
            ObliqueCandMeta& cm = scratch.cand_meta[j];
            cm.k = static_cast<std::uint8_t>(k);
            cm.has_nan = has_nan;
            cm.fn[0] = scratch.cand_num_idx[j * kObliqueMaxK + 0U];
            cm.fn[1] = scratch.cand_num_idx[j * kObliqueMaxK + 1U];
            cm.fn[2] = (k == 3) ? scratch.cand_num_idx[j * kObliqueMaxK + 2U] : cm.fn[0];
            cm.w[0] = wq[0];
            cm.w[1] = wq[1];
            cm.w[2] = (k == 3) ? wq[2] : static_cast<std::int16_t>(0);
            for (int q = 0; q < k; ++q) {
                const std::uint32_t nf = cm.fn[q];
                std::int32_t u = scratch.num_to_used[nf];
                if (u < 0) {
                    u = static_cast<std::int32_t>(n_used);
                    scratch.num_to_used[nf] = u;
                    scratch.used_num_feats[n_used++] = nf;
                }
                cm.uf[q] = static_cast<std::uint16_t>(u);
            }
            if (k == 2) {
                cm.uf[2] = cm.uf[0];
            }
        }
    }
    for (std::size_t u = 0; u < n_used; ++u) {
        scratch.num_to_used[scratch.used_num_feats[u]] = -1;
    }

    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const int n_threads =
        std::min(resolve_num_threads(params.n_jobs), static_cast<int>(sym.max_threads));
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    const std::size_t hist_elems = n_leaves * kStride;

    if (!all_k_le3) {
        // Fallback exact int64 projection path for k_proj > 3.
        const bool use_omp = (n_threads > 1) && (m >= 2U) && (n_root * m >= 32768U);
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
            for (std::ptrdiff_t js = 0; js < static_cast<std::ptrdiff_t>(m); ++js) {
                const auto j = static_cast<std::size_t>(js);
                ObliqueSplitCandidate cand;
                const std::uint8_t k = scratch.cand_k[j];
                if (k >= 2U) {
                    const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
                    const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
                    const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
                    std::int64_t pmin = std::numeric_limits<std::int64_t>::max();
                    std::int64_t pmax = std::numeric_limits<std::int64_t>::min();
                    std::int32_t vcnt = 0;
                    for (std::size_t t = 0; t < n_root; ++t) {
                        const std::size_t idx = is_contiguous ? t : leaf_rows[t];
                        bool miss = false;
                        std::int64_t acc = 0;
                        for (std::uint8_t q = 0; q < k; ++q) {
                            const std::int16_t c =
                                codes[static_cast<std::size_t>(fn[q]) * n_rows + idx];
                            if (c == kQ16SignedNanCode) {
                                miss = true;
                                break;
                            }
                            acc += static_cast<std::int64_t>(w[q]) * static_cast<std::int64_t>(c);
                        }
                        if (!miss) {
                            pmin = std::min(pmin, acc);
                            pmax = std::max(pmax, acc);
                            ++vcnt;
                        }
                    }
                    const std::int32_t c_nan_tot = static_cast<std::int32_t>(n_root) - vcnt;
                    if (vcnt >= 1 && (pmin < pmax || c_nan_tot > 0)) {
                        const std::int64_t range = pmax - pmin + 1;
                        FastDiv64 div;
                        div.prepare(static_cast<std::uint64_t>(range));
#ifdef _OPENMP
                        const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                        const std::size_t tid = 0U;
#endif
                        HistBin* hist =
                            sym.oblique_thread_hists.data() + tid * sym.max_level_leaves * kStride;
                        std::memset(static_cast<void*>(hist), 0, hist_elems * sizeof(HistBin));
                        for (std::size_t t = 0; t < n_root; ++t) {
                            const std::size_t idx = is_contiguous ? t : leaf_rows[t];
                            bool miss = false;
                            std::int64_t acc = 0;
                            for (std::uint8_t q = 0; q < k; ++q) {
                                const std::int16_t c =
                                    codes[static_cast<std::size_t>(fn[q]) * n_rows + idx];
                                if (c == kQ16SignedNanCode) {
                                    miss = true;
                                    break;
                                }
                                acc +=
                                    static_cast<std::int64_t>(w[q]) * static_cast<std::int64_t>(c);
                            }
                            const std::uint8_t b =
                                miss ? kMissingBinSentinel
                                     : static_cast<std::uint8_t>(
                                           div.div(static_cast<std::uint64_t>(acc - pmin) *
                                                   static_cast<std::uint64_t>(kObliqueBins)));
                            HistBin& dst =
                                hist[static_cast<std::size_t>(leaf_ids[t]) * kStride + b];
                            dst.g += static_cast<double>(g_seq[t]);
                            dst.h += static_cast<double>(h_seq[t]);
                            dst.c += 1;
                        }
                        double cg = -std::numeric_limits<double>::infinity();
                        std::int64_t thr = 0;
                        bool nl = true;
                        bool found = false;
                        scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score,
                                              lam, min_h, min_c, min_gain, pmin, range,
                                              c_nan_tot > 0, cg, thr, nl, found);
                        if (found) {
                            cand.gain = cg;
                            cand.k = k;
                            for (std::uint8_t q = 0; q < k; ++q) {
                                cand.features[q] = fg[q];
                                cand.num_indices[q] = fn[q];
                                cand.weights[q] = w[q];
                            }
                            cand.threshold = thr;
                            cand.nan_go_left = nl;
                            cand.valid = true;
                        }
                    }
                }
                scratch.i64_candidates[j] = cand;
            }
            for (std::size_t j = 0; j < m; ++j) {
                const ObliqueSplitCandidate& c = scratch.i64_candidates[j];
                if (c.valid && c.gain > best.gain) {
                    best = c;
                }
            }
            return best;
        }

        HistBin* hist = sym.oblique_thread_hists.data();
        for (std::size_t j = 0; j < m; ++j) {
            const std::uint8_t k = scratch.cand_k[j];
            if (k < 2U) {
                continue;
            }
            const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
            const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
            const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
            std::int64_t pmin = std::numeric_limits<std::int64_t>::max();
            std::int64_t pmax = std::numeric_limits<std::int64_t>::min();
            std::int32_t vcnt = 0;
            for (std::size_t t = 0; t < n_root; ++t) {
                const std::size_t idx = is_contiguous ? t : leaf_rows[t];
                bool miss = false;
                std::int64_t acc = 0;
                for (std::uint8_t q = 0; q < k; ++q) {
                    const std::int16_t c = codes[static_cast<std::size_t>(fn[q]) * n_rows + idx];
                    if (c == kQ16SignedNanCode) {
                        miss = true;
                        break;
                    }
                    acc += static_cast<std::int64_t>(w[q]) * static_cast<std::int64_t>(c);
                }
                if (!miss) {
                    pmin = std::min(pmin, acc);
                    pmax = std::max(pmax, acc);
                    ++vcnt;
                }
            }
            const std::int32_t c_nan_tot = static_cast<std::int32_t>(n_root) - vcnt;
            if (vcnt < 1 || (pmin >= pmax && c_nan_tot == 0)) {
                continue;
            }
            const std::int64_t range = pmax - pmin + 1;
            FastDiv64 div;
            div.prepare(static_cast<std::uint64_t>(range));
            std::memset(static_cast<void*>(hist), 0, hist_elems * sizeof(HistBin));
            for (std::size_t t = 0; t < n_root; ++t) {
                const std::size_t idx = is_contiguous ? t : leaf_rows[t];
                bool miss = false;
                std::int64_t acc = 0;
                for (std::uint8_t q = 0; q < k; ++q) {
                    const std::int16_t c = codes[static_cast<std::size_t>(fn[q]) * n_rows + idx];
                    if (c == kQ16SignedNanCode) {
                        miss = true;
                        break;
                    }
                    acc += static_cast<std::int64_t>(w[q]) * static_cast<std::int64_t>(c);
                }
                const std::uint8_t b = miss ? kMissingBinSentinel
                                            : static_cast<std::uint8_t>(div.div(
                                                  static_cast<std::uint64_t>(acc - pmin) *
                                                  static_cast<std::uint64_t>(kObliqueBins)));
                HistBin& dst = hist[static_cast<std::size_t>(leaf_ids[t]) * kStride + b];
                dst.g += static_cast<double>(g_seq[t]);
                dst.h += static_cast<double>(h_seq[t]);
                dst.c += 1;
            }
            double cg = best.gain;
            std::int64_t thr = 0;
            bool nl = true;
            bool found = false;
            scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score, lam, min_h,
                                  min_c, min_gain, pmin, range, c_nan_tot > 0, cg, thr, nl, found);
            if (found) {
                best.gain = cg;
                best.k = k;
                for (std::uint8_t q = 0; q < k; ++q) {
                    best.features[q] = fg[q];
                    best.num_indices[q] = fn[q];
                    best.weights[q] = w[q];
                }
                best.threshold = thr;
                best.nan_go_left = nl;
                best.valid = true;
            }
        }
        return best;
    }

    const bool store_proj = (n_root * m <= scratch.proj_buf.size());
    std::int32_t* proj_buf = scratch.proj_buf.data();
    const std::size_t n_chunks = (n_root + kDetChunkRows - 1U) / kDetChunkRows;
    const ObliqueCandMeta* metas = scratch.cand_meta.data();
    const std::uint32_t* used_feats = scratch.used_num_feats.data();

    // Pass 1: chunked bounds calculation (and optional proj_buf write).
    if (n_threads > 1 && n_chunks > 1U && n_root * m >= 32768U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
#ifdef _OPENMP
            const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
            const std::size_t tid = 0U;
#endif
            std::int16_t* feat_tile =
                scratch.thread_feat_buf.data() + tid * scratch.max_used_feats * kObliqueMicroBlock;
            const std::int16_t** col_ptrs =
                scratch.thread_col_ptrs.data() + tid * scratch.max_used_feats;
            sym_oblique_pass1_chunk(codes, n_rows, leaf_rows, n_root, is_contiguous, r0, r1, m,
                                    metas, n_used, used_feats, any_has_nan, store_proj, proj_buf,
                                    feat_tile, col_ptrs, scratch.chunk_pmin.data() + cu * m,
                                    scratch.chunk_pmax.data() + cu * m,
                                    scratch.chunk_valid_cnt.data() + cu * m);
        }
    } else {
        std::int16_t* feat_tile = scratch.thread_feat_buf.data();
        const std::int16_t** col_ptrs = scratch.thread_col_ptrs.data();
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t r0 = c * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
            sym_oblique_pass1_chunk(codes, n_rows, leaf_rows, n_root, is_contiguous, r0, r1, m,
                                    metas, n_used, used_feats, any_has_nan, store_proj, proj_buf,
                                    feat_tile, col_ptrs, scratch.chunk_pmin.data() + c * m,
                                    scratch.chunk_pmax.data() + c * m,
                                    scratch.chunk_valid_cnt.data() + c * m);
        }
    }

    std::size_t n_active = 0U;
    for (std::size_t j = 0; j < m; ++j) {
        std::int32_t pmin = std::numeric_limits<std::int32_t>::max();
        std::int32_t pmax = std::numeric_limits<std::int32_t>::min();
        std::int32_t vcnt = 0;
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t idx = c * m + j;
            pmin = std::min(pmin, scratch.chunk_pmin[idx]);
            pmax = std::max(pmax, scratch.chunk_pmax[idx]);
            vcnt += scratch.chunk_valid_cnt[idx];
        }
        scratch.cand_pmin[j] = pmin;
        scratch.cand_pmax[j] = pmax;
        scratch.cand_valid_cnt[j] = vcnt;
        const bool has_miss = vcnt < static_cast<std::int32_t>(n_root);
        if (scratch.cand_k[j] >= 2U && vcnt >= 1 && (pmin < pmax || (pmin == pmax && has_miss))) {
            const std::int64_t range =
                static_cast<std::int64_t>(pmax) - static_cast<std::int64_t>(pmin) + 1;
            scratch.active_cands[n_active] = static_cast<std::uint32_t>(j);
            scratch.active_ranges[n_active] = range;
            scratch.active_divs[n_active].prepare(static_cast<std::uint64_t>(range));
            ++n_active;
        }
    }
    if (n_active == 0U) {
        return best;
    }

    // Pass 2 + scan: candidate-parallel when >= 2 active candidates, else deterministic chunk-wave.
    if (n_threads > 1 && n_active >= 2U && n_root * n_active >= 32768U) {
        // Each task fills a group of up to 4 candidates' histograms in one sweep (see
        // sym_oblique_pass2_multi); smaller groups when there are too few candidates to keep every
        // thread busy. Each candidate's result lands in its own slot, so grouping changes nothing.
        const auto nt = static_cast<std::size_t>(n_threads);
        const std::size_t grp = hist_group_size(n_active, nt);
        const std::size_t n_groups = (n_active + grp - 1U) / grp;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t gs = 0; gs < static_cast<std::ptrdiff_t>(n_groups); ++gs) {
#ifdef _OPENMP
            const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
            const std::size_t tid = 0U;
#endif
            const std::size_t a0 = static_cast<std::size_t>(gs) * grp;
            const std::size_t gl = std::min(grp, n_active - a0);
            const ObliqueCandMeta* g_cms[4];
            std::int32_t g_pmins[4];
            FastDiv64 g_divs[4];
            const std::int32_t* g_pcols[4];
            HistBin* g_hists[4];
            for (std::size_t gi = 0; gi < gl; ++gi) {
                const std::uint32_t jj = scratch.active_cands[a0 + gi];
                g_cms[gi] = &metas[jj];
                g_pmins[gi] = scratch.cand_pmin[jj];
                g_divs[gi] = scratch.active_divs[a0 + gi];
                g_pcols[gi] = store_proj ? (proj_buf + jj * n_root) : nullptr;
                g_hists[gi] = sym.multi_hists.data() + (tid * 4U + gi) * hist_elems;
                std::memset(static_cast<void*>(g_hists[gi]), 0, hist_elems * sizeof(HistBin));
            }
            switch (gl) {
            case 4U:
                sym_oblique_pass2_multi<4U>(codes, n_rows, leaf_rows, n_root, is_contiguous, g_cms,
                                            g_pmins, g_divs, store_proj, g_pcols, leaf_ids, g_seq,
                                            h_seq, g_hists);
                break;
            case 3U:
                sym_oblique_pass2_multi<3U>(codes, n_rows, leaf_rows, n_root, is_contiguous, g_cms,
                                            g_pmins, g_divs, store_proj, g_pcols, leaf_ids, g_seq,
                                            h_seq, g_hists);
                break;
            case 2U:
                sym_oblique_pass2_multi<2U>(codes, n_rows, leaf_rows, n_root, is_contiguous, g_cms,
                                            g_pmins, g_divs, store_proj, g_pcols, leaf_ids, g_seq,
                                            h_seq, g_hists);
                break;
            default:
                sym_oblique_pass2_multi<1U>(codes, n_rows, leaf_rows, n_root, is_contiguous, g_cms,
                                            g_pmins, g_divs, store_proj, g_pcols, leaf_ids, g_seq,
                                            h_seq, g_hists);
                break;
            }
            for (std::size_t gi = 0; gi < gl; ++gi) {
                const std::size_t a = a0 + gi;
                const std::uint32_t j = scratch.active_cands[a];
                const ObliqueCandMeta& cm = metas[j];
                const bool check_nan = (cm.has_nan != 0U) && (scratch.cand_valid_cnt[j] <
                                                              static_cast<std::int32_t>(n_root));
                const HistBin* hist = g_hists[gi];

                ObliqueSplitCandidate cand;
                double cg = -std::numeric_limits<double>::infinity();
                std::int64_t thr = 0;
                bool nl = true;
                bool found = false;
                scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score, lam,
                                      min_h, min_c, min_gain,
                                      static_cast<std::int64_t>(scratch.cand_pmin[j]),
                                      scratch.active_ranges[a], check_nan, cg, thr, nl, found);
                if (found) {
                    cand.gain = cg;
                    cand.k = cm.k;
                    const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
                    const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
                    const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
                    for (std::uint8_t q = 0; q < cm.k; ++q) {
                        cand.features[q] = fg[q];
                        cand.num_indices[q] = fn[q];
                        cand.weights[q] = w[q];
                    }
                    cand.threshold = thr;
                    cand.nan_go_left = nl;
                    cand.valid = true;
                }
                scratch.i64_candidates[a] = cand;
            }
        }
        for (std::size_t a = 0; a < n_active; ++a) {
            const ObliqueSplitCandidate& cand = scratch.i64_candidates[a];
            if (cand.valid && cand.gain > best.gain) {
                best = cand;
            }
        }
        return best;
    }

    HistBin* hist = nullptr;
    const bool use_chunk_wave = (n_active == 1U) && (n_chunks > 1U) && (n_root >= 32768U);
    // Single-threaded, several candidates: fill their histograms kPass2Group at a time in one
    // sweep, then scan them in candidate order exactly as below.
    constexpr std::size_t kPass2Group = 4U;
    std::size_t grouped = 0U;
    if (!use_chunk_wave && n_active >= kPass2Group) {
        grouped = n_active - n_active % kPass2Group;
    }
    for (std::size_t a = 0; a < n_active; ++a) {
        const std::uint32_t j = scratch.active_cands[a];
        const ObliqueCandMeta& cm = metas[j];
        const bool check_nan =
            (cm.has_nan != 0U) && (scratch.cand_valid_cnt[j] < static_cast<std::int32_t>(n_root));
        const std::int32_t* p_col = store_proj ? (proj_buf + j * n_root) : nullptr;
        if (a < grouped) {
            const std::size_t slot = a % kPass2Group;
            if (slot == 0U) {
                const ObliqueCandMeta* g_cms[kPass2Group];
                std::int32_t g_pmins[kPass2Group];
                FastDiv64 g_divs[kPass2Group];
                const std::int32_t* g_pcols[kPass2Group];
                HistBin* g_hists[kPass2Group];
                for (std::size_t gi = 0; gi < kPass2Group; ++gi) {
                    const std::uint32_t jj = scratch.active_cands[a + gi];
                    g_cms[gi] = &metas[jj];
                    g_pmins[gi] = scratch.cand_pmin[jj];
                    g_divs[gi] = scratch.active_divs[a + gi];
                    g_pcols[gi] = store_proj ? (proj_buf + jj * n_root) : nullptr;
                    g_hists[gi] = sym.multi_hists.data() + gi * hist_elems;
                    std::memset(static_cast<void*>(g_hists[gi]), 0, hist_elems * sizeof(HistBin));
                }
                sym_oblique_pass2_multi<kPass2Group>(
                    codes, n_rows, leaf_rows, n_root, is_contiguous, g_cms, g_pmins, g_divs,
                    store_proj, g_pcols, leaf_ids, g_seq, h_seq, g_hists);
            }
            hist = sym.multi_hists.data() + slot * hist_elems;
        } else {
            hist = sym.oblique_thread_hists.data();
            std::memset(static_cast<void*>(hist), 0, hist_elems * sizeof(HistBin));
        }

        if (a < grouped) {
            // filled above
        } else if (!use_chunk_wave) {
            sym_oblique_pass2_cand(codes, n_rows, leaf_rows, n_root, is_contiguous, 0U, n_root, cm,
                                   scratch.cand_pmin[j], scratch.active_divs[a], store_proj, p_col,
                                   leaf_ids, g_seq, h_seq, hist);
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
                        HistBin* wh = sym.wave_hists.data() + wu * hist_elems;
                        std::memset(static_cast<void*>(wh), 0, hist_elems * sizeof(HistBin));
                        sym_oblique_pass2_cand(codes, n_rows, leaf_rows, n_root, is_contiguous, r0,
                                               r1, cm, scratch.cand_pmin[j], scratch.active_divs[a],
                                               store_proj, p_col, leaf_ids, g_seq, h_seq, wh);
                    }
                } else {
                    for (std::size_t wu = 0; wu < w_len; ++wu) {
                        const std::size_t c = w_base + wu;
                        const std::size_t r0 = c * kDetChunkRows;
                        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_root);
                        HistBin* wh = sym.wave_hists.data() + wu * hist_elems;
                        std::memset(static_cast<void*>(wh), 0, hist_elems * sizeof(HistBin));
                        sym_oblique_pass2_cand(codes, n_rows, leaf_rows, n_root, is_contiguous, r0,
                                               r1, cm, scratch.cand_pmin[j], scratch.active_divs[a],
                                               store_proj, p_col, leaf_ids, g_seq, h_seq, wh);
                    }
                }
                for (std::size_t w = 0; w < w_len; ++w) {
                    const HistBin* wh = sym.wave_hists.data() + w * hist_elems;
                    for (std::size_t b = 0; b < hist_elems; ++b) {
                        hist[b].g += wh[b].g;
                        hist[b].h += wh[b].h;
                        hist[b].c += wh[b].c;
                    }
                }
            }
        }

        double cg = best.gain;
        std::int64_t thr = 0;
        bool nl = true;
        bool found = false;
        scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score, lam, min_h,
                              min_c, min_gain, static_cast<std::int64_t>(scratch.cand_pmin[j]),
                              scratch.active_ranges[a], check_nan, cg, thr, nl, found);
        if (found) {
            best.gain = cg;
            best.k = cm.k;
            const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
            const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
            const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
            for (std::uint8_t q = 0; q < cm.k; ++q) {
                best.features[q] = fg[q];
                best.num_indices[q] = fn[q];
                best.weights[q] = w[q];
            }
            best.threshold = thr;
            best.nan_go_left = nl;
            best.valid = true;
        }
    }
    return best;
}

// oblique_sample_rows: choose the oblique direction on a strided sample of the level's rows, then
// rebuild that direction's 64-bin histogram on all rows and scan it exactly as the full search
// would. Sequential sums in row order keep the result independent of the thread count.
ObliqueSplitCandidate find_sym_oblique_split_sampled(
    const Dataset& ds, const std::uint32_t* rows, std::size_t n_root, bool is_contiguous,
    const std::uint8_t* leaf_ids, std::size_t n_leaves, const float* g_seq, const float* h_seq,
    const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
    const double* parent_score, std::size_t tree_index, int level, const Params& params,
    ObliqueScratch& scratch, SymmetricWorkspace& sym) {
    const auto target = static_cast<std::size_t>(params.oblique_sample_rows);
    const std::size_t stride = (n_root + target - 1U) / target;
    const std::size_t n_s = (n_root + stride - 1U) / stride;
    sym.samp_rows.resize(n_s);
    sym.samp_g.resize(n_s);
    sym.samp_h.resize(n_s);
    sym.samp_leaf.resize(n_s);
    double sg[kSymMaxLeaves] = {};
    double sh[kSymMaxLeaves] = {};
    std::int32_t sc[kSymMaxLeaves] = {};
    double sps[kSymMaxLeaves] = {};
    for (std::size_t t = 0; t < n_s; ++t) {
        const std::size_t i = t * stride;
        sym.samp_rows[t] = is_contiguous ? static_cast<std::uint32_t>(i) : rows[i];
        sym.samp_g[t] = g_seq[i];
        sym.samp_h[t] = h_seq[i];
        sym.samp_leaf[t] = leaf_ids[i];
        sg[leaf_ids[i]] += static_cast<double>(g_seq[i]);
        sh[leaf_ids[i]] += static_cast<double>(h_seq[i]);
        sc[leaf_ids[i]] += 1;
    }
    for (std::size_t l = 0; l < n_leaves; ++l) {
        sps[l] = (sg[l] * sg[l]) / (sh[l] + params.lambda_l2);
    }
    Params ps = params;
    const double frac = static_cast<double>(n_s) / static_cast<double>(n_root);
    ps.min_data_in_leaf = std::max(1, static_cast<int>(std::floor(params.min_data_in_leaf * frac)));
    ps.min_sum_hessian_in_leaf = params.min_sum_hessian_in_leaf * frac;
    ps.min_gain_to_split = 0.0;
    // Without bagging the sampled rows are the same for every tree and level of this fit, so their
    // codes are gathered once into a contiguous matrix (bounded at 64M codes) and searched in
    // place.
    const std::size_t n_all = ds.n_rows();
    const std::size_t n_num = (n_all > 0U) ? ds.codes_col_major().size() / n_all : 0U;
    constexpr std::size_t kMaxSampCodes = std::size_t{1} << 26U;
    const bool materialize = is_contiguous && n_root == n_all && n_num * n_s <= kMaxSampCodes;
    if (materialize && sym.samp_codes_stride != stride) {
        const std::int16_t* src = ds.codes_col_major().data();
        sym.samp_codes.resize(n_num * n_s);
        for (std::size_t f = 0; f < n_num; ++f) {
            const std::int16_t* col = src + f * n_all;
            std::int16_t* dst = sym.samp_codes.data() + f * n_s;
            for (std::size_t t = 0; t < n_s; ++t) {
                dst[t] = col[t * stride];
            }
        }
        sym.samp_codes_stride = stride;
    }
    const ObliqueSplitCandidate pick =
        materialize
            ? find_sym_oblique_split(ds, sym.samp_rows.data(), n_s, true, sym.samp_leaf.data(),
                                     n_leaves, sym.samp_g.data(), sym.samp_h.data(), sg, sh, sc,
                                     sps, tree_index, level, ps, scratch, sym,
                                     sym.samp_codes.data(), n_s)
            : find_sym_oblique_split(ds, sym.samp_rows.data(), n_s, false, sym.samp_leaf.data(),
                                     n_leaves, sym.samp_g.data(), sym.samp_h.data(), sg, sh, sc,
                                     sps, tree_index, level, ps, scratch, sym);
    ObliqueSplitCandidate best;
    if (!pick.valid) {
        return best;
    }

    // Exact pass for the chosen direction on all rows, over fixed row chunks: per-chunk
    // projections, ranges and histograms in parallel, reduced in chunk order, so the sums do not
    // depend on the number of threads.
    const std::size_t n_rows = ds.n_rows();
    const std::int16_t* codes = ds.codes_col_major().data();
    const int k = pick.k;
    constexpr std::int64_t kNanProj = std::numeric_limits<std::int64_t>::min();
    constexpr std::size_t kRefitChunk = 32768U;
    const std::size_t n_chunks = (n_root + kRefitChunk - 1U) / kRefitChunk;
    [[maybe_unused]] const int n_threads = std::max(resolve_num_threads(params.n_jobs), 1);
    // narrow: k <= 3 on contiguous rows, where |p| <= 3 * 2^14 * 32767 < 2^31 - 1, so projections
    // are exact in int32 (INT32_MIN marks a missing row) and the loops vectorize.
    const bool narrow = is_contiguous && k <= 3;
    constexpr std::int32_t kNanProj32 = std::numeric_limits<std::int32_t>::min();
    if (narrow) {
        sym.samp_proj32.resize(n_root);
    } else {
        sym.samp_proj.resize(n_root);
    }
    sym.samp_chunk_min.assign(n_chunks, std::numeric_limits<std::int64_t>::max());
    sym.samp_chunk_max.assign(n_chunks, kNanProj);
    std::int64_t* projv = narrow ? nullptr : sym.samp_proj.data();
    std::int32_t* projv32 = narrow ? sym.samp_proj32.data() : nullptr;
    std::int64_t* cmin = sym.samp_chunk_min.data();
    std::int64_t* cmax = sym.samp_chunk_max.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_chunks > 1U)
#endif
    for (std::ptrdiff_t ci = 0; ci < static_cast<std::ptrdiff_t>(n_chunks); ++ci) {
        const std::size_t c0 = static_cast<std::size_t>(ci) * kRefitChunk;
        const std::size_t c1 = std::min(n_root, c0 + kRefitChunk);
        std::int64_t lo = std::numeric_limits<std::int64_t>::max();
        std::int64_t hi = kNanProj;
        if (narrow) {
            // Branch-free form of the general loop below (k = 2 reuses column 0 with weight 0).
            const std::int16_t* x0 = codes + static_cast<std::size_t>(pick.num_indices[0]) * n_rows;
            const std::int16_t* x1 = codes + static_cast<std::size_t>(pick.num_indices[1]) * n_rows;
            const std::int16_t* x2 =
                (k == 3) ? codes + static_cast<std::size_t>(pick.num_indices[2]) * n_rows : x0;
            const std::int32_t w0 = pick.weights[0];
            const std::int32_t w1 = pick.weights[1];
            const std::int32_t w2 = (k == 3) ? pick.weights[2] : 0;
            std::int32_t lo32 = 0;
            std::int32_t hi32 = 0;
            project3_i32(x0, x1, x2, w0, w1, w2, c0, c1, projv32, lo32, hi32);
            if (hi32 != kNanProj32) {
                lo = lo32;
                hi = hi32;
            }
            cmin[ci] = lo;
            cmax[ci] = hi;
            continue;
        }
        for (std::size_t i = c0; i < c1; ++i) {
            const std::size_t r = is_contiguous ? i : rows[i];
            std::int64_t p = 0;
            bool missing = false;
            for (int q = 0; q < k; ++q) {
                const std::int16_t c =
                    codes[static_cast<std::size_t>(pick.num_indices[q]) * n_rows + r];
                if (c == kQ16SignedNanCode) {
                    missing = true;
                    break;
                }
                p += static_cast<std::int64_t>(pick.weights[q]) * static_cast<std::int64_t>(c);
            }
            if (missing) {
                projv[i] = kNanProj;
            } else {
                projv[i] = p;
                lo = std::min(lo, p);
                hi = std::max(hi, p);
            }
        }
        cmin[ci] = lo;
        cmax[ci] = hi;
    }
    std::int64_t pmin = std::numeric_limits<std::int64_t>::max();
    std::int64_t pmax = kNanProj;
    for (std::size_t ci = 0; ci < n_chunks; ++ci) {
        pmin = std::min(pmin, cmin[ci]);
        pmax = std::max(pmax, cmax[ci]);
    }
    if (pmax == kNanProj || pmin >= pmax) {
        return best;
    }
    const std::int64_t range = pmax - pmin + 1;
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    const std::size_t hist_size = n_leaves * kStride;
    sym.samp_hist.assign(n_chunks * hist_size, HistBin{});
    HistBin* chunk_hists = sym.samp_hist.data();
    // For k <= 3, |p| < 3 * 2^14 * 2^15, so (p - pmin) * kObliqueBins < 2^38 and range < 2^34: the
    // exact multiply-shift division applies (same quotient as oblique_bin_index, no hardware div).
    const bool fast_div = k <= 3;
    FastDiv64 div;
    div.prepare(static_cast<std::uint64_t>(range));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_chunks > 1U)
#endif
    for (std::ptrdiff_t ci = 0; ci < static_cast<std::ptrdiff_t>(n_chunks); ++ci) {
        const std::size_t c0 = static_cast<std::size_t>(ci) * kRefitChunk;
        const std::size_t c1 = std::min(n_root, c0 + kRefitChunk);
        HistBin* h = chunk_hists + static_cast<std::size_t>(ci) * hist_size;
        for (std::size_t i = c0; i < c1; ++i) {
            const std::int64_t p =
                narrow ? (projv32[i] == kNanProj32 ? kNanProj : projv32[i]) : projv[i];
            std::size_t b = static_cast<std::size_t>(kObliqueBins);
            if (p != kNanProj) {
                b = fast_div ? static_cast<std::size_t>(
                                   div.div(static_cast<std::uint64_t>(p - pmin) *
                                           static_cast<std::uint64_t>(kObliqueBins)))
                             : oblique_bin_index(p, pmin, range);
            }
            HistBin& hb = h[static_cast<std::size_t>(leaf_ids[i]) * kStride + b];
            hb.g += static_cast<double>(g_seq[i]);
            hb.h += static_cast<double>(h_seq[i]);
            hb.c += 1;
        }
    }
    bool any_nan = false;
    HistBin* hist = chunk_hists; // chunk 0 accumulates the others, in chunk order
    for (std::size_t ci = 1; ci < n_chunks; ++ci) {
        const HistBin* h = chunk_hists + ci * hist_size;
        for (std::size_t j = 0; j < hist_size; ++j) {
            hist[j].g += h[j].g;
            hist[j].h += h[j].h;
            hist[j].c += h[j].c;
        }
    }
    for (std::size_t l = 0; l < n_leaves; ++l) {
        any_nan = any_nan || (hist[l * kStride + static_cast<std::size_t>(kObliqueBins)].c > 0);
    }
    double best_gain = -std::numeric_limits<double>::infinity();
    std::int64_t best_thr = 0;
    bool best_nan_left = true;
    bool found = false;
    scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score, params.lambda_l2,
                          params.min_sum_hessian_in_leaf, params.min_data_in_leaf,
                          params.min_gain_to_split, pmin, range, any_nan, best_gain, best_thr,
                          best_nan_left, found);
    if (!found) {
        return best;
    }
    best = pick;
    best.gain = best_gain;
    best.threshold = best_thr;
    best.nan_go_left = best_nan_left;
    best.valid = true;
    return best;
}

} // namespace miter::sym_detail
