#include "core/oblique.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "core/rng.hpp"
#include "data/quantizer.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

constexpr std::int32_t kMissingProjSentinel = std::numeric_limits<std::int32_t>::min();
constexpr std::uint8_t kMissingBinSentinel = static_cast<std::uint8_t>(kObliqueBins);

void scan_oblique_hist(const HistBin* hist, double parent_g, double parent_h, std::int32_t parent_c,
                       double g_nan, double h_nan, std::int32_t c_nan, double lam, double min_h,
                       std::int32_t min_c, double min_gain, double parent_score, std::int64_t pmin,
                       std::int64_t range, double& best_gain, std::int64_t& best_thr,
                       bool& best_nan_left, bool& found) noexcept {
    double gl = 0.0;
    double hl = 0.0;
    std::int32_t cl = 0;

    if (c_nan == 0) {
        for (int b = 0; b < kObliqueBins - 1; ++b) {
            gl += hist[b].g;
            hl += hist[b].h;
            cl += hist[b].c;
            const std::int32_t cr = parent_c - cl;
            if (cl < min_c || cr < min_c) {
                continue;
            }
            const double hr = parent_h - hl;
            if (hl < min_h || hr < min_h) {
                continue;
            }
            const double gr = parent_g - gl;
            const double gain = (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam) - parent_score;
            if (gain > best_gain && gain > min_gain) {
                best_gain = gain;
                best_thr = oblique_bin_threshold(pmin, range, b);
                best_nan_left = (hl >= hr);
                found = true;
            }
        }
        return;
    }

    for (int b = 0; b < kObliqueBins - 1; ++b) {
        gl += hist[b].g;
        hl += hist[b].h;
        cl += hist[b].c;

        // Option A: missing goes left
        {
            const std::int32_t cl_a = cl + c_nan;
            const std::int32_t cr_a = parent_c - cl_a;
            if (cl_a >= min_c && cr_a >= min_c) {
                const double hl_a = hl + h_nan;
                const double hr_a = parent_h - hl_a;
                if (hl_a >= min_h && hr_a >= min_h) {
                    const double gl_a = gl + g_nan;
                    const double gr_a = parent_g - gl_a;
                    const double gain =
                        (gl_a * gl_a) / (hl_a + lam) + (gr_a * gr_a) / (hr_a + lam) - parent_score;
                    if (gain > best_gain && gain > min_gain) {
                        best_gain = gain;
                        best_thr = oblique_bin_threshold(pmin, range, b);
                        best_nan_left = true;
                        found = true;
                    }
                }
            }
        }

        // Option B: missing goes right
        {
            const std::int32_t cr_b = parent_c - cl;
            if (cl >= min_c && cr_b >= min_c) {
                const double hr_b = parent_h - hl;
                if (hl >= min_h && hr_b >= min_h) {
                    const double gr_b = parent_g - gl;
                    const double gain =
                        (gl * gl) / (hl + lam) + (gr_b * gr_b) / (hr_b + lam) - parent_score;
                    if (gain > best_gain && gain > min_gain) {
                        best_gain = gain;
                        best_thr = oblique_bin_threshold(pmin, range, b);
                        best_nan_left = false;
                        found = true;
                    }
                }
            }
        }
    }

    // Final split: all valid projections left vs missing-any-feature right.
    gl += hist[kObliqueBins - 1].g;
    hl += hist[kObliqueBins - 1].h;
    cl += hist[kObliqueBins - 1].c;
    const std::int32_t cr_fin = parent_c - cl;
    if (cl >= min_c && cr_fin >= min_c) {
        const double hr_fin = parent_h - hl;
        if (hl >= min_h && hr_fin >= min_h) {
            const double gr_fin = parent_g - gl;
            const double gain =
                (gl * gl) / (hl + lam) + (gr_fin * gr_fin) / (hr_fin + lam) - parent_score;
            if (gain > best_gain && gain > min_gain) {
                best_gain = gain;
                best_thr = oblique_bin_threshold(pmin, range, kObliqueBins - 1);
                best_nan_left = false;
                found = true;
            }
        }
    }
}

void gather_block_features(const std::int16_t* codes, std::size_t n_rows,
                           std::size_t leaf_code_offset, const std::uint32_t* leaf_rows,
                           bool is_contiguous_leaf, std::size_t mb0, std::size_t len,
                           std::size_t n_used, const std::uint32_t* used_num_feats,
                           std::int16_t* feat_tile, const std::int16_t** col_ptrs) noexcept {
    if (is_contiguous_leaf) {
        const std::size_t base_off = leaf_code_offset + mb0;
        for (std::size_t u = 0; u < n_used; ++u) {
            col_ptrs[u] = codes + static_cast<std::size_t>(used_num_feats[u]) * n_rows + base_off;
        }
        return;
    }
    const std::uint32_t* lr = leaf_rows + mb0;
    for (std::size_t u = 0; u < n_used; ++u) {
        const std::int16_t* src = codes + static_cast<std::size_t>(used_num_feats[u]) * n_rows;
        std::int16_t* dst = feat_tile + u * kObliqueMicroBlock;
        col_ptrs[u] = dst;
        std::size_t i = 0U;
        for (; i + 4U <= len; i += 4U) {
            dst[i + 0U] = src[lr[i + 0U]];
            dst[i + 1U] = src[lr[i + 1U]];
            dst[i + 2U] = src[lr[i + 2U]];
            dst[i + 3U] = src[lr[i + 3U]];
        }
        for (; i < len; ++i) {
            dst[i] = src[lr[i]];
        }
    }
}

void pass1_chunk(const std::int16_t* codes, std::size_t n_rows, std::size_t leaf_code_offset,
                 const std::uint32_t* leaf_rows, std::size_t n_leaf, bool is_contiguous_leaf,
                 std::size_t row_start, std::size_t row_end, const float* g_seq, const float* h_seq,
                 std::size_t m, const ObliqueCandMeta* metas, std::size_t n_used,
                 const std::uint32_t* used_num_feats, bool any_has_nan, bool store_proj,
                 std::int32_t* proj_buf, std::int16_t* feat_tile, const std::int16_t** col_ptrs,
                 std::int32_t* out_pmin, std::int32_t* out_pmax, std::int32_t* out_valid,
                 double* out_gnan, double* out_hnan, std::int32_t* out_cnan) noexcept {
    for (std::size_t j = 0; j < m; ++j) {
        out_pmin[j] = std::numeric_limits<std::int32_t>::max();
        out_pmax[j] = std::numeric_limits<std::int32_t>::min();
        out_valid[j] = 0;
        out_gnan[j] = 0.0;
        out_hnan[j] = 0.0;
        out_cnan[j] = 0;
    }

    for (std::size_t mb0 = row_start; mb0 < row_end; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, row_end - mb0);
        gather_block_features(codes, n_rows, leaf_code_offset, leaf_rows, is_contiguous_leaf, mb0,
                              len, n_used, used_num_feats, feat_tile, col_ptrs);

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

            if (!any_has_nan || cm.has_nan == 0U) {
                if (cm.k == 2U) {
                    if (store_proj) {
                        std::int32_t* p_dst = proj_buf + j * n_leaf + mb0;
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
                        std::int32_t* p_dst = proj_buf + j * n_leaf + mb0;
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
                const float* gm = g_seq + mb0;
                const float* hm = h_seq + mb0;
                double gn = out_gnan[j];
                double hn = out_hnan[j];
                std::int32_t cn = out_cnan[j];
                std::int32_t vc = out_valid[j];
                std::int32_t* p_dst = store_proj ? (proj_buf + j * n_leaf + mb0) : nullptr;
                for (std::size_t i = 0; i < len; ++i) {
                    const std::int16_t x0 = c0[i];
                    const std::int16_t x1 = c1[i];
                    const std::int16_t x2 = c2[i];
                    if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode ||
                        x2 == kQ16SignedNanCode) {
                        if (store_proj) {
                            p_dst[i] = kMissingProjSentinel;
                        }
                        gn += static_cast<double>(gm[i]);
                        hn += static_cast<double>(hm[i]);
                        ++cn;
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
                out_gnan[j] = gn;
                out_hnan[j] = hn;
                out_cnan[j] = cn;
            }
        }
    }
}

void accumulate_bins_no_nan(HistBin* hist, const std::uint8_t* bins, const float* gm,
                            const float* hm, std::size_t len) noexcept {
    std::size_t i = 0U;
    for (; i + 4U <= len; i += 4U) {
        const std::uint8_t b0 = bins[i + 0U];
        const std::uint8_t b1 = bins[i + 1U];
        const std::uint8_t b2 = bins[i + 2U];
        const std::uint8_t b3 = bins[i + 3U];
        hist[b0].g += static_cast<double>(gm[i + 0U]);
        hist[b0].h += static_cast<double>(hm[i + 0U]);
        hist[b0].c += 1;
        hist[b1].g += static_cast<double>(gm[i + 1U]);
        hist[b1].h += static_cast<double>(hm[i + 1U]);
        hist[b1].c += 1;
        hist[b2].g += static_cast<double>(gm[i + 2U]);
        hist[b2].h += static_cast<double>(hm[i + 2U]);
        hist[b2].c += 1;
        hist[b3].g += static_cast<double>(gm[i + 3U]);
        hist[b3].h += static_cast<double>(hm[i + 3U]);
        hist[b3].c += 1;
    }
    for (; i < len; ++i) {
        const std::uint8_t b = bins[i];
        hist[b].g += static_cast<double>(gm[i]);
        hist[b].h += static_cast<double>(hm[i]);
        hist[b].c += 1;
    }
}

void accumulate_bins_with_nan(HistBin* hist, const std::uint8_t* bins, const float* gm,
                              const float* hm, std::size_t len) noexcept {
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint8_t b = bins[i];
        if (b != kMissingBinSentinel) {
            hist[b].g += static_cast<double>(gm[i]);
            hist[b].h += static_cast<double>(hm[i]);
            hist[b].c += 1;
        }
    }
}

void pass2_chunk(const std::int16_t* codes, std::size_t n_rows, std::size_t leaf_code_offset,
                 const std::uint32_t* leaf_rows, std::size_t n_leaf, bool is_contiguous_leaf,
                 std::size_t row_start, std::size_t row_end, const float* g_seq, const float* h_seq,
                 const ObliqueCandMeta* metas, std::size_t n_used,
                 const std::uint32_t* used_num_feats, bool store_proj, const std::int32_t* proj_buf,
                 std::int16_t* feat_tile, const std::int16_t** col_ptrs, std::size_t n_active,
                 const std::uint32_t* active_cands, const std::int32_t* cand_pmin,
                 const FastDiv64* active_divs, HistBin* hists_out) noexcept {
    std::uint8_t bins[kObliqueMicroBlock];
    if (store_proj) {
        for (std::size_t a = 0; a < n_active; ++a) {
            const std::uint32_t j = active_cands[a];
            const ObliqueCandMeta& cm = metas[j];
            const std::int64_t pmin = cand_pmin[j];
            const FastDiv64 div = active_divs[a];
            const std::int32_t* p_col = proj_buf + j * n_leaf;
            HistBin* hist = hists_out + a * static_cast<std::size_t>(kObliqueBins);

            for (std::size_t mb0 = row_start; mb0 < row_end; mb0 += kObliqueMicroBlock) {
                const std::size_t len = std::min(kObliqueMicroBlock, row_end - mb0);
                const std::int32_t* pm = p_col + mb0;
                const float* gm = g_seq + mb0;
                const float* hm = h_seq + mb0;
                if (cm.has_nan == 0U) {
                    for (std::size_t i = 0; i < len; ++i) {
                        const auto num =
                            static_cast<std::uint64_t>(static_cast<std::int64_t>(pm[i]) - pmin) *
                            static_cast<std::uint64_t>(kObliqueBins);
                        bins[i] = static_cast<std::uint8_t>(div.div(num));
                    }
                    accumulate_bins_no_nan(hist, bins, gm, hm, len);
                } else {
                    for (std::size_t i = 0; i < len; ++i) {
                        const std::int32_t p32 = pm[i];
                        if (p32 == kMissingProjSentinel) {
                            bins[i] = kMissingBinSentinel;
                        } else {
                            const auto num =
                                static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin) *
                                static_cast<std::uint64_t>(kObliqueBins);
                            bins[i] = static_cast<std::uint8_t>(div.div(num));
                        }
                    }
                    accumulate_bins_with_nan(hist, bins, gm, hm, len);
                }
            }
        }
        return;
    }

    std::int32_t p_tmp[kObliqueMicroBlock];
    for (std::size_t mb0 = row_start; mb0 < row_end; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, row_end - mb0);
        gather_block_features(codes, n_rows, leaf_code_offset, leaf_rows, is_contiguous_leaf, mb0,
                              len, n_used, used_num_feats, feat_tile, col_ptrs);
        const float* gm = g_seq + mb0;
        const float* hm = h_seq + mb0;

        for (std::size_t a = 0; a < n_active; ++a) {
            const std::uint32_t j = active_cands[a];
            const ObliqueCandMeta& cm = metas[j];
            const std::int64_t pmin = cand_pmin[j];
            const FastDiv64 div = active_divs[a];
            HistBin* hist = hists_out + a * static_cast<std::size_t>(kObliqueBins);

            const std::int16_t* c0 = col_ptrs[cm.uf[0]];
            const std::int16_t* c1 = col_ptrs[cm.uf[1]];
            const std::int32_t w0 = cm.w[0];
            const std::int32_t w1 = cm.w[1];

            if (cm.has_nan == 0U) {
                if (cm.k == 2U) {
                    for (std::size_t i = 0; i < len; ++i) {
                        p_tmp[i] = w0 * c0[i] + w1 * c1[i];
                    }
                } else {
                    const std::int16_t* c2 = col_ptrs[cm.uf[2]];
                    const std::int32_t w2 = cm.w[2];
                    for (std::size_t i = 0; i < len; ++i) {
                        p_tmp[i] = w0 * c0[i] + w1 * c1[i] + w2 * c2[i];
                    }
                }
                for (std::size_t i = 0; i < len; ++i) {
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p_tmp[i]) - pmin) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    bins[i] = static_cast<std::uint8_t>(div.div(num));
                }
                accumulate_bins_no_nan(hist, bins, gm, hm, len);
            } else {
                const std::int16_t* c2 = col_ptrs[cm.uf[2]];
                const std::int32_t w2 = cm.w[2];
                for (std::size_t i = 0; i < len; ++i) {
                    const std::int16_t x0 = c0[i];
                    const std::int16_t x1 = c1[i];
                    const std::int16_t x2 = c2[i];
                    if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode ||
                        x2 == kQ16SignedNanCode) {
                        bins[i] = kMissingBinSentinel;
                    } else {
                        const std::int32_t p32 = w0 * x0 + w1 * x1 + w2 * x2;
                        const auto num =
                            static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - pmin) *
                            static_cast<std::uint64_t>(kObliqueBins);
                        bins[i] = static_cast<std::uint8_t>(div.div(num));
                    }
                }
                accumulate_bins_with_nan(hist, bins, gm, hm, len);
            }
        }
    }
}

ObliqueSplitCandidate eval_oblique_i64_candidates(
    const std::int16_t* codes, std::size_t n_rows, std::size_t leaf_code_offset,
    bool is_contiguous_leaf, const std::uint32_t* leaf_rows, std::size_t n_leaf, const float* g_seq,
    const float* h_seq, double parent_g, double parent_h, std::int32_t parent_c,
    const Params& params, std::size_t m, ObliqueScratch& scratch) noexcept {
    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const double parent_score = (parent_g * parent_g) / (parent_h + lam);

    const int n_threads = resolve_num_threads(params.n_jobs);
    const bool use_omp = (n_threads > 1) && (m >= 2U) && (n_leaf * m >= 50000U);

    auto eval_one = [&](std::size_t j) noexcept {
        ObliqueSplitCandidate cand;
        const std::uint8_t k = scratch.cand_k[j];
        if (k < 2U) {
            scratch.i64_candidates[j] = cand;
            return;
        }
        const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
        const std::int16_t* wq = scratch.cand_w.data() + j * kObliqueMaxK;

        std::int64_t pmin = std::numeric_limits<std::int64_t>::max();
        std::int64_t pmax = std::numeric_limits<std::int64_t>::min();
        double g_nan = 0.0;
        double h_nan = 0.0;
        std::int32_t c_nan = 0;
        std::int32_t c_valid = 0;

        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::size_t r = is_contiguous_leaf ? (leaf_code_offset + t) : leaf_rows[t];
            std::int64_t acc = 0;
            bool missing = false;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c = codes[static_cast<std::size_t>(fn[q]) * n_rows + r];
                if (c == kQ16SignedNanCode) {
                    missing = true;
                    break;
                }
                acc += static_cast<std::int64_t>(wq[q]) * static_cast<std::int64_t>(c);
            }
            if (missing) {
                g_nan += static_cast<double>(g_seq[t]);
                h_nan += static_cast<double>(h_seq[t]);
                ++c_nan;
            } else {
                pmin = std::min(pmin, acc);
                pmax = std::max(pmax, acc);
                ++c_valid;
            }
        }

        if (c_valid < 1 || pmin > pmax || (pmin == pmax && c_nan == 0)) {
            scratch.i64_candidates[j] = cand;
            return;
        }

        HistBin* hist = scratch.hists.data() + j * static_cast<std::size_t>(kObliqueBins);
        std::memset(static_cast<void*>(hist), 0,
                    static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));
        const std::int64_t range = pmax - pmin + 1;

        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::size_t r = is_contiguous_leaf ? (leaf_code_offset + t) : leaf_rows[t];
            std::int64_t acc = 0;
            bool missing = false;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c = codes[static_cast<std::size_t>(fn[q]) * n_rows + r];
                if (c == kQ16SignedNanCode) {
                    missing = true;
                    break;
                }
                acc += static_cast<std::int64_t>(wq[q]) * static_cast<std::int64_t>(c);
            }
            if (!missing) {
                const std::uint8_t b = oblique_bin_index(acc, pmin, range);
                hist[b].g += static_cast<double>(g_seq[t]);
                hist[b].h += static_cast<double>(h_seq[t]);
                hist[b].c += 1;
            }
        }

        double cand_best_gain = -std::numeric_limits<double>::infinity();
        std::int64_t cand_thr = 0;
        bool cand_nan_left = true;
        bool found = false;
        scan_oblique_hist(hist, parent_g, parent_h, parent_c, g_nan, h_nan, c_nan, lam, min_h,
                          min_c, min_gain, parent_score, pmin, range, cand_best_gain, cand_thr,
                          cand_nan_left, found);
        if (found) {
            cand.gain = cand_best_gain;
            cand.k = k;
            const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
            for (std::uint8_t q = 0; q < k; ++q) {
                cand.features[q] = fg[q];
                cand.num_indices[q] = fn[q];
                cand.weights[q] = wq[q];
            }
            cand.threshold = cand_thr;
            cand.nan_go_left = cand_nan_left;
            cand.valid = true;
        }
        scratch.i64_candidates[j] = cand;
    };

    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(m); ++j) {
            eval_one(static_cast<std::size_t>(j));
        }
    } else {
        for (std::size_t j = 0; j < m; ++j) {
            eval_one(j);
        }
    }

    ObliqueSplitCandidate best;
    for (std::size_t j = 0; j < m; ++j) {
        const auto& cand = scratch.i64_candidates[j];
        if (cand.valid && (!best.valid || cand.gain > best.gain)) {
            best = cand;
        }
    }
    return best;
}

} // namespace

std::int16_t round_half_away_i16(double v) noexcept {
    const double mag = std::floor(std::abs(v) + 0.5);
    const double signed_mag = (v < 0.0) ? -mag : mag;
    const double clamped = std::clamp(signed_mag, -static_cast<double>(kObliqueWeightScale),
                                      static_cast<double>(kObliqueWeightScale));
    return static_cast<std::int16_t>(clamped);
}

bool quantize_oblique_weights(const double* w, int k, std::int16_t* wq) noexcept {
    double wmax = 0.0;
    for (int q = 0; q < k; ++q) {
        wmax = std::max(wmax, std::abs(w[q]));
    }
    if (!(wmax > 0.0) || !std::isfinite(wmax)) {
        return false;
    }
    const double s = static_cast<double>(kObliqueWeightScale) / wmax;
    for (int q = 0; q < k; ++q) {
        wq[q] = round_half_away_i16(w[q] * s);
    }
    return true;
}

void ObliqueScratch::prepare(const Dataset& ds, const Params& params) {
    eligible_features.clear();
    eligible_num_indices.clear();
    num_has_nan.assign(ds.n_num_features(), 0U);

    const auto& num_indices = ds.num_feature_indices();
    for (std::size_t j = 0; j < num_indices.size(); ++j) {
        const std::size_t f = num_indices[j];
        const auto& binner = ds.binner(f);
        num_has_nan[j] = binner.has_nan ? 1U : 0U;
        if (!ds.quantizer(j).is_constant) {
            eligible_features.push_back(static_cast<std::uint32_t>(f));
            eligible_num_indices.push_back(static_cast<std::uint32_t>(j));
        }
    }

    all_eligible_features = eligible_features;
    all_eligible_num_indices = eligible_num_indices;

    const std::size_t p_elig = eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U) {
        max_m = 0U;
        cur_m = 0U;
        max_chunks = 0U;
        return;
    }

    const auto m0 = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(p_elig))));
    max_m = static_cast<std::size_t>(params.m_mult) * m0;
    cur_m = max_m;
    max_chunks = (ds.n_rows() + kDetChunkRows - 1U) / kDetChunkRows;
    max_used_feats = std::max<std::size_t>(1U, std::min(ds.n_num_features(), 3U * max_m));
    max_threads = static_cast<std::size_t>(std::max(resolve_num_threads(params.n_jobs), 1));

    const std::size_t needed_proj = std::min(ds.n_rows() * max_m, kObliqueScratchLimitInts);
    proj_buf.resize(needed_proj);
    hists.resize(max_m * static_cast<std::size_t>(kObliqueBins));
    const std::size_t wave_slots = (max_chunks > 1U) ? std::min(max_chunks, kWaveChunks) : 0U;
    wave_hists.resize(wave_slots * max_m * static_cast<std::size_t>(kObliqueBins));
    thread_feat_buf.resize(max_threads * max_used_feats * kObliqueMicroBlock);
    thread_col_ptrs.resize(max_threads * max_used_feats);

    cand_meta.resize(max_m);
    cand_k.resize(max_m);
    cand_num_idx.resize(max_m * kObliqueMaxK);
    cand_feat.resize(max_m * kObliqueMaxK);
    cand_w.resize(max_m * kObliqueMaxK);
    cand_has_nan.resize(max_m);
    used_num_feats.resize(max_used_feats);
    num_to_used.assign(ds.n_num_features(), -1);

    cand_pmin.resize(max_m);
    cand_pmax.resize(max_m);
    cand_valid_cnt.resize(max_m);
    cand_g_nan.resize(max_m);
    cand_h_nan.resize(max_m);
    cand_c_nan.resize(max_m);

    const std::size_t chunk_elems = max_chunks * max_m;
    chunk_pmin.resize(chunk_elems);
    chunk_pmax.resize(chunk_elems);
    chunk_valid_cnt.resize(chunk_elems);
    chunk_g_nan.resize(chunk_elems);
    chunk_h_nan.resize(chunk_elems);
    chunk_c_nan.resize(chunk_elems);

    active_cands.resize(max_m);
    active_ranges.resize(max_m);
    active_divs.resize(max_m);
    i64_candidates.resize(max_m);
}

void ObliqueScratch::restrict_pool(const std::vector<std::uint8_t>& selected,
                                   const Params& params) {
    eligible_features.clear();
    eligible_num_indices.clear();
    for (std::size_t i = 0; i < all_eligible_features.size(); ++i) {
        if (selected[all_eligible_features[i]] != 0U) {
            eligible_features.push_back(all_eligible_features[i]);
            eligible_num_indices.push_back(all_eligible_num_indices[i]);
        }
    }
    const std::size_t p_elig = eligible_features.size();
    if (max_m == 0U || params.m_mult <= 0 || p_elig < 2U) {
        cur_m = 0U;
        return;
    }
    const auto m0 = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(p_elig))));
    cur_m = std::min(max_m, static_cast<std::size_t>(params.m_mult) * m0);
}

ObliqueSplitCandidate find_best_oblique_split(const Dataset& ds, const std::int16_t* codes_base,
                                              std::size_t leaf_code_offset, bool is_contiguous_leaf,
                                              const std::uint32_t* leaf_rows, std::size_t n_leaf,
                                              const float* g_seq, const float* h_seq,
                                              double parent_g, double parent_h,
                                              std::int32_t parent_c, std::size_t tree_index,
                                              std::uint32_t node_id, const Params& params,
                                              ObliqueScratch& scratch) noexcept {
    ObliqueSplitCandidate best;
    const std::size_t p_elig = scratch.eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U || scratch.cur_m == 0U ||
        parent_c < 2 * params.min_data_in_leaf) {
        return best;
    }

    const std::size_t m = scratch.cur_m;
    const std::size_t n_rows = ds.n_rows();
    const std::int16_t* codes = (codes_base != nullptr) ? codes_base : ds.codes_col_major().data();

    // Draw all m candidates deterministically from counter-based RNG.
    std::uint32_t picked[kObliqueMaxK] = {};
    double raw_w[kObliqueMaxK] = {};
    std::int16_t wq[kObliqueMaxK] = {};
    bool all_k_le3 = true;
    bool any_has_nan = false;
    std::size_t n_used = 0U;

    for (std::size_t j = 0; j < m; ++j) {
        CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index),
                       static_cast<std::uint64_t>(node_id), static_cast<std::uint64_t>(j), 0ULL);
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

    if (!all_k_le3) {
        return eval_oblique_i64_candidates(codes, n_rows, leaf_code_offset, is_contiguous_leaf,
                                           leaf_rows, n_leaf, g_seq, h_seq, parent_g, parent_h,
                                           parent_c, params, m, scratch);
    }

    const bool store_proj = (n_leaf * m <= scratch.proj_buf.size());
    std::int32_t* proj_buf = scratch.proj_buf.data();
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    const ObliqueCandMeta* metas = scratch.cand_meta.data();
    const std::uint32_t* used_feats = scratch.used_num_feats.data();
    const int n_threads =
        std::min(resolve_num_threads(params.n_jobs), static_cast<int>(scratch.max_threads));

    std::size_t n_active = 0U;

    if (n_chunks == 1U) {
        if (store_proj && is_contiguous_leaf && (n_threads > 1) && (m >= 2U) &&
            (n_leaf * m >= 32768U)) {
            std::memset(static_cast<void*>(scratch.hists.data()), 0,
                        m * static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));
            std::fill(scratch.active_cands.begin(), scratch.active_cands.begin() + m, 0U);
            const double lam = params.lambda_l2;
            const double min_h = params.min_sum_hessian_in_leaf;
            const std::int32_t min_c = params.min_data_in_leaf;
            const double min_gain = params.min_gain_to_split;
            const double parent_score = (parent_g * parent_g) / (parent_h + lam);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
            for (std::ptrdiff_t j_signed = 0; j_signed < static_cast<std::ptrdiff_t>(m);
                 ++j_signed) {
                const auto j = static_cast<std::size_t>(j_signed);
                const ObliqueCandMeta& cm = metas[j];
                if (cm.k < 2U) {
                    continue;
                }
                const std::int16_t* c0 =
                    codes + static_cast<std::size_t>(cm.fn[0]) * n_rows + leaf_code_offset;
                const std::int16_t* c1 =
                    codes + static_cast<std::size_t>(cm.fn[1]) * n_rows + leaf_code_offset;
                const std::int32_t w0 = cm.w[0];
                const std::int32_t w1 = cm.w[1];
                std::int32_t* p_col = proj_buf + j * n_leaf;

                std::int32_t pmin = std::numeric_limits<std::int32_t>::max();
                std::int32_t pmax = std::numeric_limits<std::int32_t>::min();
                std::int32_t vcnt = 0;
                double gnan = 0.0;
                double hnan = 0.0;
                std::int32_t cnan = 0;

                if (!any_has_nan || cm.has_nan == 0U) {
                    if (cm.k == 2U) {
                        for (std::size_t t = 0; t < n_leaf; ++t) {
                            const std::int32_t p32 = w0 * c0[t] + w1 * c1[t];
                            p_col[t] = p32;
                            pmin = (p32 < pmin) ? p32 : pmin;
                            pmax = (p32 > pmax) ? p32 : pmax;
                        }
                    } else {
                        const std::int16_t* c2 =
                            codes + static_cast<std::size_t>(cm.fn[2]) * n_rows + leaf_code_offset;
                        const std::int32_t w2 = cm.w[2];
                        for (std::size_t t = 0; t < n_leaf; ++t) {
                            const std::int32_t p32 = w0 * c0[t] + w1 * c1[t] + w2 * c2[t];
                            p_col[t] = p32;
                            pmin = (p32 < pmin) ? p32 : pmin;
                            pmax = (p32 > pmax) ? p32 : pmax;
                        }
                    }
                    vcnt = static_cast<std::int32_t>(n_leaf);
                } else {
                    const std::int16_t* c2 =
                        codes + static_cast<std::size_t>(cm.fn[2]) * n_rows + leaf_code_offset;
                    const std::int32_t w2 = cm.w[2];
                    for (std::size_t t = 0; t < n_leaf; ++t) {
                        const std::int16_t x0 = c0[t];
                        const std::int16_t x1 = c1[t];
                        const std::int16_t x2 = c2[t];
                        if (x0 == kQ16SignedNanCode || x1 == kQ16SignedNanCode ||
                            x2 == kQ16SignedNanCode) {
                            p_col[t] = kMissingProjSentinel;
                            gnan += static_cast<double>(g_seq[t]);
                            hnan += static_cast<double>(h_seq[t]);
                            ++cnan;
                        } else {
                            const std::int32_t p32 = w0 * x0 + w1 * x1 + w2 * x2;
                            p_col[t] = p32;
                            pmin = (p32 < pmin) ? p32 : pmin;
                            pmax = (p32 > pmax) ? p32 : pmax;
                            ++vcnt;
                        }
                    }
                }

                if (vcnt < 1 || (pmin >= pmax && !(pmin == pmax && cnan > 0))) {
                    continue;
                }

                const std::int64_t range =
                    static_cast<std::int64_t>(pmax) - static_cast<std::int64_t>(pmin) + 1;
                FastDiv64 div;
                div.prepare(static_cast<std::uint64_t>(range));
                HistBin* hist = scratch.hists.data() + j * static_cast<std::size_t>(kObliqueBins);
                std::uint8_t bins[kObliqueMicroBlock];
                const std::int64_t pmin64 = pmin;

                for (std::size_t mb0 = 0; mb0 < n_leaf; mb0 += kObliqueMicroBlock) {
                    const std::size_t len = std::min(kObliqueMicroBlock, n_leaf - mb0);
                    const std::int32_t* pm = p_col + mb0;
                    const float* gm = g_seq + mb0;
                    const float* hm = h_seq + mb0;
                    if (cm.has_nan == 0U) {
                        for (std::size_t i = 0; i < len; ++i) {
                            const auto num = static_cast<std::uint64_t>(
                                                 static_cast<std::int64_t>(pm[i]) - pmin64) *
                                             static_cast<std::uint64_t>(kObliqueBins);
                            bins[i] = static_cast<std::uint8_t>(div.div(num));
                        }
                        accumulate_bins_no_nan(hist, bins, gm, hm, len);
                    } else {
                        for (std::size_t i = 0; i < len; ++i) {
                            const std::int32_t p32 = pm[i];
                            if (p32 == kMissingProjSentinel) {
                                bins[i] = kMissingBinSentinel;
                            } else {
                                const auto num = static_cast<std::uint64_t>(
                                                     static_cast<std::int64_t>(p32) - pmin64) *
                                                 static_cast<std::uint64_t>(kObliqueBins);
                                bins[i] = static_cast<std::uint8_t>(div.div(num));
                            }
                        }
                        accumulate_bins_with_nan(hist, bins, gm, hm, len);
                    }
                }

                scratch.cand_pmin[j] = pmin;
                scratch.cand_g_nan[j] = gnan;
                scratch.cand_h_nan[j] = hnan;
                scratch.cand_c_nan[j] = cnan;
                scratch.active_ranges[j] = range;
                scratch.active_cands[j] = 1U;
            }

            for (std::size_t j = 0; j < m; ++j) {
                if (scratch.active_cands[j] == 0U) {
                    continue;
                }
                const HistBin* hist =
                    scratch.hists.data() + j * static_cast<std::size_t>(kObliqueBins);
                double cand_best_gain = best.gain;
                std::int64_t cand_thr = 0;
                bool cand_nan_left = true;
                bool found = false;
                scan_oblique_hist(
                    hist, parent_g, parent_h, parent_c, scratch.cand_g_nan[j],
                    scratch.cand_h_nan[j], scratch.cand_c_nan[j], lam, min_h, min_c, min_gain,
                    parent_score, static_cast<std::int64_t>(scratch.cand_pmin[j]),
                    scratch.active_ranges[j], cand_best_gain, cand_thr, cand_nan_left, found);
                if (found) {
                    best.gain = cand_best_gain;
                    const std::uint8_t k = scratch.cand_k[j];
                    best.k = k;
                    const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
                    const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
                    const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
                    for (std::uint8_t q = 0; q < k; ++q) {
                        best.features[q] = fg[q];
                        best.num_indices[q] = fn[q];
                        best.weights[q] = w[q];
                    }
                    best.threshold = cand_thr;
                    best.nan_go_left = cand_nan_left;
                    best.valid = true;
                }
            }
            return best;
        }

        pass1_chunk(codes, n_rows, leaf_code_offset, leaf_rows, n_leaf, is_contiguous_leaf, 0U,
                    n_leaf, g_seq, h_seq, m, metas, n_used, used_feats, any_has_nan, store_proj,
                    proj_buf, scratch.thread_feat_buf.data(), scratch.thread_col_ptrs.data(),
                    scratch.cand_pmin.data(), scratch.cand_pmax.data(),
                    scratch.cand_valid_cnt.data(), scratch.cand_g_nan.data(),
                    scratch.cand_h_nan.data(), scratch.cand_c_nan.data());

        for (std::size_t j = 0; j < m; ++j) {
            if (scratch.cand_k[j] >= 2U && scratch.cand_valid_cnt[j] >= 1 &&
                (scratch.cand_pmin[j] < scratch.cand_pmax[j] ||
                 (scratch.cand_pmin[j] == scratch.cand_pmax[j] && scratch.cand_c_nan[j] > 0))) {
                const std::int64_t range = static_cast<std::int64_t>(scratch.cand_pmax[j]) -
                                           static_cast<std::int64_t>(scratch.cand_pmin[j]) + 1;
                scratch.active_cands[n_active] = static_cast<std::uint32_t>(j);
                scratch.active_ranges[n_active] = range;
                scratch.active_divs[n_active].prepare(static_cast<std::uint64_t>(range));
                ++n_active;
            }
        }
        if (n_active == 0U) {
            return best;
        }

        HistBin* hists = scratch.hists.data();
        std::memset(static_cast<void*>(hists), 0,
                    n_active * static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));
        pass2_chunk(codes, n_rows, leaf_code_offset, leaf_rows, n_leaf, is_contiguous_leaf, 0U,
                    n_leaf, g_seq, h_seq, metas, n_used, used_feats, store_proj, proj_buf,
                    scratch.thread_feat_buf.data(), scratch.thread_col_ptrs.data(), n_active,
                    scratch.active_cands.data(), scratch.cand_pmin.data(),
                    scratch.active_divs.data(), hists);
    } else {
        const bool use_omp = (n_threads > 1) && (n_leaf * m >= 50000U);

        // Pass 1 across deterministic fixed row chunks.
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                const auto cu = static_cast<std::size_t>(c);
                const std::size_t r0 = cu * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
#ifdef _OPENMP
                const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                const std::size_t tid = 0U;
#endif
                std::int16_t* feat_tile = scratch.thread_feat_buf.data() +
                                          tid * scratch.max_used_feats * kObliqueMicroBlock;
                const std::int16_t** col_ptrs =
                    scratch.thread_col_ptrs.data() + tid * scratch.max_used_feats;
                pass1_chunk(
                    codes, n_rows, leaf_code_offset, leaf_rows, n_leaf, is_contiguous_leaf, r0, r1,
                    g_seq, h_seq, m, metas, n_used, used_feats, any_has_nan, store_proj, proj_buf,
                    feat_tile, col_ptrs, scratch.chunk_pmin.data() + cu * m,
                    scratch.chunk_pmax.data() + cu * m, scratch.chunk_valid_cnt.data() + cu * m,
                    scratch.chunk_g_nan.data() + cu * m, scratch.chunk_h_nan.data() + cu * m,
                    scratch.chunk_c_nan.data() + cu * m);
            }
        } else {
            std::int16_t* feat_tile = scratch.thread_feat_buf.data();
            const std::int16_t** col_ptrs = scratch.thread_col_ptrs.data();
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t r0 = c * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                pass1_chunk(codes, n_rows, leaf_code_offset, leaf_rows, n_leaf, is_contiguous_leaf,
                            r0, r1, g_seq, h_seq, m, metas, n_used, used_feats, any_has_nan,
                            store_proj, proj_buf, feat_tile, col_ptrs,
                            scratch.chunk_pmin.data() + c * m, scratch.chunk_pmax.data() + c * m,
                            scratch.chunk_valid_cnt.data() + c * m,
                            scratch.chunk_g_nan.data() + c * m, scratch.chunk_h_nan.data() + c * m,
                            scratch.chunk_c_nan.data() + c * m);
            }
        }

        // Strict ascending chunk reduction for Pass 1 bounds & NaN gradient/hessian sums.
        for (std::size_t j = 0; j < m; ++j) {
            std::int32_t pmin = std::numeric_limits<std::int32_t>::max();
            std::int32_t pmax = std::numeric_limits<std::int32_t>::min();
            std::int32_t vcnt = 0;
            double gnan = 0.0;
            double hnan = 0.0;
            std::int32_t cnan = 0;
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t idx = c * m + j;
                pmin = std::min(pmin, scratch.chunk_pmin[idx]);
                pmax = std::max(pmax, scratch.chunk_pmax[idx]);
                vcnt += scratch.chunk_valid_cnt[idx];
                gnan += scratch.chunk_g_nan[idx];
                hnan += scratch.chunk_h_nan[idx];
                cnan += scratch.chunk_c_nan[idx];
            }
            scratch.cand_pmin[j] = pmin;
            scratch.cand_pmax[j] = pmax;
            scratch.cand_valid_cnt[j] = vcnt;
            scratch.cand_g_nan[j] = gnan;
            scratch.cand_h_nan[j] = hnan;
            scratch.cand_c_nan[j] = cnan;
        }

        for (std::size_t j = 0; j < m; ++j) {
            if (scratch.cand_k[j] >= 2U && scratch.cand_valid_cnt[j] >= 1 &&
                (scratch.cand_pmin[j] < scratch.cand_pmax[j] ||
                 (scratch.cand_pmin[j] == scratch.cand_pmax[j] && scratch.cand_c_nan[j] > 0))) {
                const std::int64_t range = static_cast<std::int64_t>(scratch.cand_pmax[j]) -
                                           static_cast<std::int64_t>(scratch.cand_pmin[j]) + 1;
                scratch.active_cands[n_active] = static_cast<std::uint32_t>(j);
                scratch.active_ranges[n_active] = range;
                scratch.active_divs[n_active].prepare(static_cast<std::uint64_t>(range));
                ++n_active;
            }
        }
        if (n_active == 0U) {
            return best;
        }

        HistBin* hists = scratch.hists.data();
        const std::size_t active_bins = n_active * static_cast<std::size_t>(kObliqueBins);
        std::memset(static_cast<void*>(hists), 0, active_bins * sizeof(HistBin));
        HistBin* wave_buf = scratch.wave_hists.data();

        // Pass 2 wave-chunked accumulation + fixed ascending chunk reduction.
        if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
            {
#ifdef _OPENMP
                const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                const std::size_t tid = 0U;
#endif
                std::int16_t* feat_tile = scratch.thread_feat_buf.data() +
                                          tid * scratch.max_used_feats * kObliqueMicroBlock;
                const std::int16_t** col_ptrs =
                    scratch.thread_col_ptrs.data() + tid * scratch.max_used_feats;

                for (std::size_t w_base = 0; w_base < n_chunks; w_base += kWaveChunks) {
                    const std::size_t w_len = std::min(kWaveChunks, n_chunks - w_base);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
                    for (std::ptrdiff_t w = 0; w < static_cast<std::ptrdiff_t>(w_len); ++w) {
                        const auto wu = static_cast<std::size_t>(w);
                        const std::size_t c = w_base + wu;
                        const std::size_t r0 = c * kDetChunkRows;
                        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                        HistBin* wh = wave_buf + wu * active_bins;
                        std::memset(static_cast<void*>(wh), 0, active_bins * sizeof(HistBin));
                        pass2_chunk(codes, n_rows, leaf_code_offset, leaf_rows, n_leaf,
                                    is_contiguous_leaf, r0, r1, g_seq, h_seq, metas, n_used,
                                    used_feats, store_proj, proj_buf, feat_tile, col_ptrs, n_active,
                                    scratch.active_cands.data(), scratch.cand_pmin.data(),
                                    scratch.active_divs.data(), wh);
                    }

                    if (n_active >= 4U) {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                        for (std::ptrdiff_t a = 0; a < static_cast<std::ptrdiff_t>(n_active); ++a) {
                            const std::size_t off = static_cast<std::size_t>(a) * kObliqueBins;
                            HistBin* dst = hists + off;
                            for (std::size_t w = 0; w < w_len; ++w) {
                                const HistBin* src = wave_buf + w * active_bins + off;
                                for (int b = 0; b < kObliqueBins; ++b) {
                                    dst[b].g += src[b].g;
                                    dst[b].h += src[b].h;
                                    dst[b].c += src[b].c;
                                }
                            }
                        }
                    } else {
#ifdef _OPENMP
#pragma omp single
#endif
                        {
                            for (std::size_t w = 0; w < w_len; ++w) {
                                const HistBin* src = wave_buf + w * active_bins;
                                for (std::size_t k = 0; k < active_bins; ++k) {
                                    hists[k].g += src[k].g;
                                    hists[k].h += src[k].h;
                                    hists[k].c += src[k].c;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            std::int16_t* feat_tile = scratch.thread_feat_buf.data();
            const std::int16_t** col_ptrs = scratch.thread_col_ptrs.data();
            for (std::size_t w_base = 0; w_base < n_chunks; w_base += kWaveChunks) {
                const std::size_t w_len = std::min(kWaveChunks, n_chunks - w_base);
                for (std::size_t w = 0; w < w_len; ++w) {
                    const std::size_t c = w_base + w;
                    const std::size_t r0 = c * kDetChunkRows;
                    const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                    std::memset(static_cast<void*>(wave_buf), 0, active_bins * sizeof(HistBin));
                    pass2_chunk(codes, n_rows, leaf_code_offset, leaf_rows, n_leaf,
                                is_contiguous_leaf, r0, r1, g_seq, h_seq, metas, n_used, used_feats,
                                store_proj, proj_buf, feat_tile, col_ptrs, n_active,
                                scratch.active_cands.data(), scratch.cand_pmin.data(),
                                scratch.active_divs.data(), wave_buf);
                    for (std::size_t k = 0; k < active_bins; ++k) {
                        hists[k].g += wave_buf[k].g;
                        hists[k].h += wave_buf[k].h;
                        hists[k].c += wave_buf[k].c;
                    }
                }
            }
        }
    }

    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const double parent_score = (parent_g * parent_g) / (parent_h + lam);

    for (std::size_t a = 0; a < n_active; ++a) {
        const std::uint32_t j = scratch.active_cands[a];
        const HistBin* hist = scratch.hists.data() + a * static_cast<std::size_t>(kObliqueBins);
        double cand_best_gain = best.gain;
        std::int64_t cand_thr = 0;
        bool cand_nan_left = true;
        bool found = false;
        scan_oblique_hist(hist, parent_g, parent_h, parent_c, scratch.cand_g_nan[j],
                          scratch.cand_h_nan[j], scratch.cand_c_nan[j], lam, min_h, min_c, min_gain,
                          parent_score, static_cast<std::int64_t>(scratch.cand_pmin[j]),
                          scratch.active_ranges[a], cand_best_gain, cand_thr, cand_nan_left, found);
        if (found) {
            best.gain = cand_best_gain;
            const std::uint8_t k = scratch.cand_k[j];
            best.k = k;
            const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
            const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
            const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
            for (std::uint8_t q = 0; q < k; ++q) {
                best.features[q] = fg[q];
                best.num_indices[q] = fn[q];
                best.weights[q] = w[q];
            }
            best.threshold = cand_thr;
            best.nan_go_left = cand_nan_left;
            best.valid = true;
        }
    }

    return best;
}

} // namespace miter
