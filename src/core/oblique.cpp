#include "core/oblique.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "core/rng.hpp"
#include "data/quantizer.hpp"

namespace miter {

namespace {

constexpr std::int32_t kMissingProjSentinel = std::numeric_limits<std::int32_t>::min();

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

ObliqueSplitCandidate eval_oblique_i64_candidates(
    const std::int16_t* codes, std::size_t p_num, const std::uint32_t* leaf_rows,
    std::size_t n_leaf, const float* g_seq, const float* h_seq, double parent_g, double parent_h,
    std::int32_t parent_c, const Params& params, std::size_t m, ObliqueScratch& scratch) noexcept {
    ObliqueSplitCandidate best;
    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const double parent_score = (parent_g * parent_g) / (parent_h + lam);

    for (std::size_t j = 0; j < m; ++j) {
        const std::uint8_t k = scratch.cand_k[j];
        if (k < 2U) {
            continue;
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
            const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
            std::int64_t acc = 0;
            bool missing = false;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c = row[fn[q]];
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
            continue;
        }

        HistBin* hist = scratch.hists.data();
        std::memset(hist, 0, static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));
        const std::int64_t range = pmax - pmin + 1;

        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
            std::int64_t acc = 0;
            bool missing = false;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c = row[fn[q]];
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

        double cand_best_gain = best.gain;
        std::int64_t cand_thr = 0;
        bool cand_nan_left = true;
        bool found = false;
        scan_oblique_hist(hist, parent_g, parent_h, parent_c, g_nan, h_nan, c_nan, lam, min_h,
                          min_c, min_gain, parent_score, pmin, range, cand_best_gain, cand_thr,
                          cand_nan_left, found);
        if (found) {
            best.gain = cand_best_gain;
            best.k = k;
            const std::uint32_t* fg = scratch.cand_feat.data() + j * kObliqueMaxK;
            for (std::uint8_t q = 0; q < k; ++q) {
                best.features[q] = fg[q];
                best.num_indices[q] = fn[q];
                best.weights[q] = wq[q];
            }
            best.threshold = cand_thr;
            best.nan_go_left = cand_nan_left;
            best.valid = true;
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

    const std::size_t p_elig = eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U) {
        max_m = 0U;
        return;
    }

    const auto m0 = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(p_elig))));
    max_m = static_cast<std::size_t>(params.m_mult) * m0;

    const std::size_t needed_proj = std::min(ds.n_rows() * max_m, kObliqueScratchLimitInts);
    proj_buf.resize(needed_proj);
    hists.resize(max_m * static_cast<std::size_t>(kObliqueBins));

    cand_k.resize(max_m);
    cand_num_idx.resize(max_m * kObliqueMaxK);
    cand_feat.resize(max_m * kObliqueMaxK);
    cand_w.resize(max_m * kObliqueMaxK);
    cand_has_nan.resize(max_m);

    cand_pmin.resize(max_m);
    cand_pmax.resize(max_m);
    cand_valid_cnt.resize(max_m);
    cand_g_nan.resize(max_m);
    cand_h_nan.resize(max_m);
    cand_c_nan.resize(max_m);
    active_cands.resize(max_m);
    active_ranges.resize(max_m);
}

ObliqueSplitCandidate find_best_oblique_split(const Dataset& ds, const std::uint32_t* leaf_rows,
                                              std::size_t n_leaf, const float* g_seq,
                                              const float* h_seq, double parent_g, double parent_h,
                                              std::int32_t parent_c, std::size_t tree_index,
                                              std::uint32_t node_id, const Params& params,
                                              ObliqueScratch& scratch) noexcept {
    ObliqueSplitCandidate best;
    const std::size_t p_elig = scratch.eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U || scratch.max_m == 0U ||
        parent_c < 2 * params.min_data_in_leaf) {
        return best;
    }

    const std::size_t m = scratch.max_m;
    const std::size_t p_num = ds.n_num_features();
    const std::int16_t* codes = ds.codes_row_major().data();

    // Draw all m candidates deterministically from counter-based RNG.
    std::uint32_t picked[kObliqueMaxK] = {};
    double raw_w[kObliqueMaxK] = {};
    std::int16_t wq[kObliqueMaxK] = {};
    bool all_k_le3 = true;

    for (std::size_t j = 0; j < m; ++j) {
        CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index),
                       static_cast<std::uint64_t>(node_id), static_cast<std::uint64_t>(j), 0ULL);
        int k =
            (params.k_proj > 0) ? params.k_proj : static_cast<int>(2U + (rng.next_u64() & 1ULL));
        k = std::min(k, static_cast<int>(p_elig));
        k = std::min(k, kObliqueMaxK);
        if (k < 2) {
            scratch.cand_k[j] = 0U;
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
    }

    if (!all_k_le3) {
        return eval_oblique_i64_candidates(codes, p_num, leaf_rows, n_leaf, g_seq, h_seq, parent_g,
                                           parent_h, parent_c, params, m, scratch);
    }

    for (std::size_t j = 0; j < m; ++j) {
        scratch.cand_pmin[j] = std::numeric_limits<std::int32_t>::max();
        scratch.cand_pmax[j] = std::numeric_limits<std::int32_t>::min();
        scratch.cand_valid_cnt[j] = 0;
        scratch.cand_g_nan[j] = 0.0;
        scratch.cand_h_nan[j] = 0.0;
        scratch.cand_c_nan[j] = 0;
    }

    const bool store_proj = (n_leaf * m <= scratch.proj_buf.size());
    std::int32_t* proj_buf = scratch.proj_buf.data();

    // Pass 1: compute projections and [Pmin, Pmax] + missing stats across all m candidates in 1
    // pass.
    for (std::size_t t = 0; t < n_leaf; ++t) {
        const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
        const double gt = static_cast<double>(g_seq[t]);
        const double ht = static_cast<double>(h_seq[t]);
        std::int32_t* out_row = store_proj ? (proj_buf + t * m) : nullptr;

        for (std::size_t j = 0; j < m; ++j) {
            const std::uint8_t k = scratch.cand_k[j];
            if (k < 2U) {
                continue;
            }
            const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
            const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;

            if (scratch.cand_has_nan[j] == 0U) {
                std::int32_t p32 = static_cast<std::int32_t>(w[0]) * row[fn[0]] +
                                   static_cast<std::int32_t>(w[1]) * row[fn[1]];
                if (k == 3U) {
                    p32 += static_cast<std::int32_t>(w[2]) * row[fn[2]];
                }
                if (store_proj) {
                    out_row[j] = p32;
                }
                if (p32 < scratch.cand_pmin[j]) {
                    scratch.cand_pmin[j] = p32;
                }
                if (p32 > scratch.cand_pmax[j]) {
                    scratch.cand_pmax[j] = p32;
                }
                scratch.cand_valid_cnt[j] += 1;
            } else {
                const std::int16_t c0 = row[fn[0]];
                const std::int16_t c1 = row[fn[1]];
                const std::int16_t c2 = (k == 3U) ? row[fn[2]] : static_cast<std::int16_t>(0);
                if (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode ||
                    (k == 3U && c2 == kQ16SignedNanCode)) {
                    if (store_proj) {
                        out_row[j] = kMissingProjSentinel;
                    }
                    scratch.cand_g_nan[j] += gt;
                    scratch.cand_h_nan[j] += ht;
                    scratch.cand_c_nan[j] += 1;
                } else {
                    std::int32_t p32 =
                        static_cast<std::int32_t>(w[0]) * c0 + static_cast<std::int32_t>(w[1]) * c1;
                    if (k == 3U) {
                        p32 += static_cast<std::int32_t>(w[2]) * c2;
                    }
                    if (store_proj) {
                        out_row[j] = p32;
                    }
                    if (p32 < scratch.cand_pmin[j]) {
                        scratch.cand_pmin[j] = p32;
                    }
                    if (p32 > scratch.cand_pmax[j]) {
                        scratch.cand_pmax[j] = p32;
                    }
                    scratch.cand_valid_cnt[j] += 1;
                }
            }
        }
    }

    std::size_t n_active = 0U;
    for (std::size_t j = 0; j < m; ++j) {
        if (scratch.cand_k[j] >= 2U && scratch.cand_valid_cnt[j] >= 1 &&
            (scratch.cand_pmin[j] < scratch.cand_pmax[j] ||
             (scratch.cand_pmin[j] == scratch.cand_pmax[j] && scratch.cand_c_nan[j] > 0))) {
            scratch.active_cands[n_active] = static_cast<std::uint32_t>(j);
            scratch.active_ranges[n_active] = static_cast<std::int64_t>(scratch.cand_pmax[j]) -
                                              static_cast<std::int64_t>(scratch.cand_pmin[j]) + 1;
            ++n_active;
        }
    }
    if (n_active == 0U) {
        return best;
    }

    HistBin* hists = scratch.hists.data();
    std::memset(hists, 0, m * static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));

    // Pass 2: bin projections into hist[j][0..63].
    if (store_proj) {
        for (std::size_t t = 0; t < n_leaf; ++t) {
            const double gt = static_cast<double>(g_seq[t]);
            const double ht = static_cast<double>(h_seq[t]);
            const std::int32_t* p_row = proj_buf + t * m;
            for (std::size_t a = 0; a < n_active; ++a) {
                const std::uint32_t j = scratch.active_cands[a];
                const std::int32_t p32 = p_row[j];
                if (p32 == kMissingProjSentinel) {
                    continue;
                }
                const std::uint8_t b = oblique_bin_index(
                    static_cast<std::int64_t>(p32), static_cast<std::int64_t>(scratch.cand_pmin[j]),
                    scratch.active_ranges[a]);
                HistBin& hb = hists[static_cast<std::size_t>(j) * kObliqueBins + b];
                hb.g += gt;
                hb.h += ht;
                hb.c += 1;
            }
        }
    } else {
        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
            const double gt = static_cast<double>(g_seq[t]);
            const double ht = static_cast<double>(h_seq[t]);
            for (std::size_t a = 0; a < n_active; ++a) {
                const std::uint32_t j = scratch.active_cands[a];
                const std::uint8_t k = scratch.cand_k[j];
                const std::uint32_t* fn = scratch.cand_num_idx.data() + j * kObliqueMaxK;
                const std::int16_t* w = scratch.cand_w.data() + j * kObliqueMaxK;
                const std::int16_t c0 = row[fn[0]];
                const std::int16_t c1 = row[fn[1]];
                const std::int16_t c2 = (k == 3U) ? row[fn[2]] : static_cast<std::int16_t>(0);
                if (scratch.cand_has_nan[j] != 0U &&
                    (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode ||
                     (k == 3U && c2 == kQ16SignedNanCode))) {
                    continue;
                }
                std::int32_t p32 =
                    static_cast<std::int32_t>(w[0]) * c0 + static_cast<std::int32_t>(w[1]) * c1;
                if (k == 3U) {
                    p32 += static_cast<std::int32_t>(w[2]) * c2;
                }
                const std::uint8_t b = oblique_bin_index(
                    static_cast<std::int64_t>(p32), static_cast<std::int64_t>(scratch.cand_pmin[j]),
                    scratch.active_ranges[a]);
                HistBin& hb = hists[static_cast<std::size_t>(j) * kObliqueBins + b];
                hb.g += gt;
                hb.h += ht;
                hb.c += 1;
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
        const HistBin* hist = hists + static_cast<std::size_t>(j) * kObliqueBins;
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
