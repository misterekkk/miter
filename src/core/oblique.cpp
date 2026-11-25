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
constexpr std::size_t kObliqueCandTile = 16U;
constexpr std::size_t kObliqueRowTile = 64U;

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

void pass1_chunk(const std::int16_t* codes, std::size_t p_num, const std::uint32_t* leaf_rows,
                 std::size_t row_start, std::size_t row_end, const float* g_seq, const float* h_seq,
                 std::size_t m, const ObliqueCandMeta* metas, bool any_has_nan, bool store_proj,
                 std::int32_t* proj_buf, std::int32_t* out_pmin, std::int32_t* out_pmax,
                 std::int32_t* out_valid, double* out_gnan, double* out_hnan,
                 std::int32_t* out_cnan) noexcept {
    for (std::size_t j = 0; j < m; ++j) {
        out_pmin[j] = std::numeric_limits<std::int32_t>::max();
        out_pmax[j] = std::numeric_limits<std::int32_t>::min();
        out_valid[j] = 0;
        out_gnan[j] = 0.0;
        out_hnan[j] = 0.0;
        out_cnan[j] = 0;
    }

    if (!any_has_nan) {
        for (std::size_t t = row_start; t < row_end; ++t) {
            const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
            std::int32_t* out_row = store_proj ? (proj_buf + t * m) : nullptr;
            for (std::size_t j = 0; j < m; ++j) {
                const ObliqueCandMeta& cm = metas[j];
                if (cm.k < 2U) {
                    continue;
                }
                const std::int32_t p32 = static_cast<std::int32_t>(cm.w[0]) * row[cm.fn[0]] +
                                         static_cast<std::int32_t>(cm.w[1]) * row[cm.fn[1]] +
                                         static_cast<std::int32_t>(cm.w[2]) * row[cm.fn[2]];
                if (store_proj) {
                    out_row[j] = p32;
                }
                if (p32 < out_pmin[j]) {
                    out_pmin[j] = p32;
                }
                if (p32 > out_pmax[j]) {
                    out_pmax[j] = p32;
                }
                out_valid[j] += 1;
            }
        }
        return;
    }

    for (std::size_t t = row_start; t < row_end; ++t) {
        const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
        const double gt = static_cast<double>(g_seq[t]);
        const double ht = static_cast<double>(h_seq[t]);
        std::int32_t* out_row = store_proj ? (proj_buf + t * m) : nullptr;
        for (std::size_t j = 0; j < m; ++j) {
            const ObliqueCandMeta& cm = metas[j];
            if (cm.k < 2U) {
                continue;
            }
            const std::int16_t c0 = row[cm.fn[0]];
            const std::int16_t c1 = row[cm.fn[1]];
            const std::int16_t c2 = row[cm.fn[2]];
            if (cm.has_nan != 0U &&
                (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode || c2 == kQ16SignedNanCode)) {
                if (store_proj) {
                    out_row[j] = kMissingProjSentinel;
                }
                out_gnan[j] += gt;
                out_hnan[j] += ht;
                out_cnan[j] += 1;
            } else {
                const std::int32_t p32 = static_cast<std::int32_t>(cm.w[0]) * c0 +
                                         static_cast<std::int32_t>(cm.w[1]) * c1 +
                                         static_cast<std::int32_t>(cm.w[2]) * c2;
                if (store_proj) {
                    out_row[j] = p32;
                }
                if (p32 < out_pmin[j]) {
                    out_pmin[j] = p32;
                }
                if (p32 > out_pmax[j]) {
                    out_pmax[j] = p32;
                }
                out_valid[j] += 1;
            }
        }
    }
}

void pass2_chunk(const std::int16_t* codes, std::size_t p_num, const std::uint32_t* leaf_rows,
                 std::size_t row_start, std::size_t row_end, const float* g_seq, const float* h_seq,
                 std::size_t m, const ObliqueCandMeta* metas, bool store_proj,
                 const std::int32_t* proj_buf, std::size_t n_active,
                 const std::uint32_t* active_cands, const std::int32_t* cand_pmin,
                 const FastDiv64* active_divs, HistBin* hists_out) noexcept {
    if (store_proj) {
        for (std::size_t t0 = row_start; t0 < row_end; t0 += kObliqueRowTile) {
            const std::size_t t1 = std::min(t0 + kObliqueRowTile, row_end);
            for (std::size_t a0 = 0; a0 < n_active; a0 += kObliqueCandTile) {
                const std::size_t a1 = std::min(a0 + kObliqueCandTile, n_active);
                for (std::size_t t = t0; t < t1; ++t) {
                    const double gt = static_cast<double>(g_seq[t]);
                    const double ht = static_cast<double>(h_seq[t]);
                    const std::int32_t* p_row = proj_buf + t * m;
                    for (std::size_t a = a0; a < a1; ++a) {
                        const std::uint32_t j = active_cands[a];
                        const std::int32_t p32 = p_row[j];
                        if (p32 == kMissingProjSentinel) {
                            continue;
                        }
                        const auto num = static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) -
                                                                    cand_pmin[j]) *
                                         static_cast<std::uint64_t>(kObliqueBins);
                        const std::size_t b = static_cast<std::size_t>(active_divs[a].div(num));
                        HistBin& hb = hists_out[a * kObliqueBins + b];
                        hb.g += gt;
                        hb.h += ht;
                        hb.c += 1;
                    }
                }
            }
        }
        return;
    }

    for (std::size_t t0 = row_start; t0 < row_end; t0 += kObliqueRowTile) {
        const std::size_t t1 = std::min(t0 + kObliqueRowTile, row_end);
        for (std::size_t a0 = 0; a0 < n_active; a0 += kObliqueCandTile) {
            const std::size_t a1 = std::min(a0 + kObliqueCandTile, n_active);
            for (std::size_t t = t0; t < t1; ++t) {
                const std::int16_t* row = codes + static_cast<std::size_t>(leaf_rows[t]) * p_num;
                const double gt = static_cast<double>(g_seq[t]);
                const double ht = static_cast<double>(h_seq[t]);
                for (std::size_t a = a0; a < a1; ++a) {
                    const std::uint32_t j = active_cands[a];
                    const ObliqueCandMeta& cm = metas[j];
                    const std::int16_t c0 = row[cm.fn[0]];
                    const std::int16_t c1 = row[cm.fn[1]];
                    const std::int16_t c2 = row[cm.fn[2]];
                    if (cm.has_nan != 0U && (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode ||
                                             c2 == kQ16SignedNanCode)) {
                        continue;
                    }
                    const std::int32_t p32 = static_cast<std::int32_t>(cm.w[0]) * c0 +
                                             static_cast<std::int32_t>(cm.w[1]) * c1 +
                                             static_cast<std::int32_t>(cm.w[2]) * c2;
                    const auto num =
                        static_cast<std::uint64_t>(static_cast<std::int64_t>(p32) - cand_pmin[j]) *
                        static_cast<std::uint64_t>(kObliqueBins);
                    const std::size_t b = static_cast<std::size_t>(active_divs[a].div(num));
                    HistBin& hb = hists_out[a * kObliqueBins + b];
                    hb.g += gt;
                    hb.h += ht;
                    hb.c += 1;
                }
            }
        }
    }
}

ObliqueSplitCandidate eval_oblique_i64_candidates(
    const std::int16_t* codes, std::size_t p_num, const std::uint32_t* leaf_rows,
    std::size_t n_leaf, const float* g_seq, const float* h_seq, double parent_g, double parent_h,
    std::int32_t parent_c, const Params& params, std::size_t m, ObliqueScratch& scratch) noexcept {
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
            scratch.i64_candidates[j] = cand;
            return;
        }

        HistBin* hist = scratch.hists.data() + j * static_cast<std::size_t>(kObliqueBins);
        std::memset(static_cast<void*>(hist), 0,
                    static_cast<std::size_t>(kObliqueBins) * sizeof(HistBin));
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

    const std::size_t needed_proj = std::min(ds.n_rows() * max_m, kObliqueScratchLimitInts);
    proj_buf.resize(needed_proj);
    hists.resize(max_m * static_cast<std::size_t>(kObliqueBins));
    const std::size_t wave_slots = (max_chunks > 1U) ? std::min(max_chunks, kWaveChunks) : 0U;
    wave_hists.resize(wave_slots * max_m * static_cast<std::size_t>(kObliqueBins));

    cand_meta.resize(max_m);
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

ObliqueSplitCandidate find_best_oblique_split(const Dataset& ds, const std::uint32_t* leaf_rows,
                                              std::size_t n_leaf, const float* g_seq,
                                              const float* h_seq, double parent_g, double parent_h,
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
    const std::size_t p_num = ds.n_num_features();
    const std::int16_t* codes = ds.codes_row_major().data();

    // Draw all m candidates deterministically from counter-based RNG.
    std::uint32_t picked[kObliqueMaxK] = {};
    double raw_w[kObliqueMaxK] = {};
    std::int16_t wq[kObliqueMaxK] = {};
    bool all_k_le3 = true;
    bool any_has_nan = false;

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
        }
    }

    if (!all_k_le3) {
        return eval_oblique_i64_candidates(codes, p_num, leaf_rows, n_leaf, g_seq, h_seq, parent_g,
                                           parent_h, parent_c, params, m, scratch);
    }

    const bool store_proj = (n_leaf * m <= scratch.proj_buf.size());
    std::int32_t* proj_buf = scratch.proj_buf.data();
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    const ObliqueCandMeta* metas = scratch.cand_meta.data();

    std::size_t n_active = 0U;

    if (n_chunks == 1U) {
        pass1_chunk(codes, p_num, leaf_rows, 0U, n_leaf, g_seq, h_seq, m, metas, any_has_nan,
                    store_proj, proj_buf, scratch.cand_pmin.data(), scratch.cand_pmax.data(),
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
        pass2_chunk(codes, p_num, leaf_rows, 0U, n_leaf, g_seq, h_seq, m, metas, store_proj,
                    proj_buf, n_active, scratch.active_cands.data(), scratch.cand_pmin.data(),
                    scratch.active_divs.data(), hists);
    } else {
        const int n_threads = resolve_num_threads(params.n_jobs);
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
                pass1_chunk(
                    codes, p_num, leaf_rows, r0, r1, g_seq, h_seq, m, metas, any_has_nan,
                    store_proj, proj_buf, scratch.chunk_pmin.data() + cu * m,
                    scratch.chunk_pmax.data() + cu * m, scratch.chunk_valid_cnt.data() + cu * m,
                    scratch.chunk_g_nan.data() + cu * m, scratch.chunk_h_nan.data() + cu * m,
                    scratch.chunk_c_nan.data() + cu * m);
            }
        } else {
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t r0 = c * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                pass1_chunk(codes, p_num, leaf_rows, r0, r1, g_seq, h_seq, m, metas, any_has_nan,
                            store_proj, proj_buf, scratch.chunk_pmin.data() + c * m,
                            scratch.chunk_pmax.data() + c * m,
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
                        pass2_chunk(codes, p_num, leaf_rows, r0, r1, g_seq, h_seq, m, metas,
                                    store_proj, proj_buf, n_active, scratch.active_cands.data(),
                                    scratch.cand_pmin.data(), scratch.active_divs.data(), wh);
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
            for (std::size_t w_base = 0; w_base < n_chunks; w_base += kWaveChunks) {
                const std::size_t w_len = std::min(kWaveChunks, n_chunks - w_base);
                for (std::size_t w = 0; w < w_len; ++w) {
                    const std::size_t c = w_base + w;
                    const std::size_t r0 = c * kDetChunkRows;
                    const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                    std::memset(static_cast<void*>(wave_buf), 0, active_bins * sizeof(HistBin));
                    pass2_chunk(codes, p_num, leaf_rows, r0, r1, g_seq, h_seq, m, metas, store_proj,
                                proj_buf, n_active, scratch.active_cands.data(),
                                scratch.cand_pmin.data(), scratch.active_divs.data(), wave_buf);
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
