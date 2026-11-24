#include "core/histogram.hpp"

#include <algorithm>
#include <cstring>

#include "util/threading.hpp"

namespace miter {

FeatureHistLayout FeatureHistLayout::build(const Dataset& ds) {
    const std::size_t p = ds.n_features();
    FeatureHistLayout layout;
    layout.offsets.assign(p, 0);
    layout.num_regular_bins.assign(p, 0);
    layout.has_nan.assign(p, 0);
    layout.is_categorical.assign(p, 0);

    std::size_t offset = 0;
    for (std::size_t f = 0; f < p; ++f) {
        if (ds.feature_kinds()[f] == FeatureKind::Numeric) {
            const auto& binner = ds.binner(f);
            if (binner.num_bins < 2U && !binner.has_nan) {
                continue;
            }
            layout.offsets[f] = offset;
            layout.num_regular_bins[f] = binner.num_bins;
            layout.has_nan[f] = binner.has_nan ? 1U : 0U;
            layout.is_categorical[f] = 0U;
            const auto fi = static_cast<std::uint32_t>(f);
            layout.active_features.push_back(fi);
            layout.active_num_features.push_back(fi);
            offset += static_cast<std::size_t>(binner.num_bins) + (binner.has_nan ? 1U : 0U);
        } else {
            const std::uint16_t nc = ds.cat_encoder(f).num_codes();
            if (nc < 2U) {
                continue;
            }
            layout.offsets[f] = offset;
            layout.num_regular_bins[f] = nc;
            layout.has_nan[f] = 0U;
            layout.is_categorical[f] = 1U;
            const auto fi = static_cast<std::uint32_t>(f);
            layout.active_features.push_back(fi);
            layout.active_cat_features.push_back(fi);
            offset += static_cast<std::size_t>(nc);
        }
    }
    layout.total_bins = offset;
    return layout;
}

HistPool::HistPool(std::size_t num_slots, std::size_t bins_per_slot) {
    reset(num_slots, bins_per_slot);
}

void HistPool::reset(std::size_t num_slots, std::size_t bins_per_slot) {
    bins_per_slot_ = bins_per_slot;
    buffer_.resize(num_slots * bins_per_slot);
}

namespace {

void build_axis_feature_range(const std::uint8_t* col, std::size_t offset,
                              const std::uint32_t* leaf_rows, std::size_t r0, std::size_t r1,
                              bool is_contiguous_root, const float* g, const float* h,
                              const float* g_ordered, const float* h_ordered,
                              HistBin* hist_out) noexcept {
    HistBin* dst = hist_out + offset;
    if (is_contiguous_root) {
        std::size_t i = r0;
        for (; i + 4U <= r1; i += 4U) {
            const std::uint8_t b0 = col[i];
            const std::uint8_t b1 = col[i + 1U];
            const std::uint8_t b2 = col[i + 2U];
            const std::uint8_t b3 = col[i + 3U];
            dst[b0].g += static_cast<double>(g[i]);
            dst[b0].h += static_cast<double>(h[i]);
            dst[b0].c += 1;
            dst[b1].g += static_cast<double>(g[i + 1U]);
            dst[b1].h += static_cast<double>(h[i + 1U]);
            dst[b1].c += 1;
            dst[b2].g += static_cast<double>(g[i + 2U]);
            dst[b2].h += static_cast<double>(h[i + 2U]);
            dst[b2].c += 1;
            dst[b3].g += static_cast<double>(g[i + 3U]);
            dst[b3].h += static_cast<double>(h[i + 3U]);
            dst[b3].c += 1;
        }
        for (; i < r1; ++i) {
            const std::uint8_t b = col[i];
            dst[b].g += static_cast<double>(g[i]);
            dst[b].h += static_cast<double>(h[i]);
            dst[b].c += 1;
        }
        return;
    }

    std::size_t t = r0;
    for (; t + 4U <= r1; t += 4U) {
        const std::uint8_t b0 = col[leaf_rows[t]];
        const std::uint8_t b1 = col[leaf_rows[t + 1U]];
        const std::uint8_t b2 = col[leaf_rows[t + 2U]];
        const std::uint8_t b3 = col[leaf_rows[t + 3U]];
        dst[b0].g += static_cast<double>(g_ordered[t]);
        dst[b0].h += static_cast<double>(h_ordered[t]);
        dst[b0].c += 1;
        dst[b1].g += static_cast<double>(g_ordered[t + 1U]);
        dst[b1].h += static_cast<double>(h_ordered[t + 1U]);
        dst[b1].c += 1;
        dst[b2].g += static_cast<double>(g_ordered[t + 2U]);
        dst[b2].h += static_cast<double>(h_ordered[t + 2U]);
        dst[b2].c += 1;
        dst[b3].g += static_cast<double>(g_ordered[t + 3U]);
        dst[b3].h += static_cast<double>(h_ordered[t + 3U]);
        dst[b3].c += 1;
    }
    for (; t < r1; ++t) {
        const std::uint8_t b = col[leaf_rows[t]];
        dst[b].g += static_cast<double>(g_ordered[t]);
        dst[b].h += static_cast<double>(h_ordered[t]);
        dst[b].c += 1;
    }
}

} // namespace

void build_axis_histograms(const Dataset& ds, const FeatureHistLayout& layout,
                           const std::uint32_t* leaf_rows, std::size_t n_leaf,
                           bool is_contiguous_root, const float* g, const float* h,
                           float* g_ordered, float* h_ordered, HistBin* hist_out, int n_jobs,
                           HistBin* wave_hist_buf) noexcept {
    if (layout.total_bins == 0U) {
        return;
    }
    std::memset(hist_out, 0, layout.total_bins * sizeof(HistBin));
    if (n_leaf == 0U) {
        return;
    }

    const int n_threads = resolve_num_threads(n_jobs);
    const std::size_t n_rows = ds.n_rows();
    const std::uint8_t* bins_base = ds.bins_col_major().data();
    const std::size_t n_active = layout.active_features.size();

    if (!is_contiguous_root) {
        if (n_threads > 1 && n_leaf >= 16384U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
            for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(n_leaf); ++t) {
                const std::uint32_t r = leaf_rows[static_cast<std::size_t>(t)];
                g_ordered[t] = g[r];
                h_ordered[t] = h[r];
            }
        } else {
            for (std::size_t t = 0; t < n_leaf; ++t) {
                const std::uint32_t r = leaf_rows[t];
                g_ordered[t] = g[r];
                h_ordered[t] = h[r];
            }
        }
    }

    // Routing between feature-parallel and chunked-per-feature depends strictly on shape so
    // summation order stays invariant to thread count.
    const bool use_chunked_axis =
        (n_active < 16U) && (n_leaf >= 32768U) && (wave_hist_buf != nullptr);

    if (!use_chunked_axis) {
        const bool par_features =
            (n_threads > 1) && (n_active >= 2U) && (n_leaf * n_active >= 20000U);
        if (par_features) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
            for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(n_active); ++idx) {
                const std::uint32_t f = layout.active_features[static_cast<std::size_t>(idx)];
                const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n_rows;
                build_axis_feature_range(col, layout.offsets[f], leaf_rows, 0U, n_leaf,
                                         is_contiguous_root, g, h, g_ordered, h_ordered, hist_out);
            }
        } else {
            for (const std::uint32_t f : layout.active_features) {
                const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n_rows;
                build_axis_feature_range(col, layout.offsets[f], leaf_rows, 0U, n_leaf,
                                         is_contiguous_root, g, h, g_ordered, h_ordered, hist_out);
            }
        }
        return;
    }

    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    const std::size_t total_bins = layout.total_bins;
    const bool use_omp = (n_threads > 1);

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
                    HistBin* wh = wave_hist_buf + wu * total_bins;
                    std::memset(wh, 0, total_bins * sizeof(HistBin));
                    for (const std::uint32_t f : layout.active_features) {
                        const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n_rows;
                        build_axis_feature_range(col, layout.offsets[f], leaf_rows, r0, r1,
                                                 is_contiguous_root, g, h, g_ordered, h_ordered,
                                                 wh);
                    }
                }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(total_bins); ++b) {
                    const auto bu = static_cast<std::size_t>(b);
                    HistBin acc = hist_out[bu];
                    for (std::size_t w = 0; w < w_len; ++w) {
                        const HistBin& src = wave_hist_buf[w * total_bins + bu];
                        acc.g += src.g;
                        acc.h += src.h;
                        acc.c += src.c;
                    }
                    hist_out[bu] = acc;
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
                std::memset(wave_hist_buf, 0, total_bins * sizeof(HistBin));
                for (const std::uint32_t f : layout.active_features) {
                    const std::uint8_t* col = bins_base + static_cast<std::size_t>(f) * n_rows;
                    build_axis_feature_range(col, layout.offsets[f], leaf_rows, r0, r1,
                                             is_contiguous_root, g, h, g_ordered, h_ordered,
                                             wave_hist_buf);
                }
                for (std::size_t b = 0; b < total_bins; ++b) {
                    hist_out[b].g += wave_hist_buf[b].g;
                    hist_out[b].h += wave_hist_buf[b].h;
                    hist_out[b].c += wave_hist_buf[b].c;
                }
            }
        }
    }
}

void subtract_histograms(HistBin* dst_large, const HistBin* parent, const HistBin* small,
                         std::size_t total_bins) noexcept {
    for (std::size_t k = 0; k < total_bins; ++k) {
        dst_large[k].g = parent[k].g - small[k].g;
        dst_large[k].h = parent[k].h - small[k].h;
        dst_large[k].c = parent[k].c - small[k].c;
    }
}

AxisSplitCandidate find_best_axis_split(const Dataset& ds, const FeatureHistLayout& layout,
                                        const HistBin* hist, double parent_g, double parent_h,
                                        std::int32_t parent_c, const Params& params) noexcept {
    AxisSplitCandidate best;
    if (layout.active_num_features.empty() || parent_c < 2 * params.min_data_in_leaf) {
        return best;
    }

    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const double parent_score = (parent_g * parent_g) / (parent_h + lam);

    for (const std::uint32_t f : layout.active_num_features) {
        const std::uint16_t nb = layout.num_regular_bins[f];
        const HistBin* fh = hist + layout.offsets[f];
        const auto& edges = ds.binner(f).edges;

        std::int32_t c_nan = 0;
        double g_nan = 0.0;
        double h_nan = 0.0;
        if (layout.has_nan[f] != 0U) {
            c_nan = fh[nb].c;
            g_nan = fh[nb].g;
            h_nan = fh[nb].h;
        }

        double gl = 0.0;
        double hl = 0.0;
        std::int32_t cl = 0;

        if (c_nan == 0) {
            for (std::uint16_t b = 0; b + 1U < nb; ++b) {
                gl += fh[b].g;
                hl += fh[b].h;
                cl += fh[b].c;
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
                if (gain > best.gain && gain > min_gain) {
                    best.gain = gain;
                    best.feature = f;
                    best.bin_idx = static_cast<std::uint8_t>(b);
                    best.threshold_u16 = edges[b];
                    best.nan_go_left = (hl >= hr);
                    best.valid = true;
                }
            }
        } else {
            for (std::uint16_t b = 0; b + 1U < nb; ++b) {
                gl += fh[b].g;
                hl += fh[b].h;
                cl += fh[b].c;

                // Option A: NaN goes left
                {
                    const std::int32_t cl_a = cl + c_nan;
                    const std::int32_t cr_a = parent_c - cl_a;
                    if (cl_a >= min_c && cr_a >= min_c) {
                        const double hl_a = hl + h_nan;
                        const double hr_a = parent_h - hl_a;
                        if (hl_a >= min_h && hr_a >= min_h) {
                            const double gl_a = gl + g_nan;
                            const double gr_a = parent_g - gl_a;
                            const double gain = (gl_a * gl_a) / (hl_a + lam) +
                                                (gr_a * gr_a) / (hr_a + lam) - parent_score;
                            if (gain > best.gain && gain > min_gain) {
                                best.gain = gain;
                                best.feature = f;
                                best.bin_idx = static_cast<std::uint8_t>(b);
                                best.threshold_u16 = edges[b];
                                best.nan_go_left = true;
                                best.valid = true;
                            }
                        }
                    }
                }

                // Option B: NaN goes right
                {
                    const std::int32_t cr_b = parent_c - cl;
                    if (cl >= min_c && cr_b >= min_c) {
                        const double hr_b = parent_h - hl;
                        if (hl >= min_h && hr_b >= min_h) {
                            const double gr_b = parent_g - gl;
                            const double gain = (gl * gl) / (hl + lam) +
                                                (gr_b * gr_b) / (hr_b + lam) - parent_score;
                            if (gain > best.gain && gain > min_gain) {
                                best.gain = gain;
                                best.feature = f;
                                best.bin_idx = static_cast<std::uint8_t>(b);
                                best.threshold_u16 = edges[b];
                                best.nan_go_left = false;
                                best.valid = true;
                            }
                        }
                    }
                }
            }

            // Final split: all finite bins left vs NaN right.
            gl += fh[nb - 1U].g;
            hl += fh[nb - 1U].h;
            cl += fh[nb - 1U].c;
            const std::int32_t cr_fin = parent_c - cl;
            if (cl >= min_c && cr_fin >= min_c) {
                const double hr_fin = parent_h - hl;
                if (hl >= min_h && hr_fin >= min_h) {
                    const double gr_fin = parent_g - gl;
                    const double gain =
                        (gl * gl) / (hl + lam) + (gr_fin * gr_fin) / (hr_fin + lam) - parent_score;
                    if (gain > best.gain && gain > min_gain) {
                        best.gain = gain;
                        best.feature = f;
                        best.bin_idx = static_cast<std::uint8_t>(nb - 1U);
                        best.threshold_u16 = kQ16NanCode;
                        best.nan_go_left = false;
                        best.valid = true;
                    }
                }
            }
        }
    }

    return best;
}

} // namespace miter
