#include "core/tree_learner.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>

#include "core/categorical.hpp"
#include "core/rng.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

// RNG purpose ids (0 is used by oblique candidates, 3 by extra_trees axis thresholds).
constexpr std::uint64_t kPurposeFeatureFraction = 1U;
constexpr std::uint64_t kPurposeBagging = 2U;

// Number of items kept for a fraction in (0, 1]: round(total * fraction), at least 1.
std::size_t fraction_count(std::size_t total, double fraction) noexcept {
    const auto k = static_cast<std::size_t>(std::llround(static_cast<double>(total) * fraction));
    return std::min(total, std::max<std::size_t>(1U, k));
}

// Exact-size sample without replacement (selection sampling over a single counter-based stream
// keyed by (random_state, tree_index, 0, 0, purpose)); bag rows ascending in row_indices[0, n_bag),
// the remaining rows ascending in oob_rows. Returns n_bag.
std::uint32_t draw_bag(const Params& params, std::size_t tree_index, std::size_t n,
                       std::uint32_t* bag, std::vector<std::uint32_t>& oob) {
    const std::size_t n_bag = fraction_count(n, params.bagging_fraction);
    CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index), 0U, 0U,
                   kPurposeBagging);
    oob.clear();
    std::size_t needed = n_bag;
    std::size_t taken = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t remaining = n - i;
        if (needed > 0U &&
            rng.next_uniform01() * static_cast<double>(remaining) < static_cast<double>(needed)) {
            bag[taken++] = static_cast<std::uint32_t>(i);
            --needed;
        } else {
            oob.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return static_cast<std::uint32_t>(taken);
}

// Per-tree feature subset (Fisher-Yates prefix over the active features, keyed by
// (random_state, tree_index, 0, 0, purpose)); narrows the layout's active lists and the oblique
// projection pool. Selected features keep ascending order, so iteration order is deterministic.
void select_features(const Params& params, std::size_t tree_index, TreeWorkspace& ws) {
    const auto& all = ws.base_active_features;
    const std::size_t k = fraction_count(all.size(), params.feature_fraction);
    std::vector<std::uint32_t>& perm = ws.feature_perm;
    std::copy(all.begin(), all.end(), perm.begin());
    CounterRng rng(params.random_state, static_cast<std::uint64_t>(tree_index), 0U, 0U,
                   kPurposeFeatureFraction);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(rng.next_u64() % (perm.size() - i));
        std::swap(perm[i], perm[j]);
    }
    std::fill(ws.feature_selected.begin(), ws.feature_selected.end(), std::uint8_t{0});
    for (std::size_t i = 0; i < k; ++i) {
        ws.feature_selected[perm[i]] = 1U;
    }
    auto narrow = [&](const std::vector<std::uint32_t>& src, std::vector<std::uint32_t>& dst) {
        dst.clear();
        for (const std::uint32_t f : src) {
            if (ws.feature_selected[f] != 0U) {
                dst.push_back(f);
            }
        }
    };
    narrow(ws.base_active_features, ws.layout.active_features);
    narrow(ws.base_active_num_features, ws.layout.active_num_features);
    narrow(ws.base_active_cat_features, ws.layout.active_cat_features);
    ws.oblique.restrict_pool(ws.feature_selected, params);
}

struct SplitChoice {
    NodeKind kind = NodeKind::Leaf;
    double gain = -std::numeric_limits<double>::infinity();
    bool valid = false;
    AxisSplitCandidate axis;
    CatSplitCandidate cat;
    ObliqueSplitCandidate oblique;
};

struct LeafState {
    std::uint32_t node_id = 0;
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    int depth = 0;
    bool gh_packed = false;
    std::size_t hist_slot = 0;
    double g_sum = 0.0;
    double h_sum = 0.0;
    SplitChoice split;
};

void sum_leaf_stats(const std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                    const float* g, const float* h, int n_jobs, TreeWorkspace& ws, double& g_sum,
                    double& h_sum) noexcept {
    const std::size_t n_leaf = static_cast<std::size_t>(end - start);
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    if (n_chunks <= 1U) {
        double gs = 0.0;
        double hs = 0.0;
        for (std::uint32_t t = start; t < end; ++t) {
            const std::uint32_t r = rows[t];
            gs += static_cast<double>(g[r]);
            hs += static_cast<double>(h[r]);
        }
        g_sum = gs;
        h_sum = hs;
        return;
    }

    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_leaf >= 32768U);
    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t t0 = static_cast<std::size_t>(start) + cu * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, static_cast<std::size_t>(end));
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                const std::uint32_t r = rows[t];
                gs += static_cast<double>(g[r]);
                hs += static_cast<double>(h[r]);
            }
            ws.chunk_g[cu] = gs;
            ws.chunk_h[cu] = hs;
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t t0 = static_cast<std::size_t>(start) + c * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, static_cast<std::size_t>(end));
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                const std::uint32_t r = rows[t];
                gs += static_cast<double>(g[r]);
                hs += static_cast<double>(h[r]);
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
    g_sum = gs;
    h_sum = hs;
}

void sum_leaf_stats_contiguous(const float* g_leaf, const float* h_leaf, std::size_t n_leaf,
                               int n_jobs, TreeWorkspace& ws, double& g_sum,
                               double& h_sum) noexcept {
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    if (n_chunks <= 1U) {
        double gs = 0.0;
        double hs = 0.0;
        for (std::size_t t = 0; t < n_leaf; ++t) {
            gs += static_cast<double>(g_leaf[t]);
            hs += static_cast<double>(h_leaf[t]);
        }
        g_sum = gs;
        h_sum = hs;
        return;
    }

    const int n_threads = resolve_num_threads(n_jobs);
    const bool use_omp = (n_threads > 1) && (n_leaf >= 32768U);
    if (use_omp) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t t0 = cu * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, n_leaf);
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                gs += static_cast<double>(g_leaf[t]);
                hs += static_cast<double>(h_leaf[t]);
            }
            ws.chunk_g[cu] = gs;
            ws.chunk_h[cu] = hs;
        }
    } else {
        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t t0 = c * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, n_leaf);
            double gs = 0.0;
            double hs = 0.0;
            for (std::size_t t = t0; t < t1; ++t) {
                gs += static_cast<double>(g_leaf[t]);
                hs += static_cast<double>(h_leaf[t]);
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
    g_sum = gs;
    h_sum = hs;
}

void gather_leaf_gh(const std::uint32_t* rows, std::size_t n_leaf, const float* g, const float* h,
                    int n_jobs, float* g_ordered, float* h_ordered) noexcept {
    const int n_threads = resolve_num_threads(n_jobs);
    if (n_threads > 1 && n_leaf >= 16384U) {
        const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const std::size_t t0 = static_cast<std::size_t>(c) * kDetChunkRows;
            const std::size_t t1 = std::min(t0 + kDetChunkRows, n_leaf);
            for (std::size_t t = t0; t < t1; ++t) {
                const std::uint32_t r = rows[t];
                g_ordered[t] = g[r];
                h_ordered[t] = h[r];
            }
        }
        return;
    }
    for (std::size_t t = 0; t < n_leaf; ++t) {
        const std::uint32_t r = rows[t];
        g_ordered[t] = g[r];
        h_ordered[t] = h[r];
    }
}

template <typename Pred>
std::uint32_t stable_partition_leaf_rows(std::uint32_t* rows, std::uint32_t start,
                                         std::uint32_t end, int n_jobs, bool save_flags,
                                         TreeWorkspace& ws, Pred pred) noexcept {
    const std::size_t n_leaf = static_cast<std::size_t>(end - start);
    std::uint32_t* tmp_left = ws.tmp_left.data();
    std::uint32_t* tmp_right = ws.tmp_right.data();
    std::uint8_t* flags = ws.split_flags.data();
    const int n_threads = resolve_num_threads(n_jobs);

    if (n_threads <= 1 || n_leaf < 32768U) {
        std::uint32_t nl = 0U;
        std::uint32_t nr = 0U;
        if (save_flags) {
            for (std::size_t t = 0; t < n_leaf; ++t) {
                const std::uint32_t r = rows[static_cast<std::size_t>(start) + t];
                const std::uint8_t fl = pred(t, r) ? 1U : 0U;
                flags[t] = fl;
                tmp_left[nl] = r;
                tmp_right[nr] = r;
                nl += fl;
                nr += (fl ^ 1U);
            }
        } else {
            for (std::size_t t = 0; t < n_leaf; ++t) {
                const std::uint32_t r = rows[static_cast<std::size_t>(start) + t];
                const std::uint32_t fl = pred(t, r) ? 1U : 0U;
                tmp_left[nl] = r;
                tmp_right[nr] = r;
                nl += fl;
                nr += (fl ^ 1U);
            }
        }
        if (nl > 0U) {
            std::copy_n(tmp_left, nl, rows + start);
        }
        if (nr > 0U) {
            std::copy_n(tmp_right, nr, rows + start + nl);
        }
        return nl;
    }

    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
    for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
        const auto cu = static_cast<std::size_t>(c);
        const std::size_t r0 = cu * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        std::uint32_t nl_c = 0U;
        for (std::size_t t = r0; t < r1; ++t) {
            const bool go_left = pred(t, rows[start + t]);
            flags[t] = go_left ? 1U : 0U;
            nl_c += go_left ? 1U : 0U;
        }
        ws.chunk_left_counts[cu] = nl_c;
    }

    std::uint32_t nl_total = 0U;
    std::uint32_t nr_total = 0U;
    for (std::size_t c = 0; c < n_chunks; ++c) {
        const std::size_t r0 = c * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        const auto len_c = static_cast<std::uint32_t>(r1 - r0);
        const std::uint32_t nl_c = ws.chunk_left_counts[c];
        ws.chunk_left_offs[c] = nl_total;
        ws.chunk_right_offs[c] = nr_total;
        nl_total += nl_c;
        nr_total += (len_c - nl_c);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
    for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
        const auto cu = static_cast<std::size_t>(c);
        const std::size_t r0 = cu * kDetChunkRows;
        const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
        std::uint32_t local_l[kDetChunkRows + 1U];
        std::uint32_t local_r[kDetChunkRows + 1U];
        std::size_t lo = 0U;
        std::size_t ro = 0U;
        for (std::size_t t = r0; t < r1; ++t) {
            const std::uint32_t r = rows[start + t];
            const std::size_t fl = flags[t];
            local_l[lo] = r;
            local_r[ro] = r;
            lo += fl;
            ro += (fl ^ 1U);
        }
        if (lo > 0U) {
            std::memcpy(tmp_left + ws.chunk_left_offs[cu], local_l, lo * sizeof(std::uint32_t));
        }
        if (ro > 0U) {
            std::memcpy(tmp_right + ws.chunk_right_offs[cu], local_r, ro * sizeof(std::uint32_t));
        }
    }

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp for schedule(static) nowait
        for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(nl_total); ++t) {
            rows[static_cast<std::size_t>(start) + static_cast<std::size_t>(t)] = tmp_left[t];
        }
#pragma omp for schedule(static)
        for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(nr_total); ++t) {
            rows[static_cast<std::size_t>(start) + nl_total + static_cast<std::size_t>(t)] =
                tmp_right[t];
        }
    }
#else
    if (nl_total > 0U) {
        std::copy_n(tmp_left, nl_total, rows + start);
    }
    if (nr_total > 0U) {
        std::copy_n(tmp_right, nr_total, rows + start + nl_total);
    }
#endif
    return nl_total;
}

void partition_work_gh(const float* src_g, const float* src_h, float* dst_g, float* dst_h,
                       std::size_t start, std::size_t n_leaf, std::size_t nl, int n_jobs,
                       TreeWorkspace& ws) noexcept {
    const std::size_t nr = n_leaf - nl;
    const std::uint8_t* flags = ws.split_flags.data();
    float* tmp_l = ws.work_f32_tmp_l.data();
    float* tmp_r = ws.work_f32_tmp_r.data();
    const float* sg = src_g + start;
    const float* sh = src_h + start;
    const int n_threads = resolve_num_threads(n_jobs);

    if (n_threads <= 1 || n_leaf < 32768U) {
        std::size_t lo = 0U;
        std::size_t ro = 0U;
        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::size_t fl = flags[t];
            tmp_l[lo] = sg[t];
            tmp_r[ro] = sg[t];
            lo += fl;
            ro += (fl ^ 1U);
        }
        if (nl > 0U) {
            std::memcpy(dst_g + start, tmp_l, nl * sizeof(float));
        }
        if (nr > 0U) {
            std::memcpy(dst_g + start + nl, tmp_r, nr * sizeof(float));
        }
        lo = 0U;
        ro = 0U;
        for (std::size_t t = 0; t < n_leaf; ++t) {
            const std::size_t fl = flags[t];
            tmp_l[lo] = sh[t];
            tmp_r[ro] = sh[t];
            lo += fl;
            ro += (fl ^ 1U);
        }
        if (nl > 0U) {
            std::memcpy(dst_h + start, tmp_l, nl * sizeof(float));
        }
        if (nr > 0U) {
            std::memcpy(dst_h + start + nl, tmp_r, nr * sizeof(float));
        }
        return;
    }

    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
            float local_l[kDetChunkRows + 1U];
            float local_r[kDetChunkRows + 1U];
            std::size_t lo = 0U;
            std::size_t ro = 0U;
            for (std::size_t t = r0; t < r1; ++t) {
                const std::size_t fl = flags[t];
                local_l[lo] = sg[t];
                local_r[ro] = sg[t];
                lo += fl;
                ro += (fl ^ 1U);
            }
            if (lo > 0U) {
                std::memcpy(tmp_l + ws.chunk_left_offs[cu], local_l, lo * sizeof(float));
            }
            if (ro > 0U) {
                std::memcpy(tmp_r + ws.chunk_right_offs[cu], local_r, ro * sizeof(float));
            }
        }
#ifdef _OPENMP
#pragma omp single nowait
#endif
        if (nl > 0U) {
            std::memcpy(dst_g + start, tmp_l, nl * sizeof(float));
        }
#ifdef _OPENMP
#pragma omp single
#endif
        if (nr > 0U) {
            std::memcpy(dst_g + start + nl, tmp_r, nr * sizeof(float));
        }
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
            const auto cu = static_cast<std::size_t>(c);
            const std::size_t r0 = cu * kDetChunkRows;
            const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
            float local_l[kDetChunkRows + 1U];
            float local_r[kDetChunkRows + 1U];
            std::size_t lo = 0U;
            std::size_t ro = 0U;
            for (std::size_t t = r0; t < r1; ++t) {
                const std::size_t fl = flags[t];
                local_l[lo] = sh[t];
                local_r[ro] = sh[t];
                lo += fl;
                ro += (fl ^ 1U);
            }
            if (lo > 0U) {
                std::memcpy(tmp_l + ws.chunk_left_offs[cu], local_l, lo * sizeof(float));
            }
            if (ro > 0U) {
                std::memcpy(tmp_r + ws.chunk_right_offs[cu], local_r, ro * sizeof(float));
            }
        }
#ifdef _OPENMP
#pragma omp single nowait
#endif
        if (nl > 0U) {
            std::memcpy(dst_h + start, tmp_l, nl * sizeof(float));
        }
#ifdef _OPENMP
#pragma omp single
#endif
        if (nr > 0U) {
            std::memcpy(dst_h + start + nl, tmp_r, nr * sizeof(float));
        }
    }
}

void partition_one_code_col(const std::int16_t* sc, std::int16_t* dc, std::int16_t* tl,
                            std::int16_t* tr, const std::uint8_t* flags, std::size_t n_leaf,
                            std::size_t nl) noexcept {
    const std::size_t nr = n_leaf - nl;
    std::size_t lo = 0U;
    std::size_t ro = 0U;
    for (std::size_t t = 0; t < n_leaf; ++t) {
        const std::size_t fl = flags[t];
        const std::int16_t v = sc[t];
        tl[lo] = v;
        tr[ro] = v;
        lo += fl;
        ro += (fl ^ 1U);
    }
    if (nl > 0U) {
        std::memcpy(dc, tl, nl * sizeof(std::int16_t));
    }
    if (nr > 0U) {
        std::memcpy(dc + nl, tr, nr * sizeof(std::int16_t));
    }
}

void partition_work_codes(const std::int16_t* src_codes, std::int16_t* dst_codes,
                          std::size_t n_rows, const std::vector<std::uint32_t>& eligible_num,
                          std::size_t start, std::size_t n_leaf, std::size_t nl, int n_jobs,
                          TreeWorkspace& ws) noexcept {
    if (eligible_num.empty() || n_leaf == 0U) {
        return;
    }
    const std::size_t n_cols = eligible_num.size();
    const bool in_place = (src_codes == dst_codes);
    const std::size_t half_stride = n_rows + 1U;
    const std::size_t thr_stride = 2U * half_stride;
    const std::uint8_t* flags = ws.split_flags.data();
    const int n_threads = resolve_num_threads(n_jobs);

    if (n_threads <= 1 || ws.work_code_threads <= 1U || (n_cols * n_leaf) < 65536U) {
        std::int16_t* tl = ws.work_col_tmp.data();
        std::int16_t* tr = tl + half_stride;
        for (std::size_t q = 0; q < n_cols; ++q) {
            const std::size_t col_off = static_cast<std::size_t>(eligible_num[q]) * n_rows + start;
            partition_one_code_col(src_codes + col_off, dst_codes + col_off, tl, tr, flags, n_leaf,
                                   nl);
        }
        return;
    }

    const int use_threads = std::min(n_threads, static_cast<int>(ws.work_code_threads));
    if (n_cols >= static_cast<std::size_t>(use_threads) || n_leaf < 65536U) {
#ifdef _OPENMP
#pragma omp parallel num_threads(use_threads)
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            std::int16_t* tl = ws.work_col_tmp.data() + static_cast<std::size_t>(tid) * thr_stride;
            std::int16_t* tr = tl + half_stride;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
            for (std::ptrdiff_t q = 0; q < static_cast<std::ptrdiff_t>(n_cols); ++q) {
                const std::size_t col_off =
                    static_cast<std::size_t>(eligible_num[static_cast<std::size_t>(q)]) * n_rows +
                    start;
                partition_one_code_col(src_codes + col_off, dst_codes + col_off, tl, tr, flags,
                                       n_leaf, nl);
            }
        }
        return;
    }

    // Few numeric columns on a huge leaf: chunk-parallel within each column.
    const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
    std::int16_t* tmp = ws.work_col_tmp.data();
#ifdef _OPENMP
#pragma omp parallel num_threads(use_threads)
#endif
    {
        for (std::size_t q = 0; q < n_cols; ++q) {
            const std::size_t col_off = static_cast<std::size_t>(eligible_num[q]) * n_rows + start;
            const std::int16_t* sc = src_codes + col_off;
            std::int16_t* dc = dst_codes + col_off;
            std::int16_t* out = in_place ? tmp : dc;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
            for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                const auto cu = static_cast<std::size_t>(c);
                const std::size_t r0 = cu * kDetChunkRows;
                const std::size_t r1 = std::min(r0 + kDetChunkRows, n_leaf);
                std::int16_t local_l[kDetChunkRows + 1U];
                std::int16_t local_r[kDetChunkRows + 1U];
                std::size_t lo = 0U;
                std::size_t ro = 0U;
                for (std::size_t t = r0; t < r1; ++t) {
                    const std::size_t fl = flags[t];
                    const std::int16_t v = sc[t];
                    local_l[lo] = v;
                    local_r[ro] = v;
                    lo += fl;
                    ro += (fl ^ 1U);
                }
                if (lo > 0U) {
                    std::memcpy(out + ws.chunk_left_offs[cu], local_l, lo * sizeof(std::int16_t));
                }
                if (ro > 0U) {
                    std::memcpy(out + nl + ws.chunk_right_offs[cu], local_r,
                                ro * sizeof(std::int16_t));
                }
            }
            if (in_place) {
#ifdef _OPENMP
#pragma omp single
#endif
                std::memcpy(dc, tmp, n_leaf * sizeof(std::int16_t));
            }
        }
    }
}

std::uint32_t partition_leaf_rows_axis(std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                                       const std::uint8_t* col, const AxisSplitCandidate& split,
                                       std::uint8_t nan_bin, bool has_nan, int n_jobs,
                                       bool save_flags, TreeWorkspace& ws) noexcept {
    const std::uint8_t thr_bin = split.bin_idx;
    if (!has_nan) {
        return stable_partition_leaf_rows(
            rows, start, end, n_jobs, save_flags, ws,
            [col, thr_bin](std::size_t /*t*/, std::uint32_t r) noexcept {
                return col[r] <= thr_bin;
            });
    }
    const bool nan_left = split.nan_go_left;
    return stable_partition_leaf_rows(
        rows, start, end, n_jobs, save_flags, ws,
        [col, thr_bin, nan_bin, nan_left](std::size_t /*t*/, std::uint32_t r) noexcept {
            const std::uint8_t b = col[r];
            return (b == nan_bin) ? nan_left : (b <= thr_bin);
        });
}

std::uint32_t partition_leaf_rows_cat(std::uint32_t* rows, std::uint32_t start, std::uint32_t end,
                                      const std::uint8_t* col, const CatBitset& bitset, int n_jobs,
                                      bool save_flags, TreeWorkspace& ws) noexcept {
    return stable_partition_leaf_rows(rows, start, end, n_jobs, save_flags, ws,
                                      [col, &bitset](std::size_t /*t*/, std::uint32_t r) noexcept {
                                          return cat_bitset_test(bitset, col[r]);
                                      });
}

std::uint32_t partition_leaf_rows_oblique(std::uint32_t* rows, std::uint32_t start,
                                          std::uint32_t end, const std::int16_t* codes,
                                          std::size_t n_rows, std::size_t codes_offset,
                                          bool codes_contiguous, const ObliqueSplitCandidate& split,
                                          int n_jobs, bool save_flags, TreeWorkspace& ws) noexcept {
    const std::uint8_t k = split.k;
    const bool nan_left = split.nan_go_left;
    const std::size_t base_off = codes_contiguous ? codes_offset : 0U;

    if (k <= 3U) {
        const auto thr = static_cast<std::int32_t>(split.threshold);
        const std::int16_t* col0 =
            codes + static_cast<std::size_t>(split.num_indices[0]) * n_rows + base_off;
        const std::int16_t* col1 =
            codes + static_cast<std::size_t>(split.num_indices[1]) * n_rows + base_off;
        const std::int16_t* col2 =
            (k == 3U) ? (codes + static_cast<std::size_t>(split.num_indices[2]) * n_rows + base_off)
                      : col0;
        const std::int32_t w0 = split.weights[0];
        const std::int32_t w1 = split.weights[1];
        const std::int32_t w2 = (k == 3U) ? split.weights[2] : 0;

        if (codes_contiguous) {
            return stable_partition_leaf_rows(
                rows, start, end, n_jobs, save_flags, ws,
                [col0, col1, col2, thr, w0, w1, w2, nan_left](std::size_t t,
                                                              std::uint32_t /*r*/) noexcept {
                    const std::int16_t c0 = col0[t];
                    const std::int16_t c1 = col1[t];
                    const std::int16_t c2 = col2[t];
                    if (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode ||
                        c2 == kQ16SignedNanCode) {
                        return nan_left;
                    }
                    const std::int32_t p32 = w0 * c0 + w1 * c1 + w2 * c2;
                    return p32 < thr;
                });
        }
        return stable_partition_leaf_rows(
            rows, start, end, n_jobs, save_flags, ws,
            [col0, col1, col2, thr, w0, w1, w2, nan_left](std::size_t /*t*/,
                                                          std::uint32_t r) noexcept {
                const std::int16_t c0 = col0[r];
                const std::int16_t c1 = col1[r];
                const std::int16_t c2 = col2[r];
                if (c0 == kQ16SignedNanCode || c1 == kQ16SignedNanCode || c2 == kQ16SignedNanCode) {
                    return nan_left;
                }
                const std::int32_t p32 = w0 * c0 + w1 * c1 + w2 * c2;
                return p32 < thr;
            });
    }

    const std::int64_t thr = split.threshold;
    if (codes_contiguous) {
        return stable_partition_leaf_rows(
            rows, start, end, n_jobs, save_flags, ws,
            [codes, n_rows, base_off, &split, k, thr, nan_left](std::size_t t,
                                                                std::uint32_t /*r*/) noexcept {
                std::int64_t acc = 0;
                for (std::uint8_t q = 0; q < k; ++q) {
                    const std::int16_t c =
                        codes[static_cast<std::size_t>(split.num_indices[q]) * n_rows + base_off +
                              t];
                    if (c == kQ16SignedNanCode) {
                        return nan_left;
                    }
                    acc +=
                        static_cast<std::int64_t>(split.weights[q]) * static_cast<std::int64_t>(c);
                }
                return acc < thr;
            });
    }
    return stable_partition_leaf_rows(
        rows, start, end, n_jobs, save_flags, ws,
        [codes, n_rows, &split, k, thr, nan_left](std::size_t /*t*/, std::uint32_t r) noexcept {
            std::int64_t acc = 0;
            for (std::uint8_t q = 0; q < k; ++q) {
                const std::int16_t c =
                    codes[static_cast<std::size_t>(split.num_indices[q]) * n_rows + r];
                if (c == kQ16SignedNanCode) {
                    return nan_left;
                }
                acc += static_cast<std::int64_t>(split.weights[q]) * static_cast<std::int64_t>(c);
            }
            return acc < thr;
        });
}

SplitChoice evaluate_leaf_splits(const Dataset& ds, LeafState& leaf, const std::int16_t* codes_base,
                                 std::size_t leaf_code_offset, bool is_contiguous_leaf,
                                 const float* g_seq, const float* h_seq, std::size_t tree_index,
                                 const Params& params, TreeWorkspace& ws) noexcept {
    SplitChoice choice;
    const auto n_leaf = static_cast<std::size_t>(leaf.end - leaf.start);
    const auto parent_c = static_cast<std::int32_t>(n_leaf);
    if (parent_c < 2 * params.min_data_in_leaf) {
        return choice;
    }

    if (!ws.layout.active_num_features.empty()) {
        choice.axis = find_best_axis_split(ds, ws.layout, ws.pool.slot(leaf.hist_slot), leaf.g_sum,
                                           leaf.h_sum, parent_c, params, tree_index, leaf.node_id);
        if (choice.axis.valid) {
            choice.kind = NodeKind::Axis;
            choice.gain = choice.axis.gain;
            choice.valid = true;
        }
    }

    if (!ws.layout.active_cat_features.empty()) {
        choice.cat = find_best_cat_split(ws.layout, ws.pool.slot(leaf.hist_slot), leaf.g_sum,
                                         leaf.h_sum, parent_c, params);
        // Fixed tie-breaking order: axis < categorical < oblique.
        if (choice.cat.valid && (!choice.valid || choice.cat.gain > choice.gain)) {
            choice.kind = NodeKind::Categorical;
            choice.gain = choice.cat.gain;
            choice.valid = true;
        }
    }

    const bool allow_oblique = (params.m_mult > 0) && (params.oblique_max_depth < 0 ||
                                                       leaf.depth <= params.oblique_max_depth);
    if (allow_oblique && ws.oblique.cur_m > 0U) {
        choice.oblique = find_best_oblique_split(
            ds, codes_base, leaf_code_offset, is_contiguous_leaf,
            ws.row_indices.data() + leaf.start, n_leaf, g_seq, h_seq, leaf.g_sum, leaf.h_sum,
            parent_c, tree_index, leaf.node_id, params, ws.oblique);
        if (choice.oblique.valid && (!choice.valid || choice.oblique.gain > choice.gain)) {
            choice.kind = NodeKind::Oblique;
            choice.gain = choice.oblique.gain;
            choice.valid = true;
        }
    }

    return choice;
}

} // namespace

void TreeWorkspace::prepare(const Dataset& ds, const Params& params) {
    layout = FeatureHistLayout::build(ds);
    base_active_features = layout.active_features;
    base_active_num_features = layout.active_num_features;
    base_active_cat_features = layout.active_cat_features;
    feature_selected.assign(ds.n_features(), 1U);
    feature_perm.assign(base_active_features.size(), 0U);
    feature_to_num_idx.assign(ds.n_features(), 0U);
    {
        const auto& num_indices = ds.num_feature_indices();
        for (std::size_t j = 0; j < num_indices.size(); ++j) {
            feature_to_num_idx[num_indices[j]] = j;
        }
    }
    oob_rows.clear();
    if (params.bagging_fraction < 1.0) {
        oob_rows.reserve(ds.n_rows());
    }
    const std::size_t slots = static_cast<std::size_t>(std::max(params.num_leaves, 1)) + 1U;
    pool.reset(slots, layout.total_bins);
    oblique.prepare(ds, params);
    const std::size_t n = ds.n_rows();
    const std::size_t max_chunks = (n + kDetChunkRows - 1U) / kDetChunkRows;
    if (layout.active_features.size() < 16U && n >= 32768U) {
        axis_wave_hists.resize(kWaveChunks * layout.total_bins);
    } else {
        axis_wave_hists.resize(0U);
    }
    row_indices.resize(n);
    tmp_left.resize(n + 1U);
    tmp_right.resize(n + 1U);
    split_flags.resize(n);
    chunk_g.resize(max_chunks);
    chunk_h.resize(max_chunks);
    chunk_left_counts.resize(max_chunks);
    chunk_left_offs.resize(max_chunks);
    chunk_right_offs.resize(max_chunks);
    g_ordered.resize(n);
    h_ordered.resize(n);

    const bool alloc_work_gh = (params.bagging_fraction >= 1.0) && (n >= 16384U) &&
                               (params.num_leaves > 1) && (params.m_mult > 0) &&
                               (params.oblique_max_depth != 0) && (oblique.cur_m > 0U);
    if (alloc_work_gh) {
        work_g.resize(n);
        work_h.resize(n);
        work_f32_tmp_l.resize(n + 1U);
        work_f32_tmp_r.resize(n + 1U);
    } else {
        work_g.resize(0U);
        work_h.resize(0U);
        work_f32_tmp_l.resize(0U);
        work_f32_tmp_r.resize(0U);
    }

    const std::size_t p_num = ds.n_num_features();
    constexpr std::size_t kMaxWorkCodeElems = 64'000'000U;
    const bool alloc_work_codes =
        alloc_work_gh && (p_num > 0U) && (p_num <= (kMaxWorkCodeElems / n));
    if (alloc_work_codes) {
        work_codes.resize(p_num * n);
        work_code_threads = static_cast<std::size_t>(resolve_num_threads(params.n_jobs));
        work_col_tmp.resize(work_code_threads * 2U * (n + 1U));
    } else {
        work_codes.resize(0U);
        work_code_threads = 0;
        work_col_tmp.resize(0U);
    }
}

Tree grow_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
               std::size_t tree_index, TreeWorkspace& ws, double* train_scores) {
    const std::size_t n = ds.n_rows();
    const auto max_leaves = static_cast<std::size_t>(std::max(params.num_leaves, 1));
    const bool bagging = params.bagging_fraction < 1.0;
    std::size_t n_root = n;
    if (bagging) {
        n_root = draw_bag(params, tree_index, n, ws.row_indices.data(), ws.oob_rows);
    } else {
        std::iota(ws.row_indices.begin(), ws.row_indices.end(), 0U);
    }
    if (params.feature_fraction < 1.0) {
        select_features(params, tree_index, ws);
    }

    Tree tree;
    tree.nodes.reserve(2U * max_leaves);
    tree.nodes.push_back(Node{});

    std::vector<LeafState> leaves;
    leaves.reserve(max_leaves);

    const std::uint8_t* bins_base = ds.bins_col_major().data();
    const std::size_t p_num = ds.n_num_features();
    const std::int16_t* codes_base = (p_num > 0U) ? ds.codes_col_major().data() : nullptr;

    LeafState root;
    root.node_id = 0U;
    root.start = 0U;
    root.end = static_cast<std::uint32_t>(n_root);
    root.depth = 0;
    root.hist_slot = 0U;
    if (!bagging) {
        sum_leaf_stats_contiguous(g, h, n_root, params.n_jobs, ws, root.g_sum, root.h_sum);
    } else {
        sum_leaf_stats(ws.row_indices.data(), root.start, root.end, g, h, params.n_jobs, ws,
                       root.g_sum, root.h_sum);
    }

    const bool root_can_split = (max_leaves > 1U) &&
                                (params.max_depth < 0 || root.depth < params.max_depth) &&
                                (static_cast<std::int32_t>(n_root) >= 2 * params.min_data_in_leaf);
    if (root_can_split) {
        if (!bagging) {
            if (ws.layout.total_bins > 0U) {
                build_axis_histograms(ds, ws.layout, ws.row_indices.data(), n, true, g, h,
                                      ws.g_ordered.data(), ws.h_ordered.data(), ws.pool.slot(0),
                                      params.n_jobs, ws.axis_wave_hists.data(), false);
            }
            root.split =
                evaluate_leaf_splits(ds, root, codes_base, 0U, true, g, h, tree_index, params, ws);
        } else {
            // The bag is a row subset: gradients are gathered into bag order.
            gather_leaf_gh(ws.row_indices.data(), n_root, g, h, params.n_jobs, ws.g_ordered.data(),
                           ws.h_ordered.data());
            if (ws.layout.total_bins > 0U) {
                build_axis_histograms(ds, ws.layout, ws.row_indices.data(), n_root, false, g, h,
                                      ws.g_ordered.data(), ws.h_ordered.data(), ws.pool.slot(0),
                                      params.n_jobs, ws.axis_wave_hists.data(), true);
            }
            root.split = evaluate_leaf_splits(ds, root, codes_base, 0U, false, ws.g_ordered.data(),
                                              ws.h_ordered.data(), tree_index, params, ws);
        }
    }
    leaves.push_back(root);

    std::size_t next_slot = 1U;
    const bool use_work_gh = !bagging && (ws.work_g.size() == n);
    const bool use_work_codes = !bagging && (p_num > 0U) && (ws.work_codes.size() == p_num * n);

    while (leaves.size() < max_leaves) {
        std::size_t best_leaf_idx = leaves.size();
        double best_gain = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < leaves.size(); ++i) {
            if (leaves[i].split.valid && leaves[i].split.gain > best_gain) {
                best_gain = leaves[i].split.gain;
                best_leaf_idx = i;
            }
        }
        if (best_leaf_idx == leaves.size()) {
            break;
        }

        const LeafState parent = leaves[best_leaf_idx];
        const auto& split = parent.split;
        const int child_depth = parent.depth + 1;
        const bool will_continue = (leaves.size() + 1U < max_leaves) &&
                                   (params.max_depth < 0 || child_depth < params.max_depth);
        const bool children_allow_oblique =
            (params.m_mult > 0) &&
            (params.oblique_max_depth < 0 || child_depth <= params.oblique_max_depth) &&
            (ws.oblique.cur_m > 0U);
        const bool need_gh_children = will_continue && use_work_gh && children_allow_oblique &&
                                      (parent.depth == 0 || parent.gh_packed);
        const bool need_codes_children = will_continue && use_work_codes && children_allow_oblique;
        const bool need_save_flags = need_gh_children || need_codes_children;

        std::uint32_t nl = 0U;
        if (split.kind == NodeKind::Axis) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(split.axis.feature) * n;
            const auto& binner = ds.binner(split.axis.feature);
            nl = partition_leaf_rows_axis(ws.row_indices.data(), parent.start, parent.end, col,
                                          split.axis, binner.nan_bin(), binner.has_nan,
                                          params.n_jobs, need_save_flags, ws);
        } else if (split.kind == NodeKind::Categorical) {
            const std::uint8_t* col = bins_base + static_cast<std::size_t>(split.cat.feature) * n;
            nl = partition_leaf_rows_cat(ws.row_indices.data(), parent.start, parent.end, col,
                                         split.cat.left_bitset, params.n_jobs, need_save_flags, ws);
        } else {
            const bool parent_codes_contig = !bagging && ((parent.depth == 0) || use_work_codes);
            const std::int16_t* parent_codes_ptr =
                (parent.depth == 0 || !use_work_codes) ? codes_base : ws.work_codes.data();
            const std::size_t parent_codes_off = parent_codes_contig ? parent.start : 0U;
            nl = partition_leaf_rows_oblique(ws.row_indices.data(), parent.start, parent.end,
                                             parent_codes_ptr, n, parent_codes_off,
                                             parent_codes_contig, split.oblique, params.n_jobs,
                                             need_save_flags, ws);
        }

        const std::uint32_t mid = parent.start + nl;
        const std::uint32_t nr = parent.end - mid;
        const auto n_parent = static_cast<std::size_t>(parent.end - parent.start);

        if (need_gh_children) {
            const float* src_g = (parent.depth == 0) ? g : ws.work_g.data();
            const float* src_h = (parent.depth == 0) ? h : ws.work_h.data();
            partition_work_gh(src_g, src_h, ws.work_g.data(), ws.work_h.data(), parent.start,
                              n_parent, nl, params.n_jobs, ws);
        }
        if (need_codes_children) {
            const std::int16_t* src_codes = (parent.depth == 0) ? codes_base : ws.work_codes.data();
            partition_work_codes(src_codes, ws.work_codes.data(), n,
                                 ws.oblique.eligible_num_indices, parent.start, n_parent, nl,
                                 params.n_jobs, ws);
        }

        const auto left_node_id = static_cast<std::uint32_t>(tree.nodes.size());
        const auto right_node_id = left_node_id + 1U;
        tree.nodes.push_back(Node{});
        tree.nodes.push_back(Node{});

        Node& pnode = tree.nodes[parent.node_id];
        pnode.kind = split.kind;
        pnode.left_child = left_node_id;
        pnode.right_child = right_node_id;
        pnode.split_gain = split.gain;

        if (split.kind == NodeKind::Axis) {
            pnode.nan_go_left = split.axis.nan_go_left;
            pnode.feature = split.axis.feature;
            pnode.threshold_u16 = split.axis.threshold_u16;
        } else if (split.kind == NodeKind::Categorical) {
            pnode.nan_go_left = split.cat.default_go_left;
            pnode.feature = split.cat.feature;
            pnode.cat_bitset_offset = static_cast<std::uint32_t>(tree.cat_bitsets.size());
            tree.cat_bitsets.push_back(split.cat.left_bitset);
        } else {
            const auto& ob = split.oblique;
            pnode.nan_go_left = ob.nan_go_left;
            pnode.oblique_k = ob.k;
            if (ob.k <= 3U) {
                for (std::uint8_t q = 0; q < ob.k; ++q) {
                    pnode.oblique_features[q] = ob.features[q];
                    pnode.oblique_weights[q] = ob.weights[q];
                }
                pnode.threshold_i32 = static_cast<std::int32_t>(ob.threshold);
            } else {
                pnode.overflow_offset = static_cast<std::uint32_t>(tree.oblique_overflow.size());
                ObliqueOverflow ov;
                ov.features.assign(ob.features, ob.features + ob.k);
                ov.weights.assign(ob.weights, ob.weights + ob.k);
                ov.threshold_i64 = ob.threshold;
                tree.oblique_overflow.push_back(std::move(ov));
            }
        }

        LeafState left_leaf;
        left_leaf.node_id = left_node_id;
        left_leaf.start = parent.start;
        left_leaf.end = mid;
        left_leaf.depth = child_depth;

        LeafState right_leaf;
        right_leaf.node_id = right_node_id;
        right_leaf.start = mid;
        right_leaf.end = parent.end;
        right_leaf.depth = child_depth;

        const bool work_gh_ready = need_gh_children;
        left_leaf.gh_packed = work_gh_ready;
        right_leaf.gh_packed = work_gh_ready;
        if (nl <= nr) {
            if (work_gh_ready) {
                sum_leaf_stats_contiguous(ws.work_g.data() + left_leaf.start,
                                          ws.work_h.data() + left_leaf.start, nl, params.n_jobs, ws,
                                          left_leaf.g_sum, left_leaf.h_sum);
            } else {
                sum_leaf_stats(ws.row_indices.data(), left_leaf.start, left_leaf.end, g, h,
                               params.n_jobs, ws, left_leaf.g_sum, left_leaf.h_sum);
            }
            right_leaf.g_sum = parent.g_sum - left_leaf.g_sum;
            right_leaf.h_sum = parent.h_sum - left_leaf.h_sum;
        } else {
            if (work_gh_ready) {
                sum_leaf_stats_contiguous(ws.work_g.data() + right_leaf.start,
                                          ws.work_h.data() + right_leaf.start, nr, params.n_jobs,
                                          ws, right_leaf.g_sum, right_leaf.h_sum);
            } else {
                sum_leaf_stats(ws.row_indices.data(), right_leaf.start, right_leaf.end, g, h,
                               params.n_jobs, ws, right_leaf.g_sum, right_leaf.h_sum);
            }
            left_leaf.g_sum = parent.g_sum - right_leaf.g_sum;
            left_leaf.h_sum = parent.h_sum - right_leaf.h_sum;
        }

        if (will_continue) {
            const std::size_t small_slot = next_slot++;
            const std::size_t large_slot = parent.hist_slot;

            auto eval_children = [&](LeafState& small_leaf, LeafState& large_leaf) noexcept {
                small_leaf.hist_slot = small_slot;
                large_leaf.hist_slot = large_slot;
                const auto n_small = static_cast<std::size_t>(small_leaf.end - small_leaf.start);
                const auto n_large = static_cast<std::size_t>(large_leaf.end - large_leaf.start);
                const bool codes_ready = need_codes_children;
                const std::int16_t* c_ptr = codes_ready ? ws.work_codes.data() : codes_base;
                const std::size_t s_off = codes_ready ? small_leaf.start : 0U;
                const std::size_t l_off = codes_ready ? large_leaf.start : 0U;

                if (work_gh_ready) {
                    float* s_g = ws.work_g.data() + small_leaf.start;
                    float* s_h = ws.work_h.data() + small_leaf.start;
                    if (ws.layout.total_bins > 0U) {
                        build_axis_histograms(ds, ws.layout,
                                              ws.row_indices.data() + small_leaf.start, n_small,
                                              false, g, h, s_g, s_h, ws.pool.slot(small_slot),
                                              params.n_jobs, ws.axis_wave_hists.data(), true);
                        subtract_histograms(ws.pool.slot(large_slot),
                                            ws.pool.slot(parent.hist_slot),
                                            ws.pool.slot(small_slot), ws.layout.total_bins);
                    }
                    small_leaf.split =
                        evaluate_leaf_splits(ds, small_leaf, c_ptr, s_off, codes_ready, s_g, s_h,
                                             tree_index, params, ws);
                    float* l_g = ws.work_g.data() + large_leaf.start;
                    float* l_h = ws.work_h.data() + large_leaf.start;
                    large_leaf.split =
                        evaluate_leaf_splits(ds, large_leaf, c_ptr, l_off, codes_ready, l_g, l_h,
                                             tree_index, params, ws);
                } else {
                    if (ws.layout.total_bins > 0U) {
                        build_axis_histograms(ds, ws.layout,
                                              ws.row_indices.data() + small_leaf.start, n_small,
                                              false, g, h, ws.g_ordered.data(), ws.h_ordered.data(),
                                              ws.pool.slot(small_slot), params.n_jobs,
                                              ws.axis_wave_hists.data(), false);
                        subtract_histograms(ws.pool.slot(large_slot),
                                            ws.pool.slot(parent.hist_slot),
                                            ws.pool.slot(small_slot), ws.layout.total_bins);
                    } else {
                        gather_leaf_gh(ws.row_indices.data() + small_leaf.start, n_small, g, h,
                                       params.n_jobs, ws.g_ordered.data(), ws.h_ordered.data());
                    }
                    small_leaf.split = evaluate_leaf_splits(
                        ds, small_leaf, c_ptr, s_off, codes_ready, ws.g_ordered.data(),
                        ws.h_ordered.data(), tree_index, params, ws);

                    const bool large_needs_oblique =
                        children_allow_oblique &&
                        (static_cast<std::int32_t>(n_large) >= 2 * params.min_data_in_leaf);
                    if (large_needs_oblique) {
                        gather_leaf_gh(ws.row_indices.data() + large_leaf.start, n_large, g, h,
                                       params.n_jobs, ws.g_ordered.data(), ws.h_ordered.data());
                    }
                    large_leaf.split = evaluate_leaf_splits(
                        ds, large_leaf, c_ptr, l_off, codes_ready, ws.g_ordered.data(),
                        ws.h_ordered.data(), tree_index, params, ws);
                }
            };

            if (nl <= nr) {
                eval_children(left_leaf, right_leaf);
            } else {
                eval_children(right_leaf, left_leaf);
            }
        }

        leaves[best_leaf_idx] = left_leaf;
        leaves.push_back(right_leaf);
    }

    tree.n_leaves = static_cast<std::uint32_t>(leaves.size());
    const double lr = params.learning_rate;
    const double lam = params.lambda_l2;
    const int n_threads = resolve_num_threads(params.n_jobs);

    for (const auto& leaf : leaves) {
        double gs = 0.0;
        double hs = 0.0;
        sum_leaf_stats(ws.row_indices.data(), leaf.start, leaf.end, g, h, params.n_jobs, ws, gs,
                       hs);
        const double val = lr * (-gs / (hs + lam));
        tree.nodes[leaf.node_id].kind = NodeKind::Leaf;
        tree.nodes[leaf.node_id].leaf_value = val;
        if (train_scores != nullptr) {
            const std::uint32_t* rows_ptr = ws.row_indices.data();
            const std::size_t n_leaf = static_cast<std::size_t>(leaf.end - leaf.start);
            if (n_threads > 1 && n_leaf >= 32768U) {
                const std::size_t n_chunks = (n_leaf + kDetChunkRows - 1U) / kDetChunkRows;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                for (std::ptrdiff_t c = 0; c < static_cast<std::ptrdiff_t>(n_chunks); ++c) {
                    const std::size_t t0 = static_cast<std::size_t>(leaf.start) +
                                           static_cast<std::size_t>(c) * kDetChunkRows;
                    const std::size_t t1 =
                        std::min(t0 + kDetChunkRows, static_cast<std::size_t>(leaf.end));
                    for (std::size_t t = t0; t < t1; ++t) {
                        train_scores[rows_ptr[t]] += val;
                    }
                }
            } else {
                for (std::uint32_t t = leaf.start; t < leaf.end; ++t) {
                    train_scores[rows_ptr[t]] += val;
                }
            }
        }
    }

    if (bagging && train_scores != nullptr && !ws.oob_rows.empty()) {
        // Rows outside the bag follow the finished tree, exactly as prediction would.
        const std::int16_t* codes = (p_num > 0U) ? ds.codes_col_major().data() : nullptr;
        Evaluator::add_tree_predictions_from_codes(tree, codes, n, p_num, ws.feature_to_num_idx,
                                                   train_scores, bins_base, params.n_jobs,
                                                   ws.oob_rows.data(), ws.oob_rows.size());
    }

    return tree;
}

Tree grow_axis_tree(const Dataset& ds, const float* g, const float* h, const Params& params,
                    TreeWorkspace& ws, double* train_scores) {
    Params ax = params;
    ax.m_mult = 0;
    return grow_tree(ds, g, h, ax, 0U, ws, train_scores);
}

} // namespace miter
