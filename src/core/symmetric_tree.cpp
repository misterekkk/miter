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
#include "core/tree_learner.hpp"
#include "data/quantizer.hpp"
#include "model/evaluator.hpp"
#include "util/fast_div.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

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

struct CatGroup {
    double g = 0.0;
    double h = 0.0;
    double score = 0.0;
    std::int32_t c = 0;
    std::int16_t bin = -1;
};

bool eval_multi_leaf_gain(const double* gl_arr, const double* hl_arr, const std::int32_t* cl_arr,
                          const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
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

void add_unique_ctr_proj(std::vector<CtrProjection>& used, const CtrProjection& proj) {
    for (const auto& existing : used) {
        if (existing == proj) {
            return;
        }
    }
    used.push_back(proj);
}

struct CtrMissGroup {
    std::size_t proj_idx = 0;
    bool need_tm = false;
    bool need_cnt = false;
    std::vector<CtrCachedCol> tm_cols;
    CtrCachedCol cnt_col;
};

struct CtrScoreJob {
    std::size_t proj_idx = 0;
    CtrKind kind = CtrKind::TargetMean;
    std::uint8_t prior_id = 0;
    double prior_value = 0.0;
    const CtrCachedCol* col = nullptr;
};

void compute_ctr_miss_group(const Dataset& ds, const CtrProjection& proj,
                            std::span<const std::uint32_t> perm, const CtrGrowContext& ctr_ctx,
                            const Params& params, CtrMissGroup& mg, CtrThreadScratch& ts) {
    const std::size_t n = ds.n_rows();
    compute_projection_hashes(ds, proj, ts.proj_hashes.data());
    if (mg.need_tm) {
        constexpr std::array<double, 3> kUnitPriors = {0.0, 0.5, 1.0};
        const double reg_prior = ctr_ctx.prior;
        const std::span<const double> prior_span = ctr_ctx.is_unit_target
                                                       ? std::span<const double>(kUnitPriors)
                                                       : std::span<const double>(&reg_prior, 1U);
        const std::size_t n_uniq = compute_online_ctr_multi_prior(
            ts.proj_hashes.data(), n, perm, ctr_ctx.target_y, ctr_ctx.sample_weight, prior_span,
            params.ctr_prior_weight, ts.scratch, ts.ctr_vals.data());
        mg.tm_cols.resize(prior_span.size());
        for (std::size_t p_idx = 0U; p_idx < prior_span.size(); ++p_idx) {
            CtrCachedCol& col = mg.tm_cols[p_idx];
            col.bins.resize(n);
            col.n_unique_keys = n_uniq;
            const float* src_vals = ts.ctr_vals.data() + p_idx * n;
            if (ctr_ctx.is_unit_target) {
                (void)build_uniform_unit_borders_and_bins(src_vals, n, params.ctr_max_borders,
                                                          col.borders, col.bins.data());
            } else {
                (void)build_ctr_borders_and_bins(src_vals, nullptr, n, true, params.ctr_max_borders,
                                                 ts.sample_vals, col.borders, col.bins.data());
            }
        }
    }
    if (mg.need_cnt) {
        const std::size_t n_uniq =
            compute_online_counter(ts.proj_hashes.data(), n, ts.scratch, ts.ctr_vals.data());
        mg.cnt_col.bins.resize(n);
        mg.cnt_col.n_unique_keys = n_uniq;
        (void)build_uniform_unit_borders_and_bins(ts.ctr_vals.data(), n, params.ctr_max_borders,
                                                  mg.cnt_col.borders, mg.cnt_col.bins.data());
    }
}

CtrSplitCandidate
find_sym_ctr_split(const Dataset& ds, const std::vector<std::uint32_t>& active_cat,
                   const std::vector<CtrProjection>& used_projs, const std::uint8_t* leaf_ids,
                   const std::uint32_t* rows, std::size_t n_root, bool is_contiguous,
                   std::size_t n_leaves, const float* g_seq, const float* h_seq,
                   const double* leaf_g, const double* leaf_h, const std::int32_t* leaf_c,
                   const double* parent_score, double lam, double min_h, std::int32_t min_c,
                   double min_gain, const Params& params, std::size_t tree_index, std::size_t depth,
                   int n_threads, SymmetricWorkspace& sym, const CtrGrowContext& ctr_ctx) {
    CtrSplitCandidate best;
    CtrTrainingState& state = *ctr_ctx.state;
    const std::size_t n = ds.n_rows();
    const int eff_max_cplx = state.effective_max_complexity;
    if (eff_max_cplx <= 0) {
        return best;
    }

    std::vector<CtrProjection> cand_projs;
    cand_projs.reserve(active_cat.size() * (used_projs.size() + 1U));
    const auto onehot_max = static_cast<std::size_t>(params.max_cat_to_onehot);
    for (const std::uint32_t f : active_cat) {
        const std::size_t cm = ds.cat_sub_index(f);
        if (state.cat_cardinalities[cm] > onehot_max) {
            cand_projs.push_back(CtrProjection{{f}});
        }
    }
    if (depth >= 1U && eff_max_cplx >= 2) {
        const auto max_cplx = static_cast<std::size_t>(eff_max_cplx);
        for (const auto& base : used_projs) {
            if (base.complexity() >= max_cplx) {
                continue;
            }
            for (const std::uint32_t f : active_cat) {
                if (std::find(base.cat_features.begin(), base.cat_features.end(), f) !=
                    base.cat_features.end()) {
                    continue;
                }
                CtrProjection next = base;
                next.cat_features.push_back(f);
                std::sort(next.cat_features.begin(), next.cat_features.end());
                cand_projs.push_back(std::move(next));
            }
        }
    }
    std::sort(cand_projs.begin(), cand_projs.end());
    cand_projs.erase(std::unique(cand_projs.begin(), cand_projs.end()), cand_projs.end());
    if (cand_projs.empty()) {
        return best;
    }

    const std::size_t n_projs = cand_projs.size();
    const auto perm_idx = static_cast<std::uint32_t>(tree_index % state.n_perms);
    const auto& perm = state.perms[perm_idx];
    const std::size_t n_priors = ctr_ctx.is_unit_target ? 3U : 1U;
    constexpr std::array<double, 3> kUnitPriors = {0.0, 0.5, 1.0};

    std::vector<std::array<const CtrCachedCol*, 3>> tm_ptrs(n_projs, {nullptr, nullptr, nullptr});
    std::vector<const CtrCachedCol*> cnt_ptrs(n_projs, nullptr);
    std::vector<CtrMissGroup> miss_groups;
    miss_groups.reserve(n_projs);

    for (std::size_t pi = 0U; pi < n_projs; ++pi) {
        bool need_tm = false;
        for (std::size_t p_idx = 0U; p_idx < n_priors; ++p_idx) {
            CtrCacheKey key{cand_projs[pi], CtrKind::TargetMean, ctr_ctx.target_id, perm_idx,
                            static_cast<std::uint8_t>(p_idx)};
            if (const auto it = state.cache.find(key); it != state.cache.end()) {
                tm_ptrs[pi][p_idx] = &it->second;
            } else {
                need_tm = true;
            }
        }
        bool need_cnt = false;
        CtrCacheKey cnt_key{cand_projs[pi], CtrKind::Counter, 0U, 0U, 0U};
        if (const auto it = state.cache.find(cnt_key); it != state.cache.end()) {
            cnt_ptrs[pi] = &it->second;
        } else {
            need_cnt = true;
        }
        if (need_tm || need_cnt) {
            CtrMissGroup mg;
            mg.proj_idx = pi;
            mg.need_tm = need_tm;
            mg.need_cnt = need_cnt;
            miss_groups.push_back(std::move(mg));
        }
    }

    if (!miss_groups.empty()) {
        const int ctr_threads =
            std::min(n_threads, static_cast<int>(state.thread_scratches.size()));
        if (ctr_threads > 1 && miss_groups.size() >= 2U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(ctr_threads)
#endif
            for (std::ptrdiff_t mi = 0; mi < static_cast<std::ptrdiff_t>(miss_groups.size());
                 ++mi) {
#ifdef _OPENMP
                const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
                const std::size_t tid = 0U;
#endif
                CtrMissGroup& mg = miss_groups[static_cast<std::size_t>(mi)];
                compute_ctr_miss_group(ds, cand_projs[mg.proj_idx], perm, ctr_ctx, params, mg,
                                       state.thread_scratches[tid]);
            }
        } else {
            for (CtrMissGroup& mg : miss_groups) {
                compute_ctr_miss_group(ds, cand_projs[mg.proj_idx], perm, ctr_ctx, params, mg,
                                       state.thread_scratches[0]);
            }
        }

        for (CtrMissGroup& mg : miss_groups) {
            const std::size_t pi = mg.proj_idx;
            if (mg.need_tm) {
                for (std::size_t p_idx = 0U; p_idx < mg.tm_cols.size(); ++p_idx) {
                    if (tm_ptrs[pi][p_idx] != nullptr) {
                        continue;
                    }
                    const std::size_t bytes = mg.tm_cols[p_idx].byte_size();
                    if (state.cache_bytes + bytes <= kMaxCtrCacheBytes) {
                        state.cache_bytes += bytes;
                        CtrCacheKey key{cand_projs[pi], CtrKind::TargetMean, ctr_ctx.target_id,
                                        perm_idx, static_cast<std::uint8_t>(p_idx)};
                        auto [it, inserted] =
                            state.cache.emplace(std::move(key), std::move(mg.tm_cols[p_idx]));
                        (void)inserted;
                        tm_ptrs[pi][p_idx] = &it->second;
                    } else {
                        tm_ptrs[pi][p_idx] = &mg.tm_cols[p_idx];
                    }
                }
            }
            if (mg.need_cnt && cnt_ptrs[pi] == nullptr) {
                const std::size_t bytes = mg.cnt_col.byte_size();
                if (state.cache_bytes + bytes <= kMaxCtrCacheBytes) {
                    state.cache_bytes += bytes;
                    CtrCacheKey cnt_key{cand_projs[pi], CtrKind::Counter, 0U, 0U, 0U};
                    auto [it, inserted] =
                        state.cache.emplace(std::move(cnt_key), std::move(mg.cnt_col));
                    (void)inserted;
                    cnt_ptrs[pi] = &it->second;
                } else {
                    cnt_ptrs[pi] = &mg.cnt_col;
                }
            }
        }
    }

    std::vector<CtrScoreJob> jobs;
    jobs.reserve(n_projs * (n_priors + 1U));
    for (std::size_t pi = 0U; pi < n_projs; ++pi) {
        for (std::size_t p_idx = 0U; p_idx < n_priors; ++p_idx) {
            const CtrCachedCol* col = tm_ptrs[pi][p_idx];
            if (col != nullptr && !col->borders.empty()) {
                const double p_val = ctr_ctx.is_unit_target ? kUnitPriors[p_idx] : ctr_ctx.prior;
                jobs.push_back(CtrScoreJob{pi, CtrKind::TargetMean,
                                           static_cast<std::uint8_t>(p_idx), p_val, col});
            }
        }
        const CtrCachedCol* c_col = cnt_ptrs[pi];
        if (c_col != nullptr && !c_col->borders.empty() && c_col->n_unique_keys > 1U) {
            jobs.push_back(CtrScoreJob{pi, CtrKind::Counter, 0U, 0.0, c_col});
        }
    }
    if (jobs.empty()) {
        return best;
    }

    const std::size_t min_rep = (n >= 40U) ? std::max<std::size_t>(20U, n / 20U) : (n / 2U);
    std::vector<CtrSplitCandidate> job_results(jobs.size());
    auto score_one = [&](std::size_t ji, std::size_t tid) {
        const CtrScoreJob& job = jobs[ji];
        const CtrCachedCol& col = *job.col;
        const auto& proj = cand_projs[job.proj_idx];
        const std::size_t cplx = proj.complexity();
        const std::size_t n_uniq = col.n_unique_keys;
        const std::size_t n_rep = (n > n_uniq) ? (n - n_uniq) : 0U;
        if (n_uniq <= 1U || (cplx >= 2U && n_rep < min_rep) ||
            (cplx == 1U && n >= 100U && n_rep < min_rep)) {
            job_results[ji] = CtrSplitCandidate{};
            return;
        }

        const auto nb = static_cast<std::uint16_t>(col.borders.size());
        const std::size_t stride = static_cast<std::size_t>(nb) + 1U;
        HistBin* hist_f = sym.axis_thread_hists.data() + tid * sym.max_level_leaves * 256U;
        std::fill_n(hist_f, n_leaves * stride, HistBin{});

        const std::uint8_t* col_bins = col.bins.data();
        if (is_contiguous) {
            for (std::size_t t = 0; t < n_root; ++t) {
                const std::size_t idx =
                    static_cast<std::size_t>(leaf_ids[t]) * stride + col_bins[t];
                hist_f[idx].g += static_cast<double>(g_seq[t]);
                hist_f[idx].h += static_cast<double>(h_seq[t]);
                hist_f[idx].c += 1;
            }
        } else {
            for (std::size_t t = 0; t < n_root; ++t) {
                const std::size_t idx =
                    static_cast<std::size_t>(leaf_ids[t]) * stride + col_bins[rows[t]];
                hist_f[idx].g += static_cast<double>(g_seq[t]);
                hist_f[idx].h += static_cast<double>(h_seq[t]);
                hist_f[idx].c += 1;
            }
        }

        double denom = 1.0;
        const double reg = params.ctr_model_size_reg;
        if (reg > 0.0) {
            const double nd = static_cast<double>(std::max<std::size_t>(n, 1U));
            const double frac_u = static_cast<double>(n_uniq) / nd;
            if (cplx >= 2U) {
                denom = 1.0 + reg * static_cast<double>(cplx - 1U) *
                                  (std::sqrt(frac_u) + 10.0 / std::sqrt(nd));
            } else {
                denom = 1.0 + 0.25 * reg * frac_u;
            }
        }

        double gl_arr[kSymMaxLeaves] = {};
        double hl_arr[kSymMaxLeaves] = {};
        std::int32_t cl_arr[kSymMaxLeaves] = {};
        CtrSplitCandidate cand;
        for (std::uint16_t b = 0U; b < nb; ++b) {
            for (std::size_t l = 0; l < n_leaves; ++l) {
                const HistBin& hb = hist_f[l * stride + b];
                gl_arr[l] += hb.g;
                hl_arr[l] += hb.h;
                cl_arr[l] += hb.c;
            }
            double raw_gain = 0.0;
            if (eval_multi_leaf_gain(gl_arr, hl_arr, cl_arr, leaf_g, leaf_h, leaf_c, parent_score,
                                     n_leaves, lam, min_h, min_c, min_gain, raw_gain)) {
                const double scored_gain = raw_gain / denom;
                if (scored_gain > min_gain && (!cand.valid || scored_gain > cand.gain)) {
                    cand.valid = true;
                    cand.gain = scored_gain;
                    cand.proj = proj;
                    cand.kind = job.kind;
                    cand.prior_id = job.prior_id;
                    cand.prior_value = job.prior_value;
                    cand.bin_idx = static_cast<std::uint8_t>(b);
                    cand.threshold = col.borders[b];
                }
            }
        }
        job_results[ji] = std::move(cand);
    };

    if (n_threads > 1 && jobs.size() >= 2U) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
        for (std::ptrdiff_t ji = 0; ji < static_cast<std::ptrdiff_t>(jobs.size()); ++ji) {
#ifdef _OPENMP
            const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
            const std::size_t tid = 0U;
#endif
            score_one(static_cast<std::size_t>(ji), tid);
        }
    } else {
        for (std::size_t ji = 0U; ji < jobs.size(); ++ji) {
            score_one(ji, 0U);
        }
    }

    std::size_t win_job = 0U;
    for (std::size_t ji = 0U; ji < jobs.size(); ++ji) {
        const CtrSplitCandidate& res = job_results[ji];
        if (res.valid && (!best.valid || res.gain > best.gain)) {
            best = res;
            win_job = ji;
        }
    }
    if (best.valid) {
        const CtrCachedCol& win_col = *jobs[win_job].col;
        state.win_ctr_bins.assign(win_col.bins.begin(), win_col.bins.end());
        state.win_ctr_bin_idx = best.bin_idx;
    }
    return best;
}

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

void sym_oblique_pass2_cand(const std::int16_t* codes, std::size_t n_rows,
                            const std::uint32_t* leaf_rows, std::size_t n_root, bool is_contiguous,
                            std::size_t r0, std::size_t r1, const ObliqueCandMeta& cm,
                            std::int32_t pmin, FastDiv64 div, bool store_proj,
                            const std::int32_t* p_col, const std::uint8_t* leaf_ids,
                            const float* g_seq, const float* h_seq, HistBin* hist_out) noexcept {
    constexpr std::size_t kStride = static_cast<std::size_t>(kObliqueBins) + 1U;
    const std::int64_t pmin64 = pmin;
    std::uint8_t bins[kObliqueMicroBlock];

    if (store_proj) {
        for (std::size_t mb0 = r0; mb0 < r1; mb0 += kObliqueMicroBlock) {
            const std::size_t len = std::min(kObliqueMicroBlock, r1 - mb0);
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
        return;
    }

    const std::int16_t* col0 = codes + static_cast<std::size_t>(cm.fn[0]) * n_rows;
    const std::int16_t* col1 = codes + static_cast<std::size_t>(cm.fn[1]) * n_rows;
    const std::int16_t* col2 = codes + static_cast<std::size_t>(cm.fn[2]) * n_rows;
    const std::int32_t w0 = cm.w[0];
    const std::int32_t w1 = cm.w[1];
    const std::int32_t w2 = cm.w[2];

    for (std::size_t mb0 = r0; mb0 < r1; mb0 += kObliqueMicroBlock) {
        const std::size_t len = std::min(kObliqueMicroBlock, r1 - mb0);
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

ObliqueSplitCandidate
find_sym_oblique_split(const Dataset& ds, const std::uint32_t* leaf_rows, std::size_t n_root,
                       bool is_contiguous, const std::uint8_t* leaf_ids, std::size_t n_leaves,
                       const float* g_seq, const float* h_seq, const double* leaf_g,
                       const double* leaf_h, const std::int32_t* leaf_c, const double* parent_score,
                       std::size_t tree_index, int level, const Params& params,
                       ObliqueScratch& scratch, SymmetricWorkspace& sym) noexcept {
    ObliqueSplitCandidate best;
    const std::size_t p_elig = scratch.eligible_features.size();
    if (params.m_mult <= 0 || p_elig < 2U || scratch.cur_m == 0U || n_root < 2U) {
        return best;
    }

    const std::size_t m = scratch.cur_m;
    const std::size_t n_rows = ds.n_rows();
    const std::int16_t* codes = ds.codes_col_major().data();

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
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
        for (std::ptrdiff_t as = 0; as < static_cast<std::ptrdiff_t>(n_active); ++as) {
            const auto a = static_cast<std::size_t>(as);
            const std::uint32_t j = scratch.active_cands[a];
            const ObliqueCandMeta& cm = metas[j];
            const bool check_nan = (cm.has_nan != 0U) &&
                                   (scratch.cand_valid_cnt[j] < static_cast<std::int32_t>(n_root));
#ifdef _OPENMP
            const auto tid = static_cast<std::size_t>(omp_get_thread_num());
#else
            const std::size_t tid = 0U;
#endif
            HistBin* hist = sym.oblique_thread_hists.data() + tid * sym.max_level_leaves * kStride;
            std::memset(static_cast<void*>(hist), 0, hist_elems * sizeof(HistBin));
            sym_oblique_pass2_cand(codes, n_rows, leaf_rows, n_root, is_contiguous, 0U, n_root, cm,
                                   scratch.cand_pmin[j], scratch.active_divs[a], store_proj,
                                   store_proj ? (proj_buf + j * n_root) : nullptr, leaf_ids, g_seq,
                                   h_seq, hist);

            ObliqueSplitCandidate cand;
            double cg = -std::numeric_limits<double>::infinity();
            std::int64_t thr = 0;
            bool nl = true;
            bool found = false;
            scan_sym_oblique_hist(hist, n_leaves, leaf_g, leaf_h, leaf_c, parent_score, lam, min_h,
                                  min_c, min_gain, static_cast<std::int64_t>(scratch.cand_pmin[j]),
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
        for (std::size_t a = 0; a < n_active; ++a) {
            const ObliqueSplitCandidate& cand = scratch.i64_candidates[a];
            if (cand.valid && cand.gain > best.gain) {
                best = cand;
            }
        }
        return best;
    }

    HistBin* hist = sym.oblique_thread_hists.data();
    const bool use_chunk_wave = (n_active == 1U) && (n_chunks > 1U) && (n_root >= 32768U);
    for (std::size_t a = 0; a < n_active; ++a) {
        const std::uint32_t j = scratch.active_cands[a];
        const ObliqueCandMeta& cm = metas[j];
        const bool check_nan =
            (cm.has_nan != 0U) && (scratch.cand_valid_cnt[j] < static_cast<std::int32_t>(n_root));
        const std::int32_t* p_col = store_proj ? (proj_buf + j * n_root) : nullptr;
        std::memset(static_cast<void*>(hist), 0, hist_elems * sizeof(HistBin));

        if (!use_chunk_wave) {
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
    const ObliqueSplitCandidate pick = find_sym_oblique_split(
        ds, sym.samp_rows.data(), n_s, false, sym.samp_leaf.data(), n_leaves, sym.samp_g.data(),
        sym.samp_h.data(), sg, sh, sc, sps, tree_index, level, ps, scratch, sym);
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
    sym.samp_proj.resize(n_root);
    sym.samp_chunk_min.assign(n_chunks, std::numeric_limits<std::int64_t>::max());
    sym.samp_chunk_max.assign(n_chunks, kNanProj);
    std::int64_t* projv = sym.samp_proj.data();
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_chunks > 1U)
#endif
    for (std::ptrdiff_t ci = 0; ci < static_cast<std::ptrdiff_t>(n_chunks); ++ci) {
        const std::size_t c0 = static_cast<std::size_t>(ci) * kRefitChunk;
        const std::size_t c1 = std::min(n_root, c0 + kRefitChunk);
        HistBin* h = chunk_hists + static_cast<std::size_t>(ci) * hist_size;
        for (std::size_t i = c0; i < c1; ++i) {
            const std::int64_t p = projv[i];
            const std::size_t b = (p == kNanProj) ? static_cast<std::size_t>(kObliqueBins)
                                                  : oblique_bin_index(p, pmin, range);
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

} // namespace

void SymmetricWorkspace::prepare(const Dataset& ds, const Params& params) {
    const std::size_t n = ds.n_rows();
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

        // 1. Axis-aligned numeric splits.
        if (n_num_act > 0U) {
            if (n_threads > 1 && n_num_act >= 2U && n_root * n_num_act >= 20000U) {
                std::vector<AxisSplitCandidate> cand_res(n_num_act);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(n_threads)
#endif
                for (std::ptrdiff_t idx_s = 0; idx_s < static_cast<std::ptrdiff_t>(n_num_act);
                     ++idx_s) {
                    const auto idx = static_cast<std::size_t>(idx_s);
                    const std::uint32_t f = active_num[idx];
                    const std::uint16_t nb = ws.layout.num_regular_bins[f];
                    const bool has_nan = ws.layout.has_nan[f] != 0U;
                    const std::size_t stride = static_cast<std::size_t>(nb) + (has_nan ? 1U : 0U);
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
                                           d, depth_limit, n_leaves, stride, false, 1, n_small,
                                           small_bit, sym, hist_f, prev_f);
                    cand_res[idx] =
                        scan_sym_axis_feature(ds, f, nb, has_nan, n_leaves, hist_f, leaf_g, leaf_h,
                                              leaf_c, parent_score, total_h, lam, min_h, min_c,
                                              min_gain, params, tree_index, static_cast<int>(d));
                }
                for (const AxisSplitCandidate& c : cand_res) {
                    if (c.valid && c.gain > choice.axis.gain) {
                        choice.axis = c;
                    }
                }
            } else {
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
                    build_sym_feature_hist(col, rows, n_root, is_contiguous, leaf_ids, g_seq, h_seq,
                                           d, depth_limit, n_leaves, stride, allow_wave, n_threads,
                                           n_small, small_bit, sym, hist_f, prev_f);
                    const AxisSplitCandidate c =
                        scan_sym_axis_feature(ds, f, nb, has_nan, n_leaves, hist_f, leaf_g, leaf_h,
                                              leaf_c, parent_score, total_h, lam, min_h, min_c,
                                              min_gain, params, tree_index, static_cast<int>(d));
                    if (c.valid && c.gain > choice.axis.gain) {
                        choice.axis = c;
                    }
                }
            }
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
