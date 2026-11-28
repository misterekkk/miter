// Symmetric trees: CatBoost-style CTR splits (target statistics over categorical combinations).

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

} // namespace

void add_unique_ctr_proj(std::vector<CtrProjection>& used, const CtrProjection& proj) {
    for (const auto& existing : used) {
        if (existing == proj) {
            return;
        }
    }
    used.push_back(proj);
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
    const std::size_t min_card =
        std::max<std::size_t>(static_cast<std::size_t>(params.max_cat_to_onehot),
                              static_cast<std::size_t>(std::max(params.ctr_min_cardinality, 0)));
    for (const std::uint32_t f : active_cat) {
        const std::size_t cm = ds.cat_sub_index(f);
        if (state.cat_cardinalities[cm] > min_card) {
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

} // namespace miter::sym_detail
