#include "core/categorical.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace miter {

namespace {

struct CatGroup {
    double g = 0.0;
    double h = 0.0;
    double score = 0.0;
    std::int32_t c = 0;
    std::int16_t bin = -1; // -1 represents the merged rare group (< min_data_per_group)
};

} // namespace

CatSplitCandidate find_best_cat_split(const FeatureHistLayout& layout, const HistBin* hist,
                                      double parent_g, double parent_h, std::int32_t parent_c,
                                      const Params& params) noexcept {
    CatSplitCandidate best;
    if (layout.active_cat_features.empty() || parent_c < 2 * params.min_data_in_leaf) {
        return best;
    }

    const double lam = params.lambda_l2;
    const double min_h = params.min_sum_hessian_in_leaf;
    const std::int32_t min_c = params.min_data_in_leaf;
    const double min_gain = params.min_gain_to_split;
    const double parent_score = (parent_g * parent_g) / (parent_h + lam);

    for (const std::uint32_t f : layout.active_cat_features) {
        const std::uint16_t nb = layout.num_regular_bins[f];
        const HistBin* fh = hist + layout.offsets[f];

        int n_used = 0;
        for (std::uint16_t b = 0; b < nb; ++b) {
            if (fh[b].c > 0) {
                ++n_used;
            }
        }
        if (n_used < 2) {
            continue;
        }

        if (n_used <= params.max_cat_to_onehot) {
            double best_feat_gain = -std::numeric_limits<double>::infinity();
            std::int32_t best_b = -1;
            double best_hl = 0.0;

            for (std::uint16_t b = 0; b < nb; ++b) {
                const std::int32_t cl = fh[b].c;
                if (cl <= 0) {
                    continue;
                }
                const std::int32_t cr = parent_c - cl;
                if (cl < min_c || cr < min_c) {
                    continue;
                }
                const double hl = fh[b].h;
                const double hr = parent_h - hl;
                if (hl < min_h || hr < min_h) {
                    continue;
                }
                const double gl = fh[b].g;
                const double gr = parent_g - gl;
                const double gain = (gl * gl) / (hl + lam) + (gr * gr) / (hr + lam) - parent_score;
                if (gain > best_feat_gain && gain > min_gain) {
                    best_feat_gain = gain;
                    best_b = static_cast<std::int32_t>(b);
                    best_hl = hl;
                }
            }

            if (best_b >= 0 && best_feat_gain > best.gain) {
                best.gain = best_feat_gain;
                best.feature = f;
                best.default_go_left = (best_hl >= (parent_h - best_hl));
                best.left_bitset = CatBitset{};
                for (std::uint16_t b = 0; b < nb; ++b) {
                    const bool go_left = (static_cast<std::int32_t>(b) == best_b) ||
                                         (fh[b].c <= 0 && best.default_go_left);
                    if (go_left) {
                        cat_bitset_set(best.left_bitset, static_cast<std::uint8_t>(b));
                    }
                }
                best.valid = true;
            }
            continue;
        }

        CatGroup groups[256];
        int n_groups = 0;
        double rare_g = 0.0;
        double rare_h = 0.0;
        std::int32_t rare_c = 0;
        const std::int32_t min_grp = std::max(params.min_data_per_group, 1);

        for (std::uint16_t b = 0; b < nb; ++b) {
            const std::int32_t c = fh[b].c;
            if (c <= 0) {
                continue;
            }
            if (c >= min_grp) {
                groups[n_groups] = CatGroup{fh[b].g, fh[b].h, 0.0, c, static_cast<std::int16_t>(b)};
                ++n_groups;
            } else {
                rare_g += fh[b].g;
                rare_h += fh[b].h;
                rare_c += c;
            }
        }
        if (rare_c > 0) {
            groups[n_groups] = CatGroup{rare_g, rare_h, 0.0, rare_c, -1};
            ++n_groups;
        }
        if (n_groups < 2) {
            continue;
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
        const int lim = std::min(std::max(params.max_cat_threshold, 1), n_groups - 1);
        double best_feat_gain = -std::numeric_limits<double>::infinity();
        int best_dir = -1;
        int best_t = 0;
        double best_hl = 0.0;

        for (int dir = 0; dir < 2; ++dir) {
            double gl = 0.0;
            double hl = 0.0;
            std::int32_t cl = 0;
            for (int t = 1; t <= lim; ++t) {
                const int q = (dir == 0) ? (t - 1) : (n_groups - t);
                gl += groups[q].g;
                hl += groups[q].h;
                cl += groups[q].c;
                const std::int32_t cr = parent_c - cl;
                if (cl < min_c || cr < min_c) {
                    continue;
                }
                const double hr = parent_h - hl;
                if (hl < min_h || hr < min_h) {
                    continue;
                }
                const double gr = parent_g - gl;
                const double gain =
                    (gl * gl) / (hl + lam2) + (gr * gr) / (hr + lam2) - parent_score;
                if (gain > best_feat_gain && gain > min_gain) {
                    best_feat_gain = gain;
                    best_dir = dir;
                    best_t = t;
                    best_hl = hl;
                }
            }
        }

        if (best_dir >= 0 && best_feat_gain > best.gain) {
            best.gain = best_feat_gain;
            best.feature = f;
            best.default_go_left = (best_hl >= (parent_h - best_hl));
            best.left_bitset = CatBitset{};

            bool rare_left = false;
            for (int t = 1; t <= best_t; ++t) {
                const int q = (best_dir == 0) ? (t - 1) : (n_groups - t);
                if (groups[q].bin < 0) {
                    rare_left = true;
                } else {
                    cat_bitset_set(best.left_bitset, static_cast<std::uint8_t>(groups[q].bin));
                }
            }
            for (std::uint16_t b = 0; b < nb; ++b) {
                const std::int32_t c = fh[b].c;
                if (c <= 0 && best.default_go_left) {
                    cat_bitset_set(best.left_bitset, static_cast<std::uint8_t>(b));
                } else if (c > 0 && c < min_grp && rare_left) {
                    cat_bitset_set(best.left_bitset, static_cast<std::uint8_t>(b));
                }
            }
            best.valid = true;
        }
    }

    return best;
}

} // namespace miter
