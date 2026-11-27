// Symmetric trees: exclusive feature bundling (EFB) of sparse numeric features.

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

// efb: greedy exclusive feature bundling of sparse numeric features, largest first (the idea is
// from Ke et al., "LightGBM", NeurIPS 2017; this implementation is miter's own). A feature joins
// the first bundle where none of its non-default rows is taken and the bins still fit.
void build_efb_bundles(const Dataset& ds, const Params& params, SymmetricWorkspace& sym,
                       std::size_t& total_bins) {
    sym.efb_bundles.clear();
    sym.efb_bundle_of.clear();
    sym.efb_offset_of.clear();
    sym.efb_default_code.clear();
    sym.efb_cols.resize(0);
    sym.efb_hists.resize(0);
    if (!params.efb) {
        return;
    }
    const std::size_t n = ds.n_rows();
    const std::size_t p = ds.n_features();
    const std::uint8_t* bins = ds.bins_col_major().data();
    sym.efb_bundle_of.assign(p, -1);
    sym.efb_offset_of.assign(p, 0U);
    sym.efb_default_code.assign(p, 0U);
    struct Cand {
        std::uint32_t f;
        std::size_t nd;
        std::uint16_t stride;
    };
    std::vector<Cand> cands;
    for (std::size_t f = 0U; f < p; ++f) {
        if (ds.feature_kinds()[f] == FeatureKind::Categorical) {
            continue;
        }
        const auto& binner = ds.binner(f);
        const std::size_t stride =
            static_cast<std::size_t>(binner.num_bins) + (binner.has_nan ? 1U : 0U);
        if (stride < 2U || stride > 64U) {
            continue;
        }
        std::array<std::size_t, 256> cnt{};
        const std::uint8_t* col = bins + f * n;
        for (std::size_t r = 0U; r < n; ++r) {
            ++cnt[col[r]];
        }
        std::size_t def = 0U;
        for (std::size_t c = 1U; c < stride; ++c) {
            if (cnt[c] > cnt[def]) {
                def = c;
            }
        }
        const std::size_t nd = n - cnt[def];
        if (nd * 4U > n) {
            continue; // dense: rarely exclusive, not worth the checks
        }
        sym.efb_default_code[f] = static_cast<std::uint8_t>(def);
        cands.push_back({static_cast<std::uint32_t>(f), nd, static_cast<std::uint16_t>(stride)});
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.nd > b.nd; });
    constexpr std::size_t kMaxBundles = 32U;
    std::vector<std::vector<std::uint8_t>> cols;
    std::vector<SymmetricWorkspace::EfbBundle> bundles;
    for (const Cand& c : cands) {
        const std::uint8_t* col = bins + static_cast<std::size_t>(c.f) * n;
        const std::uint8_t def = sym.efb_default_code[c.f];
        const std::size_t slots = static_cast<std::size_t>(c.stride) - 1U;
        std::size_t chosen = bundles.size();
        for (std::size_t b = 0U; b < bundles.size(); ++b) {
            if (static_cast<std::size_t>(bundles[b].n_bins) + slots > 256U) {
                continue;
            }
            const std::uint8_t* bc = cols[b].data();
            bool ok = true;
            for (std::size_t r = 0U; r < n && ok; ++r) {
                ok = (col[r] == def) || (bc[r] == 0U);
            }
            if (ok) {
                chosen = b;
                break;
            }
        }
        if (chosen == bundles.size()) {
            if (bundles.size() >= kMaxBundles) {
                continue;
            }
            bundles.push_back({});
            bundles.back().n_bins = 1U;
            cols.emplace_back(n, std::uint8_t{0});
        }
        auto& bun = bundles[chosen];
        const std::uint16_t off = bun.n_bins;
        std::uint8_t* bc = cols[chosen].data();
        for (std::size_t r = 0U; r < n; ++r) {
            const std::uint8_t v = col[r];
            if (v != def) {
                bc[r] = static_cast<std::uint8_t>(off + (v < def ? v : v - 1U));
            }
        }
        bun.members.push_back(c.f);
        sym.efb_offset_of[c.f] = off;
        bun.n_bins = static_cast<std::uint16_t>(bun.n_bins + slots);
    }
    std::vector<std::size_t> keep;
    for (std::size_t b = 0U; b < bundles.size(); ++b) {
        if (bundles[b].members.size() >= 2U) {
            keep.push_back(b);
        }
    }
    sym.efb_cols.resize(keep.size() * n);
    for (std::size_t i = 0U; i < keep.size(); ++i) {
        auto bun = bundles[keep[i]];
        std::memcpy(sym.efb_cols.data() + i * n, cols[keep[i]].data(), n);
        for (const std::uint32_t f : bun.members) {
            sym.efb_bundle_of[f] = static_cast<std::int32_t>(i);
        }
        bun.prev_offset = total_bins;
        total_bins += bun.n_bins;
        sym.efb_bundles.push_back(std::move(bun));
    }
    sym.efb_hists.resize(sym.efb_bundles.size() * sym.max_level_leaves * 256U);
}

// A member feature's histogram from its bundle's: its own slots, and its default bin as the leaf
// total minus those slots.
void expand_efb_member(const SymmetricWorkspace& sym, std::uint32_t f, std::size_t stride,
                       std::size_t n_leaves, const double* leaf_g, const double* leaf_h,
                       const std::int32_t* leaf_c, HistBin* hist_f) noexcept {
    const auto bi = static_cast<std::size_t>(sym.efb_bundle_of[f]);
    const auto& bun = sym.efb_bundles[bi];
    const std::size_t nb = bun.n_bins;
    const HistBin* bh = sym.efb_hists.data() + bi * sym.max_level_leaves * 256U;
    const std::size_t off = sym.efb_offset_of[f];
    const std::size_t def = sym.efb_default_code[f];
    for (std::size_t l = 0U; l < n_leaves; ++l) {
        HistBin* hf = hist_f + l * stride;
        double gs = 0.0;
        double hs = 0.0;
        std::int32_t cs = 0;
        for (std::size_t c = 0U; c < stride; ++c) {
            if (c == def) {
                continue;
            }
            const HistBin& x = bh[l * nb + off + (c < def ? c : c - 1U)];
            hf[c] = x;
            gs += x.g;
            hs += x.h;
            cs += x.c;
        }
        hf[def].g = leaf_g[l] - gs;
        hf[def].h = leaf_h[l] - hs;
        hf[def].c = leaf_c[l] - cs;
    }
}

} // namespace miter::sym_detail
