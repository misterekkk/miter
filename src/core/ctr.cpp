#include "core/ctr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include "core/rng.hpp"
#include "util/threading.hpp"

namespace miter {

namespace {

// Google CityHash64 (v1, MIT License) matching CatBoost & lignum::city_hash64 bit-for-bit.
constexpr std::uint64_t kCityK0 = 0xc3a5c85c97cb3127ULL;
constexpr std::uint64_t kCityK1 = 0xb492b66fbe98f273ULL;
constexpr std::uint64_t kCityK2 = 0x9ae16a3b2f90404fULL;
constexpr std::uint64_t kCityK3 = 0xc949d7c7509e6557ULL;

std::uint64_t fetch64(const char* p) noexcept {
    std::uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

std::uint32_t fetch32(const char* p) noexcept {
    std::uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

std::uint64_t rotate64(std::uint64_t v, int s) noexcept {
    return (s == 0) ? v : ((v >> s) | (v << (64 - s)));
}

std::uint64_t rotate_by_at_least1(std::uint64_t v, int s) noexcept {
    return (v >> s) | (v << (64 - s));
}

std::uint64_t shift_mix(std::uint64_t v) noexcept {
    return v ^ (v >> 47);
}

std::uint64_t hash_len16(std::uint64_t u, std::uint64_t v) noexcept {
    constexpr std::uint64_t kMul = 0x9ddfea08eb382d69ULL;
    std::uint64_t a = (u ^ v) * kMul;
    a ^= (a >> 47);
    std::uint64_t b = (v ^ a) * kMul;
    b ^= (b >> 47);
    return b * kMul;
}

std::uint64_t hash_len0to16(const char* s, std::size_t len) noexcept {
    if (len > 8U) {
        const std::uint64_t a = fetch64(s);
        const std::uint64_t b = fetch64(s + len - 8U);
        return hash_len16(a, rotate_by_at_least1(b + len, static_cast<int>(len))) ^ b;
    }
    if (len >= 4U) {
        const std::uint64_t a = fetch32(s);
        return hash_len16(len + (a << 3U), fetch32(s + len - 4U));
    }
    if (len > 0U) {
        const auto a = static_cast<std::uint8_t>(s[0]);
        const auto b = static_cast<std::uint8_t>(s[len >> 1U]);
        const auto c = static_cast<std::uint8_t>(s[len - 1U]);
        const std::uint32_t y =
            static_cast<std::uint32_t>(a) + (static_cast<std::uint32_t>(b) << 8U);
        const std::uint32_t z =
            static_cast<std::uint32_t>(len) + (static_cast<std::uint32_t>(c) << 2U);
        return shift_mix((y * kCityK2) ^ (z * kCityK3)) * kCityK2;
    }
    return kCityK2;
}

std::uint64_t hash_len17to32(const char* s, std::size_t len) noexcept {
    const std::uint64_t a = fetch64(s) * kCityK1;
    const std::uint64_t b = fetch64(s + 8U);
    const std::uint64_t c = fetch64(s + len - 8U) * kCityK2;
    const std::uint64_t d = fetch64(s + len - 16U) * kCityK0;
    return hash_len16(rotate64(a - b, 43) + rotate64(c, 30) + d,
                      a + rotate64(b ^ kCityK3, 20) - c + len);
}

std::pair<std::uint64_t, std::uint64_t> weak_hash_len32(std::uint64_t w, std::uint64_t x,
                                                        std::uint64_t y, std::uint64_t z,
                                                        std::uint64_t a, std::uint64_t b) noexcept {
    a += w;
    b = rotate64(b + a + z, 21);
    const std::uint64_t c = a;
    a += x;
    a += y;
    b += rotate64(a, 44);
    return {a + z, b + c};
}

std::pair<std::uint64_t, std::uint64_t> weak_hash_len32(const char* s, std::uint64_t a,
                                                        std::uint64_t b) noexcept {
    return weak_hash_len32(fetch64(s), fetch64(s + 8U), fetch64(s + 16U), fetch64(s + 24U), a, b);
}

std::uint64_t hash_len33to64(const char* s, std::size_t len) noexcept {
    std::uint64_t z = fetch64(s + 24U);
    std::uint64_t a = fetch64(s) + (len + fetch64(s + len - 16U)) * kCityK0;
    std::uint64_t b = rotate64(a + z, 52);
    std::uint64_t c = rotate64(a, 37);
    a += fetch64(s + 8U);
    c += rotate64(a, 7);
    a += fetch64(s + 16U);
    const std::uint64_t vf = a + z;
    const std::uint64_t vs = b + rotate64(a, 31) + c;
    a = fetch64(s + 16U) + fetch64(s + len - 32U);
    z = fetch64(s + len - 8U);
    b = rotate64(a + z, 52);
    c = rotate64(a, 37);
    a += fetch64(s + len - 24U);
    c += rotate64(a, 7);
    a += fetch64(s + len - 16U);
    const std::uint64_t wf = a + z;
    const std::uint64_t ws = b + rotate64(a, 31) + c;
    const std::uint64_t r = shift_mix((vf + ws) * kCityK2 + (wf + vs) * kCityK0);
    return shift_mix(r * kCityK0 + vs) * kCityK2;
}

std::uint64_t city_hash64(const char* s, std::size_t len) noexcept {
    if (len <= 32U) {
        return (len <= 16U) ? hash_len0to16(s, len) : hash_len17to32(s, len);
    }
    if (len <= 64U) {
        return hash_len33to64(s, len);
    }
    std::uint64_t x = fetch64(s);
    std::uint64_t y = fetch64(s + len - 16U) ^ kCityK1;
    std::uint64_t z = fetch64(s + len - 56U) ^ kCityK0;
    auto v = weak_hash_len32(s + len - 64U, len, y);
    auto w = weak_hash_len32(s + len - 32U, len * kCityK1, kCityK0);
    z += shift_mix(v.second) * kCityK1;
    x = rotate64(z + x, 39) * kCityK1;
    y = rotate64(y, 33) * kCityK1;
    len = (len - 1U) & ~static_cast<std::size_t>(63U);
    do {
        x = rotate64(x + y + v.first + fetch64(s + 16U), 37) * kCityK1;
        y = rotate64(y + v.second + fetch64(s + 48U), 42) * kCityK1;
        x ^= w.second;
        y ^= v.first;
        z = rotate64(z ^ w.first, 33);
        v = weak_hash_len32(s, v.second * kCityK1, x + w.first);
        w = weak_hash_len32(s + 32U, z + w.second, y);
        std::swap(z, x);
        s += 64;
        len -= 64U;
    } while (len != 0U);
    return hash_len16(hash_len16(v.first, w.first) + shift_mix(y) * kCityK1 + z,
                      hash_len16(v.second, w.second) + x);
}

std::size_t next_pow2_min(std::size_t min_cap) noexcept {
    std::size_t cap = 256U;
    while (cap < min_cap) {
        cap <<= 1U;
    }
    return cap;
}

std::size_t find_or_insert_slot(CtrScratch& s, std::uint64_t key, bool& is_new_out) noexcept {
    std::uint64_t idx = ctr_slot_of(key, s.mask);
    for (;;) {
        if (s.gens[idx] != s.cur_gen) {
            s.gens[idx] = s.cur_gen;
            s.keys[idx] = key;
            s.w_sum[idx] = 0.0;
            s.wy_sum[idx] = 0.0;
            s.cnt[idx] = 0U;
            is_new_out = true;
            return static_cast<std::size_t>(idx);
        }
        if (s.keys[idx] == key) {
            is_new_out = false;
            return static_cast<std::size_t>(idx);
        }
        idx = (idx + 1U) & s.mask;
    }
}

std::size_t find_slot_const(const CtrScratch& s, std::uint64_t key) noexcept {
    std::uint64_t idx = ctr_slot_of(key, s.mask);
    for (;;) {
        if (s.gens[idx] != s.cur_gen || s.keys[idx] == key) {
            return static_cast<std::size_t>(idx);
        }
        idx = (idx + 1U) & s.mask;
    }
}

} // namespace

std::uint32_t cat_hash_string(std::string_view s) noexcept {
    return static_cast<std::uint32_t>(city_hash64(s.data(), s.size()));
}

float lookup_ctr_table(const CtrTable& table, std::uint64_t key) noexcept {
    if (table.slots.empty()) {
        return table.default_value;
    }
    std::uint64_t idx = ctr_slot_of(key, table.mask);
    for (;;) {
        const std::uint32_t entry = table.slots[idx];
        if (entry == 0U) {
            return table.default_value;
        }
        if (table.keys[idx] == key) {
            return table.values[entry - 1U];
        }
        idx = (idx + 1U) & table.mask;
    }
}

void CtrScratch::prepare(std::size_t n_rows) {
    const std::size_t cap = next_pow2_min(std::max<std::size_t>(n_rows * 2U + 16U, 256U));
    keys.assign(cap, 0ULL);
    gens.assign(cap, 0U);
    w_sum.assign(cap, 0.0);
    wy_sum.assign(cap, 0.0);
    cnt.assign(cap, 0U);
    mask = static_cast<std::uint64_t>(cap - 1U);
    cur_gen = 0U;
}

void CtrScratch::next_generation() noexcept {
    if (cur_gen == std::numeric_limits<std::uint32_t>::max()) {
        std::fill(gens.begin(), gens.end(), 0U);
        cur_gen = 1U;
    } else {
        ++cur_gen;
    }
}

std::size_t CtrCacheKeyHash::operator()(const CtrCacheKey& k) const noexcept {
    std::size_t h = static_cast<std::size_t>(k.kind);
    auto mix = [&h](std::size_t v) noexcept {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U);
    };
    mix(k.target_id);
    mix(k.perm_idx);
    mix(k.prior_id);
    mix(k.cat_features.size());
    for (const std::uint32_t f : k.cat_features) {
        mix(f);
    }
    return h;
}

void CtrTrainingState::prepare(const Dataset& ds, const Params& params) {
    n_rows = ds.n_rows();
    n_perms = static_cast<std::size_t>(std::max(params.ctr_n_permutations, 1));
    perms.resize(n_perms);
    constexpr std::uint64_t kPurposeCtrPerm = 7U;
    for (std::size_t p = 0; p < n_perms; ++p) {
        auto& perm = perms[p];
        perm.resize(n_rows);
        for (std::size_t i = 0; i < n_rows; ++i) {
            perm[i] = static_cast<std::uint32_t>(i);
        }
        CounterRng rng(params.random_state, static_cast<std::uint64_t>(p), 0U, 0U, kPurposeCtrPerm);
        for (std::size_t i = n_rows; i > 1U; --i) {
            const std::size_t j = static_cast<std::size_t>(rng.next_u64() % i);
            std::swap(perm[i - 1U], perm[j]);
        }
    }

    scratch.prepare(n_rows);
    const std::size_t n_thr =
        static_cast<std::size_t>(std::clamp(resolve_num_threads(params.n_jobs), 1, 8));
    thread_scratches.resize(n_thr);
    for (auto& ts : thread_scratches) {
        ts.scratch.prepare(n_rows);
        ts.proj_hashes.resize(n_rows);
        ts.ctr_vals.resize(3U * n_rows);
        ts.sample_vals.reserve(2048U);
    }

    cat_cardinalities.assign(ds.n_cat_features(), 0U);
    const std::uint32_t* hash_base = ds.cat_hashes_col_major().data();
    if (hash_base != nullptr && n_rows > 0U) {
        CtrScratch& sc = thread_scratches[0].scratch;
        for (const std::size_t f : ds.cat_feature_indices()) {
            const std::size_t cm = ds.cat_sub_index(f);
            const std::uint32_t* hcol = hash_base + cm * n_rows;
            sc.next_generation();
            std::size_t uniq = 0U;
            for (std::size_t i = 0U; i < n_rows; ++i) {
                bool is_new = false;
                (void)find_or_insert_slot(sc, static_cast<std::uint64_t>(hcol[i]), is_new);
                if (is_new) {
                    ++uniq;
                }
            }
            cat_cardinalities[cm] = uniq;
        }
    }

    cache.clear();
    cache_bytes = 0U;
    win_ctr_bins.resize(n_rows);
    win_ctr_bin_idx = 0U;
    specs.clear();
    finalized_count = 0U;
}

std::uint32_t CtrTrainingState::register_ctr(const std::vector<std::uint32_t>& cat_features,
                                             CtrKind kind, std::uint32_t target_id,
                                             std::uint8_t prior_id, double prior_value,
                                             Model& model) {
    for (std::size_t id = 0; id < specs.size(); ++id) {
        if (specs[id].kind == kind && specs[id].target_id == target_id &&
            specs[id].prior_id == prior_id && specs[id].cat_features == cat_features) {
            return static_cast<std::uint32_t>(id);
        }
    }

    // Deduplicate projection across TargetMean/Counter/target_id/prior_id.
    std::uint32_t proj_idx = 0U;
    bool found_proj = false;
    for (std::size_t pi = 0; pi < model.ctr_projections.size(); ++pi) {
        if (model.ctr_projections[pi].cat_features == cat_features) {
            proj_idx = static_cast<std::uint32_t>(pi);
            found_proj = true;
            break;
        }
    }
    if (!found_proj) {
        proj_idx = static_cast<std::uint32_t>(model.ctr_projections.size());
        model.ctr_projections.push_back(CtrProjection{cat_features});
    }

    const auto new_id = static_cast<std::uint32_t>(specs.size());
    specs.push_back(CtrRegisteredSpec{cat_features, kind, target_id, prior_id, prior_value});

    CtrTable tbl;
    tbl.projection = proj_idx;
    tbl.kind = kind;
    model.ctr_tables.push_back(std::move(tbl));
    return new_id;
}

void compute_projection_hashes(const Dataset& ds, std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept {
    const std::size_t n = ds.n_rows();
    const std::uint32_t* base = ds.cat_hashes_col_major().data();
    if (cat_features.empty()) {
        std::fill_n(out_hashes, n, 0ULL);
        return;
    }
    const std::size_t c0 = ds.cat_sub_index(cat_features[0]);
    const std::uint32_t* h0 = base + c0 * n;
    for (std::size_t i = 0; i < n; ++i) {
        out_hashes[i] = static_cast<std::uint64_t>(h0[i]);
    }
    for (std::size_t m = 1U; m < cat_features.size(); ++m) {
        const std::size_t cm = ds.cat_sub_index(cat_features[m]);
        const std::uint32_t* hm = base + cm * n;
        for (std::size_t i = 0; i < n; ++i) {
            out_hashes[i] = combine_cat_hash(out_hashes[i], static_cast<std::uint64_t>(hm[i]));
        }
    }
}

void compute_projection_hashes(const TransformedData& td,
                               const std::vector<std::size_t>& feature_to_cat_idx,
                               std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept {
    const std::size_t n = td.n_rows;
    const std::uint32_t* base = td.cat_hashes_col_major.data();
    if (cat_features.empty() || base == nullptr) {
        std::fill_n(out_hashes, n, 0ULL);
        return;
    }
    const std::size_t c0 = feature_to_cat_idx[cat_features[0]];
    const std::uint32_t* h0 = base + c0 * n;
    for (std::size_t i = 0; i < n; ++i) {
        out_hashes[i] = static_cast<std::uint64_t>(h0[i]);
    }
    for (std::size_t m = 1U; m < cat_features.size(); ++m) {
        const std::size_t cm = feature_to_cat_idx[cat_features[m]];
        const std::uint32_t* hm = base + cm * n;
        for (std::size_t i = 0; i < n; ++i) {
            out_hashes[i] = combine_cat_hash(out_hashes[i], static_cast<std::uint64_t>(hm[i]));
        }
    }
}

std::size_t compute_online_ctr_multi_prior(const std::uint64_t* proj_hashes, std::size_t n_rows,
                                           std::span<const std::uint32_t> perm, const double* y,
                                           const double* sample_weight,
                                           std::span<const double> priors, double prior_weight,
                                           CtrScratch& scratch,
                                           float* out_vals_col_major) noexcept {
    scratch.next_generation();
    const std::size_t n_priors = priors.size();
    std::size_t n_unique = 0U;
    for (std::size_t pos = 0; pos < n_rows; ++pos) {
        const std::size_t i = perm[pos];
        const std::uint64_t key = proj_hashes[i];
        bool is_new = false;
        const std::size_t slot = find_or_insert_slot(scratch, key, is_new);
        if (is_new) {
            ++n_unique;
        }
        const double wy = scratch.wy_sum[slot];
        const double denom = scratch.w_sum[slot] + prior_weight;
        for (std::size_t p = 0U; p < n_priors; ++p) {
            out_vals_col_major[p * n_rows + i] =
                static_cast<float>((wy + prior_weight * priors[p]) / denom);
        }

        const double w = (sample_weight != nullptr) ? sample_weight[i] : 1.0;
        scratch.w_sum[slot] += w;
        scratch.wy_sum[slot] += w * y[i];
    }
    return n_unique;
}

std::size_t compute_online_counter(const std::uint64_t* proj_hashes, std::size_t n_rows,
                                   CtrScratch& scratch, float* out_vals) noexcept {
    scratch.next_generation();
    std::size_t n_unique = 0U;
    std::uint32_t max_cnt = 0U;
    for (std::size_t i = 0; i < n_rows; ++i) {
        bool is_new = false;
        const std::size_t slot = find_or_insert_slot(scratch, proj_hashes[i], is_new);
        if (is_new) {
            ++n_unique;
        }
        const std::uint32_t c = ++scratch.cnt[slot];
        if (c > max_cnt) {
            max_cnt = c;
        }
    }
    const double log_denom = (max_cnt > 0U) ? std::log1p(static_cast<double>(max_cnt)) : 1.0;
    for (std::size_t i = 0; i < n_rows; ++i) {
        const std::size_t slot = find_slot_const(scratch, proj_hashes[i]);
        out_vals[i] =
            static_cast<float>(std::log1p(static_cast<double>(scratch.cnt[slot])) / log_denom);
    }
    return n_unique;
}

void compute_online_ctr(const std::uint64_t* proj_hashes, std::size_t n_rows,
                        std::span<const std::uint32_t> perm, const double* y,
                        const double* sample_weight, double prior, double prior_weight,
                        CtrKind kind, CtrScratch& scratch, float* out_vals) noexcept {
    if (kind == CtrKind::TargetMean) {
        (void)compute_online_ctr_multi_prior(proj_hashes, n_rows, perm, y, sample_weight,
                                             std::span<const double>(&prior, 1U), prior_weight,
                                             scratch, out_vals);
        return;
    }
    (void)compute_online_counter(proj_hashes, n_rows, scratch, out_vals);
}

std::uint16_t build_uniform_unit_borders_and_bins(const float* ctr_vals, std::size_t n_rows,
                                                  int max_borders, std::vector<float>& borders_out,
                                                  std::uint8_t* out_bins) {
    borders_out.clear();
    if (n_rows < 2U || max_borders < 1) {
        std::fill_n(out_bins, n_rows, std::uint8_t{0});
        return 0U;
    }
    const int nb = std::clamp(max_borders, 1, 255);
    const float f_steps = static_cast<float>(nb + 1);
    borders_out.resize(static_cast<std::size_t>(nb));
    for (int m = 0; m < nb; ++m) {
        borders_out[static_cast<std::size_t>(m)] = static_cast<float>(m + 1) / f_steps;
    }

    const float* bp = borders_out.data();
    for (std::size_t i = 0; i < n_rows; ++i) {
        const float v = ctr_vals[i];
        int b = std::clamp(static_cast<int>(v * f_steps), 0, nb);
        while (b > 0 && v < bp[b - 1]) {
            --b;
        }
        while (b < nb && v >= bp[b]) {
            ++b;
        }
        out_bins[i] = static_cast<std::uint8_t>(b);
    }
    return static_cast<std::uint16_t>(nb);
}

std::uint16_t build_ctr_borders_and_bins(const float* ctr_vals, const std::uint32_t* active_rows,
                                         std::size_t n_active, bool is_contiguous, int max_borders,
                                         std::vector<float>& sample_buf,
                                         std::vector<float>& borders_out, std::uint8_t* out_bins) {
    borders_out.clear();
    if (n_active < 2U || max_borders < 1) {
        std::fill_n(out_bins, n_active, std::uint8_t{0});
        return 0U;
    }

    constexpr std::size_t kMaxSample = 2048U;
    sample_buf.clear();
    if (n_active <= kMaxSample) {
        sample_buf.resize(n_active);
        for (std::size_t t = 0; t < n_active; ++t) {
            const std::size_t idx = is_contiguous ? t : active_rows[t];
            sample_buf[t] = ctr_vals[idx];
        }
    } else {
        sample_buf.resize(kMaxSample);
        for (std::size_t s = 0; s < kMaxSample; ++s) {
            const std::size_t t = (s * (n_active - 1U)) / (kMaxSample - 1U);
            const std::size_t idx = is_contiguous ? t : active_rows[t];
            sample_buf[s] = ctr_vals[idx];
        }
    }

    std::sort(sample_buf.begin(), sample_buf.end());
    if (sample_buf.front() == sample_buf.back()) {
        std::fill_n(out_bins, n_active, std::uint8_t{0});
        return 0U;
    }

    const std::size_t n_sample = sample_buf.size();
    const auto mb = static_cast<std::size_t>(std::min(max_borders, 255));
    borders_out.reserve(mb);
    for (std::size_t m = 1U; m <= mb; ++m) {
        const std::size_t idx = (m * (n_sample - 1U)) / (mb + 1U);
        const float b = sample_buf[idx];
        if (b > sample_buf.front() && (borders_out.empty() || b > borders_out.back())) {
            borders_out.push_back(b);
        }
    }
    if (borders_out.empty()) {
        borders_out.push_back(sample_buf.back());
    }

    const std::size_t nb = borders_out.size();
    const float* bp = borders_out.data();
    for (std::size_t t = 0; t < n_active; ++t) {
        const std::size_t idx = is_contiguous ? t : active_rows[t];
        const float v = ctr_vals[idx];
        std::size_t lo = 0U;
        std::size_t hi = nb;
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2U;
            if (v < bp[mid]) {
                hi = mid;
            } else {
                lo = mid + 1U;
            }
        }
        out_bins[t] = static_cast<std::uint8_t>(lo);
    }
    return static_cast<std::uint16_t>(nb);
}

void finalize_pending_ctr_tables(const Dataset& ds, CtrTrainingState& state, const double* target_y,
                                 double prior, const double* sample_weight, double prior_weight,
                                 std::uint32_t current_target_id, Model& model,
                                 CtrScratch& scratch) {
    (void)prior;
    if (state.finalized_count >= state.specs.size()) {
        return;
    }
    const std::size_t n = ds.n_rows();
    std::vector<std::uint64_t> hashes(n);

    for (std::size_t id = state.finalized_count; id < state.specs.size(); ++id) {
        const CtrRegisteredSpec& spec = state.specs[id];
        if (spec.target_id != current_target_id) {
            continue;
        }
        compute_projection_hashes(ds, spec.cat_features, hashes.data());
        scratch.next_generation();

        std::vector<std::size_t> occupied;
        occupied.reserve(std::min<std::size_t>(n, 4096U));

        std::uint32_t max_cnt = 0U;
        if (spec.kind == CtrKind::TargetMean) {
            for (std::size_t i = 0; i < n; ++i) {
                bool is_new = false;
                const std::size_t slot = find_or_insert_slot(scratch, hashes[i], is_new);
                if (is_new) {
                    occupied.push_back(slot);
                }
                const double w = (sample_weight != nullptr) ? sample_weight[i] : 1.0;
                scratch.w_sum[slot] += w;
                scratch.wy_sum[slot] += w * target_y[i];
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                bool is_new = false;
                const std::size_t slot = find_or_insert_slot(scratch, hashes[i], is_new);
                if (is_new) {
                    occupied.push_back(slot);
                }
                const std::uint32_t c = ++scratch.cnt[slot];
                if (c > max_cnt) {
                    max_cnt = c;
                }
            }
        }

        // Sort occupied by key for bit-identical serialization order.
        std::sort(occupied.begin(), occupied.end(),
                  [&](std::size_t a, std::size_t b) { return scratch.keys[a] < scratch.keys[b]; });

        const std::size_t n_entries = occupied.size();
        const std::size_t tbl_cap = next_pow2_min(n_entries * 2U + 1U);
        CtrTable& dst = model.ctr_tables[id];
        dst.kind = spec.kind;
        dst.default_value =
            (spec.kind == CtrKind::TargetMean) ? static_cast<float>(spec.prior_value) : 0.0f;
        dst.mask = static_cast<std::uint64_t>(tbl_cap - 1U);
        dst.keys.assign(tbl_cap, 0ULL);
        dst.slots.assign(tbl_cap, 0U);
        dst.values.resize(n_entries);

        const double wp = prior_weight * spec.prior_value;
        const double log_denom = (max_cnt > 0U) ? std::log1p(static_cast<double>(max_cnt)) : 1.0;
        for (std::size_t e = 0; e < n_entries; ++e) {
            const std::size_t src_slot = occupied[e];
            const std::uint64_t key = scratch.keys[src_slot];
            if (spec.kind == CtrKind::TargetMean) {
                dst.values[e] = static_cast<float>((scratch.wy_sum[src_slot] + wp) /
                                                   (scratch.w_sum[src_slot] + prior_weight));
            } else {
                dst.values[e] = static_cast<float>(
                    std::log1p(static_cast<double>(scratch.cnt[src_slot])) / log_denom);
            }
            std::uint64_t t_idx = ctr_slot_of(key, dst.mask);
            while (dst.slots[t_idx] != 0U) {
                t_idx = (t_idx + 1U) & dst.mask;
            }
            dst.keys[t_idx] = key;
            dst.slots[t_idx] = static_cast<std::uint32_t>(e + 1U);
        }
    }
    state.finalized_count = state.specs.size();
}

} // namespace miter
