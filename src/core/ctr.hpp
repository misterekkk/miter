#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "data/dataset.hpp"

namespace miter {

inline constexpr std::uint32_t kMissingCatHash = 0x9e3779b9U;
inline constexpr std::uint64_t kCtrHashMagic = 0x4906ba494954cb65ULL;
inline constexpr std::uint64_t kCtrNumLeftHash = 0x517cc1b727220a95ULL;
inline constexpr std::uint64_t kCtrNumRightHash = 0x94d049bb133111ebULL;
inline constexpr std::size_t kMaxCtrCacheBytes = 128ULL * 1024ULL * 1024ULL;

std::uint32_t cat_hash_string(std::string_view s) noexcept;

inline std::uint64_t combine_cat_hash(std::uint64_t a, std::uint64_t b) noexcept {
    return kCtrHashMagic * (a + kCtrHashMagic * b);
}

inline std::uint64_t ctr_slot_of(std::uint64_t key, std::uint64_t mask) noexcept {
    return (key ^ (key >> 29U)) & mask;
}

float lookup_ctr_table(const CtrTable& table, std::uint64_t key) noexcept;

struct CtrScratch {
    std::vector<std::uint64_t> keys;
    std::vector<std::uint32_t> gens;
    std::vector<double> w_sum;
    std::vector<double> wy_sum;
    std::vector<std::uint32_t> cnt;
    std::uint64_t mask = 0;
    std::uint32_t cur_gen = 0;

    void prepare(std::size_t n_rows);
    void next_generation() noexcept;
};

struct CtrSplitCandidate {
    CtrProjection proj;
    CtrKind kind = CtrKind::TargetMean;
    std::uint8_t prior_id = 0;
    double prior_value = 0.0;
    std::uint8_t bin_idx = 0;
    float threshold = 0.0f;
    double gain = -1.0;
    bool valid = false;
};

struct CtrRegisteredSpec {
    CtrProjection proj;
    CtrKind kind = CtrKind::TargetMean;
    std::uint32_t target_id = 0;
    std::uint8_t prior_id = 0;
    double prior_value = 0.0;
};

struct CtrCacheKey {
    CtrProjection proj;
    CtrKind kind = CtrKind::TargetMean;
    std::uint32_t target_id = 0;
    std::uint32_t perm_idx = 0;
    std::uint8_t prior_id = 0;

    bool operator==(const CtrCacheKey& other) const = default;
};

struct CtrCacheKeyHash {
    std::size_t operator()(const CtrCacheKey& k) const noexcept;
};

struct CtrCachedCol {
    std::vector<std::uint8_t> bins;
    std::vector<float> borders;
    std::size_t n_unique_keys = 0;

    std::size_t byte_size() const noexcept { return bins.size() + borders.size() * sizeof(float); }
};

struct CtrThreadScratch {
    CtrScratch scratch;
    std::vector<std::uint64_t> proj_hashes;
    std::vector<float> ctr_vals;
    std::vector<float> sample_vals;
};

class CtrTrainingState {
public:
    void prepare(const Dataset& ds, const Params& params, std::size_t n_classes = 1U);

    std::size_t n_rows = 0;
    std::size_t n_perms = 0;
    int effective_max_complexity = 0;
    std::vector<std::vector<std::uint32_t>> perms;
    CtrScratch scratch;
    std::vector<CtrThreadScratch> thread_scratches;
    std::vector<std::size_t> cat_cardinalities;

    // Quantized 1-byte CTR cache across levels and trees.
    std::unordered_map<CtrCacheKey, CtrCachedCol, CtrCacheKeyHash> cache;
    std::size_t cache_bytes = 0;

    // Winning split bins copied once per level for advancing tree leaf_ids.
    std::vector<std::uint8_t> win_ctr_bins;
    std::uint8_t win_ctr_bin_idx = 0;

    // Model-wide deduplicated CTR definitions registered during boosting.
    std::vector<CtrRegisteredSpec> specs;
    std::size_t finalized_count = 0;

    std::uint32_t register_ctr(const CtrProjection& proj, CtrKind kind, std::uint32_t target_id,
                               std::uint8_t prior_id, double prior_value, Model& model);
};

// Resolve effective max CTR complexity (-1 = auto, else clamp/pass params.max_ctr_complexity).
int resolve_effective_ctr_complexity(const Dataset& ds,
                                     const std::vector<std::size_t>& cat_cardinalities,
                                     const Params& params, std::size_t n_classes = 1U) noexcept;

// Compute projection hashes for all rows [0, n_rows) in dataset.
void compute_projection_hashes(const Dataset& ds, const CtrProjection& proj,
                               std::uint64_t* out_hashes) noexcept;

void compute_projection_hashes(const Dataset& ds, std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept;

// Compute projection hashes for all rows [0, td.n_rows) in TransformedData.
void compute_projection_hashes(const TransformedData& td,
                               const std::vector<std::size_t>& feature_to_cat_idx,
                               std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept;

// Compute online ordered CTRs into out_vals[0..n_rows) (indexed by row id) along perm.
// For CtrKind::Counter, computes normalized log1p(count) / log1p(max_count) independent of y/perm.
void compute_online_ctr(const std::uint64_t* proj_hashes, std::size_t n_rows,
                        std::span<const std::uint32_t> perm, const double* y,
                        const double* sample_weight, double prior, double prior_weight,
                        CtrKind kind, CtrScratch& scratch, float* out_vals) noexcept;

// Compute TargetMean online ordered CTRs for multiple priors in a single permutation pass.
// out_vals_col_major must have space for priors.size() * n_rows floats. Returns unique key count.
std::size_t compute_online_ctr_multi_prior(const std::uint64_t* proj_hashes, std::size_t n_rows,
                                           std::span<const std::uint32_t> perm, const double* y,
                                           const double* sample_weight,
                                           std::span<const double> priors, double prior_weight,
                                           CtrScratch& scratch, float* out_vals_col_major) noexcept;

// Compute normalized Counter CTRs into out_vals[0..n_rows). Returns unique key count.
std::size_t compute_online_counter(const std::uint64_t* proj_hashes, std::size_t n_rows,
                                   CtrScratch& scratch, float* out_vals) noexcept;

// Uniform O(1) binning on [0, 1] matching bin <= b <=> val < borders[b] bit-for-bit.
std::uint16_t build_uniform_unit_borders_and_bins(const float* ctr_vals, std::size_t n_rows,
                                                  int max_borders, std::vector<float>& borders_out,
                                                  std::uint8_t* out_bins);

// Extract up to max_borders data-quantile thresholds from active rows and bin active rows.
std::uint16_t build_ctr_borders_and_bins(const float* ctr_vals, const std::uint32_t* active_rows,
                                         std::size_t n_active, bool is_contiguous, int max_borders,
                                         std::vector<float>& sample_buf,
                                         std::vector<float>& borders_out, std::uint8_t* out_bins);

// Finalize any newly registered CTR tables into model.ctr_tables.
void finalize_pending_ctr_tables(const Dataset& ds, CtrTrainingState& state, const double* target_y,
                                 double prior, const double* sample_weight, double prior_weight,
                                 std::uint32_t current_target_id, Model& model,
                                 CtrScratch& scratch);

} // namespace miter
