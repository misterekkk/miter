#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "data/dataset.hpp"

namespace miter {

inline constexpr std::uint32_t kMissingCatHash = 0x9e3779b9U;
inline constexpr std::uint64_t kCtrHashMagic = 0x4906ba494954cb65ULL;

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
    std::vector<std::uint32_t> cat_features;
    CtrKind kind = CtrKind::TargetMean;
    float threshold = 0.0f;
    double gain = -1.0;
    bool valid = false;
};

struct CtrRegisteredSpec {
    std::vector<std::uint32_t> cat_features;
    CtrKind kind = CtrKind::TargetMean;
    std::uint32_t target_id = 0;
};

class CtrTrainingState {
public:
    void prepare(const Dataset& ds, const Params& params);

    std::size_t n_rows = 0;
    std::size_t n_perms = 0;
    std::vector<std::vector<std::uint32_t>> perms;
    CtrScratch scratch;

    // Reusable buffers during symmetric tree level candidate scoring.
    std::vector<std::uint64_t> proj_hashes;
    std::vector<float> ctr_vals;
    std::vector<float> win_ctr_vals;
    std::vector<std::uint8_t> ctr_bins;
    std::vector<float> borders;
    std::vector<float> sample_vals;

    // Model-wide deduplicated CTR definitions registered during boosting.
    std::vector<CtrRegisteredSpec> specs;
    std::size_t finalized_count = 0;

    std::uint32_t register_ctr(const std::vector<std::uint32_t>& cat_features, CtrKind kind,
                               std::uint32_t target_id, Model& model);
};

// Compute projection hashes for all rows [0, n_rows) in dataset.
void compute_projection_hashes(const Dataset& ds, std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept;

// Compute projection hashes for all rows [0, td.n_rows) in TransformedData.
void compute_projection_hashes(const TransformedData& td,
                               const std::vector<std::size_t>& feature_to_cat_idx,
                               std::span<const std::uint32_t> cat_features,
                               std::uint64_t* out_hashes) noexcept;

// Compute online ordered CTRs into out_vals[0..n_rows) (indexed by row id) along perm.
// For CtrKind::Counter, computes LOO log1p(count - 1) independent of y/perm.
void compute_online_ctr(const std::uint64_t* proj_hashes, std::size_t n_rows,
                        std::span<const std::uint32_t> perm, const double* y,
                        const double* sample_weight, double prior, double prior_weight,
                        CtrKind kind, CtrScratch& scratch, float* out_vals) noexcept;

// Extract up to max_borders strictly increasing float thresholds from active rows and bin active
// rows into out_bins[0..n_active) (values in [0, n_borders]). Returns n_borders.
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
