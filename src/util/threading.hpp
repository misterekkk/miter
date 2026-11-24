#pragma once

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miter {

// Fixed row chunk and wave widths ensure chunk reduction order is invariant to thread count.
inline constexpr std::size_t kDetChunkRows = 4096U;
inline constexpr std::size_t kWaveChunks = 32U;

/**
 * @brief Resolve the active thread count for a booster or evaluator call.
 *
 * Returns 1 when n_jobs == 1 or OpenMP is unavailable; callers must not enter
 * an OpenMP parallel region when this returns 1.
 */
inline int resolve_num_threads(int n_jobs) noexcept {
#ifdef _OPENMP
    if (n_jobs == 1) {
        return 1;
    }
    const int max_t = std::max(omp_get_max_threads(), 1);
    if (n_jobs <= 0) {
        return max_t;
    }
    return std::min(n_jobs, max_t);
#else
    (void)n_jobs;
    return 1;
#endif
}

} // namespace miter
