#include <miter/miter.hpp>

#include <lignum/lignum.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace miter {

std::string_view version() noexcept {
    return MITER_VERSION;
}

bool has_openmp() noexcept {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

int openmp_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

bool lignum_linked() {
    // Call into lignum so the linker verifies the dependency at build and test time.
    return lignum::catboost_hash("miter") != 0U;
}

std::uint32_t lignum_smoke_hash(std::string_view text) {
    return lignum::catboost_hash(text);
}

} // namespace miter
