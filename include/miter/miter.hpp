#pragma once

#include <miter/model.hpp>
#include <miter/params.hpp>

#include <cstdint>
#include <string_view>

namespace miter {

/**
 * @brief Return the miter C++ library version string.
 */
std::string_view version() noexcept;

/**
 * @brief Return true if miter was compiled with OpenMP support.
 */
bool has_openmp() noexcept;

/**
 * @brief Return the maximum number of OpenMP threads (1 if compiled without OpenMP).
 */
int openmp_max_threads() noexcept;

/**
 * @brief Return true when the local lignum library is linked and callable.
 */
bool lignum_linked();

/**
 * @brief Compute a deterministic smoke-test hash via lignum::catboost_hash.
 */
std::uint32_t lignum_smoke_hash(std::string_view text);

} // namespace miter
