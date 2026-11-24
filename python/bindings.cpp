#include <miter/miter.hpp>

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <string>
#include <utility>

namespace nb = nanobind;

namespace {

std::string hello() {
    return "miter " + std::string(miter::version());
}

std::string cpp_version() {
    return std::string(miter::version());
}

std::pair<std::size_t, std::size_t>
array_shape(const nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>& arr) {
    const std::size_t rows = arr.shape(0);
    const std::size_t cols = arr.shape(1);
    nb::gil_scoped_release release;
    return {rows, cols};
}

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "miter C++20 core module";

    m.def("hello", &hello, "Return a greeting with the miter version.");
    m.def("cpp_version", &cpp_version, "Return the miter C++ library version.");
    m.def("has_openmp", &miter::has_openmp, "Return whether miter was built with OpenMP.");
    m.def("openmp_max_threads", &miter::openmp_max_threads,
          "Return the maximum number of OpenMP threads.");
    m.def("lignum_linked", &miter::lignum_linked,
          "Return whether the local lignum library is linked and callable.");
    m.def("lignum_smoke_hash", &miter::lignum_smoke_hash, nb::arg("text"),
          "Compute a deterministic hash via lignum::catboost_hash.");
    m.def("array_shape", &array_shape, nb::arg("arr").noconvert(),
          "Return (rows, cols) for a 2D float64 NumPy array without copying.");
}
