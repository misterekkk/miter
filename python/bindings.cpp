#include <miter/miter.hpp>

#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "data/dataset.hpp"

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

template <typename T> nb::ndarray<nb::numpy, T, nb::ndim<1>> vec_to_ndarray_1d(std::vector<T> vec) {
    auto* heap_vec = new std::vector<T>(std::move(vec));
    nb::capsule owner(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    const std::size_t shape[1] = {heap_vec->size()};
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(heap_vec->data(), 1, shape, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T, nb::ndim<2>> vec_to_ndarray_2d(std::vector<T> vec, std::size_t rows,
                                                         std::size_t cols) {
    auto* heap_vec = new std::vector<T>(std::move(vec));
    nb::capsule owner(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {static_cast<std::int64_t>(cols), 1};
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(heap_vec->data(), 2, shape, owner, strides);
}

template <typename FloatT>
std::vector<miter::NumericColumnView>
views_from_2d(const nb::ndarray<const FloatT, nb::ndim<2>, nb::device::cpu>& arr) {
    const std::size_t n_rows = arr.shape(0);
    const std::size_t n_cols = arr.shape(1);
    if (n_rows == 0 || n_cols == 0) {
        throw std::invalid_argument("Input array must have at least 1 row and 1 column");
    }
    const std::int64_t s0 = arr.stride(0);
    const std::int64_t s1 = arr.stride(1);
    if ((n_rows > 1 && s0 <= 0) || (n_cols > 1 && s1 <= 0) || s0 < 0 || s1 < 0) {
        throw std::invalid_argument(
            "Input array must have positive strides; pass np.ascontiguousarray(X)");
    }

    const FloatT* base = arr.data();
    const std::size_t row_stride = (n_rows == 1) ? 1U : static_cast<std::size_t>(s0);
    std::vector<miter::NumericColumnView> views;
    views.reserve(n_cols);
    for (std::size_t j = 0; j < n_cols; ++j) {
        miter::NumericColumnView v;
        const FloatT* col_ptr = base + j * static_cast<std::size_t>(s1);
        if constexpr (std::is_same_v<FloatT, double>) {
            v.f64 = col_ptr;
        } else {
            v.f32 = col_ptr;
        }
        v.row_stride = row_stride;
        views.push_back(v);
    }
    return views;
}

template <typename DataHolder>
nb::ndarray<nb::numpy, const std::int16_t, nb::ndim<2>> make_codes_view(nb::object self) {
    const auto& obj = nb::cast<const DataHolder&>(self);
    const std::size_t rows = obj.n_rows;
    const std::size_t cols = obj.n_num_features;
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {static_cast<std::int64_t>(cols), 1};
    const std::int16_t* ptr = (cols == 0 || rows == 0) ? nullptr : obj.codes_row_major.data();
    return nb::ndarray<nb::numpy, const std::int16_t, nb::ndim<2>>(ptr, 2, shape, self, strides);
}

template <>
nb::ndarray<nb::numpy, const std::int16_t, nb::ndim<2>>
make_codes_view<miter::Dataset>(nb::object self) {
    const auto& ds = nb::cast<const miter::Dataset&>(self);
    const std::size_t rows = ds.n_rows();
    const std::size_t cols = ds.n_num_features();
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {static_cast<std::int64_t>(cols), 1};
    const std::int16_t* ptr = (cols == 0 || rows == 0) ? nullptr : ds.codes_row_major().data();
    return nb::ndarray<nb::numpy, const std::int16_t, nb::ndim<2>>(ptr, 2, shape, self, strides);
}

template <typename DataHolder>
nb::ndarray<nb::numpy, std::uint16_t, nb::ndim<2>> make_unsigned_codes(const DataHolder& obj) {
    const std::size_t rows = obj.n_rows;
    const std::size_t cols = obj.n_num_features;
    std::vector<std::uint16_t> u(rows * cols);
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = miter::Q16Quantizer::to_unsigned_code(obj.codes_row_major[i]);
    }
    return vec_to_ndarray_2d(std::move(u), rows, cols);
}

template <>
nb::ndarray<nb::numpy, std::uint16_t, nb::ndim<2>>
make_unsigned_codes<miter::Dataset>(const miter::Dataset& ds) {
    const std::size_t rows = ds.n_rows();
    const std::size_t cols = ds.n_num_features();
    std::vector<std::uint16_t> u(rows * cols);
    const auto& codes = ds.codes_row_major();
    for (std::size_t i = 0; i < u.size(); ++i) {
        u[i] = miter::Q16Quantizer::to_unsigned_code(codes[i]);
    }
    return vec_to_ndarray_2d(std::move(u), rows, cols);
}

template <typename DataHolder>
nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<2>> make_bins_view(nb::object self) {
    const auto& obj = nb::cast<const DataHolder&>(self);
    const std::size_t rows = obj.n_rows;
    const std::size_t cols = obj.n_features;
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {1, static_cast<std::int64_t>(rows)};
    const std::uint8_t* ptr = (cols == 0 || rows == 0) ? nullptr : obj.bins_col_major.data();
    return nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<2>>(ptr, 2, shape, self, strides);
}

template <>
nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<2>>
make_bins_view<miter::Dataset>(nb::object self) {
    const auto& ds = nb::cast<const miter::Dataset&>(self);
    const std::size_t rows = ds.n_rows();
    const std::size_t cols = ds.n_features();
    const std::size_t shape[2] = {rows, cols};
    const std::int64_t strides[2] = {1, static_cast<std::int64_t>(rows)};
    const std::uint8_t* ptr = (cols == 0 || rows == 0) ? nullptr : ds.bins_col_major().data();
    return nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<2>>(ptr, 2, shape, self, strides);
}

std::vector<miter::NumericColumnView> extract_num_col_views(const nb::list& num_cols,
                                                            std::size_t n_rows) {
    std::vector<miter::NumericColumnView> views;
    views.reserve(num_cols.size());
    for (nb::handle item : num_cols) {
        nb::ndarray<const double, nb::ndim<1>, nb::device::cpu> arr_f64;
        if (nb::try_cast(item, arr_f64)) {
            if (arr_f64.shape(0) != n_rows) {
                throw std::invalid_argument("Numeric column length does not match n_rows");
            }
            const std::int64_t s0 = arr_f64.stride(0);
            if (n_rows > 1 && s0 <= 0) {
                throw std::invalid_argument("Numeric column must have positive stride");
            }
            views.push_back(miter::NumericColumnView{
                arr_f64.data(), nullptr, (n_rows <= 1) ? 1U : static_cast<std::size_t>(s0)});
            continue;
        }
        nb::ndarray<const float, nb::ndim<1>, nb::device::cpu> arr_f32;
        if (nb::try_cast(item, arr_f32)) {
            if (arr_f32.shape(0) != n_rows) {
                throw std::invalid_argument("Numeric column length does not match n_rows");
            }
            const std::int64_t s0 = arr_f32.stride(0);
            if (n_rows > 1 && s0 <= 0) {
                throw std::invalid_argument("Numeric column must have positive stride");
            }
            views.push_back(miter::NumericColumnView{
                nullptr, arr_f32.data(), (n_rows <= 1) ? 1U : static_cast<std::size_t>(s0)});
            continue;
        }
        throw std::invalid_argument("Numeric columns must be 1D float64 or float32 CPU arrays");
    }
    return views;
}

std::vector<miter::CategoricalColumnView> extract_cat_col_views(
    std::vector<std::vector<std::string>> vocabs,
    const std::vector<nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>&
        indices,
    std::size_t n_rows) {
    if (vocabs.size() != indices.size()) {
        throw std::invalid_argument("cat_vocabs and cat_indices length mismatch");
    }
    std::vector<miter::CategoricalColumnView> views;
    views.reserve(vocabs.size());
    for (std::size_t c = 0; c < vocabs.size(); ++c) {
        if (indices[c].shape(0) != n_rows) {
            throw std::invalid_argument("Categorical index array length does not match n_rows");
        }
        miter::CategoricalColumnView view;
        view.vocab = std::move(vocabs[c]);
        view.indices = indices[c].data();
        views.push_back(std::move(view));
    }
    return views;
}

miter::Dataset fit_numeric_f64(const nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>& arr,
                               int max_bin, std::vector<std::string> feature_names) {
    const std::size_t n_rows = arr.shape(0);
    const std::size_t n_cols = arr.shape(1);
    auto views = views_from_2d(arr);
    std::vector<miter::FeatureKind> kinds(n_cols, miter::FeatureKind::Numeric);
    miter::DataConfig cfg;
    cfg.max_bin = max_bin;
    nb::gil_scoped_release release;
    return miter::Dataset::fit_mixed(n_rows, std::move(kinds), std::move(feature_names), views, {},
                                     cfg);
}

miter::Dataset fit_numeric_f32(const nb::ndarray<const float, nb::ndim<2>, nb::device::cpu>& arr,
                               int max_bin, std::vector<std::string> feature_names) {
    const std::size_t n_rows = arr.shape(0);
    const std::size_t n_cols = arr.shape(1);
    auto views = views_from_2d(arr);
    std::vector<miter::FeatureKind> kinds(n_cols, miter::FeatureKind::Numeric);
    miter::DataConfig cfg;
    cfg.max_bin = max_bin;
    nb::gil_scoped_release release;
    return miter::Dataset::fit_mixed(n_rows, std::move(kinds), std::move(feature_names), views, {},
                                     cfg);
}

miter::Dataset fit_mixed_py(
    std::size_t n_rows, const std::vector<bool>& is_cat, std::vector<std::string> feature_names,
    const nb::list& num_cols, std::vector<std::vector<std::string>> cat_vocabs,
    const std::vector<nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>&
        cat_indices,
    int max_bin, int min_data_per_group, int max_categories) {
    std::vector<miter::FeatureKind> kinds;
    kinds.reserve(is_cat.size());
    for (const bool flag : is_cat) {
        kinds.push_back(flag ? miter::FeatureKind::Categorical : miter::FeatureKind::Numeric);
    }
    auto num_views = extract_num_col_views(num_cols, n_rows);
    auto cat_views = extract_cat_col_views(std::move(cat_vocabs), cat_indices, n_rows);
    miter::DataConfig cfg;
    cfg.max_bin = max_bin;
    cfg.min_data_per_group = min_data_per_group;
    cfg.max_categories = max_categories;
    nb::gil_scoped_release release;
    return miter::Dataset::fit_mixed(n_rows, std::move(kinds), std::move(feature_names), num_views,
                                     cat_views, cfg);
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

    nb::class_<miter::TransformedData>(m, "TransformedData")
        .def_ro("n_rows", &miter::TransformedData::n_rows)
        .def_ro("n_features", &miter::TransformedData::n_features)
        .def_ro("n_num_features", &miter::TransformedData::n_num_features)
        .def("codes", &make_codes_view<miter::TransformedData>)
        .def("unsigned_codes", &make_unsigned_codes<miter::TransformedData>)
        .def("bins", &make_bins_view<miter::TransformedData>);

    nb::class_<miter::Dataset>(m, "Dataset")
        .def_prop_ro("n_rows", &miter::Dataset::n_rows)
        .def_prop_ro("n_features", &miter::Dataset::n_features)
        .def_prop_ro("n_num_features", &miter::Dataset::n_num_features)
        .def_prop_ro("n_cat_features", &miter::Dataset::n_cat_features)
        .def_prop_ro("feature_names", &miter::Dataset::feature_names)
        .def_prop_ro("is_categorical",
                     [](const miter::Dataset& ds) {
                         std::vector<bool> flags;
                         flags.reserve(ds.n_features());
                         for (const auto k : ds.feature_kinds()) {
                             flags.push_back(k == miter::FeatureKind::Categorical);
                         }
                         return flags;
                     })
        .def("codes", &make_codes_view<miter::Dataset>)
        .def("unsigned_codes", &make_unsigned_codes<miter::Dataset>)
        .def("bins", &make_bins_view<miter::Dataset>)
        .def("quantizer_params",
             [](const miter::Dataset& ds) {
                 const std::size_t p_num = ds.n_num_features();
                 std::vector<double> lo(p_num);
                 std::vector<double> scale(p_num);
                 std::vector<std::uint8_t> is_const(p_num);
                 for (std::size_t j = 0; j < p_num; ++j) {
                     const auto& q = ds.quantizer(j);
                     lo[j] = q.lo;
                     scale[j] = q.scale;
                     is_const[j] = q.is_constant ? 1U : 0U;
                 }
                 return std::make_tuple(vec_to_ndarray_1d(std::move(lo)),
                                        vec_to_ndarray_1d(std::move(scale)),
                                        vec_to_ndarray_1d(std::move(is_const)));
             })
        .def("bin_edges",
             [](const miter::Dataset& ds, std::size_t feature_idx) {
                 return vec_to_ndarray_1d(ds.binner(feature_idx).edges);
             })
        .def("num_bins_for_feature", &miter::Dataset::num_bins_for_feature)
        .def("cat_categories",
             [](const miter::Dataset& ds, std::size_t feature_idx) {
                 return ds.cat_encoder(feature_idx).categories;
             })
        .def("cat_other_code",
             [](const miter::Dataset& ds, std::size_t feature_idx) {
                 return ds.cat_encoder(feature_idx).other_code();
             })
        .def("cat_missing_code",
             [](const miter::Dataset& ds, std::size_t feature_idx) {
                 return ds.cat_encoder(feature_idx).missing_code();
             })
        .def(
            "transform_numeric_f64",
            [](const miter::Dataset& ds,
               const nb::ndarray<const double, nb::ndim<2>, nb::device::cpu>& arr) {
                if (ds.n_cat_features() != 0 || arr.shape(1) != ds.n_num_features()) {
                    throw std::invalid_argument("Feature count or kind mismatch in transform");
                }
                const std::size_t n_rows = arr.shape(0);
                auto views = views_from_2d(arr);
                nb::gil_scoped_release release;
                return ds.transform_mixed(n_rows, views, {});
            },
            nb::arg("arr").noconvert())
        .def(
            "transform_numeric_f32",
            [](const miter::Dataset& ds,
               const nb::ndarray<const float, nb::ndim<2>, nb::device::cpu>& arr) {
                if (ds.n_cat_features() != 0 || arr.shape(1) != ds.n_num_features()) {
                    throw std::invalid_argument("Feature count or kind mismatch in transform");
                }
                const std::size_t n_rows = arr.shape(0);
                auto views = views_from_2d(arr);
                nb::gil_scoped_release release;
                return ds.transform_mixed(n_rows, views, {});
            },
            nb::arg("arr").noconvert())
        .def("transform_mixed",
             [](const miter::Dataset& ds, std::size_t n_rows, const nb::list& num_cols,
                std::vector<std::vector<std::string>> cat_vocabs,
                const std::vector<nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig,
                                              nb::device::cpu>>& cat_indices) {
                 auto num_views = extract_num_col_views(num_cols, n_rows);
                 auto cat_views = extract_cat_col_views(std::move(cat_vocabs), cat_indices, n_rows);
                 nb::gil_scoped_release release;
                 return ds.transform_mixed(n_rows, num_views, cat_views);
             });

    m.def("fit_numeric_f64", &fit_numeric_f64, nb::arg("arr").noconvert(), nb::arg("max_bin") = 255,
          nb::arg("feature_names") = std::vector<std::string>{});
    m.def("fit_numeric_f32", &fit_numeric_f32, nb::arg("arr").noconvert(), nb::arg("max_bin") = 255,
          nb::arg("feature_names") = std::vector<std::string>{});
    m.def("fit_mixed", &fit_mixed_py);
}
