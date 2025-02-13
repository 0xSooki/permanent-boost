#include "matrix.hpp"
#include "numpy_utils.hpp"
#include "permanent.hpp"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#include <complex>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <type_traits>
#include <utility>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
namespace ffi = xla::ffi;

template <typename T>
T permanent_np(py::array_t<T, py::array::c_style | py::array::forcecast> array,
               py::array_t<int> rows, py::array_t<int> cols) {
  Matrix<T> matrix = numpy_to_matrix(array);

  std::vector<int> row_mult = numpy_to_vec(rows);
  std::vector<int> col_mult = numpy_to_vec(cols);

  T result =
      permanent<std::complex<double>, double>(matrix, row_mult, col_mult);

  return result;
}

template <ffi::DataType T>
std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<T> &buffer) {
  auto dims = buffer.dimensions();

  if (dims.size() == 0) {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error PermanentImpl(ffi::Buffer<ffi::C128> x, ffi::Buffer<ffi::U64> rows,
                         ffi::Buffer<ffi::U64> cols,
                         ffi::ResultBuffer<ffi::C128> y) {
  auto [totalSize, n] = get_dims(x);

  if (n == 0) {
    return ffi::Error::InvalidArgument("RmsNorm input must be an array");
  }
  std::vector<int> row_mult(&(rows.typed_data()[0]),
                            &(rows.typed_data()[0]) + n);
  std::vector<int> col_mult(&(cols.typed_data()[0]),
                            &(cols.typed_data()[0]) + n);
  Matrix<std::complex<double>> matrix(n, n, &(x.typed_data()[0]));

  y->typed_data()[0] =
      permanent<std::complex<double>, double>(matrix, row_mult, col_mult);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(Permanent, PermanentImpl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Arg<ffi::Buffer<ffi::U64>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

template <typename T> py::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        Permanent calculator plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           permanent
    )pbdoc";
  m.def("permanent", &permanent_np<std::complex<double>>, R"pbdoc(
        Compute the permanent of a matrix with specified row and column multiplicities.
    )pbdoc");
  m.def("registrations", []() {
    py::dict registrations;
    registrations["perm"] = EncapsulateFfiHandler(Permanent);
    return registrations;
  });

  m.attr("__version__") = "dev";
}
