#include "matrix.hpp"
#include "numpy_utils.hpp"
#include "permanent.hpp"
#include <complex>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

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
  m.attr("__version__") = "dev";
}
