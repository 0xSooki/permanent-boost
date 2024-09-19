#ifndef NUMPY_TO_MATRIX_H
#define NUMPY_TO_MATRIX_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "matrix.hpp"

namespace py = pybind11;

/**
 * Create a numpy scalar from a C++ native data type.
 *
 * @note Not sure if this is the right way to create a numpy scalar. This just
 * returns a 0-dimensional array instead of a scalar, which is almost the same,
 * but not quite.
 *
 * Source: https://stackoverflow.com/a/44682603
 */
template <typename T> py::object create_numpy_scalar(T input) {
  T *ptr = new T;
  (*ptr) = input;

  py::capsule free_when_done(ptr, [](void *f) {
    T *ptr = reinterpret_cast<T *>(f);
    delete ptr;
  });

  return py::array_t<T>({}, {}, ptr, free_when_done);
}

/**
 * Creates a Matrix from a numpy array with shared memory.
 */
template <typename T>
Matrix<T> numpy_to_matrix(
    py::array_t<T, py::array::c_style | py::array::forcecast> numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();

  size_t rows = bufferinfo.shape[0];
  size_t cols = bufferinfo.shape[1];

  T *data = static_cast<T *>(bufferinfo.ptr);

  Matrix<T> matrix(rows, cols, data);

  return matrix;
}

/**
 * Creates a Vector from a numpy array with shared memory.
 */
template <typename T> std::vector<T> numpy_to_vec(py::array_t<T> numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();

  size_t rows = bufferinfo.shape[0];

  T *data = static_cast<T *>(bufferinfo.ptr);

  std::vector<T> vec(data, data + rows);
  return vec;
}

#endif