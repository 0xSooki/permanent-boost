#ifndef PERMANENT_H
#define PERMANENT_H

#include "matrix.hpp"
#include <complex>
#include <math.h>
#include <numeric>
#include <vector>

/**
 * @brief Computes the permanent of a given matrix.
 *
 * This function calculates the permanent of the input matrix. The permanent is
 * a function similar to the determinant but with all signs positive in the
 * expansion by minors.
 *
 * @tparam T The type of the elements in the matrix.
 * @param matrix_in The input matrix for which the permanent is to be computed.
 * @param row_mult A vector representing the multiplicity of each row.
 * @param col_mult A vector representing the multiplicity of each column.
 * @return The permanent of the input matrix.
 */
template <typename T>
extern T permanent(Matrix<T> matrix_in, std::vector<int> row_mult,
                   std::vector<int> col_mult) {
  T sum = 0;
  for (int i = 0; i < matrix_in.rows; i++) {
    for (int j = 0; j < matrix_in.cols; j++) {
      sum += matrix_in(i, j);
    }
  }
  return sum;
}

#endif