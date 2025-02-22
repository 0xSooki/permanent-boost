#ifndef PERMANENT_HPP
#define PERMANENT_HPP

#include "matrix.hpp"
#include "utils.hpp"
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "n_aryGrayCodeCounter.hpp"
#include <thread>

#ifdef __SIZEOF_INT128__ // Check if __int128 is supported
using int_type = __int128;
#else
using int_type = int64_t; // Fallback to int64_t
#endif
/**
 * @brief Computes the permanent of a given matrix.
 *
 * This function calculates the permanent of the input matrix. The permanent is
 * a function similar to the determinant but with all signs positive in the
 * expansion by minors.
 *
 * @tparam T The type of the elements in the matrix.
 * @param A The input matrix for which the permanent is to be computed.
 * @param rows A vector representing the multiplicity of each row.
 * @param cols A vector representing the multiplicity of each column.
 * @return The permanent of the input matrix.
 */
template <typename scalar_type, typename precision_type>
std::complex<double> permanent(Matrix<std::complex<double>> &A,
                               std::vector<int> &rows, std::vector<int> &cols) {

  size_t min_idx = 0;
  int minelem = 0;

  for (int i = 0; i < rows.size(); i++) {
    if (minelem == 0 || rows[i] < minelem && rows[i] != 0) {
      minelem = rows[i];
      min_idx = i;
    }
  }

  if (rows.size() > 0 && minelem != 0) {
    std::vector<int> rows_(rows.size() + 1);

    rows_[0] = 1;

    for (int i = 0; i < rows.size(); i++) {
      rows_[i + 1] = rows[i];
    }
    rows_[1 + min_idx] -= 1;

    Matrix<scalar_type> mtx_(A.rows + 1, A.cols);

    for (int j = 0; j < A.cols; j++) {
      mtx_(0, j) = A(min_idx, j);
    }

    for (int i = 0; i < A.rows; i++) {
      for (int j = 0; j < A.cols; j++) {
        mtx_(i + 1, j) = A(i, j);
      }
    }

    rows = rows_;
    A = mtx_;
  }

  int sum_rows = sum(rows);
  int sum_cols = sum(cols);

  if (sum_rows != sum_cols) {
    std::string error("BBFGPermanentCalculatorRepeated_Tasks::calculate:  "
                      "Number of input and output states should be equal");
    throw error;
  }

  if (A.rows == 0 || A.cols == 0 || sum_rows == 0 || sum_cols == 0)
    // the permanent of an empty matrix is 1 by definition
    return std::complex<double>(1.0, 0.0);

  if (A.rows == 1) {
    scalar_type ret(1.0, 0.0);
    for (size_t idx = 0; idx < cols.size(); idx++) {
      for (size_t jdx = 0; jdx < cols[idx]; jdx++) {
        ret *= A[idx];
      }
    }

    return std::complex<double>(ret.real(), ret.imag());
  }

  Matrix<std::complex<double>> mtx2 =
      Matrix<std::complex<double>>(A.rows, A.cols);
  for (size_t idx = 0; idx < A.size(); idx++) {
    mtx2[idx] = A[idx] * 2.0;
  }

  std::vector<int> n_ary_limits(rows.size() - 1);
  for (size_t idx = 0; idx < n_ary_limits.size(); idx++) {
    n_ary_limits[idx] = rows[idx + 1] + 1;
  }

  uint64_t Idx_max = n_ary_limits[0];
  for (size_t idx = 1; idx < n_ary_limits.size(); idx++) {
    Idx_max *= n_ary_limits[idx];
  }

  std::complex<double> permanent(0.0, 0.0);

  // determine the concurrency of the calculation
  unsigned int nthreads = std::thread::hardware_concurrency();
  int64_t concurrency = (int64_t)nthreads * 4;
  concurrency = concurrency < Idx_max ? concurrency : (int64_t)Idx_max;

  // tbb::parallel_for((int64_t)0, concurrency, (int64_t)1, [&](int64_t
  // job_idx)
  // {

  // for (int64_t job_idx = 0; job_idx < concurrency; job_idx++) {
  std::complex<double> partial_permanent(0.0, 0.0);

  //   // initial offset and upper boundary of the gray code counter
  //   int64_t work_batch = Idx_max / concurrency;
  //   int64_t initial_offset = job_idx * work_batch;
  //   int64_t offset_max = (job_idx + 1) * work_batch - 1;
  //   if (job_idx == concurrency - 1) {
  //     offset_max = Idx_max - 1;
  //   }

  // n_aryGrayCodeCounter gcode_counter(n_ary_limits, initial_offset);

  // gcode_counter.set_offset_max(offset_max);

  n_aryGrayCodeCounter gcode_counter(n_ary_limits);

  std::vector<int> gcode = gcode_counter.get();

  // calculate the initial column sum and binomial coefficient
  int_type binomial_coeff = 1;

  Matrix<scalar_type> colsum(1, cols.size());
  std::uninitialized_copy_n(A.data, colsum.size(), colsum.data);
  auto mtx_data = A.data + A.stride;

  // variable to count all the -1 elements in the delta vector
  int minus_signs_all = 0;

  int row_idx = 1;

  for (size_t idx = 0; idx < gcode.size(); idx++) {

    // the value of the element of the gray code stand for the number of
    // \delta_i=-1 elements in the subset of multiplicated rows
    const int minus_signs = gcode[idx];
    int rows_current = rows[idx + 1];

    for (size_t col_idx = 0; col_idx < cols.size(); col_idx++) {
      colsum[col_idx] += (scalar_type)A(row_idx, col_idx) *
                         (precision_type)(rows_current - 2 * minus_signs);
    }

    minus_signs_all += minus_signs;

    // update the binomial coefficient
    binomial_coeff *= binomialCoeffInt128(rows_current, minus_signs);

    // mtx_data += A.stride;
    row_idx += 1;
  }

  // variable to refer to the parity of the delta vector (+1 if the
  // number of -1 elements in delta vector is even, -1 otherwise)
  char parity = (minus_signs_all % 2 == 0) ? 1 : -1;

  scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);
  for (size_t idx = 0; idx < cols.size(); idx++) {
    for (size_t jdx = 0; jdx < cols[idx]; jdx++) {
      colsum_prod *= colsum[idx];
    }
  }

  // add the initial addend to the permanent
  // scalar_type &addend_loc = permanent;
  partial_permanent += colsum_prod * (precision_type)binomial_coeff;

  // iterate over gray codes to calculate permanent addends
  for (int64_t idx = gcode_counter.get_offset() + 1;
       idx < gcode_counter.get_offset_max() + 1; idx++) {

    int changed_index, value_prev, value;
    if (gcode_counter.next(changed_index, value_prev, value)) {
      break;
    }

    parity = -parity;

    // update column sum and calculate the product of the elements
    int row_offset = (changed_index + 1); //* A.stride;
    // auto mtx_data = mtx2.data + row_offset;
    scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);

    for (size_t col_idx = 0; col_idx < cols.size(); col_idx++) {
      if (value_prev < value) {
        colsum[col_idx] -= mtx2(row_offset, col_idx);
      } else {
        colsum[col_idx] += mtx2(row_offset, col_idx);
      }

      for (size_t jdx = 0; jdx < cols[col_idx]; jdx++) {
        colsum_prod *= colsum[col_idx];
      }
    }

    // update binomial factor
    int rows_current = rows[changed_index + 1];
    binomial_coeff = value < value_prev
                         ? binomial_coeff * value_prev / (rows_current - value)
                         : binomial_coeff * (rows_current - value_prev) / value;
    // binomial_coeff /= binomialCoeffInt64(rows_current,
    // value_prev); binomial_coeff *=
    // binomialCoeffInt64(rows_current, value);

    partial_permanent += colsum_prod * (precision_type)binomial_coeff;
  }

  permanent += partial_permanent;

  for (size_t n = colsum.size(); n > 0; --n)
    colsum[n - 1].~scalar_type();

  //});

  // priv_addend.combine_each([&](scalar_type &a) { permanent += a; });

  permanent /= std::pow(2, sum_rows - 1);
  return permanent;
}

Matrix<std::complex<double>> grad_perm(Matrix<std::complex<double>> &A,
                                       std::vector<int> &rows,
                                       std::vector<int> &cols) {
  int n = rows.size();

  Matrix<std::complex<double>> perm_grad(n, n);

  for (int i = 0; i < n; ++i) {
    if (rows[i] == 0)
      continue;

    for (int j = 0; j < n; ++j) {
      if (cols[j] == 0)
        continue;

      std::vector<int> grad_rows(rows);
      grad_rows[i] -= 1;

      Matrix<std::complex<double>> A_(A);

      std::vector<int> grad_cols(cols);
      grad_cols[j] -= 1;

      perm_grad(i, j) =
          (double)rows[i] * (double)cols[j] *
          permanent<std::complex<double>, double>(A_, grad_rows, grad_cols);
    }
  }

  return perm_grad;
}

#endif