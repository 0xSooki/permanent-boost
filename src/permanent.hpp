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

/**
 * @brief Computes the permanent of a given matrix.
 *
 * This function calculates the permanent of the input matrix. The permanent is
 * a function similar to the determinant but with all signs positive in the
 * expansion by minors.
 *
 * @tparam T The type of the elements in the matrix.
 * @param mtx The input matrix for which the permanent is to be computed.
 * @param row_mult A vector representing the multiplicity of each row.
 * @param col_mult A vector representing the multiplicity of each column.
 * @return The permanent of the input matrix.
 */
template <typename scalar_type, typename precision_type>
std::complex<double> permanent(Matrix<std::complex<double>> &mtx,
                               std::vector<int> &col_mult,
                               std::vector<int> &row_mult) {

  int sum_row_mult = sum(row_mult);
  int sum_col_mult = sum(col_mult);
  ;
  if (sum_row_mult != sum_col_mult) {
    std::string error("BBFGPermanentCalculatorRepeated_Tasks::calculate:  "
                      "Number of input and output states should be equal");
    throw error;
  }

  if (mtx.rows == 0 || mtx.cols == 0 || sum_row_mult == 0 || sum_col_mult == 0)
    // the permanent of an empty matrix is 1 by definition
    return std::complex<double>(1.0, 0.0);

  if (mtx.rows == 1) {

    scalar_type ret(1.0, 0.0);
    for (size_t idx = 0; idx < col_mult.size(); idx++) {
      for (size_t jdx = 0; jdx < col_mult[idx]; jdx++) {
        ret *= mtx[idx];
      }
    }

    return std::complex<double>(ret.real(), ret.imag());
  }

  Matrix<std::complex<double>> mtx2 =
      Matrix<std::complex<double>>(mtx.rows, mtx.cols);
  for (size_t idx = 0; idx < mtx.size(); idx++) {
    mtx2[idx] = mtx[idx] * 2.0;
  }

  // row_mult.print_matrix();
  std::vector<int> n_ary_limits(row_mult.size() - 1);
  for (size_t idx = 0; idx < n_ary_limits.size(); idx++) {
    n_ary_limits[idx] = row_mult[idx + 1] + 1;
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

  // tbb::parallel_for((int64_t)0, concurrency, (int64_t)1, [&](int64_t job_idx)
  // {
  for (int64_t job_idx = 0; job_idx < concurrency; job_idx++) {
    std::complex<double> partial_permanent(0.0, 0.0);

    // initial offset and upper boundary of the gray code counter
    int64_t work_batch = Idx_max / concurrency;
    int64_t initial_offset = job_idx * work_batch;
    int64_t offset_max = (job_idx + 1) * work_batch - 1;
    if (job_idx == concurrency - 1) {
      offset_max = Idx_max - 1;
    }

    n_aryGrayCodeCounter gcode_counter(n_ary_limits, initial_offset);

    gcode_counter.set_offset_max(offset_max);
    std::vector<int> gcode = gcode_counter.get();

    // print gcode
    // std::cout << "gcode: ";
    // for (size_t idx = 0; idx < gcode.size(); idx++) {
    //   std::cout << gcode[idx] << " ";
    // }
    // std::cout << std::endl;

    // calculate the initial column sum and binomial coefficient
    __int128 binomial_coeff = 1;

    Matrix<scalar_type> colsum(1, col_mult.size());
    std::uninitialized_copy_n(mtx.data, colsum.size(), colsum.data);
    auto mtx_data = mtx.data + mtx.stride;

    // variable to count all the -1 elements in the delta vector
    int minus_signs_all = 0;

    int row_idx = 1;

    for (size_t idx = 0; idx < gcode.size(); idx++) {

      // the value of the element of the gray code stand for the number of
      // \delta_i=-1 elements in the subset of multiplicated rows
      const int minus_signs = gcode[idx];
      int row_mult_current = row_mult[idx + 1];

      for (size_t col_idx = 0; col_idx < col_mult.size(); col_idx++) {
        colsum[col_idx] += (scalar_type)mtx_data[col_idx] *
                           (precision_type)(row_mult_current - 2 * minus_signs);
      }

      minus_signs_all += minus_signs;

      // update the binomial coefficient
      binomial_coeff *= binomialCoeffInt128(row_mult_current, minus_signs);

      mtx_data += mtx.stride;
    }

    // variable to refer to the parity of the delta vector (+1 if the
    // number of -1 elements in delta vector is even, -1 otherwise)
    char parity = (minus_signs_all % 2 == 0) ? 1 : -1;

    scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);
    for (size_t idx = 0; idx < col_mult.size(); idx++) {
      for (size_t jdx = 0; jdx < col_mult[idx]; jdx++) {
        colsum_prod *= colsum[idx];
      }
    }

    // add the initial addend to the permanent
    // scalar_type &addend_loc = permanent;
    partial_permanent += colsum_prod * (precision_type)binomial_coeff;

    // iterate over gray codes to calculate permanent addends
    for (int64_t idx = initial_offset + 1; idx < offset_max + 1; idx++) {

      int changed_index, value_prev, value;
      if (gcode_counter.next(changed_index, value_prev, value)) {
        break;
      }

      parity = -parity;

      // update column sum and calculate the product of the elements
      int row_offset = (changed_index + 1) * mtx.stride;
      auto mtx_data = mtx2.data + row_offset;
      scalar_type colsum_prod((precision_type)parity, (precision_type)0.0);
      for (size_t col_idx = 0; col_idx < col_mult.size(); col_idx++) {
        if (value_prev < value) {
          colsum[col_idx] -= mtx_data[col_idx];
        } else {
          colsum[col_idx] += mtx_data[col_idx];
        }

        for (size_t jdx = 0; jdx < col_mult[col_idx]; jdx++) {
          colsum_prod *= colsum[col_idx];
        }
      }

      // update binomial factor
      int row_mult_current = row_mult[changed_index + 1];
      binomial_coeff =
          value < value_prev
              ? binomial_coeff * value_prev / (row_mult_current - value)
              : binomial_coeff * (row_mult_current - value_prev) / value;
      // binomial_coeff /= binomialCoeffInt64(row_mult_current,
      // value_prev); binomial_coeff *=
      // binomialCoeffInt64(row_mult_current, value);

      partial_permanent += colsum_prod * (precision_type)binomial_coeff;
      std::cout << "permanent: " << partial_permanent << std::endl;
    }

    for (size_t n = colsum.size(); n > 0; --n)
      colsum[n - 1].~scalar_type();

    permanent += partial_permanent;
  }
  //});

  // priv_addend.combine_each([&](scalar_type &a) { permanent += a; });

  permanent /= (precision_type)ldexp(1.0, sum_row_mult - 1);

  return permanent;
}

#endif