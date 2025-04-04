#include "kernels.h"
#include <cuda/std/complex>
#include <cuComplex.h>
#include "matrix.hpp"
#include "utils.hpp"
#include "n_aryGrayCodeCounter.hpp"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

__global__ void FooFwdKernel(const cuda::std::complex<double> *a, const cuda::std::complex<double> *b, cuda::std::complex<double> *c,
                             cuda::std::complex<double> *b_plus_1, // intermediate output b+1
                             size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
  {
    b_plus_1[i] = b[i] + 1.0;
    c[i] = a[i] * b_plus_1[i];
  }
}

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::C128> a,
                      ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> c,
                      ffi::ResultBuffer<ffi::C128> b_plus_1, size_t n)
{
  const int block_dim = 128;
  const int grid_dim = 1;
  // Note how we access regular Buffer data vs Result Buffer data:
  FooFwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      reinterpret_cast<const cuda::std::complex<double> *>(a.typed_data()),
      reinterpret_cast<const cuda::std::complex<double> *>(b.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(c->typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(b_plus_1->typed_data()),
      n);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void FooBwdKernel(const float *c_grad,   // incoming gradient wrt c
                             const float *a,        // original input a
                             const float *b_plus_1, // intermediate output b+1
                             float *a_grad,         // outgoing gradient wrt a
                             float *b_grad,         // outgoing gradient wrt b
                             size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
  {
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      size_t n)
{
  const int block_dim = 128;
  const int grid_dim = 1;
  FooBwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      c_grad.typed_data(), a.typed_data(), b_plus_1->typed_data(),
      a_grad->typed_data(), b_grad->typed_data(), n);
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__device__ void atAddComplex(cuDoubleComplex *a, cuDoubleComplex b)
{
  // transform the addresses of real and imag. parts to double pointers
  double *x = (double *)a;
  double *y = x + 1;
  // use atomicAdd for double variables
  atomicAdd(x, cuCreal(b));
  atomicAdd(y, cuCimag(b));
}

template <ffi::DataType T>
std::pair<int64_t, int64_t> get_dims(const ffi::Buffer<T> &buffer)
{
  auto dims = buffer.dimensions();

  if (dims.size() == 0)
  {
    return std::make_pair(0, 0);
  }
  return std::make_pair(buffer.element_count(), dims.back());
}

__global__ void PermanentKernelMatrix(Matrix<cuDoubleComplex> A, uint64_t *rows, size_t rows_size,
                                      uint64_t *cols, size_t cols_size, cuDoubleComplex *result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid == 1)
  {
    size_t min_idx = 0;
    uint64_t minelem = 0;

    for (size_t i = 0; i < rows_size; i++)
    {
      int current = rows[i];
      if (minelem == 0 || (current < minelem && current != 0))
      {
        minelem = current;
        min_idx = i;
      }
    }

    if (rows_size > 0 && minelem != 0)
    {
      size_t new_size = rows_size + 1;
      uint64_t *rows_new = new uint64_t[new_size];

      rows_new[0] = 1;

      for (size_t i = 0; i < rows_size; i++)
      {
        rows_new[i + 1] = rows[i];
      }
      rows_new[1 + min_idx] -= 1;

      Matrix<cuDoubleComplex> mtx_(A.rows + 1, A.cols);

      for (size_t j = 0; j < A.cols; j++)
      {
        mtx_(0, j) = A(min_idx, j);
      }

      for (size_t i = 0; i < A.rows; i++)
      {
        for (size_t j = 0; j < A.cols; j++)
        {
          mtx_(i + 1, j) = A(i, j);
        }
      }

      delete[] rows;
      rows = rows_new;
      rows_size = new_size;

      A = mtx_;
    }

    int sum_rows = 0;
    for (size_t i = 0; i < rows_size; i++)
    {
      sum_rows += rows[i];
    }

    int sum_cols = 0;
    for (size_t i = 0; i < cols_size; i++)
    {
      sum_cols += cols[i];
    }

    if (sum_rows != sum_cols)
    {
      *result = make_cuDoubleComplex(0.0, 0.0);
      return;
    }

    if (A.rows == 0 || A.cols == 0 || sum_rows == 0 || sum_cols == 0)
    {
      *result = make_cuDoubleComplex(1.0, 0.0);
      return;
    }

    if (A.rows == 1)
    {
      cuDoubleComplex ret = make_cuDoubleComplex(1.0, 0.0);
      for (size_t idx = 0; idx < cols_size; idx++)
      {
        for (size_t jdx = 0; jdx < cols[idx]; jdx++)
        {
          ret = cuCmul(ret, A[idx]);
        }
      }
      *result = ret;
      return;
    }

    Matrix<cuDoubleComplex> mtx2(A.rows, A.cols);
    for (size_t idx = 0; idx < A.size(); idx++)
    {
      mtx2[idx] = cuCmul(A[idx], make_cuDoubleComplex(2.0, 0.0));
    }

    size_t n_ary_size = rows_size - 1;
    int *n_ary_limits = new int[n_ary_size];
    for (size_t idx = 0; idx < n_ary_size; idx++)
    {
      n_ary_limits[idx] = rows[idx + 1] + 1;
    }

    uint64_t idx_max = n_ary_limits[0];
    for (size_t idx = 1; idx < n_ary_size; idx++)
    {
      idx_max *= n_ary_limits[idx];
    }

    n_aryGrayCodeCounter gcode_counter(n_ary_limits, n_ary_size, 0);
    int *gcode = gcode_counter.get();

    int binomial_coeff = 1;

    Matrix<cuDoubleComplex> colsum(1, cols_size);
    for (size_t i = 0; i < cols_size; i++)
    {
      colsum[i] = A[i];
    }

    auto mtx_data = A.data + A.stride;

    int minus_signs_all = 0;

    for (size_t idx = 0; idx < n_ary_size; idx++)
    {
      const int &minus_signs = gcode[idx];
      int row_mult_current = rows[idx + 1];

      for (size_t col_idx = 0; col_idx < cols_size; col_idx++)
      {
        cuDoubleComplex factor = make_cuDoubleComplex(
            (double)(row_mult_current - 2 * minus_signs), 0.0);
        cuDoubleComplex product = cuCmul(mtx_data[col_idx], factor);
        colsum[col_idx] = cuCadd(colsum[col_idx], product);
      }

      minus_signs_all += minus_signs;

      binomial_coeff *= binomialCoeffManual<int>(row_mult_current, minus_signs);

      mtx_data += A.stride;
    }

    char parity = (minus_signs_all % 2 == 0) ? 1 : -1;

    cuDoubleComplex colsum_prod = make_cuDoubleComplex((double)parity, 0.0);
    for (size_t idx = 0; idx < cols_size; idx++)
    {
      for (size_t jdx = 0; jdx < cols[idx]; jdx++)
      {
        colsum_prod = cuCmul(colsum_prod, colsum[idx]);
      }
    }

    cuDoubleComplex permanent_accum = cuCmul(colsum_prod,
                                             make_cuDoubleComplex((double)binomial_coeff, 0.0));

    for (uint64_t idx = 1; idx < idx_max; idx++)
    {
      int changed_index, value_prev, value;
      if (gcode_counter.next(changed_index, value_prev, value))
      {
        break;
      }

      parity = -parity;

      int row_offset = (changed_index + 1) * A.stride;
      auto mtx_data = mtx2.data + row_offset;
      cuDoubleComplex colsum_prod = make_cuDoubleComplex((double)parity, 0.0);

      for (size_t col_idx = 0; col_idx < cols_size; col_idx++)
      {
        if (value_prev < value)
        {
          colsum[col_idx] = cuCsub(colsum[col_idx], mtx_data[col_idx]);
        }
        else
        {
          colsum[col_idx] = cuCadd(colsum[col_idx], mtx_data[col_idx]);
        }

        for (size_t jdx = 0; jdx < cols[col_idx]; jdx++)
        {
          colsum_prod = cuCmul(colsum_prod, colsum[col_idx]);
        }
      }

      int row_mult_current = rows[changed_index + 1];
      if (value < value_prev)
      {
        binomial_coeff = binomial_coeff * value_prev / (row_mult_current - value);
      }
      else
      {
        binomial_coeff = binomial_coeff * (row_mult_current - value_prev) / value;
      }

      cuDoubleComplex term = cuCmul(colsum_prod,
                                    make_cuDoubleComplex((double)binomial_coeff, 0.0));
      permanent_accum = cuCadd(permanent_accum, term);
    }

    double scale_factor = 1.0 / (1ULL << (sum_rows - 1));
    permanent_accum = cuCmul(permanent_accum, make_cuDoubleComplex(scale_factor, 0.0));

    delete[] n_ary_limits;

    *result = permanent_accum;
  }
}

ffi::Error PermanentHostMatrixFromBuffer(cudaStream_t stream, ffi::Buffer<ffi::C128> A,
                                         ffi::Buffer<ffi::U64> rows,
                                         ffi::Buffer<ffi::U64> cols,
                                         ffi::ResultBuffer<ffi::C128> permanent)
{
  // const int block_dim = 128;
  auto [total_size, n] = get_dims(A);
  Matrix<cuDoubleComplex> m(n, n, reinterpret_cast<cuDoubleComplex *>(A.typed_data()));
  // const int grid_dim = (n * n + block_dim - 1) / block_dim;

  const int block_dim = 128;
  const int grid_dim = 64;

  cudaMemset(permanent->typed_data(), 0, sizeof(cuDoubleComplex));

  // Reset result to zero.
  // cudaMemset(permanent->typed_data(), 0, sizeof(cuDoubleComplex));

  PermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(m, rows.typed_data(),
                                                            rows.element_count(),
                                                            cols.typed_data(),
                                                            cols.element_count(),
                                                            reinterpret_cast<cuDoubleComplex *>(permanent->typed_data()));

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}