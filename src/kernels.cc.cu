#include "kernels.h"
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

__global__ void FooFwdKernel(const cuDoubleComplex *a, const cuDoubleComplex *b, cuDoubleComplex *c,
                             cuDoubleComplex *b_plus_1, // intermediate output b+1
                             size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t j = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
  {
    b_plus_1[i] = cuCadd(b[i], make_cuDoubleComplex(1.0, 0.0));
    c[i] = cuCmul(a[i], b_plus_1[i]);
  }
}

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::C128> a,
                      ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> c,
                      ffi::ResultBuffer<ffi::C128> b_plus_1, size_t n)
{
  const int block_dim = 128;
  const int grid_dim = 2;
  // Note how we access regular Buffer data vs Result Buffer data:
  FooFwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      reinterpret_cast<const cuDoubleComplex *>(a.typed_data()),
      reinterpret_cast<const cuDoubleComplex *>(b.typed_data()),
      reinterpret_cast<cuDoubleComplex *>(c->typed_data()),
      reinterpret_cast<cuDoubleComplex *>(b_plus_1->typed_data()),
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

__global__ void OPermanentKernelMatrix(Matrix<cuDoubleComplex> A, uint64_t *rows, size_t rows_size,
                                       uint64_t *cols, size_t cols_size,
                                       int *n_ary_limits, size_t n_ary_size, uint64_t idx_max,
                                       int64_t host_max_concurrent_warps, int sum_rows, cuDoubleComplex *result)
{
  extern __shared__ cuDoubleComplex addends[];

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  cuDoubleComplex local_result = make_cuDoubleComplex(0.0, 0.0);

  unsigned int nthreads = 32;
  int64_t concurrency = min(host_max_concurrent_warps, static_cast<int64_t>(idx_max));

  for (uint64_t job_idx = tid; job_idx < concurrency; job_idx += stride)
  {
    int64_t work_batch = idx_max / concurrency;
    int64_t initial_offset = job_idx * work_batch;
    int64_t offset_max = (job_idx + 1) * work_batch - 1;
    if (job_idx == concurrency - 1)
    {
      offset_max = idx_max - 1;
    }

    n_aryGrayCodeCounter gcode_counter(n_ary_limits, n_ary_size, initial_offset);
    gcode_counter.set_offset_max(offset_max);

    int *gcode = gcode_counter.get();
    int binomial_coeff = 1;
    int minus_signs_all = 0;

    cuDoubleComplex colsum[64];
    for (size_t i = 0; i < cols_size && i < 64; i++)
    {
      colsum[i] = A(0, i);
    }

    for (size_t row = 0; row < n_ary_size; row++)
    {
      int minus_signs = gcode[row];
      int row_mult = rows[row + 1];
      for (size_t col = 0; col < cols_size && col < 64; col++)
      {
        double factor = row_mult - 2.0 * minus_signs;
        cuDoubleComplex scaled = cuCmul(A(row + 1, col),
                                        make_cuDoubleComplex(factor, 0.0));
        colsum[col] = cuCadd(colsum[col], scaled);
      }

      minus_signs_all += minus_signs;

      binomial_coeff *= binomialCoeffManual<int>(row_mult, minus_signs);
    }

    int parity = (minus_signs_all % 2 == 0) ? 1 : -1;

    cuDoubleComplex colsum_prod = make_cuDoubleComplex((double)parity, 0.0);

    for (size_t i = 0; i < cols_size && i < 64; i++)
    {
      for (size_t j = 0; j < cols[i]; j++)
      {
        colsum_prod = cuCmul(colsum_prod, colsum[i]);
      }
    }

    colsum_prod = cuCmul(colsum_prod, make_cuDoubleComplex((double)binomial_coeff, 0.0));

    local_result = cuCadd(local_result, colsum_prod);

    for (int64_t idx = initial_offset + 1; idx < offset_max + 1; idx++)
    {
      int changed_index, value_prev, value;
      if (gcode_counter.next(changed_index, value_prev, value))
      {
        break;
      }

      parity = -parity;

      int row_offset = changed_index + 1;
      cuDoubleComplex colsum_prod = make_cuDoubleComplex(parity, 0.0);
      for (size_t col_idx = 0; col_idx < cols_size; col_idx++)
      {
        if (value_prev < value)
        {
          colsum[col_idx] = cuCsub(colsum[col_idx], cuCmul(make_cuDoubleComplex(2.0, 0.0), A(row_offset, col_idx)));
        }
        else
        {
          colsum[col_idx] = cuCadd(colsum[col_idx], cuCmul(make_cuDoubleComplex(2.0, 0.0), A(row_offset, col_idx)));
        }

        for (size_t jdx = 0; jdx < cols[col_idx]; jdx++)
        {
          colsum_prod = cuCmul(colsum_prod, colsum[col_idx]);
        }
      }

      int row_mult_current = rows[changed_index + 1];
      binomial_coeff =
          value < value_prev
              ? binomial_coeff * value_prev / (row_mult_current - value)
              : binomial_coeff * (row_mult_current - value_prev) / value;

      colsum_prod = cuCmul(colsum_prod, make_cuDoubleComplex((double)binomial_coeff, 0.0));
      local_result = cuCadd(local_result, colsum_prod);
    }
  }
  double scale_factor = 1.0 / (1ULL << (sum_rows - 1));
  local_result = cuCmul(local_result, make_cuDoubleComplex(scale_factor, 0.0));

  atAddComplex(result, local_result);
}

ffi::Error PermanentHostMatrixFromBuffer(cudaStream_t stream, ffi::Buffer<ffi::C128> A,
                                         ffi::Buffer<ffi::U64> rows,
                                         ffi::Buffer<ffi::U64> cols,
                                         ffi::ResultBuffer<ffi::C128> permanent)
{

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  auto [total_size, n] = get_dims(A);

  uint64_t *h_rows = new uint64_t[rows.element_count()];
  size_t rows_size = rows.element_count();
  int sum_rows = 0;
  cudaMemcpy(h_rows, rows.typed_data(), rows_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < rows_size; i++)
  {
    sum_rows += h_rows[i];
  }

  size_t min_idx = 0;
  uint64_t minelem = 0;

  for (size_t i = 0; i < rows_size; i++)
  {
    int current = h_rows[i];
    if (minelem == 0 || (current < minelem && current != 0))
    {
      minelem = current;
      min_idx = i;
    }
  }

  Matrix<cuDoubleComplex> m(n, n, reinterpret_cast<cuDoubleComplex *>(A.typed_data()));

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  unsigned int warps_per_sm = 32;
  int64_t host_max_concurrent_warps = (int64_t)props.multiProcessorCount * warps_per_sm;

  if (rows_size > 0 && minelem != 0)
  {
    size_t new_size = rows_size + 1;
    uint64_t *new_rows = new uint64_t[new_size];

    new_rows[0] = 1;
    for (size_t i = 0; i < rows_size; i++)
    {
      new_rows[i + 1] = h_rows[i];
    }
    new_rows[1 + min_idx] -= 1;

    cuDoubleComplex *h_matrix = new cuDoubleComplex[n * n];
    cudaMemcpy(h_matrix, A.typed_data(), n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    cuDoubleComplex *h_new_matrix = new cuDoubleComplex[(n + 1) * n];

    for (size_t j = 0; j < n; j++)
    {
      h_new_matrix[j] = h_matrix[min_idx * n + j];
    }

    for (size_t i = 0; i < n; i++)
    {
      for (size_t j = 0; j < n; j++)
      {
        h_new_matrix[(i + 1) * n + j] = h_matrix[i * n + j];
      }
    }

    cuDoubleComplex *d_new_matrix;
    uint64_t *d_new_rows;
    cudaMalloc(&d_new_matrix, (n + 1) * n * sizeof(cuDoubleComplex));
    cudaMalloc(&d_new_rows, new_size * sizeof(uint64_t));

    cudaMemcpy(d_new_matrix, h_new_matrix, (n + 1) * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new_rows, new_rows, new_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    Matrix<cuDoubleComplex> modified_m(n + 1, n, d_new_matrix);

    cudaMemset(permanent->typed_data(), 0, sizeof(cuDoubleComplex));

    size_t n_ary_size = new_size - 1;
    int *h_n_ary_limits = new int[n_ary_size];
    for (size_t idx = 0; idx < n_ary_size; idx++)
    {
      h_n_ary_limits[idx] = new_rows[idx + 1] + 1;
    }

    uint64_t idx_max = n_ary_size > 0 ? h_n_ary_limits[0] : 0;
    for (size_t idx = 1; idx < n_ary_size; idx++)
    {
      idx_max *= h_n_ary_limits[idx];
    }

    int *d_n_ary_limits;
    cudaMalloc(&d_n_ary_limits, n_ary_size * sizeof(int));
    cudaMemcpy(d_n_ary_limits, h_n_ary_limits, n_ary_size * sizeof(int), cudaMemcpyHostToDevice);

    const int block_dim = 256;
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        OPermanentKernelMatrix,
        block_dim,
        0);

    int max_concurrent_blocks_gpu = props.multiProcessorCount * max_blocks_per_sm;

    int min_blocks_for_work = (int)((idx_max + block_dim - 1) / block_dim);

    const int grid_dim = std::min(max_concurrent_blocks_gpu, min_blocks_for_work);

    size_t shared_mem_size = block_dim * sizeof(cuDoubleComplex);

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);

    OPermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
        modified_m, d_new_rows, new_size,
        cols.typed_data(), cols.element_count(),
        d_n_ary_limits, n_ary_size, idx_max,
        host_max_concurrent_warps, sum_rows,
        reinterpret_cast<cuDoubleComplex *>(permanent->typed_data()));

    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    printf("Kernel execution time: %f ms\n", milliseconds2);

    delete[] h_matrix;
    delete[] h_new_matrix;
    delete[] new_rows;
    cudaFree(d_new_matrix);
    cudaFree(d_new_rows);
  }
  else
  {
    size_t n_ary_size = rows_size - 1;
    int *h_n_ary_limits = new int[n_ary_size];
    for (size_t idx = 0; idx < n_ary_size; idx++)
    {
      h_n_ary_limits[idx] = h_rows[idx + 1] + 1;
    }

    uint64_t idx_max = h_n_ary_limits[0];
    for (size_t idx = 1; idx < n_ary_size; idx++)
    {
      idx_max *= h_n_ary_limits[idx];
    }

    int *d_n_ary_limits;
    cudaMalloc(&d_n_ary_limits, n_ary_size * sizeof(int));
    cudaMemcpy(d_n_ary_limits, h_n_ary_limits, n_ary_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(permanent->typed_data(), 0, sizeof(cuDoubleComplex));

    const int block_dim = 256;
    const int grid_dim = (int)((idx_max + block_dim - 1) / block_dim);

    size_t shared_mem_size = block_dim * sizeof(cuDoubleComplex);

    OPermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
        m, rows.typed_data(),
        rows.element_count(),
        cols.typed_data(),
        cols.element_count(),
        d_n_ary_limits,
        n_ary_size,
        idx_max,
        host_max_concurrent_warps, sum_rows,
        reinterpret_cast<cuDoubleComplex *>(permanent->typed_data()));
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Execution time: %f ms\n", milliseconds);

  delete[] h_rows;

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void PermanentKernelMatrix(Matrix<cuDoubleComplex> A, uint64_t *rows, size_t rows_size,
                                      uint64_t *cols, size_t cols_size, cuDoubleComplex *result)
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

  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  cuDoubleComplex local_result = make_cuDoubleComplex(0.0, 0.0);

  for (uint64_t idx = tid; idx < idx_max; idx += stride)
  {

    uint64_t temp_idx = idx;
    int minus_signs_all = 0;
    int binomial_coeff = 1;
    int gcode[32];

    for (size_t pos = 0; pos < n_ary_size; pos++)
    {
      gcode[pos] = temp_idx % n_ary_limits[pos];
      temp_idx /= n_ary_limits[pos];
      minus_signs_all += gcode[pos];
      binomial_coeff *= binomialCoeffManual<int>(rows[pos + 1], gcode[pos]);
    }

    cuDoubleComplex colsum[32];
    for (size_t i = 0; i < cols_size && i < 32; i++)
    {
      colsum[i] = A(0, i);
    }

    for (size_t row = 0; row < n_ary_size; row++)
    {
      int minus_signs = gcode[row];
      int row_mult = rows[row + 1];

      for (size_t col = 0; col < cols_size && col < 32; col++)
      {
        double factor = row_mult - 2.0 * minus_signs;
        cuDoubleComplex scaled = cuCmul(A(row + 1, col),
                                        make_cuDoubleComplex(factor, 0.0));
        colsum[col] = cuCadd(colsum[col], scaled);
      }
    }

    int parity = (minus_signs_all % 2 == 0) ? 1 : -1;

    cuDoubleComplex term = make_cuDoubleComplex((double)parity, 0.0);
    for (size_t i = 0; i < cols_size && i < 32; i++)
    {
      for (size_t j = 0; j < cols[i]; j++)
      {
        term = cuCmul(term, colsum[i]);
      }
    }

    term = cuCmul(term, make_cuDoubleComplex((double)binomial_coeff, 0.0));

    local_result = cuCadd(local_result, term);
  }

  double scale_factor = 1.0 / (1ULL << (sum_rows - 1));
  local_result = cuCmul(local_result, make_cuDoubleComplex(scale_factor, 0.0));

  atAddComplex(result, local_result);
}

ffi::Error PermFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> y,
                       ffi::ResultBuffer<ffi::C128> res)
{
  const int block_dim = 128;
  auto [total_size, n] = get_dims(A);
  Matrix<cuDoubleComplex> m(n, n, reinterpret_cast<cuDoubleComplex *>(A.typed_data()));
  const int grid_dim = (n * n + block_dim - 1) / block_dim;

  cudaMemset(res->typed_data(), 0, sizeof(cuDoubleComplex));
  cudaMemset(y->typed_data(), 0, sizeof(cuDoubleComplex));

  PermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
      m, rows.typed_data(),
      rows.element_count(),
      cols.typed_data(),
      cols.element_count(),
      reinterpret_cast<cuDoubleComplex *>(y->typed_data()));

  PermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
      m, rows.typed_data(),
      rows.element_count(),
      cols.typed_data(),
      cols.element_count(),
      reinterpret_cast<cuDoubleComplex *>(res->typed_data()));

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

ffi::Error PermBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> res, ffi::Buffer<ffi::C128> A,
                       ffi::Buffer<ffi::U64> rows, ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> ct_x)
{
  auto [total_size, n] = get_dims(A);
  if (n == 0)
  {
    return ffi::Error::InvalidArgument("Permanent backward inputs must be arrays");
  }

  uint64_t *h_rows = new uint64_t[n];
  cudaMemcpy(h_rows, rows.typed_data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  uint64_t *h_cols = new uint64_t[n];
  cudaMemcpy(h_cols, cols.typed_data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  Matrix<cuDoubleComplex> m(n, n, reinterpret_cast<cuDoubleComplex *>(A.typed_data()));

  cudaMemset(ct_x->typed_data(), 0, n * n * sizeof(cuDoubleComplex));

  for (size_t i = 0; i < n; ++i)
  {
    if (h_rows[i] == 0)
      continue;

    for (size_t j = 0; j < n; ++j)
    {
      if (h_cols[j] == 0)
        continue;

      uint64_t *grad_rows = new uint64_t[n];
      uint64_t *grad_cols = new uint64_t[n];

      // Copy and decrement
      memcpy(grad_rows, h_rows, n * sizeof(uint64_t));
      memcpy(grad_cols, h_cols, n * sizeof(uint64_t));
      grad_rows[i] -= 1;
      grad_cols[j] -= 1;

      uint64_t *d_grad_rows, *d_grad_cols;
      cudaMalloc(&d_grad_rows, n * sizeof(uint64_t));
      cudaMalloc(&d_grad_cols, n * sizeof(uint64_t));
      cudaMemcpyAsync(d_grad_rows, grad_rows, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_grad_cols, grad_cols, n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

      cuDoubleComplex *d_entry_result;
      cudaMalloc(&d_entry_result, sizeof(cuDoubleComplex));
      cudaMemset(d_entry_result, 0, sizeof(cuDoubleComplex));

      const int block_dim = 128;
      const int grid_dim = (n * n + block_dim - 1) / block_dim;
      PermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
          m, d_grad_rows, n, d_grad_cols, n, d_entry_result);

      cuDoubleComplex h_entry_result;
      cudaMemcpyAsync(&h_entry_result, d_entry_result, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);

      cudaStreamSynchronize(stream);

      double scale = static_cast<double>(h_rows[i]) * static_cast<double>(h_cols[j]);
      h_entry_result = cuCmul(h_entry_result, make_cuDoubleComplex(scale, 0.0));

      cudaMemcpyAsync(
          reinterpret_cast<cuDoubleComplex *>(ct_x->typed_data()) + i * n + j,
          &h_entry_result,
          sizeof(cuDoubleComplex),
          cudaMemcpyHostToDevice,
          stream);

      cudaFree(d_grad_rows);
      cudaFree(d_grad_cols);
      cudaFree(d_entry_result);

      delete[] grad_rows;
      delete[] grad_cols;
    }
  }
  delete[] h_rows;
  delete[] h_cols;
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }

  return ffi::Error::Success();
}