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

__global__ void PermanentKernelMatrix(Matrix<cuDoubleComplex> A, uint64_t *rows, size_t rows_size,
                                      uint64_t *cols, size_t cols_size,
                                      int *h_n_ary_limits, size_t n_ary_size, uint64_t idx_max,
                                      int64_t host_max_concurrent_warps, int sum_rows, cuDoubleComplex *result)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  cuDoubleComplex local_result = make_cuDoubleComplex(0.0, 0.0);

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

    n_aryGrayCodeCounter gcode_counter(h_n_ary_limits, n_ary_size, initial_offset);
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
      int changed_index, prev_value, value;
      if (gcode_counter.next(changed_index, prev_value, value))
      {
        break;
      }

      parity = -parity;

      int row_offset = changed_index + 1;
      cuDoubleComplex colsum_prod = make_cuDoubleComplex(parity, 0.0);
      for (size_t col_idx = 0; col_idx < cols_size; col_idx++)
      {
        if (prev_value < value)
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
          value < prev_value
              ? binomial_coeff * prev_value / (row_mult_current - value)
              : binomial_coeff * (row_mult_current - prev_value) / value;

      colsum_prod = cuCmul(colsum_prod, make_cuDoubleComplex((double)binomial_coeff, 0.0));
      local_result = cuCadd(local_result, colsum_prod);
    }
  }
  double scale_factor = 1.0 / (1ULL << (sum_rows - 1));
  local_result = cuCmul(local_result, make_cuDoubleComplex(scale_factor, 0.0));

  atAddComplex(result, local_result);
}

cudaError_t calculatePermanent(cudaStream_t stream,
                               cuDoubleComplex *A_data, size_t n,
                               uint64_t *rows_data, size_t rows_size,
                               uint64_t *cols_data, size_t cols_size,
                               cuDoubleComplex *permanent_data)
{
  std::vector<uint64_t> h_rows(rows_size);
  cudaError_t cuda_err = cudaMemcpy(h_rows.data(), rows_data, rows_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  if (cuda_err != cudaSuccess)
    return cuda_err;

  int sum_rows = 0;
  for (auto r : h_rows)
    sum_rows += r;

  std::vector<uint64_t> h_cols(cols_size);
  cuda_err = cudaMemcpy(h_cols.data(), cols_data, cols_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  if (cuda_err != cudaSuccess)
    return cuda_err;

  int sum_cols = 0;
  for (auto c : h_cols)
    sum_cols += c;

  size_t min_idx = 0;
  int minelem = 0;

  for (int i = 0; i < h_rows.size(); i++)
  {
    if (minelem == 0 || (h_rows[i] < minelem && h_rows[i] != 0))
    {
      minelem = h_rows[i];
      min_idx = i;
    }
  }

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  unsigned int warps_per_sm = props.warpSize > 0 ? (props.maxThreadsPerMultiProcessor / props.warpSize) : 32;
  int64_t host_max_concurrent_warps = (int64_t)props.multiProcessorCount * warps_per_sm;

  cuda_err = cudaMemsetAsync(permanent_data, 0, sizeof(cuDoubleComplex), stream);
  if (cuda_err != cudaSuccess)
    return cuda_err;

  if (h_rows.size() > 0 && minelem != 0)
  {
    size_t new_rows_size = rows_size + 1;
    std::vector<uint64_t> h_new_rows(new_rows_size);
    h_new_rows[0] = 1;
    for (size_t i = 0; i < rows_size; i++)
    {
      h_new_rows[i + 1] = h_rows[i];
    }
    h_new_rows[1 + min_idx] -= 1;

    Matrix<cuDoubleComplex> h_orig_matrix(n, n);
    cuda_err = cudaMemcpy(h_orig_matrix.data, A_data, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess)
      return cuda_err;

    Matrix<cuDoubleComplex> mtx_(n + 1, n);
    for (size_t j = 0; j < n; j++)
    {
      mtx_[j] = h_orig_matrix(min_idx, j);
    }
    for (size_t i = 0; i < n; i++)
    {
      for (size_t j = 0; j < n; j++)
      {
        mtx_(i + 1, j) = h_orig_matrix(i, j);
      }
    }

    cuDoubleComplex *d_new_matrix = nullptr;
    uint64_t *d_new_rows = nullptr;
    int *d_n_ary_limits = nullptr;

    cuda_err = cudaMalloc(&d_new_matrix, (n + 1) * n * sizeof(cuDoubleComplex));
    if (cuda_err != cudaSuccess)
    {
      return cuda_err;
    }
    cuda_err = cudaMalloc(&d_new_rows, new_rows_size * sizeof(uint64_t));
    if (cuda_err != cudaSuccess)
    {
      cudaFree(d_new_matrix);
      return cuda_err;
    }

    cuda_err = cudaMemcpyAsync(d_new_matrix, mtx_.data, (n + 1) * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
      cudaFree(d_new_matrix);
      cudaFree(d_new_rows);
      return cuda_err;
    }
    cuda_err = cudaMemcpyAsync(d_new_rows, h_new_rows.data(), new_rows_size * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
      cudaFree(d_new_matrix);
      cudaFree(d_new_rows);
      return cuda_err;
    }

    Matrix<cuDoubleComplex> modified_m(n + 1, n, d_new_matrix);

    size_t n_ary_size = new_rows_size - 1;
    std::vector<int> h_n_ary_limits(n_ary_size);
    for (size_t i = 0; i < n_ary_size; i++)
    {
      h_n_ary_limits[i] = h_new_rows[i + 1] + 1;
    }

    uint64_t idx_max = h_n_ary_limits[0];
    for (size_t i = 1; i < n_ary_size; i++)
    {
      idx_max *= h_n_ary_limits[i];
    }

    cuda_err = cudaMalloc(&d_n_ary_limits, n_ary_size * sizeof(int));
    if (cuda_err != cudaSuccess)
    {
      cudaFree(d_new_matrix);
      cudaFree(d_new_rows);
      return cuda_err;
    }

    cuda_err = cudaMemcpyAsync(d_n_ary_limits, h_n_ary_limits.data(), n_ary_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (cuda_err != cudaSuccess)
    {
      cudaFree(d_new_matrix);
      cudaFree(d_new_rows);
      cudaFree(d_n_ary_limits);
      return cuda_err;
    }

    const int block_dim = 256;
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, PermanentKernelMatrix, block_dim, 0);
    int max_concurrent_blocks_gpu = props.multiProcessorCount * max_blocks_per_sm;
    int min_blocks_for_work = (idx_max == 0) ? 0 : (int)((idx_max + block_dim - 1) / block_dim);
    const int grid_dim = std::min(max_concurrent_blocks_gpu, min_blocks_for_work);

    if (grid_dim > 0)
    {
      PermanentKernelMatrix<<<grid_dim, block_dim, 0, stream>>>(
          modified_m, d_new_rows, new_rows_size,
          cols_data, cols_size,
          d_n_ary_limits, n_ary_size, idx_max,
          host_max_concurrent_warps, sum_rows,
          permanent_data);
      cuda_err = cudaGetLastError();
    }

    cudaError_t free_err = cudaFree(d_new_matrix);
    if (cuda_err == cudaSuccess && free_err != cudaSuccess)
      cuda_err = free_err;
    free_err = cudaFree(d_new_rows);
    if (cuda_err == cudaSuccess && free_err != cudaSuccess)
      cuda_err = free_err;
    free_err = cudaFree(d_n_ary_limits);
    if (cuda_err == cudaSuccess && free_err != cudaSuccess)
      cuda_err = free_err;

    if (cuda_err != cudaSuccess)
      return cuda_err;
  }

  return cudaSuccess;
}

ffi::Error PermImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A,
                    ffi::Buffer<ffi::U64> rows,
                    ffi::Buffer<ffi::U64> cols,
                    ffi::ResultBuffer<ffi::C128> permanent)
{
  auto [total_size, n] = get_dims(A);
  size_t rows_size = rows.element_count();
  size_t cols_size = cols.element_count();

  cuDoubleComplex *A_data = reinterpret_cast<cuDoubleComplex *>(A.typed_data());
  uint64_t *rows_data = rows.typed_data();
  uint64_t *cols_data = cols.typed_data();
  cuDoubleComplex *permanent_data = reinterpret_cast<cuDoubleComplex *>(permanent->typed_data());

  cudaError_t calc_err = calculatePermanent(stream, A_data, n,
                                            rows_data, rows_size,
                                            cols_data, cols_size,
                                            permanent_data);

  if (calc_err == cudaSuccess)
  {
    return ffi::Error::Success();
  }
  else if (calc_err == cudaErrorInvalidValue)
  {
    return ffi::Error::InvalidArgument(std::string("Invalid input detected during permanent calculation: ") + cudaGetErrorString(calc_err));
  }
  else
  {
    return ffi::Error::Internal(std::string("CUDA error during permanent calculation: ") + cudaGetErrorString(calc_err));
  }
}

ffi::Error PermBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> res_grad,
                       ffi::Buffer<ffi::C128> A,
                       ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> ct_x)
{
  auto [total_size, n] = get_dims(A);
  if (n == 0)
  {
    cudaMemsetAsync(ct_x->typed_data(), 0, total_size * sizeof(cuDoubleComplex), stream);
    return ffi::Error::Success();
  }
  size_t rows_size = rows.element_count();
  size_t cols_size = cols.element_count();
  if (n != rows_size || n != cols_size)
  {
    return ffi::Error::InvalidArgument("Matrix dimension mismatch with row/col vector sizes in PermBwdImpl.");
  }

  std::vector<uint64_t> h_rows(n);
  cudaError_t cuda_err = cudaMemcpy(h_rows.data(), rows.typed_data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  if (cuda_err != cudaSuccess)
    return ffi::Error::Internal(std::string("CUDA memcpy error (h_rows): ") + cudaGetErrorString(cuda_err));

  std::vector<uint64_t> h_cols(n);
  cuda_err = cudaMemcpy(h_cols.data(), cols.typed_data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
  if (cuda_err != cudaSuccess)
    return ffi::Error::Internal(std::string("CUDA memcpy error (h_cols): ") + cudaGetErrorString(cuda_err));

  cuDoubleComplex *A_data = reinterpret_cast<cuDoubleComplex *>(A.typed_data());
  cuDoubleComplex *ct_x_data = reinterpret_cast<cuDoubleComplex *>(ct_x->typed_data());

  cuda_err = cudaMemsetAsync(ct_x_data, 0, n * n * sizeof(cuDoubleComplex), stream);
  if (cuda_err != cudaSuccess)
    return ffi::Error::Internal(std::string("CUDA memset error (ct_x): ") + cudaGetErrorString(cuda_err));

  for (size_t i = 0; i < n; ++i)
  {
    if (h_rows[i] == 0)
      continue;

    for (size_t j = 0; j < n; ++j)
    {
      if (h_cols[j] == 0)
        continue;

      std::vector<uint64_t> grad_rows_host = h_rows;
      std::vector<uint64_t> grad_cols_host = h_cols;
      grad_rows_host[i] -= 1;
      grad_cols_host[j] -= 1;

      uint64_t *d_grad_rows = nullptr;
      uint64_t *d_grad_cols = nullptr;
      cuDoubleComplex *d_entry_result = nullptr;

      cuda_err = cudaMalloc(&d_grad_rows, n * sizeof(uint64_t));
      if (cuda_err != cudaSuccess)
      {
        return ffi::Error::Internal(std::string("CUDA malloc error (d_grad_rows): ") + cudaGetErrorString(cuda_err));
      }

      cuda_err = cudaMalloc(&d_grad_cols, n * sizeof(uint64_t));
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        return ffi::Error::Internal(std::string("CUDA malloc error (d_grad_cols): ") + cudaGetErrorString(cuda_err));
      }

      cuda_err = cudaMalloc(&d_entry_result, sizeof(cuDoubleComplex));
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        return ffi::Error::Internal(std::string("CUDA malloc error (d_entry_result): ") + cudaGetErrorString(cuda_err));
      }

      cudaMemsetAsync(d_entry_result, 0, sizeof(cuDoubleComplex), stream);
      cuda_err = cudaMemcpyAsync(d_grad_rows, grad_rows_host.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        cudaFree(d_entry_result);
        return ffi::Error::Internal(std::string("CUDA memcpy error (d_grad_rows): ") + cudaGetErrorString(cuda_err));
      }

      cuda_err = cudaMemcpyAsync(d_grad_cols, grad_cols_host.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        cudaFree(d_entry_result);
        return ffi::Error::Internal(std::string("CUDA memcpy error (d_grad_cols): ") + cudaGetErrorString(cuda_err));
      }

      cudaError_t sub_perm_cuda_err = calculatePermanent(stream, A_data, n,
                                                         d_grad_rows, n,
                                                         d_grad_cols, n,
                                                         d_entry_result);

      if (sub_perm_cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        cudaFree(d_entry_result);
        if (sub_perm_cuda_err == cudaErrorInvalidValue)
        {
          return ffi::Error::InvalidArgument(std::string("Invalid input during sub-permanent calculation: ") + cudaGetErrorString(sub_perm_cuda_err));
        }
        else
        {
          return ffi::Error::Internal(std::string("CUDA error during sub-permanent calculation: ") + cudaGetErrorString(sub_perm_cuda_err));
        }
      }

      cuDoubleComplex h_entry_result;
      cuda_err = cudaMemcpy(&h_entry_result, d_entry_result, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        cudaFree(d_entry_result);
        return ffi::Error::Internal(std::string("CUDA memcpy error (h_entry_result): ") + cudaGetErrorString(cuda_err));
      }

      double scale = static_cast<double>(h_rows[i]) * static_cast<double>(h_cols[j]);
      cuDoubleComplex scaled_result = cuCmul(h_entry_result, make_cuDoubleComplex(scale, 0.0));

      cuda_err = cudaMemcpyAsync(ct_x_data + i * n + j, &scaled_result, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
      if (cuda_err != cudaSuccess)
      {
        cudaFree(d_grad_rows);
        cudaFree(d_grad_cols);
        cudaFree(d_entry_result);
        return ffi::Error::Internal(std::string("CUDA memcpy error (scaled_result): ") + cudaGetErrorString(cuda_err));
      }

      cudaFree(d_grad_rows);
      cudaFree(d_grad_cols);
      cudaFree(d_entry_result);
    }
  }

  cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA stream sync error at end of PermBwdImpl: ") + cudaGetErrorString(cuda_err));
  }

  return ffi::Error::Success();
}

ffi::Error PermFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> y,
                       ffi::ResultBuffer<ffi::C128> res)
{
  ffi::Error perm_err = PermImpl(stream, A, rows, cols, y);

  cudaError_t cuda_err = cudaMemcpyAsync(
      reinterpret_cast<cuDoubleComplex *>(res->typed_data()),
      reinterpret_cast<cuDoubleComplex *>(y->typed_data()),
      sizeof(cuDoubleComplex),
      cudaMemcpyDeviceToDevice,
      stream);

  if (cuda_err != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA memcpy error (res): ") + cudaGetErrorString(cuda_err));
  }

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}
