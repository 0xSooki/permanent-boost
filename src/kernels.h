#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::C128> a,
                      ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> c,
                      ffi::ResultBuffer<ffi::C128> b_plus_1, size_t n);

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      size_t n);

ffi::Error PermanentHostMatrixFromBuffer(cudaStream_t stream, ffi::Buffer<ffi::C128> A,
                                         ffi::Buffer<ffi::U64> rows,
                                         ffi::Buffer<ffi::U64> cols,
                                         ffi::ResultBuffer<ffi::C128> permanent);

ffi::Error PermFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> A, ffi::Buffer<ffi::U64> rows,
                       ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> y,
                       ffi::ResultBuffer<ffi::C128> res);

ffi::Error PermBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::C128> res, ffi::Buffer<ffi::C128> A,
                       ffi::Buffer<ffi::U64> rows, ffi::Buffer<ffi::U64> cols,
                       ffi::ResultBuffer<ffi::C128> ct_x);

#endif // KERNELS_H_