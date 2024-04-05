#ifndef _TRANSPOSE_
#define _TRANSPOSE_

#include "transpose_util.h"

#define M_TILE 32
#define N_TILE 32

template<typename T> __global__ void transpose_kernel(T* input, T* output, const long M, const long N) {
  const long x_0 = blockIdx.x * N_TILE;
  const long y_0 = blockIdx.y * M_TILE;

  const long x = x_0 + threadIdx.x;
  const long y = y_0 + threadIdx.y;

  //output[x*M+y] = input[y*N+x];
  output[y*N+x] = input[x*M+y];
}

// show performance difference when read is strided instead of write


template<typename T> void transpose(T* d_input, T* d_output, const long M, const long N) {
  dim3 blockDim(M_TILE, N_TILE);
  dim3 gridDim(M/M_TILE, N/N_TILE);
  transpose_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, M, N);
  cudaDeviceSynchronize();
  checkCUDAError();
}
#endif
