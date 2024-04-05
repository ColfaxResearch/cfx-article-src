#ifndef _TRANSPOSE_
#define _TRANSPOSE_

#include "transpose_util.h"

#define M_TILE 32
#define N_TILE 32

template<typename T> __global__ void transpose_kernel(T* input, T* output, const long M, const long N) {
  __shared__ T tile[N_TILE][M_TILE];

  const long x_load_0 = blockIdx.x * N_TILE;
  const long y_load_0 = blockIdx.y * M_TILE;
  const long x = threadIdx.x;

  for(long y = 0; y < M_TILE; y++) 
    tile[x][y] = input[(y_load_0+y)*N+(x_load_0+x)];

  __syncthreads();

  const long x_store_0 = blockIdx.y * M_TILE;
  const long y_store_0 = blockIdx.x * N_TILE;

  for(long y = 0; y < N_TILE; y++) 
    output[(y_store_0+y)*M+(x_store_0+x)] = tile[y][x];
}

template<typename T> void transpose(T* d_input, T* d_output, const long M, const long N) {
  dim3 blockDim(M_TILE, 1);
  dim3 gridDim(M/M_TILE, N/N_TILE);
  transpose_kernel<T><<<gridDim, blockDim>>>(d_input, d_output, M, N);
  cudaDeviceSynchronize();
  checkCUDAError();
}
#endif
