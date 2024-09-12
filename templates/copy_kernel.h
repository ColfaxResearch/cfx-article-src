#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cutlass/numeric_types.h"
#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

// template <...>
__global__ static void // __launch_bounds__(NumThreads, 1)
    copy_kernel() {    
    using namespace cute;

    // ...
}

template <int TILE_M = 128, int TILE_N = 128, int THREADS = 128>
int copy_host(int M, int N, int iterations = 1) {  
    using namespace cute;

    printf("Example copy kernel to be implemented.\n");

    printf("M, N, TILE_M, TILE_N, THREADS = [%d, %d, %d, %d, %d].\n", M, N, TILE_M, TILE_N, THREADS);

    using Element = float;

    auto tensor_shape = make_shape(M, N);

    // Allocate and initialize
    thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
    thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

    for (size_t i = 0; i < h_S.size(); ++i) {    
        h_S[i] = static_cast<Element>(i);
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    Element* ptr_S = thrust::raw_pointer_cast(d_S.data());
    Element* ptr_D = thrust::raw_pointer_cast(d_D.data());

    //
    // Define Layouts on host
    //

    //
    // Determine grid and block dimensions
    //

    dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
    dim3 blockDim(THREADS);

    int smem_size = 0;    
    // void* kernel_ptr = (void *)copy_kernel<...>;
    //   printf("\nsmem size: %d.\n", smem_size);
    //   if (smem_size >= 48 * 1024) {
    //     CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    //         kernel_ptr,
    //         cudaFuncAttributeMaxDynamicSharedMemorySize,
    //         smem_size));
    //   }

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // copy_kernel<...><<<gridDim, blockDim, smem_size>>>(...);

    cudaError result = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s), "
              << 2e-9 * M * N * sizeof(Element) << " GB."
              << std::endl;
  }

  //
  // Verify
  //

  h_D = d_D;

  int good = 0, bad = 0;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_D[i] == h_S[i])
      good++;
    else
      bad++;
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;
  
  return 0;
}