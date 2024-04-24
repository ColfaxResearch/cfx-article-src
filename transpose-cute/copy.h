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
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "cutlass/detail/layout.hpp"

#include "shared_storage.h"

template <class TensorS, class TensorD, class ThreadLayoutS,
          class ThreadLayoutD>
__global__ static void __launch_bounds__(256, 1)
    copyKernel(TensorS const S, TensorD const D,
                         ThreadLayoutS const tS, ThreadLayoutD const tD) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  Tensor tSgS = local_partition(gS, tS, threadIdx.x); // (ThrValM, ThrValN)
  Tensor tDgD = local_partition(gD, tD, threadIdx.x);

  Tensor rmem = make_tensor_like(tSgS);               // (ThrValM, ThrValN)

  copy(tSgS, rmem);
  copy(rmem, tDgD);
}

int copy_host_kernel(int M, int N) {
  printf("Baseline copy; No transpose\n");

  using Element = float;
  using namespace cute;

  auto tensor_shape = make_shape(M, N);

  // Allocate and initialize
  thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(i);

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  //
  // Tile tensors
  //

  using bM = Int<32>;
  using bN = Int<32>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  auto threadLayoutS =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});
  auto threadLayoutD =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayoutS)); // 256 threads

  int iterations = 10;

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    copyKernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                                threadLayoutS, threadLayoutD);
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
              << 2e-6 * M * N * sizeof(Element) / time_ms << " GB/s)"
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
