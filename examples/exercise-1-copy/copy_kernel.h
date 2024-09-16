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

using namespace cute;

template <int NumThreads_, int TILE_M_, int TILE_N_, typename Element>
struct KernelTraits {
  
  constexpr static int NumThreads = NumThreads_;
  constexpr static int TILE_M = TILE_M_;
  constexpr static int TILE_N = TILE_N_;
  // using ThrLayout = decltype(make_layout(make_shape(Int<NumThreads/TILE_N>{}, Int<TILE_N>{}), LayoutRight{}));
  using TileShape = decltype(make_shape(Int<TILE_M>{}, Int<TILE_N>{}));

  using CopyAtom = Copy_Atom<UniversalCopy<uint128_t>, Element>;
  // assume working with 32-bit type
  static_assert(NumThreads % (TILE_N/4) == 0, "Have enough threads.");
  static_assert(TILE_M % (NumThreads/(TILE_N/4)) == 0, "TiledCopy Tiler divides into the (Data) Tile Shape.");
  // e.g. TILE_N = 128. 128 threads. 32 threads per row.
  using ThrLayout = decltype(make_layout(Shape<Int<NumThreads/(TILE_N/4)>, Int<TILE_N/4>>{}, LayoutRight{}));
  // using ThrLayout = make_layout(Shape<Int<1>{}, Int<NumThreads>{}>, LayoutRight{});
  using ValLayout = decltype(make_layout(Shape<_1, _4>{}, LayoutRight{}));
  using TiledCopy = decltype(make_tiled_copy(CopyAtom{}, ThrLayout{}, ValLayout{}));
};

template <typename Ktraits, typename TensorS, typename TensorD>
__global__ static void __launch_bounds__(Ktraits::NumThreads, 1)
    copy_kernel(TensorS mS, TensorD mD) {    
    using namespace cute;
    using TileShape = typename Ktraits::TileShape;
    // using ThrLayout = typename Ktraits::ThrLayout;
    using TiledCopy = typename Ktraits::TiledCopy;
    // auto tile_shape = TileShape{};
    constexpr int TILE_M = Ktraits::TILE_M;
    constexpr int TILE_N = Ktraits::TILE_N;
    auto tile_shape = make_shape(Int<TILE_M>{}, Int<TILE_N>{});
    // auto thr_layout = ThrLayout{};
    
    int x = blockIdx.x; int y = blockIdx.y;
    auto gS = local_tile(mS, tile_shape, make_coord(x,y));
    auto gD = local_tile(mD, tile_shape, make_coord(x,y));
    
    // auto tAgS = local_partition(gS, thr_layout, threadIdx.x);
    // auto tAgD = local_partition(gD, thr_layout, threadIdx.x);  

    auto tiled_copy = TiledCopy{};
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto tAgS = thr_copy.partition_S(gS);
    auto tAgD = thr_copy.partition_D(gD);

    Tensor fragment = make_tensor_like(tAgS);
    // (V, M, N) -> (V, M, N)    
    cute::copy(tiled_copy, tAgS, fragment);
    cute::copy(tiled_copy, fragment, tAgD);
    // cute::copy(tAgS, tAgD);

    // if(cute::thread0()) {
    //   print("tAgS"); print(tAgS); print("\n");
    //   print("tAgD"); print(tAgD); print("\n");
    // }
}

template <int TILE_M = 128, int TILE_N = 128, int THREADS = 256>
int copy_host(int M, int N, int iterations = 1) {  
    using namespace cute;

    printf("Example copy kernel to be implemented.\n");

    printf("M, N, TILE_M, TILE_N, THREADS = [%d, %d, %d, %d, %d].\n",
      M, N, TILE_M, TILE_N, THREADS);

    using Element = float;

    // Allocate and initialize
    thrust::host_vector<Element> h_S(M*N); // (M, N)
    thrust::host_vector<Element> h_D(M*N); // (M, N)

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

    auto tensor_shape = make_shape(M, N);
    Layout gmem_layout = make_layout(tensor_shape, make_stride(N, Int<1>{}));
    // Layout gmem_layout_row = make_layout(tensor_shape, LayoutRight{});
    // Layout gmem_layout_default = make_layout(tensor_shape);
    // Layout gmem_layout_col = make_layout(tensor_shape, LayoutLeft{});
    // Layout gmem_layout_padded = make_layout(tensor_shape, make_stride(N+1, 1));

    Tensor mS = make_tensor(make_gmem_ptr(ptr_S), gmem_layout);
    Tensor mD = make_tensor(make_gmem_ptr(ptr_D), gmem_layout);

    auto tile_shape = make_shape(Int<TILE_M>{}, Int<TILE_N>{});

    // size thread layout = num of threads
    // e.g. (1, 128) : (0, 1)
    // auto thread_layout_A = make_layout(make_shape(Int<THREADS/TILE_N>{}, Int<TILE_N>{}), LayoutRight{});
    // e.g. (128, 1) : (1, 0)
    // auto thread_layout_B = make_layout(make_shape(Int<TILE_M>{}, Int<THREADS/TILE_M>{}), LayoutLeft{});

    // print("\nThrLayoutA: "); print(thread_layout_A); 
    // print("\nThrLayoutB: "); print(thread_layout_B); 

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
  
  using Ktraits = KernelTraits<THREADS, TILE_M, TILE_N, Element>;

  using TiledCopy = typename Ktraits::TiledCopy;
//   print(TiledCopy{});

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();

    copy_kernel<Ktraits><<<gridDim, blockDim, smem_size>>>(mS, mD);

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