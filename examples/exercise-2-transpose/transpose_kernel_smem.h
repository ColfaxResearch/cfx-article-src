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

#include "cutlass/detail/layout.hpp"

namespace cfx {

using namespace cute;

// Shared Storage for aligned addresses
template <class Element, class SmemLayout>
struct SharedStorage {
    cute::array_aligned<Element, cute::cosize_v<SmemLayout>> smem;
};

template <int kNumThreads_, int kBlockM_, int kBlockN_, typename element_type_>
struct KernelTraits {

    using Element = element_type_;
    constexpr static int kNumThreads = kNumThreads_;
    constexpr static int kNumWarps = ceil_div(kNumThreads, 32);
    constexpr static int kBlockM = kBlockM_;
    constexpr static int kBlockN = kBlockN_;

    using TileShape = decltype(make_shape(Int<kBlockM>{}, Int<kBlockN>{})); // (bM, bN)
    using TileShapeDst = decltype(make_shape(Int<kBlockN>{}, Int<kBlockM>{})); // (bN, bM)

    // for 128 bit vectorized copy
    static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<Element>);
    // for row major copy src: (bM, bN)
    // use 32-bit access since copying to swizzled or padded SMEM
    // constexpr static int kNumThreadsPerRow = kBlockN / kNumVecElem;
    // constexpr static int kNumRows = kNumThreads / kNumThreadsPerRow;
    constexpr static int kNumThreadsPerRow = kBlockN;
    constexpr static int kNumRows = kNumThreads / kNumThreadsPerRow;
    // for row major copy dst: (bN, bM)
    // use 32-bit access size since reading strided from SMEM
    constexpr static int kNumThreadsPerRowDst = kBlockM;
    constexpr static int kNumRowsDst = kNumThreads / kBlockM;

    // static_assert((kBlockN % kNumVecElem == 0) && (kNumThreads % kNumThreadsPerRow == 0)
    //    && (kNumThreads % kBlockM == 0),
    //     "Divisibility constraints for tiled copy not satisfied.");
    static_assert((kNumThreads % kBlockN == 0) && (kNumThreads % kBlockM == 0),
        "Divisibility constraints for tiled copy not satisfied.");
    
    using TiledCopyThrLayout =
        decltype(make_layout(make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}), LayoutRight{}));
    
    // using TiledCopyValLayout = 
    //     decltype(make_layout(make_shape(Int<1>{}, Int<kNumVecElem>{}), LayoutRight{}));

    using TiledCopyAsync = decltype(make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<Element>, Element>{}, 
        TiledCopyThrLayout{}));
    // using TiledCopyAsync = decltype(make_tiled_copy(
    //     Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, Element>{}, 
    //     TiledCopyThrLayout{}, TiledCopyValLayout{}));

    using TiledCopyThrLayoutDst =
        decltype(make_layout(make_shape(Int<kNumRowsDst>{}, Int<kNumThreadsPerRowDst>{}), LayoutRight{}));
    
    // 32 bit access size
    using TiledCopy = decltype(make_tiled_copy(
        Copy_Atom<UniversalCopy<Element>, Element>{},
        TiledCopyThrLayoutDst{}));
    
    // using SmemLayout =  decltype(make_layout(TileShape{}, LayoutRight{}));
    // using SmemLayout = decltype(make_layout(TileShape{}, make_stride(Int<kBlockN+1>{}, _1{})));
    // using SmemLayout = 
    //     decltype(composition(Swizzle<5, 0, 5>{}, make_layout(TileShape{}, LayoutRight{})));    
    using SmemLayout = 
        decltype(composition(Swizzle<6, 0, 7>{}, make_layout(TileShape{}, LayoutRight{})));

    // using SmemLayoutDst =
    //     decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}), LayoutLeft{}));
    
    // This method transposes the shape and deduces the stride for SmemLayoutDst.
    // Padding/swizzling on SmemLayout then comes along 'for free'.
    using FactoringLayout = decltype(make_layout(TileShapeDst{}, LayoutRight{}));
    using SmemLayoutDst = decltype(composition(SmemLayout{}, FactoringLayout{}));

    using SharedStorage = SharedStorage<Element, SmemLayout>;
};

// Host side kernel arguments
template <typename Element, bool Is_row_major = true>
struct KernelArguments {
    using GmemShapeT = cute::Shape<int32_t, int32_t>;
    using GmemLayoutT = std::conditional_t<Is_row_major,
        cute::Layout<GmemShapeT, cute::Stride<int32_t, _1>>,
        cute::Layout<GmemShapeT, cute::Stride<_1, int32_t>>
    >;
    Element* ptr_S;
    GmemLayoutT const layout_S;
    Element* ptr_D;
    GmemLayoutT const layout_D;
};

// Device side kernel params
template <typename Element, bool Is_row_major = true>
struct KernelParams {
    using GmemLayoutT = typename KernelArguments<Element, Is_row_major>::GmemLayoutT;
    Element* ptr_S;
    GmemLayoutT const layout_S;
    Element* ptr_D;
    GmemLayoutT const layout_D;
};

template <typename Element, bool Is_row_major = true>
static KernelParams<Element, Is_row_major>
to_underlying_arguments(KernelArguments<Element, Is_row_major> const& args) {
    return {args.ptr_S, args.layout_S, args.ptr_D, args.layout_D};
}

} // namespace cfx

template <typename Ktraits, typename Params>
__global__ static void __launch_bounds__(Ktraits::kNumThreads, 1)
    copy_kernel(CUTE_GRID_CONSTANT Params const params) {
    using namespace cute;

    using Element = typename Ktraits::Element;
    using TileShape = typename Ktraits::TileShape;
    using TileShapeDst = typename Ktraits::TileShapeDst;
    using SmemLayout = typename Ktraits::SmemLayout;
    using SmemLayoutDst = typename Ktraits::SmemLayoutDst;
    using TiledCopy = typename Ktraits::TiledCopy;
    using TiledCopyAsync = typename Ktraits::TiledCopyAsync;

    // Use Shared Storage structure to allocate aligned SMEM addresses.
    extern __shared__ char shared_memory[];
    using SharedStorage = typename Ktraits::SharedStorage;
    SharedStorage &shared_storage =
        *reinterpret_cast<SharedStorage *>(shared_memory);

    // Create SMEM tensors, two different views
    Tensor sS = make_tensor(make_smem_ptr(shared_storage.smem.data()), SmemLayout{});
    Tensor sD = make_tensor(make_smem_ptr(shared_storage.smem.data()), SmemLayoutDst{});

    Tensor mS = make_tensor(make_gmem_ptr(params.ptr_S), params.layout_S); // (bM, bN)
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), params.layout_D); // (bN, bM)
    
    Tensor gS = local_tile(mS, TileShape{}, make_coord(blockIdx.x, blockIdx.y)); // (bM, bN)
    Tensor gD = local_tile(mD, TileShapeDst{}, make_coord(blockIdx.y, blockIdx.x)); // (bN, bM)

    // if(cute::thread0()) {
    //     print("mS: "); print(mS); print("\n");
    //     print("mD: "); print(mD); print("\n");
    //     print("gS: "); print(gS); print("\n");
    //     print("gD: "); print(gD); print("\n");
    //     print("sS: "); print(sS); print("\n");
    //     print("sD: "); print(sD); print("\n");
    // }

    auto tiled_copy = TiledCopy{}; // dst copy
    auto tiled_copy_async = TiledCopyAsync{}; // source copy
    auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto thr_copy_async = tiled_copy_async.get_thread_slice(threadIdx.x);    

    Tensor tSgS = thr_copy_async.partition_S(gS); // (COPY_OP, COPY_M, COPY_N)
    Tensor tSsS = thr_copy_async.partition_D(sS); // (COPY_OP, COPY_M, COPY_N)
    
    Tensor tDsD = thr_copy.partition_S(sD); // (COPY_OP, COPY_M, COPY_N)
    Tensor tDgD = thr_copy.partition_D(gD); // (COPY_OP, COPY_M, COPY_N)

    copy(tiled_copy_async, tSgS, tSsS);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(tiled_copy, tDsD, tDgD);

    // if(cute::thread0()) {
    //     print("tSgS: "); print(tSgS); print("\n");
    //     print("tSsS: "); print(tSsS); print("\n");
    //     print("tDsS_out: "); print(tDsD); print("\n");
    //     print("tDgD: "); print(tDgD); print("\n");
    // }
    // __syncthreads();
  
}

template <int TILE_M = 64, int TILE_N = 128, int THREADS = 256>
int transpose_host(int M, int N, int iterations = 1) { 
    using namespace cute;

    printf("Example transpose kernel.\n");

    printf("M, N, TILE_M, TILE_N, THREADS = [%d, %d, %d, %d, %d].\n", M, N, TILE_M, TILE_N, THREADS);

    using Element = float;

    auto tensor_shape = cute::make_shape(M, N);
    auto tensor_shape_trans = cute::make_shape(N, M);

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

    // Define kernel traits
    using Ktraits = cfx::KernelTraits<THREADS, TILE_M, TILE_N, Element>;

    using TiledCopy = typename Ktraits::TiledCopy;
    using TiledCopyAsync = typename Ktraits::TiledCopyAsync;
    // print(TiledCopyAsync{});

    // Construct kernel params

    using Params = cfx::KernelParams<Element>;
    Params params =
    cfx::to_underlying_arguments<Element>({
        ptr_S,
        make_layout(tensor_shape, LayoutRight{}),
        ptr_D,
        make_layout(tensor_shape_trans, LayoutRight{})
    });

    // Layout gmem_layout = make_layout(tensor_shape, LayoutRight{});
    // Tensor mS = make_tensor(make_gmem_ptr(ptr_S), gmem_layout);
    // Tensor mD = make_tensor(make_gmem_ptr(ptr_D), gmem_layout);

    //
    // Determine grid and block dimensions
    //

    dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
    dim3 blockDim(THREADS);

    void* kernel_ptr = (void *)copy_kernel<Ktraits, Params>;

    int smem_size = sizeof(typename Ktraits::SharedStorage);
    printf("\nsmem size: %d.\n", smem_size);
    if (smem_size >= 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
    }

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    copy_kernel<Ktraits, Params><<<gridDim, blockDim, smem_size>>>(params);
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
              << 2 * M * N * sizeof(Element) << " bytes."
              << std::endl;
  }

  //
  // Verify
  //

#if 1

  h_D = d_D;

  int good = 0, bad = 0;

  auto transposeFunction = make_layout(tensor_shape, LayoutRight{});

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_D[i] == h_S[transposeFunction(i)])
      good++;
    else
      bad++;
  }

  std::cout << "Success " << good << ", Fail " << bad << std::endl;

#endif

  return 0;
}

void swizzle_test() {
    using namespace cute;

    using M = Int<32>;
    using N = Int<64>;
    // using N = Int<32>;

    auto tile_shape = make_shape(M{}, N{});
    auto smem_layout = composition(Swizzle<5, 0, 6>{}, make_layout(tile_shape, LayoutRight{}));
    // auto smem_layout = composition(Swizzle<5, 0, 5>{}, make_layout(tile_shape, LayoutRight{}));

    print(smem_layout); print("\n");

    for(int i = 0; i < M{}; ++i) {
        for(int j = 0; j < N{}; ++j) {
            print(smem_layout(i,j)%32); print(" ");
        }
        print("\n");
    }

}