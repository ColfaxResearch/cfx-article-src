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
#include "util.h"

template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ static void __launch_bounds__(256, 1)
    copyKernel(TensorS const S, TensorD const D, ThreadLayout, VecLayout) {
  using namespace cute;
  using Element = typename TensorS::value_type;

  Tensor gS = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (bM, bN)
  Tensor gD = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (bN, bM)

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(VecLayout{})>;

  // A copy atom corresponds to one hardware memory access.
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  //
  // Note, this assumes the vector and thread layouts are aligned with contigous data
  // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
  // reads. Alternative vector layouts are also possible, though incompatible layouts
  // will result in compile time errors.
  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor tSgS = thr_copy.partition_S(gS);             // (CopyOp, CopyM, CopyN)
  Tensor tDgD = thr_copy.partition_D(gD);             // (CopyOp, CopyM, CopyN)

//  Tensor tSgS = local_partition(gS, ThreadLayout{}, threadIdx.x); // (ThrValM, ThrValN)
//  Tensor tDgD = local_partition(gD, ThreadLayout{}, threadIdx.x);

  Tensor rmem = make_tensor_like(tSgS);               // (ThrValM, ThrValN)

  copy(tSgS, rmem);
  copy(rmem, tDgD);
}

template <typename T> void copy_baseline(TransposeParams<T> params) {

  using Element = float;
  using namespace cute;

  //
  // Make tensors
  //
  auto tensor_shape = make_shape(params.M, params.N);
  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(make_gmem_ptr(params.input), gmemLayoutS);
  Tensor tensor_D = make_tensor(make_gmem_ptr(params.output), gmemLayoutD);
 
  //
  // Tile tensors
  //
  using bM = Int<32>;
  using bN = Int<1024>;

  auto block_shape = make_shape(bM{}, bN{});       // (bM, bN)

  Tensor tiled_tensor_S =
      tiled_divide(tensor_S, block_shape); // ((bM, bN), m', n')
  Tensor tiled_tensor_D =
      tiled_divide(tensor_D, block_shape); // ((bN, bM), n', m')

  auto threadLayout =
      make_layout(make_shape(Int<8>{}, Int<32>{}), LayoutRight{});

  auto vec_layout = make_layout(make_shape(Int<4>{}, Int<1>{}));

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim(
      size<1>(tiled_tensor_S),
      size<2>(tiled_tensor_S)); // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(threadLayout)); // 256 threads

  copyKernel<<<gridDim, blockDim>>>(tiled_tensor_S, tiled_tensor_D,
                                       threadLayout,  vec_layout);
}
