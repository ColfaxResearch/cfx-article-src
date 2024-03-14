#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <ATen/autocast_mode.h>

#include <pybind11/pybind11.h>

#include <cstdio>
#include <iostream>


template<typename DataType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, DataType* ptrC);

template<typename DataType> void cutlass_gemm_unpack(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  const int M = A.sizes()[0];
  const int K = B.sizes()[0]; 
  const int N = B.sizes()[1];
  DataType const *ptrA = reinterpret_cast<DataType*>(A.data_ptr());
  DataType const *ptrB = reinterpret_cast<DataType*>(B.data_ptr());
  DataType *ptrC = reinterpret_cast<DataType*>(C.data_ptr());
  cutlass_gemm_wrapper<DataType>(M, N, K, ptrA, ptrB, ptrC);
}

torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  torch::Tensor _A = A.contiguous();
  torch::Tensor _B = B.contiguous();
  torch::Tensor _C = C.contiguous();

  if(!(A.device().is_cuda() && B.device().is_cuda() && C.device().is_cuda()))
    throw std::invalid_argument("cutlass_gemm only supports GPU device. Use .to(device=torch.device('cuda'))");

  if(A.dtype() == torch::kFloat16)
    cutlass_gemm_unpack<cutlass::half_t>(A, B, _C);
  else if(A.dtype() == torch::kFloat32)
    cutlass_gemm_unpack<float>(A, B, _C);
  else
    throw std::invalid_argument("Unsupported precision type");

  if(!C.is_contiguous())
    C.copy_(_C);
  return C;
}

torch::Tensor cutlass_gemm(
    torch::Tensor A,     
    torch::Tensor B) {

  const int M = A.sizes()[0];
  const int N = B.sizes()[1];
  torch::Tensor C = torch::zeros({M, N}, torch::TensorOptions().device(torch::kCUDA).dtype(A.dtype()));

  return cutlass_gemm(A,B,C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm", py::overload_cast<torch::Tensor,torch::Tensor>(&cutlass_gemm))
   .def("mm", py::overload_cast<torch::Tensor,torch::Tensor,torch::Tensor>(&cutlass_gemm), py::arg("A"), py::arg("B"), py::kw_only(), py::arg("out"));
}

// CUTLASS 2.X syntax GEMM
#ifndef COMPILE_3X_HOPPER
template<typename DataType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, DataType* ptrC) {
  using Gemm = cutlass::gemm::device::Gemm<
    DataType,                     // ElementA
    cutlass::layout::RowMajor,    // LayoutA
    DataType,                     // ElementB
    cutlass::layout::RowMajor,    // LayoutB
    DataType,                     // ElementOutput
    cutlass::layout::RowMajor,    // LayoutOutput
    float                         // ElementAccumulator
  >;

  float alpha = 1.0f;
  float beta = 0.0f;

  int lda = M;
  int ldb = K;
  int ldc = M;

  Gemm gemm_op;
  gemm_op({
    {M, N, K},
    {ptrA, lda},            // TensorRef to A device tensor
    {ptrB, ldb},            // TensorRef to B device tensor
    {ptrC, ldc},            // TensorRef to C device tensor
    {ptrC, ldc},            // TensorRef to D device tensor - may be the same as C
    {alpha, beta}           // epilogue operation arguments
  });
}

// CUTLASS 3.X syntax GEMM
#else
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

template<class DataType> void cutlass_gemm_wrapper(int M, int N, int K, DataType const* ptrA, DataType const* ptrB, DataType* ptrC) {


  // A matrix configuration
  using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
  constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
  constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<DataType>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
  constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<DataType>::value;

  // Core kernel configurations
  using ElementAccumulator  = float;                                          // Element type for internal accumulation
  using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
  using TilesShape          = Shape<_128,_128,_64>;                           // Threadblock-level tile size
  using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
  using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder 

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      DataType, LayoutA, AlignmentA,
      DataType, LayoutB, AlignmentB,
      ElementAccumulator,
      TilesShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TilesShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    DataType, LayoutC, AlignmentC,
    DataType, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;
  cutlass::Status status;

  //
  // Define the problem size
  //

  float alpha = 1.00f;
  float beta = 0.0f;

  //
  // Allocate device memory
  //
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, Int<1>{}));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, Int<1>{}));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, Int<1>{}));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, Int<1>{}));

  //
  // Launch GEMM on the device
  //
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {ptrA, stride_A, ptrB, stride_B},
    {{alpha, beta}, ptrC, stride_C, ptrC, stride_D}
  };
  
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  gemm_op.can_implement(arguments);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  gemm_op.initialize(arguments, workspace.get());

  // Correctness / Warmup iteration
  gemm_op.run();

}
#endif





