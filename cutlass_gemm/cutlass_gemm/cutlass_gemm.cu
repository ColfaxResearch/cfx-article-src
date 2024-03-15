#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <ATen/autocast_mode.h>

#include <pybind11/pybind11.h>

#include <cstdio>
#include <iostream>

#include "cutlass_gemm.hpp"

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
