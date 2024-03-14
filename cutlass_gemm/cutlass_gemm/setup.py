import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cutlass_dir = os.environ.get("CUTLASS_DIR", "")
if not os.path.isdir(cutlass_dir):
  raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation") 

_cutlass_include_dirs = ["tools/util/include","include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

nvcc_flags=["-O3","-DNDEBUG","-std=c++17"]
ld_flags=[]


if "COMPILE_3X_HOPPER" in os.environ:
  nvcc_flags += ["--generate-code=arch=compute_90a,code=[sm_90a]", "-DCOMPILE_3X_HOPPER"] 
  ld_flags += ["cuda"]

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
                name="cutlass_gemm",  
                sources=["cutlass_gemm.cu"],
                include_dirs=cutlass_include_dirs,
                extra_compile_args={'nvcc': nvcc_flags},
                libraries=ld_flags)
   ],
    cmdclass={
        'build_ext': BuildExtension
    })
