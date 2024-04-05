#include <cstdio>
#include <iostream>
#include <chrono>
#include "util.h"

// Selecting the "version" by uncommenting the version you want to test.
//#include "versions/copy.h"
//#include "versions/transpose_naive.h"
#include "versions/shared_mem_transpose.h"
//#include "versions/shared_mem_transpose_multirow.h"

int main() {

  using DTYPE = float;

  const long M = 1L<<13;
  const long N = 1L<<13;

  std::cout << "Creating the matrices on CPU" << std::endl;
  DTYPE* input  = (DTYPE*) malloc(M*N*sizeof(DTYPE)); 
  DTYPE* output = (DTYPE*) malloc(M*N*sizeof(DTYPE)); 
  DTYPE* ref_output = (DTYPE*) malloc(M*N*sizeof(DTYPE)); 
  randomMatrix<DTYPE>(input, M, N);

  // Prepare the reference
  std::cout << "Reference result on CPU" << std::endl;
  naiveTranspose(input, ref_output, M, N);

  std::cout << "Copying to GPU" << std::endl;
  float* d_input;
  float* d_output;
  cudaMalloc(&d_input, M*N*sizeof(DTYPE));
  cudaMalloc(&d_output,M*N*sizeof(DTYPE));
  cudaMemcpy(d_input, input, M*N*sizeof(DTYPE), cudaMemcpyHostToDevice);


  std::cout << "Running on GPU" << std::endl;
  for (int i = 0; i < 10; i++) {     
    // Transpose call
    auto t1 = std::chrono::high_resolution_clock::now();
    transpose<DTYPE>(d_input, d_output, M, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;

    // Performance. Factor of 2 in there because it is read and write.
    std::cout << "Trial " << i << " Completed in " << tDiff.count() << "ms (" << 2e-6*M*N*sizeof(DTYPE)/tDiff.count() << " GB/s)" << std::endl;

    // Check the result on the first iteration
    if(i == 0) {
      cudaMemcpy(output, d_output, M*N*sizeof(DTYPE), cudaMemcpyDeviceToHost);
      if(!matrixEqual<DTYPE>(ref_output, output, M, N)) {
        std::cout << "Validation failed" << std::endl;
        //printMatrix(ref_output, N, M);
        //printMatrix(output, N, M);
        break;
      }
    }
  }

  std::cout << "Done" << std::endl;

  cudaFree(d_input); 
  cudaFree(d_output); 
  free(input);
  free(output);
  free(ref_output);

}
