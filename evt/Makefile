CUTLASS_DIR=/usr/include/cutlass
REPO_DIR=..
CXX=nvcc
APPS=evt_gemm_cute 

CXXFLAGS=--generate-code=arch=compute_90a,code=[compute_90a] -std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${REPO_DIR}/include/utils --expt-relaxed-constexpr $(CFLAGS)

LDFLAGS= 

LDLIBS=-lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl
.SUFFIXES: .cu

default: $(APPS)

$(APPS): % : %.cu
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean: 
	rm -f $(APPS)
