CXX=nvcc
CXXFLAGS=--generate-code=arch=compute_90a,code=[compute_90a] -std=c++17 -O3 -Xcompiler -fopenmp

OBJECTS = main.o 

.SUFFIXES: .o .cc

default: clean app

app: $(OBJECTS)
	$(CXX) $(CXXFLAGS)  -o $@ $(OBJECTS)

main.o:
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

clean: 
	rm -f $(OBJECTS) app
