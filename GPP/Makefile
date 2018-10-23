#EXE = gpp_GPUComplex.ex
#SRC1 = gpp_GPUComplex.cpp 
#SRC2 = gpp_GPUComplex_CUDAKernel.cu
EXE = gpp_CUDA.ex
SRC1 = gpp.cpp 
SRC2 = gpp_CUDAKernel.cu


CXX = nvcc

LINK = ${CXX}

ifeq ($(CXX),nvcc)
	CXXFLAGS= -g -O3 -std=c++11 -Wno-deprecated-gpu-targets 
	CXXFLAGS+=-Xptxas -v #--maxrregcount=100 #output usage of registers
	CXXFLAGS+=-fmad=true
	#CXXFLAGS+=-arch=sm_60
	CXXFLAGS+=-arch=sm_70
	LINKFLAGS=-Wno-deprecated-gpu-targets
endif 

OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cu=.o)

$(EXE): $(OBJ2) $(OBJ1) 
	$(CXX) $(OBJ2) $(OBJ1) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC1) 
	$(CXX) -c $(SRC1) $(CXXFLAGS)

$(OBJ2): $(SRC2) 
	$(CXX) -c $(SRC2) $(CXXFLAGS)

clean: 
	rm -f $(OBJ1) $(OBJ2) $(EXE)

