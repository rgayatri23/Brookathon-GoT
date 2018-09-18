EXE = ff_GPUComplex.ex
SRC1 = ff_GPUComplex.cpp 
SRC2 = ff_GPUComplex_CUDAKernel.cu

CXX = nvcc

LINK = ${CXX}

ifeq ($(CXX),nvcc)
	CXXFLAGS= -g -O3 -std=c++11 -Wno-deprecated-gpu-targets 
#	CXXFLAGS+= -Xptxas -v --maxrregcount=150
 #   CXXFLAG+=-arch=sm_35
	    CXXFLAGS+=-arch=sm_70 -lineinfo
	LINKFLAGS=-Wno-deprecated-gpu-targets
endif 

OBJ1 = $(SRC1:.cpp=.o)
OBJ2 = $(SRC2:.cu=.o)

$(EXE): $(OBJ1) $(OBJ2) 
	$(CXX) $(OBJ1) $(OBJ2) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC1) 
	$(CXX) -c $(SRC1) $(CXXFLAGS)

$(OBJ2): $(SRC2) 
	$(CXX) -c $(SRC2) $(CXXFLAGS)

clean: 
	rm -f $(OBJ1) $(OBJ2) $(EXE)

