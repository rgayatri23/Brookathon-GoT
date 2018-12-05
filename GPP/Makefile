EXE = gpp.ex
SRC = gpp.cpp 

#CXX = xlc++_r
#CXX = icc 
#CXX = g++
CXX = clang++

LINK = ${CXX}

ifeq ($(CXX),icc)
    CXXFLAGS=-O3 -std=c++11 -qopenmp -qopt-report=5
    CXXFLAGS+=-qopenmp-offload=host #For offloading
    #CXXFLAGS+=-qno-openmp-offload #For disabling offloading
    #CXXFLAGS+=-xCORE-AVX2
    CXXFLAGS+=-xMIC-AVX512
    LINKFLAGS=-qopenmp #-qopenmp-offload=mic
    #LINKFLAGS+=-dynamic #Dynamic flag gives runtime issues with OMP4.5
endif 

ifeq ($(CXX),g++)
	CXXFLAGS= -g -O3 -std=c++11 -fopenmp -foffload="-lm" -foffload=nvptx-none
	LINKFLAGS=-fopenmp -foffload="-lm" -foffload=nvptx-none
endif 

ifeq ($(CXX),xlc++_r)
	CXXFLAGS=-O3 -std=gnu++11 -qsmp=noauto:omp -qoffload #-Xptxas -v
	LINKFLAGS=-qsmp=noauto:omp -qoffload 
endif 

ifeq ($(CXX),clang++)
	CXXFLAGS=-O3 -std=gnu++11 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${OLCF_CUDA_ROOT}
	LINKFLAGS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${OLCF_CUDA_ROOT}
endif 

ifeq ($(CXX),icc)
	CXXFLAGS=-O3 -qopenmp -qopt-report=5 -g
	CXXFLAGS+=-xCORE_AVX2
#	CXXFLAGS+=-xMIC_AVX512
	LINKFLAGS=-qopenmp -dynamic
endif 

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)  
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ1): $(SRC) 
	$(CXX) -c $(SRC) $(CXXFLAGS)

clean: 
	rm -f $(OBJ) $(EXE)

