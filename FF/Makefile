EXE = ff.ex
#SRC = testOpenACC.cpp
SRC = ff.cpp

#CXX = xlc++_r
CXX = pgc++

LINK = ${CXX}

ifeq ($(CXX),pgc++)
	CXXFLAGS=-O3 -std=c++11 --gnu_extensions -g -Munroll -mcmodel=medium
	CXXFLAGS+=-acc -ta=tesla:cc70
	CXXFLAGS+=-ta=tesla:managed
	CXXFLAGS+=-Minfo=accel
	#CXXFLAGS+=-Mlarge_arrays
	LINKFLAGS=-acc
	LINKFLAGS+=-ta=tesla:cc70
	#LINKFLAGS+=-ta=tesla:cc60
	LINKFLAGS+=-ta=tesla:managed
endif

ifeq ($(CXX),xlc++_r)
	CXXFLAGS=-O3 -std=gnu++11 -g -qsmp
	LINKFLAGS=-qsmp
endif


OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC)
	$(CXX) -c $(SRC) $(CXXFLAGS)


clean:
	rm -f $(OBJ) $(EXE)

