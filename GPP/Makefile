EXE = gpp.ex
SRC = gpp.cpp

CXX = pgc++

LINK = ${CXX}

ifeq ($(CXX),pgc++)
	CXXFLAGS=-fast -std=c++11 --gnu_extensions -g -Munroll
	#CXXFLAGS+=-acc -ta=multicore
	CXXFLAGS+=-acc -ta=tesla:cc70
	CXXFLAGS+=-Minfo=accel
	CXXFLAGS+=-ta=tesla:keepgpu -ta=tesla:keepptx
	LINKFLAGS=-acc
	LINKFLAGS+=-ta=tesla:cc70
	#LINKFLAGS+=-ta=tesla:cc60
endif

OBJ = $(SRC:.cpp=.o)

$(EXE): $(OBJ)
	$(CXX) $(OBJ) -o $(EXE) $(LINKFLAGS)

$(OBJ): $(SRC)
	$(CXX) -c $(SRC) $(CXXFLAGS)


clean:
	rm -f $(OBJ) $(EXE)

