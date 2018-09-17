EXE = gpp.ex
SRC = gpp.cpp

CXX = pgc++

LINK = ${CXX}

ifeq ($(CXX),pgc++)
	CXXFLAGS=-O3 -std=c++11 --gnu_extensions -g -Munroll
	CXXFLAGS+=-acc -ta=nvidia
	CXXFLAGS+=-Minfo=accel
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

