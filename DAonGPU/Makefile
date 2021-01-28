INCPATH 	= include
OBJ		= obj
SRC		= src

CXX 		= g++
NVCC 		= nvcc

LIBS		= -lm
CXXFLAGS 	= -std=c++11 -I$(INCPATH) -g 
NVCCFLAGS	=  -rdc=true -std=c++11 -I$(INCPATH) -g -G
NVCCLIBS	= -lcudart

TARGET		= main.exe
DEPS		= main gpuDAVertexer
CUDEPS		= main gpuDAVertexer

OBJ_  := $(DEPS:%=$(OBJ)/%.o)
OBJCU_  := $(CUDEPS:%=$(OBJ)/%cu.o)

all:	$(TARGET) 

$(TARGET) : $(OBJ_) $(OBJCU_)
	$(NVCC)  -o $(TARGET) $(OBJ_) $(OBJCU_)

$(OBJ)/%.o : $(SRC)/%.cc
	$(CXX) $(CXXFLAGS) $(LIBS) -o $@ -c $^

$(OBJ)/%cu.o : $(SRC)/%.cu 
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) -c $^  -o $@


clean :
	@rm obj/*.o
	@rm *.exe
