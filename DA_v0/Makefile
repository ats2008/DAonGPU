INCPATH 	= include
OBJ		= obj
SRC		= src

CXX 		= g++
NVCC 		= nvcc

LIBS		= -lm
CXXFLAGS 	= -std=c++14 -I$(INCPATH) 
NVCCFLAGS	= -std=c++14 -I$(INCPATH)
NVCCLIBS	= -lcudart

TARGET		= da.exe
DEPS		= 
CUDEPS		= sketch_DA_v7

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
