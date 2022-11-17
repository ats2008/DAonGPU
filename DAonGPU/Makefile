INCPATH 	= include
OBJ		= obj
SRC		= src

CXX 		= g++
NVCC 		= nvcc

LIBS		= -lm
CXXFLAGS 	= -std=c++11 -I$(INCPATH) -g -rdynamic 
# Add -Xptxas -v to NVCCFLAGS for doing compiletime register analysis
NVCCFLAGS	=  -rdc=true -std=c++11 -I$(INCPATH) -g  --generate-line-info -Xcompiler -rdynamic  
NVCCGENCODE     =  
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
	$(NVCC) $(NVCCGENCODE) $(NVCCFLAGS) $(NVCCLIBS) -c $^  -o $@


clean :
	@rm *.exe  obj/*.o
cclean :
	@rm src/*.orig

