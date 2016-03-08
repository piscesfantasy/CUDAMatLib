CXX=g++
CFLAGS=-O3 -Wall
NVCC=nvcc -arch=sm_21 -w #-Xcompiler "-Wall"
#NVCCFLAGS=-Xcompiler -fPIC #-std=c++11
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas #-lstdc++ -lpthread
#INCLUDE=-I /usr/local/cuda/samples/common/inc/ -I /usr/local/cuda/include

SRC=src
OBJ=obj

SOURCES=vector_add.cu
OBJECTS:=$(addprefix $(OBJ)/, $(addsuffix .o,$(basename $(SOURCES))))

BIN=vector_add

#vpath %.h include/
vpath %.cc src/
vpath %.cu src/

.PHONY: dir clean

all: dir $(BIN)

dir:
	mkdir -p $(OBJ)

$(OBJ)/%.o: %.cc
	$(CXX) $(CFLAGS) -c -o $@ $<

$(OBJ)/%.o : %.cu
	$(NVCC) -c -o $@ $<

$(BIN): $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS) 

clean:
	rm -rf $(OBJ)
