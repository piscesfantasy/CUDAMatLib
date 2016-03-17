CXX=g++
CFLAGS=-O3 -Wall
NVCC=nvcc -arch=sm_21 -w #-Xcompiler "-Wall"
#NVCCFLAGS=-Xcompiler -fPIC #-std=c++11
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas #-lstdc++ -lpthread
INCLUDE=-I /usr/local/cuda/samples/common/inc/ -I /usr/local/cuda/include -I include/

SRC=src
OBJ=obj
BIN=bin

SOURCES=test_array.cu test_mat.cu
EXECUTABLES=test_array test_mat

OBJECTS:=$(addprefix $(OBJ)/, $(addsuffix .o,$(basename $(SOURCES))))

vpath %.h include/
vpath %.cc src/
vpath %.cu src/

.PHONY: dir exe clean

all: dir test_array test_mat

dir:
	mkdir -p $(BIN) $(OBJ)

$(OBJ)/%.o: %.cc
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJ)/%.o : %.cu
	$(NVCC) $(INCLUDE) -c -o $@ $<

test_array: $(OBJ)/test_array.o
	$(CXX) -o $(BIN)/test_array $^ $(LDFLAGS) 

test_mat: $(OBJ)/test_mat.o
	$(CXX) -o $(BIN)/test_mat $^ $(LDFLAGS) 

clean:
	rm -rf $(BIN) $(OBJ)
