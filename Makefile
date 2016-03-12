CXX=g++
CFLAGS=-O3 -Wall
NVCC=nvcc -arch=sm_21 -w #-Xcompiler "-Wall"
#NVCCFLAGS=-Xcompiler -fPIC #-std=c++11
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas #-lstdc++ -lpthread
INCLUDE=-I /usr/local/cuda/samples/common/inc/ -I /usr/local/cuda/include -I include/

SRC=src
OBJ=obj

SOURCES=test_array.cu

OBJECTS:=$(addprefix $(OBJ)/, $(addsuffix .o,$(basename $(SOURCES))))

vpath %.h include/
vpath %.cc src/
vpath %.cu src/

.PHONY: dir exe clean

all: dir exe

dir:
	mkdir -p $(OBJ)

$(OBJ)/%.o: %.cc
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJ)/%.o : %.cu
	$(NVCC) $(INCLUDE) -c -o $@ $<

exe: $(OBJECTS)
	$(CXX) -o test $^ $(LDFLAGS) 

clean:
	rm -rf $(OBJ)
