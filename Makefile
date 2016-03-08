CXX=g++
CFLAGS=-O3 -Wall
NVCC=nvcc -arch=sm_21 -w #-Xcompiler "-Wall"
#NVCCFLAGS=-Xcompiler -fPIC #-std=c++11
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas #-lstdc++ -lpthread
#INCLUDE=-I /usr/local/cuda/samples/common/inc/ -I /usr/local/cuda/include

SRC=src
OBJ=obj

SOURCES=vector_add.cu \
    vector_add_stream.cu

OBJECTS:=$(addprefix $(OBJ)/, $(addsuffix .o,$(basename $(SOURCES))))

#vpath %.h include/
vpath %.cc src/
vpath %.cu src/

.PHONY: dir exe clean

all: dir exe

dir:
	mkdir -p $(OBJ)

$(OBJ)/%.o: %.cc
	$(CXX) $(CFLAGS) -c -o $@ $<

$(OBJ)/%.o : %.cu
	$(NVCC) -c -o $@ $<

exe: $(OBJECTS)
	$(CXX) -o test.ext $^ $(LDFLAGS) 

clean:
	rm -rf $(OBJ)
