CUDAMatLib
==========
A library offering GPU parallel computation for array and matrix data types with CUDA.

Requirements
------------
* Linux/Unix
* g++ 4.2 or higher
* NVIDIA VGA card
* [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (>= CUDA 5.0)
    * After installation, add the following 2 lines in ~/.bashrc
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```

Installation & Test
-------------------
* Use the header files in include/ folder directly.
* Refer to src/\*.cu for use cases.
* `make` to create binaries in test/ for testing.
