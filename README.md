CUDAMatLib
==========
A library offering GPU parallel computation for array and matrix data types with CUDA.

Requirements
------------
* Linux/Unix
* g++ 4.2 or higher
* NVIDIA VGA card
* [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (>= CUDA 5.0)

Installation & Test
-------------------
* Use the header files in include/ folder directly.
* Refer to src/\*.cu for use cases.
* `make` to create binaries in bin/ for testing.

To-do
-----
* Test and fix functions in CUDA\_array
* Complete declaired functions in CUDA\_matrix
* Figure out how to better incorporate different implementation of matrix multiplication
* Decide whether to incorporate every archived function
