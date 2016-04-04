# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MACOSX_RPATH 1)

project(GPUAccel)

find_package(CUDA QUIET REQUIRED)

#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_20,code=sm_20
#    )

# Specify binary name and source file to build it from
CUDA_ADD_LIBRARY(
    GPUAccel SHARED
    kernel.cu)
