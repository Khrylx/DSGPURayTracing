# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MACOSX_RPATH 1)

project(GPUAccel)

find_package(CUDA QUIET REQUIRED)

if(APPLE)

elseif(LINUX)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif(LINUX)


# Specify binary name and source file to build it from
CUDA_ADD_LIBRARY(
    GPUAccel SHARED
    kernel.cu)