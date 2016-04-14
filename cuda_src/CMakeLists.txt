# CMakeLists.txt to build hellocuda.cu
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MACOSX_RPATH 1)

project(GPUAccel)

find_package(CUDA QUIET REQUIRED)

if(APPLE)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
    LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
elseif(LINUX)
	list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
	LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
	SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

file(GLOB GPUSRC
    "setup.cu"
)

# Specify binary name and source file to build it from
CUDA_ADD_LIBRARY(
GPUAccel SHARED
${GPUSRC})

if(APPLE)
link_directories(
${GLEW_LIBRARY_DIRS}
${GLFW_LIBRARY_DIRS}
${CMU462_LIBRARY_DIRS}
${FREETYPE_LIBRARY_DIRS}
)

target_link_libraries( GPUAccel
CMU462 ${CMU462_LIBRARIES}
)
endif()