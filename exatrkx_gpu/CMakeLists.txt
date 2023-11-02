cmake_minimum_required(VERSION 3.20.1)

project(ExatrkxGPU VERSION 0.1.0 LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

set(supported_archs "60" "62" "70" "72" "75" "80")
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

option(CUDA_STATIC_RUNTIME "Use CUDA static runtime" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND EXATRKX_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "-Wno-error=format-truncation")

find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(cugraph REQUIRED)

add_subdirectory(libFRNN)

add_library(
    ExaTrkXGPU SHARED
    ExaTrkXTrackFinding.cpp
    ExaTrkXUtils.cpp
)
target_link_libraries(
    ExaTrkXGPU
  PUBLIC
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    python3.8
)
target_link_libraries(
    ExaTrkXGPU
  PRIVATE
    frnn
)
set_target_properties(ExaTrkXGPU
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

add_executable(inference-gpu inference-gpu.cpp)
target_link_libraries(inference-gpu PUBLIC tbb ExaTrkXGPU)
target_compile_definitions(inference-gpu PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
set_target_properties(inference-gpu
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

install(
  TARGETS ExaTrkXGPU
  DESTINATION lib
  EXPORT ExaTrkXGPUTargets
)

file(GLOB HeaderFiles *.hpp)
install(FILES ${HeaderFiles}
  DESTINATION include/exatrkxgpu
)

add_executable(inference-gpu-throughput inference-gpu-throughput.cpp)
target_link_libraries(inference-gpu-throughput PUBLIC tbb ExaTrkXGPU)
target_compile_definitions(inference-gpu-throughput PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
set_target_properties(inference-gpu-throughput
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)
