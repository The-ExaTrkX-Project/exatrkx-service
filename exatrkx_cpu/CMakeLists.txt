cmake_minimum_required(VERSION 3.20.1)

project(ExatrkxCPU VERSION 0.1.0 LANGUAGES C CXX)


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND EXATRKX_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_FLAGS "-Wno-error=format-truncation") 

find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)
find_package(BLAS REQUIRED)
# find_package(OpenMP REQUIRED)
# find_package(faiss REQUIRED)

add_library(
  ExaTrkXCPU SHARED
  ExaTrkXTrackFinding.cpp
  ExaTrkXUtils.cpp
)
target_link_libraries(
  ExaTrkXCPU
  PUBLIC
  ${TORCH_LIBRARIES}
  PRIVATE
  libfaiss.so
)

set_target_properties(ExaTrkXCPU
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

add_executable(inference-cpu inference-cpu.cpp)
target_link_libraries(inference-cpu PUBLIC tbb ExaTrkXCPU)
set_target_properties(inference-cpu
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

install(
  TARGETS ExaTrkXCPU
  DESTINATION lib
  EXPORT ExaTrkXCPUTargets
)

file(GLOB HeaderFiles *.hpp)
install(FILES ${HeaderFiles}
  DESTINATION include/exatrkxcpu
)

add_executable(inference-cpu-throughput inference-cpu-throughput.cpp)
target_link_libraries(inference-cpu-throughput PUBLIC tbb ExaTrkXCPU)
set_target_properties(inference-cpu-throughput
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)
