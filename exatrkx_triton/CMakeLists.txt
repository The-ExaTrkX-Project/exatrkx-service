cmake_minimum_required(VERSION 3.20.1)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(ExatrkxTriton VERSION 0.1.0 LANGUAGES C CXX)

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

# find_package(Torch REQUIRED)
# find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

# find_package(TritonCore CONFIG REQUIRED)
find_package(TritonCommon CONFIG REQUIRED)
find_package(TritonClient CONFIG REQUIRED COMPONENTS TritonClient::triton-client)

add_executable(inference-aas inference-aas.cpp ExaTrkXTrackFindingTriton.cpp)
target_link_libraries(inference-aas PUBLIC tbb)
set_target_properties(inference-aas
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)

target_link_libraries(
  inference-aas
  PRIVATE
    libgrpcclient.so
    libprotobuf.a
)

# install packages
install(TARGETS inference-aas
        DESTINATION bin
)
        
        