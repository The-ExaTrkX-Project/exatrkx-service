# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified by Haoran Zhao
# 2023/08/26

cmake_minimum_required(VERSION 3.17)

project(tritonexatrkxgpubackend LANGUAGES C CXX VERSION 0.1)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because exatrkxgpu backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

# change "main" to "r22.02"
set(TRITON_COMMON_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r22.02" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

find_package(Torch REQUIRED)
#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_exatrkxgpu.ldscript libtriton_exatrkxgpu.ldscript COPYONLY)

add_library(
  triton-exatrkxgpu-backend SHARED
  src/exatrkxgpu.cc
)

add_library(
  TritonExatrkxGPURuntimeBackend::triton-exatrkxgpu-backend ALIAS triton-exatrkxgpu-backend
)

target_compile_features(triton-exatrkxgpu-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-exatrkxgpu-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)

target_link_libraries(
  triton-exatrkxgpu-backend
  PRIVATE
    ExaTrkXGPU
    ${TORCH_LIBRARIES}
    triton-core-serverapi   # from repo-core
    triton-core-backendapi  # from repo-core
    triton-core-serverstub  # from repo-core
    triton-backend-utils    # from repo-backend
)

if(WIN32)
  set_target_properties(
    triton-exatrkxgpu-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_exatrkxgpu
  )
else()
  set_target_properties(
    triton-exatrkxgpu-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_exatrkxgpu
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_exatrkxgpu.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_exatrkxgpu.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonExatrkxGPUBackend)

install(
  TARGETS
    triton-exatrkxgpu-backend
  EXPORT
    triton-exatrkxgpu-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/exatrkxgpu
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/exatrkxgpu
)

install(
  EXPORT
    triton-exatrkxgpu-backend-targets
  FILE
    TritonExatrkxGPUBackendTargets.cmake
  NAMESPACE
    TritonExatrkxGPUBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonExatrkxGPUBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonExatrkxGPUBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonExatrkxGPUBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-exatrkxgpu-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonExatrkxGPUBackendTargets.cmake
  NAMESPACE TritonExatrkxGPUBackend::
)

export(PACKAGE TritonExatrkxGPUBackend)
