#!/bin/bash
cmake -S . -B build \
 -DCMAKE_C_COMPILER=`which gcc` \
 -DCMAKE_CXX_COMPILER=`which g++` \
 -DCMAKE_INSTALL_PREFIX:PATH=install \
 -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"


 cmake --build build --target install
 cp -r build/install/install/backends/exatrkxcpu /opt/tritonserver/backends