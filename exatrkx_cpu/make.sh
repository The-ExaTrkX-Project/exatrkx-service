#!/bin/bash
# Usage ./make.sh -j 4
if [ $# -eq 0 ]
  then
    BUILD_ARGS="-j 2"
else
    BUILD_ARGS="$@"
fi
cmake -S . -B build -DCMAKE_C_COMPILER=`which gcc` \
	-DCMAKE_CXX_COMPILER=`which g++` \
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"

cmake --build build --target inference-cpu -- $BUILD_ARGS
cmake --build build --target inference-cpu-throughput -- $BUILD_ARGS
cmake --build build --target install -- $BUILD_ARGS