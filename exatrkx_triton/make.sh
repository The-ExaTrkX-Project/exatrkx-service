#!/bin/bash
# Usage ./make.sh -j 4
if [ $# -eq 0 ]
  then
    BUILD_ARGS="-j 2"
else
    BUILD_ARGS="$@"
fi
cmake -S . -B build -DCMAKE_PREFIX_PATH="/usr/local/lib/cmake/TritonCommon"

cmake --build build --target inference-aas -- $BUILD_ARGS
cmake --build build --target install -- $BUILD_ARGS