#!/bin/bash
cmake -S . -B build \
	-DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)');/workspace/build/third-party/protobuf/lib/cmake" \
	-DCMAKE_C_COMPILER=`which gcc` \
	-DCMAKE_CXX_COMPILER=`which g++`


cmake --build build -j 2