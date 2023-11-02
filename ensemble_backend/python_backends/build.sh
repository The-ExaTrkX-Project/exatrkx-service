#!/usr/bin/env bash

#=========================================
# 
# Title: build.sh
# Author: Andrew Naylor
# Date: May 23
# Brief: Build envs + conda-pack
#
# Usage: ./build.sh [frnn|cugraph] (optional: -o [output_file_path] -j [number_of_threads])
#
#=========================================

#Args
if [ "$#" -lt 1 ]
then
        echo "./build.sh requires [frnn|cugraph]"
        exit 1
fi

#Variables
BUILD_TYPE=$1
shift 

while test $# -gt 0; do
  case "$1" in
    -o)
      shift
      ARG_OUTPUT_FILE_PATH=$1
      shift
      ;;
    -j)
      shift
      ARG_NTHREADS=$1
      shift
      ;;
    *)
      break
      ;;
  esac
done

OUTPUT_FILE_PATH=${ARG_OUTPUT_FILE_PATH:="${BUILD_TYPE}.tar.gz"}
NTHREADS=${ARG_NTHREADS:=4}

#Build
case "$BUILD_TYPE" in 
    frnn)
        echo "<> Building frnn env"
        # Install PyTorch
        PYTORCH_VERSION=2.0.1
        mamba install -y numpy pytorch==${PYTORCH_VERSION} pytorch-cuda=11.8 -c pytorch -c nvidia 
        mamba clean -itcly

        # Install Frnn
        CUDA_COMPUTE="-gencode=arch=compute_80,code=sm_80"
        FRNN_VERSION=asnaylor
        export NCORES=${NTHREADS} 
        export CUDA_COMPUTE=${CUDA_COMPUTE} 
        git clone --recursive https://github.com/${FRNN_VERSION}/FRNN.git 
        cd FRNN 
        make
        ;;
    cugraph)
        echo "<> Building cugraph env"
        CUGRAPH_VER="23.04"
        mamba install -y -c rapidsai -c conda-forge -c nvidia \
              cugraph=${CUGRAPH_VER}
        mamba clean -itcly
        ;;
    *)
        echo "Unsupported $BUILD_TYPE"
        exit 1
        ;;
esac

#Pack
echo "<> Packing conda-env"
conda pack -o $OUTPUT_FILE_PATH -j $NTHREADS