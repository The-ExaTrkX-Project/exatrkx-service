#!/usr/bin/env bash

#=========================================
# 
# Title: start_tritonserver.sh
# Author: Andrew Naylor
# Date: Mar 23
# Brief: Wrapper script for deploying tritronsever
#
#=========================================

TRITON_LOG_VERBOSE_FLAGS=""
TRITON_SEVER_NAME="TritonServer_${SLURMD_NODENAME}"

#Setup Triton flags
if [ "$TRITON_LOG_VERBOSE" = true ]
then
    TRITON_LOG_VERBOSE_FLAGS="--log-verbose=3 --log-info=1 --log-warning=1 --log-error=1"
fi

#Start Triton
echo "[slurm] starting $TRITON_SEVER_NAME"
podman-hpc run -it --rm --gpu \
    --volume="$TRITON_PY_BACKENDS:/python_backends/" \
    --volume="$TRITON_MODELS:/models/" \
    --shm-size=20GB -p 8002:8002 -p 8001:8001 \
    $TRITON_IMAGE \
    tritonserver \
        --model-repository=/models/ \
        $TRITON_LOG_VERBOSE_FLAGS \
        > $TRITON_LOGS/$TRITON_SEVER_NAME.log 2>&1
