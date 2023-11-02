#!/usr/bin/env bash

#=========================================
# 
# Title: deploy_triton.sh
# Author: Andrew Naylor
# Date: Mar 23
# Brief: Triton service deploy script can be either compute or interactive 
#
#=========================================

#Functions
export_vars(){
    for var in "$@"; do
        export $var
    done
}

# Load config
source setup_env.cfg
export_vars TRITON_LOG_VERBOSE TRITON_PY_BACKENDS TRITON_MODELS TRITON_IMAGE


#Create folder and symlink
export TRITON_LOGS=$TRITON_JOBS_DIR/$SLURM_JOB_ID
rm -r $TRITON_LOGS &>/dev/null; mkdir -p $TRITON_LOGS
ln -fns $TRITON_LOGS $TRITON_JOBS_DIR/latest


#Launch Triton Server
echo "[slurm] Launching $SLURM_JOB_NUM_NODES Triton Servers..."
srun -C gpu \
    --hint=nomultithread \
    --cpus-per-task=64 \
    --gpus-per-task=4 \
    --gpu-bind=closest \
    --ntasks-per-node=1 \
    ./start_tritonserver.sh

exit
