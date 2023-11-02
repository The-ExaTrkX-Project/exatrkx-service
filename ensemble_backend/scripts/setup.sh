#!/usr/bin/env bash

#=========================================
# 
# Title: setup.sh
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Setup environment + compile ExaTrkx
#
#=========================================


#Functions
check_images(){
    echo "<> Pulling Images"
    for img in "$@"; do
        podman-hpc pull $img
    done
}
check_folder(){
    if [ ! -d "$2" ]; then
        echo "<Warning> $1:$2 does not exist"
    fi 
}
create_folders(){
    echo "<> Create folders"
    mkdir -p "$@"
}
build_bin(){
    echo "<> Building $2 binary" 
    podman-hpc run -it --rm --gpu --volume="$(pwd)/$2/:/workdir/" $1 /bin/bash -c "cd /workdir; ./make.sh -j 32"
}
# build_legacy_bin(){
#     echo "<> Building $2 binary" 
#     mkdir -p $2/build 
#     podman-hpc run -it --rm --gpu --volume="$(pwd)/$2/:/workdir/" $1 /bin/bash -c "cd /workdir/build; ../make.sh -j 32"
# }

# Load config
source setup_env.cfg
IMAGE_LIST="$TRITON_IMAGE $EXATRKX_IMAGE $EXATRKX_CPU_IMAGE $EXATRKX_GPU_IMAGE $PROMETHEUS_IMAGE $GRAFANA_IMAGE $NGINX_IMAGE"


#Main
main() {
    check_images $IMAGE_LIST

    check_folder "TRITON_MODELS" $TRITON_MODELS
    check_folder "TRITON_PY_BACKENDS" $TRITON_PY_BACKENDS

    create_folders $PROMETHEUS_DB_DIR $GRAFANA_DIR $TRITON_JOBS_DIR

    build_bin $EXATRKX_CPU_IMAGE exatrkx_cpu
    build_bin $EXATRKX_GPU_IMAGE exatrkx_gpu
    build_bin $EXATRKX_GPU_IMAGE exatrkx_triton

    echo "<> Setup complete"
}


main