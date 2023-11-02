#!/usr/bin/env bash

#=========================================
# 
# Title: run_exatrkx.sh
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Run exatrkx on tritron
#
# Usage: ./run_exatrkx.sh server_address [optional: -d data_folder -n njobs -j cpu_threads_per_job -q/--quiet]
#
#=========================================


#Args
if [ "$#" -lt 1 ]
then
        echo "./run_exatrkx.sh requires server_address"
        exit 1
fi


#Functions
execute_exatrkx() {
    if [ "$QUIET" = true ] ; then
        _REDIRECT='/dev/null'
    else  
        _REDIRECT='/dev/tty'
    fi

    _SRUN=false
    if [ "$_SRUN" = true ] ; then
        srun -C gpu \
            -n $NJOBS \
            -c $NTHREADS \
            -G 4 \
            podman-hpc run -it --rm --gpu \
                --volume="$(pwd):/workdir/" \
                --volume="$INPUT_DATA_DIR:/data/" \
                --net=host \
                $EXATRKX_IMAGE \
                /bin/bash -c \
                    "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/workdir/exatrkx_pipeline/build/lib/; \
                    /workdir/exatrkx_pipeline/build/bin/inference \
                    -s 4 \
                    -d /data/ \
                    -t $NTHREADS \
                    -u $ARG_SERVER_ADDRESS" &> $_REDIRECT
    else
        for i in `seq $NJOBS`
        do
            podman-hpc run -it --rm --gpu \
                --volume="$(pwd):/workdir/" \
                --volume="$INPUT_DATA_DIR:/data/" \
                --net=host \
                $EXATRKX_IMAGE \
                /bin/bash -c \
                    "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/workdir/exatrkx_pipeline/build/lib/; \
                    /workdir/exatrkx_pipeline/build/bin/inference \
                    -s 4 \
                    -d /data/ \
                    -t $NTHREADS \
                    -u $ARG_SERVER_ADDRESS" &> $_REDIRECT &
        done
        wait
    fi
    
}


# Load config
source setup_env.cfg

#Input Args
ARG_SERVER_ADDRESS=$1
shift 

while test $# -gt 0; do
  case "$1" in
    -d)
      shift
      ARG_INPUT_DATA_DIR=$1
      shift
      ;;
    -n)
      shift
      ARG_NJOBS=$1
      shift
      ;;
    -j)
      shift
      ARG_NTHREADS=$1
      shift
      ;;
    -q|--quiet)
      ARG_QUIET=true
      shift
      ;;
    *)
      break
      ;;
  esac
done
INPUT_DATA_DIR=${ARG_INPUT_DATA_DIR:=$EXATRKX_HOME/exatrkx_pipeline/datanmodels}
NJOBS=${ARG_NJOBS:=1}
NTHREADS=${ARG_NTHREADS:=1}
QUIET=${ARG_QUIET:=false}


#Main
main() {
    echo "<> Launching test on $ARG_SERVER_ADDRESS"
    echo "    - data_dir: $INPUT_DATA_DIR"
    echo "    - njobs: $NJOBS"
    echo "    - cpu_threads_per_job: $NTHREADS"
    request_start=`date +%s.%N`
    execute_exatrkx
    request_end=`date +%s.%N`
    total_runtime=$( echo "$request_end - $request_start" | bc -l )
    echo "<> Total Runtime: $total_runtime"
}

main
