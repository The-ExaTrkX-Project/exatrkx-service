#!/usr/bin/env bash

#=========================================
# 
# Title: monitor_triton.sh
# Author: Andrew Naylor
# Date: Jun 23
# Brief: Monitor Triton
#
# Usage: ./monitor_triton.sh slurm_jobid
#
#=========================================


#Args
if [ "$#" -lt 1 ]
then
        echo "./monitor_triton.sh requires slurm_jobid"
        exit 1
fi


#Functions
convert_nodename_to_ip () {
  python3 -c "import socket; print(socket.gethostbyname('$1'))"
}

generate_config_files() {
    cp triton_service/prometheus_template.yml $PROMETHEUS_CONFIG
    printf "" > $NGINX_UPSTREAM_CONF
    trion_nodes=$(scontrol show hostnames $(squeue -j $ARG_SLURM_JOB -O NodeList | tail -n +2))
    for triton_node_name in $trion_nodes
    do 
        trion_node_ip=$(convert_nodename_to_ip $triton_node_name)
        echo "          - $trion_node_ip:8002" >> $PROMETHEUS_CONFIG
        echo "server $trion_node_ip:8001;" >> $NGINX_UPSTREAM_CONF
    done
}


# Load config
source setup_env.cfg

#Input Args
ARG_SLURM_JOB=$1
TRITON_JOB_DIR=$TRITON_JOBS_DIR/$ARG_SLURM_JOB
PROMETHEUS_CONFIG=$TRITON_JOB_DIR/prometheus.yml
NGINX_CONF=triton_service/nginx.conf
NGINX_UPSTREAM_CONF=$TRITON_JOB_DIR/tritonservers_upstream.conf



#Main
main() {
    echo "<> Generating config files"
    generate_config_files

    echo "<> Launching Prometheus..."
    # podman-hpc run -it --rm \ #TO-DO: Fix permissions issue
    #     --net=host \
    #     --volume="$PROMETHEUS_DB_DIR:/prometheus/" \
    #     --mount type=bind,source=$PROMETHEUS_CONFIG,destination=/etc/prometheus/prometheus.yml \
    #     $PROMETHEUS_IMAGE \
    #         > $TRITON_JOB_DIR/prometheus.log 2>&1 & 
    shifter \
        --image=$PROMETHEUS_IMAGE \
        --volume=$PROMETHEUS_DB_DIR:/prometheus \
        /bin/prometheus \
        --config.file=$PROMETHEUS_CONFIG \
        --storage.tsdb.path=/prometheus \
        > $TRITON_JOB_DIR/prometheus.log 2>&1 & 

    echo "<> Launching Grafana..."
    # podman-hpc run -it --rm \ #TO-DO: Fix permissions issue
    #     --net host \
    #     --volume="$GRAFANA_DIR:/grafana/" \
    #     --env GF_PATHS_DATA=/grafana \
    #     --env GF_PATHS_PLUGINS=/grafana/plugins \
    #     $GRAFANA_IMAGE \
    #         > $TRITON_JOB_DIR/grafana.log 2>&1 &
    shifter \
        --image=$GRAFANA_IMAGE \
        --volume=$GRAFANA_DIR:/grafana \
        --env GF_PATHS_DATA=/grafana \
        --env GF_PATHS_PLUGINS=/grafana/plugins \
        --entrypoint \
        > $TRITON_JOB_DIR/grafana.log 2>&1 &

    echo "<> Launching Nginx..."
    podman-hpc run -it --rm \
               --net host \
               --mount type=bind,source=$NGINX_CONF,destination=/etc/nginx/conf.d/default.conf \
               --mount type=bind,source=$NGINX_UPSTREAM_CONF,destination=/etc/nginx/tritonservers_upstream.conf \
               $NGINX_IMAGE \
               > $TRITON_JOB_DIR/nginx.log 2>&1 &

    TRITON_SERVER_IP=$(convert_nodename_to_ip "$HOSTNAME"):9191
    echo "<> Triton server running on: $TRITON_SERVER_IP"
    wait
}

main
