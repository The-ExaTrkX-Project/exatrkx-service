# CPU-based Customized Backend

Start the container *as a server*:
```bash!
podman-hpc run -it --rm --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ username/exatrkx:v1.0 bash
```

## Build the backend
The customized backend depends on the `ExaTrkXCPU` library, which can be compiled from the `exatrkx_cpu`.

Then compile the customized backend by running the `make.sh` inside `backend` folder.

## Start the server

Launch the server:
```bash!
cp -r /workspace/custom_backend_cpu/backend/build/install/install/backends/exatrkxcpu /opt/tritonserver/backends && tritonserver --model-repository=/workspace/custom_backend_cpu/model_repo --log-verbose=4
```

## Start the client
Start a new container *as a client*, see [Container](../README.md#container) for details. And compile the ExaTrkX client code in the `exatrkx_triton` folder: 
`cd /workspace/exatrkx_triton && ./make.sh`.

## Check on 100 events
```bash!
time ./build/bin/inference-aas -d /workspace/exatrkx_pipeline/datanmodels/lrt/inputs
```
