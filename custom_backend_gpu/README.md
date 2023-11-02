# GPU-based customized backend

Start the container *as a server*:
```bash!
podman-hpc run -it --rm --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ username/exatrkx:v1.0 bash
```

## Build the backend 
The GPU-based customized backend depends on the `ExaTrkXGPU` library, which can be compiled from the `exatrkx_gpu`.

Then compile the customized backend by running the `make.sh` inside `backend` folder.

## Start the server

```bash!
cp -r /workspace/custom_backend_gpu/backend/build/install/install/backends/exatrkxgpu/ /opt/tritonserver/backends && tritonserver --model-repository=/workspace/custom_backend_gpu/model_repo --log-verbose=4
```
## Start the client 
Start a new container *as a client*:
```bash!
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ username/custom_backend:v1.0 bash
```
And compile the ExaTrkX client code in the `exatrkx_triton` folder: 
`cd /workspace/exatrkx_triton && ./make.sh`.


## Check on 100 events
``` bash!
time ./build/bin/inference-aas -m exatrkxgpu -d /workspace/exatrkx_pipeline/datanmodels/lrt/inputs
```
