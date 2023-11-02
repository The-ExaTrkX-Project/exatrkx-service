# ExaTrkX Client

Start a container *as a client*:
```bash!
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ username/custom_backend:v1.0 bash
```
And compile the ExaTrkX client code in the `exatrkx_triton` folder: 
`cd /workspace/exatrkx_triton && ./make.sh`.

The code only works if there is server setup. Check `./build/bin/inference-aas -h` for help.
