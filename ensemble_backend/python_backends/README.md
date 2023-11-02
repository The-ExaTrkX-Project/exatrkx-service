# Python Backends
Serveral stages used in the ExaTrkX pipeline use the Triton Python backend. These stages use custom execution environment which are deployed via conda-pack (see [triton docs](https://github.com/triton-inference-server/python_backend#creating-custom-execution-environments) for more information).


**Stage**|**Python Backend Name**
:-----|:-----
`frnn`| frnn
`applyfilter`| frnn
`wcc`| cugraph

# Build Base Image

Create the base mamba environment:
```bash
podman-hpc build --format docker -f base-cuda11-mamba-py38.Dockerfile -t base_mamba:cuda11-py38
```

> **Note**
> Note these instructions use `podman-hpc` but you can also use `podman`/`docker`


# Create conda envs

Run the container:
```bash
podman-hpc run -it --gpu -v $PWD:/workdir -v $SCRATCH:/scratch base_mamba:cuda11-py38 /bin/bash
```

Inside the container, install the relevant libraries and pack:
```
./build.sh [frnn|cugraph] (optional: -o [output_file_path] -j [number_of_threads])
```

# Update the triton models folder

Update the file paths (`EXECUTION_ENV_PATH`) in the `config.pbtxt` for the `applyfilter`, `frnn` & `wcc` models:
```json
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/exatrxk/python_backends/frnn_cuda11_py38.tar.gz"}
}
```
