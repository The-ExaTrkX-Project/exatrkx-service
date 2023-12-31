# Exa.TrkX as a Service

The "ExaTrkX as a service" presented in the CTD 2023 workshop. 

## Introduction 
The ExaTrkX pipeline contains 6 stages: Embedding (`embed`), Fixed-radius nearest neighbour (`frnn`), Filtering (`filter`), 
Edge Classification (`gnn`), and Weakly-connected components (`wcc`). 

The pipeline can be run in two modes: direct inference and server inference. We created a [docker file](Dockerfile) that works for both modes. 
See the *Container* section for details. And the service is based on the [Triton inference server](https://github.com/triton-inference-server).

You can first build a "direct inference" and then set up a "server inference". 

## Direct Inference
Direct inference means that the algorithm directly runs on CPUs or GPUs without a server. However, we can use the same code in a server and run it on a server.

There are three C++ implementations of the ExaTrkX pipeline: the baseline pipeline that runs on either CPUs or GPUs in [exatrkx_pipeline](exatrkx_pipeline),
the CPU-only pipeline [exatrkx_cpu](exatrkx_cpu), and the GPU-only pipeline [exatrkx_gpu](exatrkx_gpu). Please see instructions in each folder to compile and run the code.

The CPU-only and GPU-only pipeline are duplicates of the baseline pipeline. We wrote them for R&D purposes. You can also find the baseline implementation in [exatrkx-acat2023](https://github.com/exatrkx/exatrkx-acat2021) and [exatrkx-cpp-ctd2022](https://github.com/exatrkx/exatrxk-cpp-ctd2022).

## Triton Server

There are three ExaTrkX-as-a-Service implementations: Ensemble backend in [ensemble_backend](ensemble_backend),
CPU-based customized backend in [custom_backend_cpu](custom_backend_cpu),
and GPU-based customized backend in [custom_backend_gpu](custom_backend_gpu).

## Evaluation

We use the tool `perf_analyzer` [link to Triton doc](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) from Triton to evaluate the performance. Details can be found in [evaluation](evaluation).

## Container
The container can be launched in two modes: client or server. And in all README files, we assume the container is launched from the parent directory.

*As a client*: 
```bash!
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ username/custom_backend:v1.0 bash
```

*As a server*:
```bash!
podman-hpc run -it --rm --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ username/exatrkx:v1.0 bash
```

## Code Structure

The code structure is as follows:

Direct inferences:
```bash
├── exatrkx_cpu
├── exatrkx_gpu
├── exatrkx_pipeline
```

Triton Server:
```bash
├── custom_backend_cpu
├── custom_backend_gpu
├── ensemble_backend
```

ExaTrkX Client
```bash
├── exatrkx_triton
```

Evaluation
```bash
├── evaluation
```
