# ExaTrkX in GPUs
This is the preliminary implementation of the ExaTrkX algorithm in GPUs.
The torch models are executed through the `libtorch` library. The fixed
radius clustering is implemented via the `frnn` library. 

## Build
Start the container *as a client*:
```bash!
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ username/custom_backend:v1.0 bash
```

To compile the code, simply `cd exatrkx_gpu` and run `./make.sh`.

## Latency evaluation
```bash!
./build/bin/inference-gpu -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv
```
See `./build/bin/inference-gpu -h` for help.

Results
```text
Input file: ../../exatrkx_pipeline/datanmodels/in_e1000.csv
Running Inference with local GPUs
Total 37 tracks in 1 events.
1) embedding: 0.0398
2) building:  0.0015
3) filtering: 0.0088
4) gnn:       0.0141
5) labeling:  0.0016
6) total:     0.0659
-----------------------------------------------------
Summary of the first event
1) embedding:  0.0398
2) building:   0.0015
3) filtering:  0.0088
4) gnn:        0.0141
5) labeling:   0.0016
6) total:      0.0659
-----------------------------------------------------
Summary of without first 1 event
Not enough data. 1 total and 1 skipped
Summary of the last event
1) embedding:  0.0398
2) building:   0.0015
3) filtering:  0.0088
4) gnn:        0.0141
5) labeling:   0.0016
6) total:      0.0659
```

## Throughput evaluation

```bash!
./build/bin/inference-gpu-throughput -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv -t 1 
```
See `./build/bin/inference-gpu-throughput -h` for help.
