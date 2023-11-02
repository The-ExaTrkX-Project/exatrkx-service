# ExaTrkX in CPUs
This is the preliminary implementation of the ExaTrkX algorithm in CPUs.
The torch models are executed through the `libtorch` library. The fixed
radius clustering is implemented via the `faiss-cpu` library.

## Build
Start the container *as a client*:
```bash!
podman-hpc run -it --rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/ username/custom_backend:v1.0 bash
```

To compile the code, simply `cd exatrkx_cpu` and run `./make.sh`.

## Latency evaluation
Run the direct inference and evaluate the latency:
```bash!
./build/bin/inference-cpu -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv -t 1
```
See `./build/bin/inference-cpu -h` for help.

Results
```text
Input file: ../../exatrkx_pipeline/datanmodels/in_e1000.csv
Models loaded successfully
Running Inference with local CPUs
Embedding model run successfully
is_trained = true
Total 39 tracks in 1 events.
1) embedding: 0.4282
2) building:  0.0166
3) filtering: 3.4804
4) gnn:       0.2146
5) labeling:  0.0023
6) total:     4.1421
-----------------------------------------------------
Summary of the first event
1) embedding:  0.4282
2) building:   0.0166
3) filtering:  3.4804
4) gnn:        0.2146
5) labeling:   0.0023
6) total:      4.1421
-----------------------------------------------------
Summary of without first 1 event
Not enough data. 1 total and 1 skipped
Summary of the last event
1) embedding:  0.4282
2) building:   0.0166
3) filtering:  3.4804
4) gnn:        0.2146
5) labeling:   0.0023
6) total:      4.1421
```

## Throughput evaluation

```bash
./build/bin/inference-cpu-throughput -m ../../exatrkx_pipeline/datanmodels -d ../../exatrkx_pipeline/datanmodels/in_e1000.csv -t 1 
```
See `./build/bin/inference-cpu-throughput -h` for help.
