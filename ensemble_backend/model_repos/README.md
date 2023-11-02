# Model Repositories

This folder contains two model repository folders for the nvidia triton inference server to use. It is recommended to use the models contained with the `models` folder. The `models_trackML` was prepared for use with the track ML dataset.

**Figure 1**: ExaTrkX Triton server pipeline

```mermaid
---
title: ExaTrkX ensemble model
---
stateDiagram-v2
    direction LR
    
    classDef pytorch_style fill:#f00,color:white,font-weight:bold,stroke-width:2px,stroke:black
    classDef python_backend_style fill:#46eb34,color:white,font-weight:bold,stroke-width:2px,stroke:yellow
    

    [*] --> embed:::pytorch_style : SP
    embed --> frnn:::python_backend_style : new SP
    [*] --> filter : SP
    frnn --> filter:::pytorch_style : Edges
    filter --> applyfilter:::python_backend_style : Edge Scores
    frnn --> applyfilter : Edges
    applyfilter --> gnn : Edges
    [*] --> gnn:::pytorch_style : SP
    applyfilter --> wcc:::python_backend_style : Edges
    gnn --> wcc : Edge Scores
    wcc --> [*] : Tracks

    state backend_legend {
        direction LR
            pytorch
            python_backend
        }
    

    class pytorch pytorch_style
    class python_backend python_backend_style
```


## Testing models
Use the `test_triton_model.py` to test models on the triton server:
```bash
python test_triton_model.py [triton_model] [inputs] -t/--triton_address [ip_address]
```

Example:
```bash
python3 test_triton_model.py exatrkx ../../data/exatrkx_input_FEATURES.csv -t 128.55.65.210:8001
```
