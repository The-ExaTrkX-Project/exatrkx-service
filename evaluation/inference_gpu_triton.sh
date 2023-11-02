#!/bin/bash
dir=${1:-/workspace/exatrkx_pipeline/datanmodels/lrt/inputs}
inference-aas -m exatrkxgpu -d ${dir}