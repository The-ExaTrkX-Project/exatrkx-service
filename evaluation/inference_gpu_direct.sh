#!/bin/bash
dir=${1:-/workspace/exatrkx_pipeline/datanmodels/lrt/inputs}
inference-gpu -m /workspace/exatrkx_pipeline/datanmodels -d ${dir}