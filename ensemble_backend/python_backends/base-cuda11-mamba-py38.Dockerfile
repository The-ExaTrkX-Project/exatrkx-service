#Taken from https://github.com/rapidsai/gpuci-build-environment/blob/branch-23.06/miniforge-cuda/Dockerfile
#& https://gitlab.com/NERSC/nersc-official-images/-/blob/main/nersc/cupy/11.5/Containerfile

ARG FROM_IMAGE=nvidia/cuda
ARG CUDA_VER=11.8.0
ARG IMAGE_TYPE=devel
ARG LINUX_VER=ubuntu20.04
FROM ${FROM_IMAGE}:${CUDA_VER}-cudnn8-${IMAGE_TYPE}-${LINUX_VER}

# Define versions and download locations
ARG ARCH_TYPE="x86_64"
ARG MINICONDA_VERSION=23.3.1-0
ARG PYTHON_VERSION=py38
ARG MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${PYTHON_VERSION}_${MINICONDA_VERSION}-Linux-${ARCH_TYPE}.sh"
ENV MINICONDA_ROOT_PREFIX="/opt/conda"

# Set environment
ENV PATH=${MINICONDA_ROOT_PREFIX}/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA_VERSION as in some 'nvidia/cuda' images this is not set
## A lot of scripts and conda recipes depend on this env var
ENV CUDA_VERSION=${CUDA_VER}

# Add the right libraries for Ubuntu
RUN apt-get update \
 && apt-get upgrade --yes \
 && apt-get install --yes --no-install-recommends \
      build-essential      \
      ninja-build          \
      git                  \
      wget                 \
 && apt-get clean all \
 && rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget --quiet ${MINICONDA_URL} -O conda_installer.sh && \
    /bin/bash conda_installer.sh -b -p ${MINICONDA_ROOT_PREFIX} && \
    rm -rf conda_installer.sh

# Install Mamba + conda-pack + clean-up
RUN conda install mamba conda-pack -c conda-forge -y \
 && mamba clean -itcly
