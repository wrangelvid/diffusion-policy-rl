# By default, we build without Cuda drivers
FROM mambaorg/micromamba:1.4.2-focal

# Use instead the following line to build with Cuda drivers
# (should only be used if intending to run Docker image on system
# with Nvidia GPUs)
#FROM mambaorg/micromamba:git-3208378-focal-cuda-11.6.2

# See https://github.com/mamba-org/micromamba-docker#quick-start
# for details on using Micromamba in Docker

USER root
RUN apt-get update && apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

USER $MAMBA_USER

RUN mkdir src 
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_environment_docker.yaml /src/conda_environment_docker.yaml
RUN micromamba install -y -n base -f /src/conda_environment_docker.yaml && \
  micromamba clean --all --yes

# Activate mamba environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1  

# Install python dependencies
RUN pip install mani_skill2

# Copy over all source code
WORKDIR /src/
COPY . /src/

# Set environment for RL
ENV ENV_ID PickSingleEGAD-v0
