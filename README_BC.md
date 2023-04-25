# Diffusion Policy Sensorimotor Learning Project
## 🛠️ Installation
### 🖥️ Simulation
To reproduce our simulation benchmark results, install our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```

The `conda_environment_macos.yaml` file is only for development on MacOS and does not have full support for benchmarks.
## 🖥️ Reproducing Simulation Benchmark Results 
### Download ManiSkill Data
Under the repo root, create data subdirectory:
```console
[diffusion_policy]$ mkdir -p data && cd data
```

Download training data:
```console
[data]$ python -m mani_skill2.utils.download_demo all  
```
