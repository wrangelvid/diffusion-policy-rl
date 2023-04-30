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

This will create a demos folder within data/ and download all the training data.

Here, ${ENV_ID} is the environment id of the environment you want to train on. For example, it could be `PickSingleEGAD-v0`

```console
[data]$ python -m mani_skill2.utils.download_demo ${ENV_ID} 
```
Then we need to run the trajectory replay script to generate the replay data. 

```console
[diffusion_policy]$  python -m mani_skill2.trajectory.replay_trajectory --traj-path data/demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj -obs-mode state --target-control-mode pd_ee_delta_pose --num-procs 30
```
Lastly, we need to convert the data to get the right format for training.
```console
[diffusion_policy]$ python maniskill_data_converter.py ./data/demos/rigid_body --observation_mode=state --control_mode=pd_ee_delta_p
ose --env_id=PickSingleEGAD-v0
```
### Training using state observation
`maniskill_state_diffusion_policy_cnn.yaml` is the config file for training using the `PickSingleEGAD-v0` environment.
```console
[diffusion_policy]$ python train.py --config-dir=. --config-name=maniskill_state_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
