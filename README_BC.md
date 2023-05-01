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
##  Reproducing State Based Results 
### Download ManiSkill Data
Under the repo root, create data subdirectory:
```console
[diffusion_policy]$ mkdir -p data && cd data
```

Download training data:

This will create a demos folder within data/ and download all the training data.

Here, ${ENV_ID} is the environment id of the environment you want to train on. For example, it could be `PickSingleEGAD-v0`
```console
[data]$ export ENV_ID="PickSingleEGAD-v0"
```

To download the assets and demonstrations run the following commands:
```console
[data]$ python -m mani_skill2.utils.download_asset ${ENV_ID}
[data]$ python -m mani_skill2.utils.download_demo ${ENV_ID} 
```
Then we need to run the trajectory replay script to generate the replay data. 

```console
[diffusion_policy]$  python -m mani_skill2.trajectory.replay_trajectory --traj-path data/demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode state --target-control-mode pd_ee_delta_pose --num-procs 30
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
##  Reproducing Hybrid Based Results 
The observations have the following structure:
```json
{
    "state": [2*nq + 7], qpos, qvel, base_pose concatenated togther
    "rgbd": [8, H, W], rgd_1, depth_1, rgd_2, depth_2, stacked together. Here we have base_camera and hand_camera. This may be adjusted for different environments.
}
```
Download training data:

This will create a demos folder within data/ and download all the training data.

Here, ${ENV_ID} is the environment id of the environment you want to train on. For example, it could be `PickSingleEGAD-v0`

```console
[data]$ python -m mani_skill2.utils.download_demo ${ENV_ID} 
```
Then we need to run the trajectory replay script to generate the replay data. 

```console
[diffusion_policy]$  python -m mani_skill2.trajectory.replay_trajectory --traj-path data/demos/rigid_body/PickSingleEGAD-v0/trajectory.h5 --save-traj --obs-mode state --target-control-mode pd_ee_delta_pose --num-procs 30
```
Lastly, we need to convert the data to get the right format for training.
```console
[diffusion_policy]$ python maniskill_data_converter.py ./data/demos/rigid_body --observation_mode=rgbd --control_mode=pd_ee_delta_p
ose --env_id=PickSingleEGAD-v0
```
### Training using state observation
`maniskill_rgbd_diffusion_policy_cnn.yaml` is the config file for training using the `PickSingleEGAD-v0` environment.
```console
[diffusion_policy]$ python train.py --config-dir=. --config-name=maniskill_rgbd_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/PickSingleEGAD-v0/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

## Running in Docker
The docker image can be built either with or without Cuda drivers. By default, it is built without, to be able to run on devices without NVidia GPUs. To enable cuda, see the Dockerfile for details.

To build the Docker image, run
```console
docker build . --tag diffusion --platform linux/amd64
```
(this will probably take ~10 minutes the first time).

Next, we want to run the Docker file, and mount our local data folder so that the Docker image can create and manipulate the data folder.
First, navigate to this repo on your local system, and make sure to create a folder for data:
```console
mkdir -p data/demos
```
Next, let us run the docker image in interactive mode, while mounting our `data` folder as a volume in the image:
```console
docker run --platform linux/amd64 -v $PWD/data:/src/data/ -it diffusion 
```
(now, the Docker image will be able to save data between runs to `data`)

You should now be able to run code in the interactive Docker shell. For instance, try downloading the demo and asset data for the current environment `ENV_ID` (change by setting i.e. `export ENV_ID="PickSingleEGAD-v0"`):
```
python -m mani_skill2.utils.download_demo ${ENV_ID} -o data/demos
python -m mani_skill2.utils.download_asset ${ENV_ID} -o data
```

In theory, it should also work to replay trajectories through Docker. However, currently, the following seems to be very slow
```
python -m mani_skill2.trajectory.replay_trajectory --traj-path data/demos/rigid_body/${ENV_ID}/trajectory.h5 --save-traj --obs-mode state --target-control-mode pd_ee_delta_pose
```
