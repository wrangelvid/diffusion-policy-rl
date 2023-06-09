import h5py
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
import numpy as np
import zarr
import click
import os


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


@click.command()
@click.option('--env_id',
              type=str,
              default='PickSingleEGAD-v0',
              help='maniskill2 environment id')
@click.option('--observation_mode',
              type=str,
              default='state',
              help='observation mode')
@click.option('--control_mode',
              type=str,
              default='pd_ee_delta_pose',
              help='control mode')
@click.option('--load_count',
              type=int,
              default=-1,
              help='number of episodes to load, -1 means all')
@click.argument('rootdir', type=click.Path(exists=True))
def convert_data(rootdir, env_id, observation_mode, control_mode, load_count):

    dataset_file = os.path.join(
        rootdir, f"{env_id}/trajectory.{observation_mode}.{control_mode}.h5")
    output_file = os.path.join(
        rootdir, f"{env_id}/trajectory.{observation_mode}.{control_mode}.zarr")

    data = h5py.File(dataset_file, "r")
    json_path = dataset_file.replace(".h5", ".json")
    json_data = load_json(json_path)
    episodes = json_data["episodes"]
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]

    observations = []
    actions = []
    # Marks one-past the last index for each episode
    episode_ends = [0]
    if load_count == -1:
        load_count = len(episodes)
    with click.progressbar(range(load_count),
                           label="Loading episodes",
                           empty_char='🥚',
                           fill_char='🐣') as pbar:
        for eps_id in pbar:
            eps = episodes[eps_id]
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            if observation_mode == "state":
                observations.append(trajectory["obs"][:-1])
            else:
                observations.append(trajectory["obs"])
            actions.append(trajectory["actions"])
            episode_ends.append(episode_ends[-1] + len(trajectory["actions"]))
        episode_ends = np.array(episode_ends[1:])

# All demonstration episodes are concatinated in the first dimension N
    if observation_mode == "state":
        zarr_data = {
            'meta': {
                'episode_ends': episode_ends
            },
            'data': {
                'action': np.vstack(actions).astype(np.float32),
                'obs': np.vstack(observations).astype(np.float32)
            }
        }

        click.echo(click.style("Saving zarr ...", fg="yellow"))
        # save the zarr data dictonary to a zarr file.
        z = zarr.open(output_file, 'w')
        z.create_group('meta')
        z.create_dataset('meta/episode_ends',
                         data=zarr_data['meta']['episode_ends'])
        z.create_group('data')
        z.create_dataset('data/action', data=zarr_data['data']['action'])
        z.create_dataset('data/obs', data=zarr_data['data']['obs'])

    if observation_mode == "rgbd":
        state_list = []
        rgbd_list = []

        with click.progressbar(observations,
                               label="Concatenating episodes",
                               empty_char='🐣',
                               fill_char='🐤') as pbar:
            for obs in pbar:
                # concatenate agent states
                state = np.hstack([
                    flatten_state_dict(obs["agent"])[:-1],
                    flatten_state_dict(obs["extra"])[:-1],
                ])
                state_list.append(state)
                images = []

                for cam_uid, cam_obs in obs["image"].items():
                    rgb = cam_obs["rgb"] / 255.0
                    depth = cam_obs["depth"]

                    images.append(rgb)
                    images.append(depth)

                # Concatenate all the images
                rgbd_list.append(np.einsum('nijk->nkij', np.concatenate(images, axis=-1)[:-1]))

        zarr_data = {
            'meta': {
                'episode_ends': episode_ends
            },
            'data': {
                'action': np.vstack(actions).astype(np.float32),
                'rgbd': np.vstack(rgbd_list).astype(np.float32),
                'state': np.vstack(state_list).astype(np.float32),
            }
        }
        click.echo(click.style("Saving zarr ...", fg="yellow"))

        # save the zarr data dictonary to a zarr file.
        z = zarr.open(output_file, 'w')
        z.create_group('meta')
        z.create_dataset('meta/episode_ends',
                         data=zarr_data['meta']['episode_ends'])
        z.create_group('data')
        z.create_dataset('data/rgbd', data=zarr_data['data']['rgbd'])
        z.create_dataset('data/state', data=zarr_data['data']['state'])
        z.create_dataset('data/action', data=zarr_data['data']['action'])

    click.echo(
        click.style(f"Saved {len(episode_ends)} episodes to {output_file}",
                    fg="green"))

if __name__ == "__main__":
    convert_data()
