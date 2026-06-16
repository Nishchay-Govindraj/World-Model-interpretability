"""
data/collector.py

Trajectory collection pipeline.

Handles:
  - Parallel or sequential trajectory collection from any BaseEnvironment
  - HDF5 storage (efficient for large numeric arrays)
  - Progress logging to W&B
  - Deterministic seeding (each trajectory gets a unique seed derived from base seed)
  - Train/validation split

HDF5 file structure:
    trajectories/
        {split}/          (train | val)
            {i}/          (trajectory index)
                observations   (T, *obs_shape) uint8 or int64
                actions        (T,) int64
                rewards        (T,) float32
                dones          (T,) bool
                states/
                    {var_name} (T,) float32 or int64

Usage:
    collector = TrajectoryCollector(env, config, output_path="data/trajectories/minigrid.hdf5")
    collector.collect(num_trajectories=200_000, policy=random_policy)
"""

import time
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from tqdm import tqdm

from environments.base_env import BaseEnvironment, Trajectory


def random_policy(observation: np.ndarray, action_space_size: int) -> int:
    """Uniform random policy — baseline for data collection."""
    return np.random.randint(0, action_space_size)


class TrajectoryCollector:
    """
    Collects trajectories from an environment and saves them to HDF5.

    Args:
        env:              BaseEnvironment instance (MiniGridEnv or PhysicsEnv)
        config:           full config dict (from YAML)
        output_path:      path to the HDF5 output file
        use_wandb:        whether to log collection progress to W&B
    """

    def __init__(
        self,
        env: BaseEnvironment,
        config: dict,
        output_path: str,
        use_wandb: bool = True,
    ):
        self.env = env
        self.config = config
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self._wandb = None

        if use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb
            wandb.init(
                project=self.config.get("wandb", {}).get("project", "world-model-interpretability"),
                name=f"data-collection-{self.env.env_name}",
                config={
                    "env": self.env.env_name,
                    "obs_shape": self.env.observation_shape,
                    "action_space_size": self.env.action_space_size,
                    "state_variables": self.env.state_variable_names,
                },
                tags=["data-collection"],
            )
            self._wandb = wandb
        except ImportError:
            print("W&B not installed — skipping logging. Run: pip install wandb")
            self.use_wandb = False

    def collect(
        self,
        num_trajectories: int,
        policy: Callable | None = None,
        max_steps: int | None = None,
        base_seed: int = 0,
        validation_split: float = 0.1,
    ) -> dict:
        """
        Collect trajectories and write to HDF5.

        Args:
            num_trajectories: total number of episodes to collect
            policy:           callable(obs, action_space_size) -> int
                              defaults to uniform random
            max_steps:        max steps per trajectory (overrides config)
            base_seed:        seed for trajectory i = base_seed + i
            validation_split: fraction of trajectories held out for val set

        Returns:
            dict with collection statistics
        """
        if policy is None:
            policy = lambda obs: random_policy(obs, self.env.action_space_size)

        if max_steps is None:
            max_steps = self.config["data_collection"].get(
                "max_steps_per_episode",
                self.config["data_collection"].get("steps_per_trajectory", 500),
            )

        num_val = int(num_trajectories * validation_split)
        num_train = num_trajectories - num_val

        stats = {
            "total_trajectories": num_trajectories,
            "train": num_train,
            "val": num_val,
            "mean_episode_length": 0.0,
            "mean_total_reward": 0.0,
        }

        episode_lengths = []
        episode_rewards = []
        start_time = time.time()

        with h5py.File(self.output_path, "w") as f:
            train_grp = f.create_group("trajectories/train")
            val_grp = f.create_group("trajectories/val")

            pbar = tqdm(range(num_trajectories), desc=f"Collecting {self.env.env_name}")

            for i in pbar:
                episode_seed = base_seed + i
                split = "val" if i < num_val else "train"
                grp = val_grp if split == "val" else train_grp
                traj_idx = i if split == "val" else i - num_val

                trajectory = self.env.collect_trajectory(
                    policy=policy,
                    max_steps=max_steps,
                    episode_seed=episode_seed,
                )

                self._write_trajectory(grp, traj_idx, trajectory)

                episode_lengths.append(len(trajectory))
                episode_rewards.append(trajectory.total_reward)

                # Log to W&B every 500 trajectories
                if self.use_wandb and self._wandb and i % 500 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    self._wandb.log({
                        "collected": i,
                        "mean_episode_length": np.mean(episode_lengths[-500:]),
                        "mean_episode_reward": np.mean(episode_rewards[-500:]),
                        "trajectories_per_second": i / elapsed,
                    })
                    pbar.set_postfix({
                        "mean_len": f"{np.mean(episode_lengths[-500:]):.1f}",
                        "t/s": f"{i / elapsed:.1f}",
                    })

            # Write metadata
            f.attrs["env_name"] = self.env.env_name
            f.attrs["num_train"] = num_train
            f.attrs["num_val"] = num_val
            f.attrs["state_variable_names"] = self.env.state_variable_names
            f.attrs["obs_shape"] = list(self.env.observation_shape)
            f.attrs["action_space_size"] = self.env.action_space_size

        stats["mean_episode_length"] = float(np.mean(episode_lengths))
        stats["mean_total_reward"] = float(np.mean(episode_rewards))

        elapsed = time.time() - start_time
        print(f"\nCollection complete in {elapsed:.1f}s")
        print(f"  Trajectories: {num_train} train / {num_val} val")
        print(f"  Mean episode length: {stats['mean_episode_length']:.1f} steps")
        print(f"  Mean total reward:   {stats['mean_total_reward']:.3f}")
        print(f"  Saved to: {self.output_path}")

        if self.use_wandb and self._wandb:
            self._wandb.log(stats)
            self._wandb.finish()

        return stats

    def _write_trajectory(
        self,
        group: h5py.Group,
        idx: int,
        trajectory: Trajectory,
    ) -> None:
        """Write a single Trajectory to an HDF5 group."""
        traj_grp = group.create_group(str(idx))

        # Observations: int64 for MiniGrid (token IDs), uint8 for physics (pixels)
        obs = trajectory.observations
        if obs.dtype == np.int64:
            traj_grp.create_dataset("observations", data=obs, compression="gzip")
        else:
            traj_grp.create_dataset(
                "observations", data=obs, dtype=np.uint8, compression="gzip"
            )

        traj_grp.create_dataset("actions", data=trajectory.actions.astype(np.int64))
        traj_grp.create_dataset(
            "rewards",
            data=np.array([t.reward for t in trajectory.timesteps], dtype=np.float32),
        )
        traj_grp.create_dataset(
            "dones",
            data=np.array([t.done for t in trajectory.timesteps], dtype=bool),
        )

        # Ground-truth state variables — stored per variable for fast probe access
        state_grp = traj_grp.create_group("states")
        states = trajectory.states
        for var_name, values in states.items():
            dtype = np.int64 if values.dtype in [np.int32, np.int64] else np.float32
            state_grp.create_dataset(var_name, data=values.astype(dtype))

        traj_grp.attrs["episode_seed"] = trajectory.episode_seed
        traj_grp.attrs["total_reward"] = trajectory.total_reward
        traj_grp.attrs["length"] = len(trajectory)
