"""
data/dataset.py

PyTorch Dataset for training the transformer world model.

MiniGrid — Observation Prediction Protocol:
  - Input:  flattened grid observation at step t (1083 cell tokens)
  - Target: flattened grid observation at step t+1 (next observation)
  - Each cell value is one integer token (vocab_size=32)
  - context_length = 1083 = one full observation
  - Windows slide one observation at a time across the trajectory
  - The model sees raw cell values and must predict the next grid state
  - World model structure emerges from this prediction objective alone

  This directly follows Li et al. (2023) Othello-GPT protocol:
  the model input never contains explicit state labels —
  position/direction/goal encoding must emerge internally.

Physics — Observation Prediction Protocol (future):
  - Input:  VQ-VAE token sequence from rendered frames
  - Target: next VQ-VAE token sequence

State variables are returned alongside each sample for probe training.
They are NEVER seen by the transformer during training.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Observation-level sliding window dataset.

    Each sample:
        tokens:  (context_length,) int64 — flattened observation at step t
        targets: (context_length,) int64 — flattened observation at step t+1
        states:  dict of var_name -> scalar tensor — ground-truth state at step t

    Args:
        hdf5_path:       path to HDF5 file from TrajectoryCollector
        split:           "train" or "val"
        context_length:  number of tokens per observation (1083 for FourRooms)
        stride_steps:    number of observations to skip between windows (default 1)
        state_vars:      state variable names to return (None = all)
        mode:            "observations" (MiniGrid) or "vqvae" (physics, future)
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        context_length: int = 1083,
        stride_steps: int = 1,
        state_vars: Optional[list[str]] = None,
        mode: str = "observations",
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"

        self.hdf5_path      = Path(hdf5_path)
        self.split          = split
        self.context_length = context_length
        self.stride_steps   = stride_steps
        self.mode           = mode

        self._windows: list[tuple[int, int]] = []  # (traj_idx, step_idx)
        self._traj_lengths: list[int] = []
        self._state_var_names: list[str] = []
        self._obs_flat_size: int = context_length

        self._build_index(state_vars)

    def _build_index(self, state_vars: Optional[list[str]]) -> None:
        """
        Build list of all valid (trajectory, step) windows.
        Each window = one observation at step t, target = observation at step t+1.
        Requires at least 2 steps per trajectory.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            split_grp = f[f"trajectories/{self.split}"]
            num_trajectories = len(split_grp)

            # Get state variable names from first trajectory
            first_traj = split_grp["0"]
            all_state_vars = list(first_traj["states"].keys())

            if state_vars is not None:
                missing = [v for v in state_vars if v not in all_state_vars]
                if missing:
                    raise ValueError(
                        f"State variables not found: {missing}\n"
                        f"Available: {all_state_vars}"
                    )
                self._state_var_names = state_vars
            else:
                self._state_var_names = all_state_vars

            # Get actual observation flat size from first trajectory
            first_obs = first_traj["observations"][0]
            self._obs_flat_size = int(np.prod(first_obs.shape))

            for traj_idx in range(num_trajectories):
                traj_grp = split_grp[str(traj_idx)]
                num_steps = int(traj_grp.attrs["length"])
                self._traj_lengths.append(num_steps)

                # Need at least 2 steps (current + next observation)
                if num_steps < 2:
                    continue

                # Each window is one step — slide by stride_steps
                for step in range(0, num_steps - 1, self.stride_steps):
                    self._windows.append((traj_idx, step))

        print(
            f"Dataset [{self.split}]: {len(self._windows):,} windows "
            f"from {len(self._traj_lengths):,} trajectories "
            f"(obs_flat_size={self._obs_flat_size}, stride={self.stride_steps})"
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one training sample.

        tokens:  flattened observation at step t    — (obs_flat_size,) int64
        targets: flattened observation at step t+1  — (obs_flat_size,) int64
        states:  ground-truth state at step t       — dict of scalar tensors
        """
        traj_idx, step = self._windows[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            traj_grp = f[f"trajectories/{self.split}/{traj_idx}"]

            # Load two consecutive observations
            obs_t   = traj_grp["observations"][step].flatten().astype(np.int64)
            obs_t1  = traj_grp["observations"][step + 1].flatten().astype(np.int64)

            # Ground-truth state at step t (probe targets — never seen by model)
            states_raw = {
                var: traj_grp[f"states/{var}"][step]
                for var in self._state_var_names
            }

        # Clamp to vocab range to prevent embedding index errors
        vocab_size = 32
        obs_t  = np.clip(obs_t,  0, vocab_size - 1)
        obs_t1 = np.clip(obs_t1, 0, vocab_size - 1)

        tokens  = torch.from_numpy(obs_t)   # (obs_flat_size,)
        targets = torch.from_numpy(obs_t1)  # (obs_flat_size,)

        states = {}
        for var, value in states_raw.items():
            if np.issubdtype(type(value), np.integer) or isinstance(value, (int, np.int32, np.int64)):
                states[var] = torch.tensor(int(value), dtype=torch.int64)
            else:
                states[var] = torch.tensor(float(value), dtype=torch.float32)

        return {"tokens": tokens, "targets": targets, "states": states}

    @property
    def state_variable_names(self) -> list[str]:
        return self._state_var_names

    @property
    def obs_flat_size(self) -> int:
        return self._obs_flat_size


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch samples — handles nested states dict."""
    tokens  = torch.stack([b["tokens"] for b in batch])   # (B, obs_flat_size)
    targets = torch.stack([b["targets"] for b in batch])  # (B, obs_flat_size)

    state_vars = batch[0]["states"].keys()
    states = {
        var: torch.stack([b["states"][var] for b in batch])  # (B,)
        for var in state_vars
    }
    return {"tokens": tokens, "targets": targets, "states": states}
