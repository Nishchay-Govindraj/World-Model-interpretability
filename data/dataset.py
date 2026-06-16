"""
data/dataset.py

PyTorch Dataset for training the transformer world model.

Reads trajectories from HDF5 files produced by TrajectoryCollector.
Produces fixed-length token sequences for next-token prediction.

Protocol (following Li et al., 2023):
  - Input:  token sequence of length context_length
  - Target: same sequence shifted by 1 (next token prediction)
  - The transformer learns to predict the next observation token.
    No state supervision — world model structure must emerge from
    the prediction objective alone.

State variables are also returned per-position so the probe training
code can directly index into Dataset samples without a second pass.

Usage:
    dataset = TrajectoryDataset(
        hdf5_path="data/trajectories/minigrid.hdf5",
        split="train",
        context_length=256,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    for batch in loader:
        tokens = batch["tokens"]      # (B, T) int64
        targets = batch["targets"]    # (B, T) int64  (tokens shifted by 1)
        states = batch["states"]      # dict: var_name -> (B, T) float32
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Sliding-window dataset over trajectory token sequences.

    Each sample is a window of length context_length drawn from a trajectory.
    Windows are sampled from positions where a full window fits.

    Args:
        hdf5_path:       path to the HDF5 file from TrajectoryCollector
        split:           "train" or "val"
        context_length:  token sequence length for the transformer
        stride:          step between successive windows (default: context_length // 2)
                         smaller stride = more windows per trajectory (data augmentation)
        state_vars:      list of state variable names to return.
                         None = return all variables in the dataset.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        context_length: int = 256,
        stride: Optional[int] = None,
        state_vars: Optional[list[str]] = None,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"

        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length // 2

        # Open file once to build the index; keep closed during training
        # (h5py file handles are not picklable, so we reopen in __getitem__)
        self._windows: list[tuple[int, int]] = []  # (trajectory_idx, start_pos)
        self._traj_lengths: list[int] = []
        self._state_var_names: list[str] = []

        self._build_index(state_vars)

    def _build_index(self, state_vars: Optional[list[str]]) -> None:
        """
        Scan HDF5 file to build a list of all valid (trajectory, start) windows.
        Called once at construction.
        """
        with h5py.File(self.hdf5_path, "r") as f:
            split_grp = f[f"trajectories/{self.split}"]
            num_trajectories = len(split_grp)

            # Determine state variable names from first trajectory
            first_traj = split_grp["0"]
            all_state_vars = list(first_traj["states"].keys())
            if state_vars is not None:
                missing = [v for v in state_vars if v not in all_state_vars]
                if missing:
                    raise ValueError(
                        f"Requested state variables not found in dataset: {missing}\n"
                        f"Available: {all_state_vars}"
                    )
                self._state_var_names = state_vars
            else:
                self._state_var_names = all_state_vars

            for traj_idx in range(num_trajectories):
                traj_grp = split_grp[str(traj_idx)]
                length = int(traj_grp.attrs["length"])
                self._traj_lengths.append(length)

                # Sliding windows: need context_length + 1 tokens (input + target)
                required = self.context_length + 1
                if length < required:
                    continue  # trajectory too short for even one window

                for start in range(0, length - required + 1, self.stride):
                    self._windows.append((traj_idx, start))

        print(
            f"Dataset [{self.split}]: {len(self._windows)} windows "
            f"from {len(self._traj_lengths)} trajectories "
            f"(context_length={self.context_length}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        """
        Return one training sample.

        Returns a dict with:
            tokens:  (context_length,) int64 — input token sequence
            targets: (context_length,) int64 — target sequence (tokens shifted +1)
            states:  dict of var_name -> (context_length,) float32 — ground-truth state
                     aligned with the INPUT token positions (not shifted)
        """
        traj_idx, start = self._windows[idx]
        end = start + self.context_length + 1

        # Reopen file per __getitem__ (required for multi-worker DataLoader)
        with h5py.File(self.hdf5_path, "r") as f:
            traj_grp = f[f"trajectories/{self.split}/{traj_idx}"]
            obs = traj_grp["observations"][start:end]   # (T+1, *obs_shape) or (T+1,)
            states_raw = {
                var: traj_grp[f"states/{var}"][start:end]
                for var in self._state_var_names
            }

        # For MiniGrid: obs is already (T+1,) int64
        # For Physics: obs is (T+1, H, W, C) uint8 — tokenised separately by VQ-VAE
        # At Dataset level we return raw observations; tokenisation happens in the model
        tokens = torch.from_numpy(obs[:-1].astype(np.int64))    # (T,)
        targets = torch.from_numpy(obs[1:].astype(np.int64))    # (T,) shifted by 1

        # State tensors aligned with input positions (not shifted)
        states = {}
        for var, values in states_raw.items():
            arr = values[:-1]  # align with token positions
            # Use float32 for continuous variables; int64 for categoricals
            if arr.dtype in [np.int32, np.int64, np.bool_]:
                states[var] = torch.from_numpy(arr.astype(np.int64))
            else:
                states[var] = torch.from_numpy(arr.astype(np.float32))

        return {
            "tokens": tokens,
            "targets": targets,
            "states": states,
        }

    @property
    def state_variable_names(self) -> list[str]:
        return self._state_var_names

    @property
    def vocab_size(self) -> int:
        """
        Infer vocab size from the dataset metadata.
        Returns max token value + 1 (assumes integer tokens starting at 0).
        """
        with h5py.File(self.hdf5_path, "r") as f:
            return int(f.attrs.get("vocab_size", 512))


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate to handle the nested 'states' dict.
    Stacks all tensors across the batch dimension.
    """
    tokens = torch.stack([b["tokens"] for b in batch])      # (B, T)
    targets = torch.stack([b["targets"] for b in batch])    # (B, T)

    # Collect all state variable names (consistent across batch)
    state_vars = batch[0]["states"].keys()
    states = {
        var: torch.stack([b["states"][var] for b in batch])  # (B, T)
        for var in state_vars
    }

    return {"tokens": tokens, "targets": targets, "states": states}
