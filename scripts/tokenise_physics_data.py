"""
scripts/tokenise_physics_data.py

Pre-tokenise the Physics Sandbox HDF5 dataset using the trained VQ-VAE,
caching the discrete token sequences to a new HDF5 file. This avoids
re-running the VQ-VAE encoder on every training step — tokenisation
happens once, then the transformer training script reads cached tokens
directly, exactly like MiniGrid's data/dataset.py already does.

Output HDF5 structure mirrors the original physics.hdf5 but replaces
'observations' (raw RGB frames) with 'observations' (VQ-VAE token
sequences, shape (T, 64) int64 instead of (T, 64, 64, 3) uint8).
Ground-truth states are copied through unchanged — they remain the
probe targets, never seen by the transformer.

Usage:
    python scripts/tokenise_physics_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
from tqdm import tqdm

from models.tokeniser import VQVAE, VQVAEConfig


def load_vqvae(checkpoint_path: str, device: torch.device) -> VQVAE:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vqvae = VQVAE(config).to(device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()
    return vqvae


def tokenise_split(
    vqvae: VQVAE,
    source_path: str,
    dest_file: h5py.File,
    split: str,
    device: torch.device,
    batch_size: int = 256,
) -> None:
    """
    Tokenise all frames in one split (train/val) and write to dest_file.
    Processes frames in batches across trajectories for GPU efficiency,
    then re-segments back into per-trajectory groups matching the
    original structure.
    """
    with h5py.File(source_path, "r") as src:
        src_split = src[f"trajectories/{split}"]
        n_traj = len(src_split)
        dest_split = dest_file.create_group(f"trajectories/{split}")

        for traj_idx in tqdm(range(n_traj), desc=f"Tokenising [{split}]"):
            traj_grp = src_split[str(traj_idx)]
            frames = traj_grp["observations"][:]  # (T, 64, 64, 3) uint8
            T = frames.shape[0]

            # Tokenise in batches to avoid VRAM overflow on long trajectories
            all_tokens = []
            for i in range(0, T, batch_size):
                batch = frames[i:i + batch_size].astype(np.float32) / 255.0
                batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)  # (B,3,64,64)
                tokens = vqvae.encode_to_tokens(batch_t)  # (B, 64)
                all_tokens.append(tokens.cpu().numpy())

            token_sequence = np.concatenate(all_tokens, axis=0)  # (T, 64)

            # Write new trajectory group with tokenised observations
            dest_traj = dest_split.create_group(str(traj_idx))
            dest_traj.create_dataset("observations", data=token_sequence, compression="gzip")
            dest_traj.create_dataset("actions", data=traj_grp["actions"][:])
            dest_traj.create_dataset("rewards", data=traj_grp["rewards"][:])
            dest_traj.create_dataset("dones", data=traj_grp["dones"][:])

            # Copy ground-truth states unchanged — these remain probe targets
            dest_states = dest_traj.create_group("states")
            for var_name in traj_grp["states"].keys():
                dest_states.create_dataset(var_name, data=traj_grp[f"states/{var_name}"][:])

            dest_traj.attrs["episode_seed"] = traj_grp.attrs["episode_seed"]
            dest_traj.attrs["total_reward"] = traj_grp.attrs["total_reward"]
            dest_traj.attrs["length"] = traj_grp.attrs["length"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vqvae_checkpoint = "checkpoints/vqvae_physics.pt"
    source_path = "data/trajectories/physics/physics.hdf5"
    dest_path = "data/trajectories/physics/physics_tokenised.hdf5"

    print(f"Loading VQ-VAE from {vqvae_checkpoint}")
    vqvae = load_vqvae(vqvae_checkpoint, device)

    print(f"\nTokenising {source_path} -> {dest_path}")

    with h5py.File(source_path, "r") as src:
        env_name = src.attrs.get("env_name", "PhysicsSandbox-v0")
        num_train = src.attrs.get("num_train")
        num_val = src.attrs.get("num_val")
        state_var_names = src.attrs.get("state_variable_names")
        action_space_size = src.attrs.get("action_space_size")

    with h5py.File(dest_path, "w") as dest:
        tokenise_split(vqvae, source_path, dest, "train", device)
        tokenise_split(vqvae, source_path, dest, "val", device)

        dest.attrs["env_name"] = env_name
        dest.attrs["num_train"] = num_train
        dest.attrs["num_val"] = num_val
        dest.attrs["state_variable_names"] = state_var_names
        dest.attrs["action_space_size"] = action_space_size
        dest.attrs["vocab_size"] = vqvae.config.num_embeddings
        dest.attrs["tokens_per_frame"] = vqvae.config.tokens_per_frame

    print(f"\nTokenisation complete. Saved to: {dest_path}")
    print(f"Tokens per frame: {vqvae.config.tokens_per_frame}")
    print(f"Vocab size: {vqvae.config.num_embeddings}")


if __name__ == "__main__":
    main()
