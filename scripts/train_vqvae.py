"""
scripts/train_vqvae.py

Train the VQ-VAE tokeniser on Physics Sandbox observations.

This must be run BEFORE training the transformer world model on Physics,
since the transformer consumes VQ-VAE token sequences, not raw pixels.

Usage:
    python scripts/train_vqvae.py --no-wandb

    # Adjust training duration
    python scripts/train_vqvae.py --n-epochs 50 --no-wandb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.tokeniser import VQVAE, VQVAEConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PhysicsFrameDataset(Dataset):
    """
    Loads individual frames (not trajectories) from the Physics HDF5 file
    for VQ-VAE training. The VQ-VAE is trained per-frame, independent of
    sequence structure — it only needs to learn good frame reconstructions.
    """

    def __init__(self, hdf5_path: str, split: str = "train", max_frames: int = 50000):
        self.hdf5_path = hdf5_path
        self.split = split
        self._index: list[tuple[int, int]] = []  # (traj_idx, step)

        with h5py.File(hdf5_path, "r") as f:
            split_grp = f[f"trajectories/{split}"]
            n_traj = len(split_grp)

            for traj_idx in range(n_traj):
                traj_grp = split_grp[str(traj_idx)]
                num_steps = int(traj_grp.attrs["length"])
                for step in range(num_steps):
                    self._index.append((traj_idx, step))
                    if len(self._index) >= max_frames:
                        break
                if len(self._index) >= max_frames:
                    break

        print(f"PhysicsFrameDataset [{split}]: {len(self._index):,} frames")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        traj_idx, step = self._index[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            frame = f[f"trajectories/{self.split}/{traj_idx}/observations"][step]  # (H, W, 3) uint8

        # Normalise to [0, 1], convert to (C, H, W) for conv layers
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
        return frame


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE tokeniser for Physics Sandbox")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-frames", type=int, default=50000,
                        help="Max frames to use for training (subsampled from full dataset)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    physics_config = load_config(f"{args.config_dir}/physics_config.yaml")
    tok_cfg = physics_config["tokenisation"]

    vqvae_config = VQVAEConfig(
        obs_dim=tok_cfg["obs_dim"],
        latent_dim=tok_cfg["latent_dim"],
        num_embeddings=tok_cfg["num_embeddings"],
        commitment_cost=tok_cfg["commitment_cost"],
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
    )

    print(f"\nVQ-VAE config: obs_dim={vqvae_config.obs_dim}, "
          f"latent_dim={vqvae_config.latent_dim}, "
          f"num_embeddings={vqvae_config.num_embeddings}")
    print(f"Tokens per frame: {vqvae_config.tokens_per_frame} "
          f"({vqvae_config.latent_spatial_size}x{vqvae_config.latent_spatial_size})")

    hdf5_path = "data/trajectories/physics/physics.hdf5"
    train_dataset = PhysicsFrameDataset(hdf5_path, split="train", max_frames=args.max_frames)
    val_dataset = PhysicsFrameDataset(hdf5_path, split="val", max_frames=args.max_frames // 10)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    vqvae = VQVAE(vqvae_config).to(device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=args.learning_rate)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb_cfg = physics_config.get("wandb", {})
            wandb.init(
                project="world-model-interpretability",
                name="vqvae_physics",
                config=vars(args),
                tags=["track-a", "vqvae", "physics"],
            )
        except ImportError:
            print("wandb not installed — skipping logging")
            use_wandb = False

    print(f"\nTraining VQ-VAE for {args.n_epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(args.n_epochs):
        vqvae.train()
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        n_batches = 0

        for frames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs}"):
            frames = frames.to(device)

            x_hat, vq_loss, _ = vqvae(frames)
            recon_loss = F.mse_loss(x_hat, frames)
            total_loss = recon_loss + vq_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            n_batches += 1

        epoch_recon_loss /= n_batches
        epoch_vq_loss /= n_batches

        # Validation
        vqvae.eval()
        val_recon_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for frames in val_loader:
                frames = frames.to(device)
                x_hat, _, _ = vqvae(frames)
                val_recon_loss += F.mse_loss(x_hat, frames).item()
                n_val_batches += 1
        val_recon_loss /= max(n_val_batches, 1)

        print(f"Epoch {epoch+1:3d}/{args.n_epochs} | "
              f"train_recon={epoch_recon_loss:.5f} | "
              f"train_vq={epoch_vq_loss:.5f} | "
              f"val_recon={val_recon_loss:.5f}")

        if use_wandb:
            wandb.log({
                "vqvae/train_recon_loss": epoch_recon_loss,
                "vqvae/train_vq_loss": epoch_vq_loss,
                "vqvae/val_recon_loss": val_recon_loss,
            }, step=epoch)

        if val_recon_loss < best_val_loss:
            best_val_loss = val_recon_loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                "model_state_dict": vqvae.state_dict(),
                "config": vqvae_config,
                "epoch": epoch,
                "val_recon_loss": val_recon_loss,
            }, "checkpoints/vqvae_physics.pt")
            print(f"  Saved best checkpoint (val_recon_loss={val_recon_loss:.5f})")

    print(f"\nVQ-VAE training complete. Best val reconstruction loss: {best_val_loss:.5f}")
    print(f"Checkpoint: checkpoints/vqvae_physics.pt")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
