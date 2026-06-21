"""
scripts/inspect_sae_features.py

Verify what a specific SAE feature actually encodes, and report
dictionary health metrics (dead features, reconstruction quality).

This closes two gaps from the initial SAE pass:
  1. MI scores tell us a feature CORRELATES with a variable, not WHAT
     it represents. This script pulls the actual top-activating examples
     for a feature and checks them against concrete hypotheses
     (e.g. "does F867 encode room identity in FourRooms?").
  2. Reports dead features (never activate) and reconstruction quality
     in interpretable terms — standard SAE methodology reporting.

Usage:
    python scripts/inspect_sae_features.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --sae-checkpoint checkpoints/sae_minigrid_layer5.pt \\
        --env minigrid --feature 867
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from interpretability.sae import SAEConfig, SparseAutoencoder, collect_layer_activations
from models.transformer import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_hdf5_path(env: str) -> str:
    paths = {
        "minigrid": "data/trajectories/minigrid/minigrid.hdf5",
        "physics":  "data/trajectories/physics/physics.hdf5",
    }
    return paths[env]


def load_sae(sae_checkpoint_path: str, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(sae_checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    sae = SparseAutoencoder(config).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def infer_fourrooms_quadrant(x: int, y: int, grid_size: int = 19) -> str:
    """
    Classify (x, y) into FourRooms quadrant based on standard layout.
    FourRooms has a cross-shaped wall dividing it into 4 roughly equal rooms.
    Centre wall is approximately at grid_size // 2.
    """
    mid = grid_size // 2
    horizontal = "top" if y < mid else "bottom"
    vertical = "left" if x < mid else "right"
    return f"{horizontal}-{vertical}"


def report_dead_features(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    device: torch.device,
) -> dict:
    """
    Identify dead features (never activate above zero across the dataset)
    and report dictionary utilisation — standard SAE health metric.
    """
    sae.eval()
    with torch.no_grad():
        x = torch.from_numpy(activations).float().to(device)
        x_hat, features = sae(x)
        features_np = features.cpu().numpy()

        # Reconstruction quality: fraction of variance explained (like R^2)
        x_np = x.cpu().numpy()
        x_hat_np = x_hat.cpu().numpy()
        residual_var = np.var(x_np - x_hat_np, axis=0).sum()
        total_var = np.var(x_np, axis=0).sum()
        fvu = residual_var / total_var  # fraction of variance unexplained
        recon_r2 = 1.0 - fvu

    # Dead feature = never activates (max activation across all samples is 0)
    max_activation_per_feature = features_np.max(axis=0)  # (d_hidden,)
    dead_mask = max_activation_per_feature == 0.0
    n_dead = int(dead_mask.sum())
    n_total = features_np.shape[1]

    # Activation frequency: fraction of samples each feature fires on
    activation_freq = (features_np > 0).mean(axis=0)  # (d_hidden,)

    return {
        "n_dead_features": n_dead,
        "n_total_features": n_total,
        "dead_fraction": n_dead / n_total,
        "n_alive_features": n_total - n_dead,
        "reconstruction_r2": float(recon_r2),
        "mean_activation_freq_alive": float(activation_freq[~dead_mask].mean()) if n_dead < n_total else 0.0,
    }


def inspect_feature(
    feature_idx: int,
    sae: SparseAutoencoder,
    model,
    hdf5_path: str,
    layer_idx: int,
    device: torch.device,
    n_trajectories: int = 300,
    max_steps_per_traj: int = 30,
    top_n_examples: int = 20,
) -> None:
    """
    Find the top-activating examples for a specific feature and report
    their ground-truth state — lets us verify what the feature actually encodes.
    """
    print(f"\n=== Inspecting Feature F{feature_idx} ===")

    activations, states = collect_layer_activations(
        model=model, hdf5_path=hdf5_path, layer_idx=layer_idx, device=device,
        n_trajectories=n_trajectories, max_steps_per_traj=max_steps_per_traj,
    )

    sae.eval()
    with torch.no_grad():
        x = torch.from_numpy(activations).float().to(device)
        _, features = sae(x)
        feature_activations = features[:, feature_idx].cpu().numpy()  # (N,)

    # Get top-N activating examples
    top_indices = np.argsort(feature_activations)[::-1][:top_n_examples]

    print(f"\nTop {top_n_examples} activating examples for F{feature_idx}:")
    print(f"{'Activation':>10} | {'agent_x':>7} {'agent_y':>7} | {'Quadrant':>12}")
    print("-" * 50)

    quadrant_counts = {}
    for idx in top_indices:
        act = feature_activations[idx]
        ax = int(states["agent_x"][idx])
        ay = int(states["agent_y"][idx])
        quadrant = infer_fourrooms_quadrant(ax, ay)
        quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
        print(f"{act:10.3f} | {ax:7d} {ay:7d} | {quadrant:>12}")

    print(f"\nQuadrant distribution among top activations: {quadrant_counts}")

    # Test the room-identity hypothesis: if F867 encodes one specific quadrant,
    # top activations should overwhelmingly cluster in ONE quadrant
    if quadrant_counts:
        dominant_quadrant = max(quadrant_counts, key=quadrant_counts.get)
        dominant_fraction = quadrant_counts[dominant_quadrant] / top_n_examples
        print(f"\nDominant quadrant: {dominant_quadrant} "
              f"({dominant_fraction*100:.0f}% of top activations)")
        if dominant_fraction > 0.7:
            print("VERDICT: Strong evidence this feature encodes room/quadrant identity.")
        elif dominant_fraction > 0.4:
            print("VERDICT: Weak/partial evidence of quadrant correspondence.")
        else:
            print("VERDICT: No clear quadrant correspondence — feature likely encodes "
                  "something else (e.g. raw position magnitude, not room identity).")


def main():
    parser = argparse.ArgumentParser(description="Inspect SAE features and dictionary health")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Transformer checkpoint path")
    parser.add_argument("--sae-checkpoint", type=str, required=True,
                        help="SAE checkpoint path")
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature", type=int, default=None,
                        help="Specific feature index to inspect (e.g. 867)")
    parser.add_argument("--n-trajectories", type=int, default=300)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    model = load_model(args.checkpoint, model_config, scale=args.scale, device=str(device))
    sae = load_sae(args.sae_checkpoint, device)

    # Dictionary health check
    print("\n=== SAE Dictionary Health ===")
    activations, _ = collect_layer_activations(
        model=model, hdf5_path=get_hdf5_path(args.env), layer_idx=args.layer,
        device=device, n_trajectories=args.n_trajectories,
    )
    health = report_dead_features(sae, activations, device)
    print(f"Total features:        {health['n_total_features']}")
    print(f"Dead features:         {health['n_dead_features']} "
          f"({health['dead_fraction']*100:.1f}%)")
    print(f"Alive features:        {health['n_alive_features']}")
    print(f"Mean activation freq (alive features): "
          f"{health['mean_activation_freq_alive']*100:.2f}% of samples")
    print(f"Reconstruction R^2:    {health['reconstruction_r2']:.4f}")

    # Feature-specific inspection
    if args.feature is not None:
        inspect_feature(
            feature_idx=args.feature,
            sae=sae,
            model=model,
            hdf5_path=get_hdf5_path(args.env),
            layer_idx=args.layer,
            device=device,
            n_trajectories=args.n_trajectories,
        )


if __name__ == "__main__":
    main()
