"""
scripts/train_sae.py

Train a sparse autoencoder on a specific layer's residual stream activations,
then analyse feature-to-variable correspondence via mutual information.

Usage:
    python scripts/train_sae.py --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 5

    # Adjust SAE hyperparameters
    python scripts/train_sae.py --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 5 --expansion-factor 16 --l1-coefficient 5e-4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from interpretability.sae import (
    SAEConfig, collect_layer_activations, train_sae, compute_feature_correspondence
)
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


def plot_feature_correspondence(
    correspondence: dict[str, list[tuple[int, float]]],
    output_path: str,
) -> None:
    """Bar chart of top mutual-information features per state variable."""
    variables = [v for v in correspondence if len(correspondence[v]) > 0]
    n_vars = len(variables)

    if n_vars == 0:
        print("No variables with valid feature correspondence to plot.")
        return

    fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 4), squeeze=False)
    axes = axes[0]

    for ax, var in zip(axes, variables):
        feature_ids = [f"F{idx}" for idx, _ in correspondence[var]]
        scores = [score for _, score in correspondence[var]]
        ax.barh(feature_ids[::-1], scores[::-1], color="steelblue")
        ax.set_title(var)
        ax.set_xlabel("Mutual Information")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nFeature correspondence plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train SAE on transformer residual stream")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True,
                        help="Which transformer layer's residual stream to train SAE on")
    parser.add_argument("--expansion-factor", type=int, default=8,
                        help="SAE dictionary size = expansion_factor * d_model")
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--n-trajectories", type=int, default=1000,
                        help="Number of held-out trajectories to collect activations from")
    parser.add_argument("--max-steps-per-traj", type=int, default=30)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    model = load_model(args.checkpoint, model_config, scale=args.scale, device=str(device))

    hdf5_path = get_hdf5_path(args.env)

    # Step 1: Collect activations from the target layer
    activations, states = collect_layer_activations(
        model=model,
        hdf5_path=hdf5_path,
        layer_idx=args.layer,
        device=device,
        n_trajectories=args.n_trajectories,
        max_steps_per_traj=args.max_steps_per_traj,
    )
    print(f"Collected {activations.shape[0]:,} activation samples "
          f"(d_model={activations.shape[1]})")

    # Step 2: Set up W&B if requested
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_cfg = model_config.get("wandb", {})
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"sae_{args.env}_layer{args.layer}",
                config={
                    "layer": args.layer,
                    "expansion_factor": args.expansion_factor,
                    "l1_coefficient": args.l1_coefficient,
                    "n_epochs": args.n_epochs,
                    "n_samples": activations.shape[0],
                },
                tags=["track-a", "sae", args.env],
            )
        except ImportError:
            print("wandb not installed — skipping logging")
            use_wandb = False

    # Step 3: Train the SAE
    sae_config = SAEConfig(
        d_model=activations.shape[1],
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient,
        n_epochs=args.n_epochs,
    )
    sae = train_sae(activations, sae_config, device, use_wandb=use_wandb, wandb_run=wandb_run)

    # Step 4: Save SAE checkpoint
    Path("checkpoints").mkdir(exist_ok=True)
    sae_path = f"checkpoints/sae_{args.env}_layer{args.layer}.pt"
    torch.save({
        "model_state_dict": sae.state_dict(),
        "config": sae_config,
        "layer": args.layer,
        "env": args.env,
    }, sae_path)
    print(f"SAE checkpoint saved: {sae_path}")

    # Step 5: Feature-to-variable correspondence analysis
    print("\n=== Feature-to-Variable Correspondence (Mutual Information) ===")
    correspondence = compute_feature_correspondence(sae, activations, states, device)

    for var, features in correspondence.items():
        if not features:
            print(f"{var:16s}: SKIPPED (degenerate variable)")
            continue
        top_feature, top_score = features[0]
        print(f"{var:16s}: top feature F{top_feature} (MI={top_score:.4f})")

    plot_path = f"results/{args.env}_layer{args.layer}_sae_correspondence.png"
    plot_feature_correspondence(correspondence, plot_path)

    if use_wandb and wandb_run is not None:
        wandb_run.log({"feature_correspondence": wandb.Image(plot_path)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
