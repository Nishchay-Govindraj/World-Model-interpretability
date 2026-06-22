"""
scripts/run_probes.py

Run the linear probe suite on a trained checkpoint and produce
a layer x variable heatmap — the key dissertation figure for Track A.

Usage:
    python scripts/run_probes.py --checkpoint checkpoints/minigrid_small_step1000.pt --env minigrid

    # More trajectories for a more reliable estimate (slower)
    python scripts/run_probes.py --checkpoint checkpoints/minigrid_small_step5000.pt \\
        --env minigrid --n-trajectories 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

from interpretability.probes import run_probe_suite, results_to_matrix
from models.transformer import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_hdf5_path(env: str) -> str:
    paths = {
        "minigrid": "data/trajectories/minigrid/minigrid.hdf5",
        "physics":  "data/trajectories/physics/physics_tokenised.hdf5",
    }
    return paths[env]


def get_config_key(env: str, scale: str) -> str:
    return f"physics_{scale}" if env == "physics" else scale


def plot_heatmap(
    matrix: np.ndarray,
    variables: list[str],
    layers: list[str],
    output_path: str,
    title: str,
) -> None:
    """
    Plot and save the layer x variable probe score heatmap.

    Continuous variables (position) show R^2 (0 to 1, can be negative).
    Categorical variables (direction, carrying) show accuracy (0 to 1).
    Both are colour-scaled 0-1 for visual comparability, with R^2 clipped at 0.
    """
    matrix_clipped = np.clip(np.nan_to_num(matrix, nan=0.0), 0, 1)

    fig, ax = plt.subplots(figsize=(8, max(4, len(layers) * 0.6)))
    sns.heatmap(
        matrix_clipped,
        annot=matrix,            # show raw (unclipped, possibly NaN) values as text
        fmt=".2f",
        cmap="viridis",
        vmin=0, vmax=1,
        xticklabels=variables,
        yticklabels=layers,
        cbar_kws={"label": "Probe score (R\u00b2 / accuracy)"},
        mask=np.isnan(matrix),   # grey out degenerate (single-class) variables
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("State Variable")
    ax.set_ylabel("Transformer Layer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nHeatmap saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run linear probe suite")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--n-trajectories", type=int, default=500,
                        help="Number of held-out trajectories to sample for probing")
    parser.add_argument("--max-steps-per-traj", type=int, default=20,
                        help="Max steps sampled per trajectory")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))

    hdf5_path = get_hdf5_path(args.env)

    results = run_probe_suite(
        model=model,
        hdf5_path=hdf5_path,
        device=device,
        n_trajectories=args.n_trajectories,
        max_steps_per_traj=args.max_steps_per_traj,
    )

    matrix, variables, layers = results_to_matrix(results)

    output_path = f"results/{args.env}_probe_heatmap.png"
    plot_heatmap(
        matrix, variables, layers, output_path,
        title=f"Linear Probe Results — {args.env} ({args.scale} model)",
    )

    # Print summary: which variables are best/worst encoded, and at which layer
    print("\n=== Summary ===")
    for j, var in enumerate(variables):
        col = matrix[:, j]
        if np.all(np.isnan(col)):
            print(f"{var:20s}: SKIPPED (only one class present in dataset)")
            continue
        best_layer = int(np.nanargmax(col))
        best_score = col[best_layer]
        print(f"{var:20s}: best score {best_score:.3f} at layer {best_layer}")

    if not args.no_wandb:
        try:
            import wandb
            wandb_cfg = model_config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"probes_{args.env}_{args.scale}",
                tags=["track-a", "probes", args.env],
            )
            wandb.log({"probe_heatmap": wandb.Image(output_path)})
            for r in results:
                wandb.log({
                    f"probe/{r.variable}/layer_{r.layer}": r.score,
                })
            wandb.finish()
        except ImportError:
            print("wandb not installed — skipping logging")


if __name__ == "__main__":
    main()
