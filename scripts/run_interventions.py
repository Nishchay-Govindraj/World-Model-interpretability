"""
scripts/run_interventions.py

Run causal intervention experiments: patch the linear probe direction
for each state variable and measure how much of the clean/corrupted
loss gap is recovered. This tests whether probed representations are
CAUSALLY involved in model predictions, not merely correlated.

Usage:
    python scripts/run_interventions.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 5

    # Test multiple layers for comparison
    python scripts/run_interventions.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from interpretability.interventions import (
    run_linear_direction_intervention, summarise_intervention_results
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


VARIABLES_TO_TEST = ["agent_x", "agent_y", "goal_x", "goal_y"]


def plot_recovery_fractions(summaries: dict, output_path: str) -> None:
    """Bar chart of mean recovery fraction per variable."""
    variables = list(summaries.keys())
    means = [summaries[v]["mean_recovery_fraction"] for v in variables]
    stds = [summaries[v]["std_recovery_fraction"] for v in variables]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue" if m > 0.3 else "lightcoral" for m in means]
    ax.bar(variables, means, yerr=stds, color=colors, capsize=5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Full recovery")
    ax.axhline(y=0.0, color="grey", linestyle="-", alpha=0.5, label="No effect")
    ax.set_ylabel("Mean Recovery Fraction")
    ax.set_title("Causal Intervention: Linear Probe Direction Patching")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run causal intervention experiments")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-pairs", type=int, default=150,
                        help="Number of clean/corrupted trajectory pairs per variable")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    model = load_model(args.checkpoint, model_config, scale=args.scale, device=str(device))

    hdf5_path = get_hdf5_path(args.env)

    print(f"\nRunning causal interventions on layer {args.layer}")
    print(f"Variables: {VARIABLES_TO_TEST}")
    print(f"Pairs per variable: {args.n_pairs}\n")

    summaries = {}
    for variable in VARIABLES_TO_TEST:
        print(f"\n--- Testing {variable} ---")
        results = run_linear_direction_intervention(
            model=model,
            hdf5_path=hdf5_path,
            layer_idx=args.layer,
            variable=variable,
            device=device,
            n_pairs=args.n_pairs,
        )
        summary = summarise_intervention_results(results)
        summaries[variable] = summary

        print(f"  Valid pairs tested:     {summary['n_pairs']}")
        print(f"  Mean recovery fraction: {summary['mean_recovery_fraction']:.3f} "
              f"(+/- {summary['std_recovery_fraction']:.3f})")
        print(f"  Mean clean loss:        {summary['mean_clean_loss']:.4f}")
        print(f"  Mean corrupted loss:    {summary['mean_corrupted_loss']:.4f}")
        print(f"  Mean patched loss:      {summary['mean_patched_loss']:.4f}")

    print("\n=== Summary: Causal Involvement by Variable ===")
    for var, summary in summaries.items():
        verdict = "STRONG causal evidence" if summary["mean_recovery_fraction"] > 0.6 else \
                  "MODERATE causal evidence" if summary["mean_recovery_fraction"] > 0.3 else \
                  "WEAK/NO causal evidence"
        print(f"{var:16s}: recovery={summary['mean_recovery_fraction']:.3f} -> {verdict}")

    plot_path = f"results/{args.env}_layer{args.layer}_interventions.png"
    plot_recovery_fractions(summaries, plot_path)

    if not args.no_wandb:
        try:
            import wandb
            wandb_cfg = model_config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"interventions_{args.env}_layer{args.layer}",
                tags=["track-a", "interventions", args.env],
            )
            for var, summary in summaries.items():
                wandb.log({f"intervention/{var}/recovery_fraction": summary["mean_recovery_fraction"]})
            wandb.log({"interventions_plot": wandb.Image(plot_path)})
            wandb.finish()
        except ImportError:
            print("wandb not installed — skipping logging")


if __name__ == "__main__":
    main()
