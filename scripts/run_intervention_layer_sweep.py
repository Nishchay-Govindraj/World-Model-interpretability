"""
scripts/run_intervention_layer_sweep.py

Sweep causal interventions (Mode C — full residual patch, local window)
across ALL layers for both environments.

This tests WHERE in the model's computation causal position representations
become load-bearing. If Mode C recovery is perfect at all layers, the
representation is built in early and maintained throughout. If it grows
across layers, it emerges progressively. If it peaks at specific layers
and then drops, later layers may transform or compress the representation.

This directly addresses the frontier lab question: not just "is position
represented" but "where in the computational graph does position information
become causally sufficient for prediction?"

Usage:
    python scripts/run_intervention_layer_sweep.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --scale small

    python scripts/run_intervention_layer_sweep.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from interpretability.interventions import run_intervention_mode_c, summarise_intervention_results
from models.transformer import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_hdf5_path(env: str) -> str:
    return {
        "minigrid": "data/trajectories/minigrid/minigrid.hdf5",
        "physics":  "data/trajectories/physics/physics_tokenised.hdf5",
    }[env]


def get_config_key(env: str, scale: str) -> str:
    return f"physics_{scale}" if env == "physics" else scale


VARIABLES = {
    "minigrid": ["agent_x", "agent_y"],
    "physics":  ["pos_x_0", "pos_y_0"],
}


def main():
    parser = argparse.ArgumentParser(description="Layer sweep for causal interventions (Mode C)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--n-pairs", type=int, default=80,
                        help="Pairs per variable per layer (lower since we run all layers)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))
    n_layers = model.config.n_layers

    hdf5_path = get_hdf5_path(args.env)
    variables = VARIABLES[args.env]

    print(f"\nLayer sweep: {n_layers} layers x {len(variables)} variables x {args.n_pairs} pairs")
    print(f"Mode: filtered_full (full residual patch, local window evaluation)\n")

    # results[var][layer] = recovery_fraction
    results = {var: {} for var in variables}

    for layer_idx in range(n_layers):
        print(f"\n--- Layer {layer_idx} ---")
        for var in variables:
            pairs = run_intervention_mode_c(
                model=model, hdf5_path=hdf5_path,
                layer_idx=layer_idx, variable=var,
                device=device, env=args.env,
                n_pairs=args.n_pairs,
            )
            summary = summarise_intervention_results(pairs)
            recovery = summary["mean_recovery_fraction"]
            results[var][layer_idx] = recovery
            print(f"  {var:16s}: recovery = {recovery:.3f} (n={summary['n_pairs']})")

    print(f"\n\n=== Layer Sweep Summary — Mode C Recovery Fraction ===")
    header = f"{'Layer':>6s}" + "".join(f" | {v:>12s}" for v in variables)
    print(header)
    print("-" * len(header))
    for layer_idx in range(n_layers):
        row = f"{layer_idx:>6d}"
        for var in variables:
            val = results[var].get(layer_idx, float("nan"))
            row += f" | {val:12.3f}"
        print(row)

    # Plot recovery by layer
    fig, ax = plt.subplots(figsize=(8, 5))
    for var in variables:
        ys = [results[var].get(l, float("nan")) for l in range(n_layers)]
        ax.plot(range(n_layers), ys, marker="o", label=var)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mode C Recovery Fraction")
    ax.set_title(f"Causal Recovery by Layer — {args.env} (Mode C: full residual)")
    ax.set_xticks(range(n_layers))
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.4, label="Full recovery")
    ax.axhline(0.0, color="grey", linestyle="-", alpha=0.3)
    plt.tight_layout()

    plot_path = f"results/{args.env}_layer_sweep_interventions.png"
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
