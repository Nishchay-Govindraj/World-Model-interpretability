"""
scripts/run_interventions.py

Run all three causal intervention modes (last-position, agent-cell-position,
filtered-full-patch) for each state variable and compare results side by side.

Usage:
    python scripts/run_interventions.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 5
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
    run_intervention_mode_a_or_b, run_intervention_mode_c, summarise_intervention_results
)
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


MINIGRID_VARIABLES = ["agent_x", "agent_y", "goal_x", "goal_y"]
PHYSICS_VARIABLES  = ["pos_x_0", "pos_y_0", "pos_x_1", "pos_y_1"]  # position variables only
MODES = ["last", "agent_cell", "filtered_full"]


def plot_comparison(all_summaries: dict, variables: list, output_path: str) -> None:
    """Grouped bar chart: variable x mode comparison of recovery fractions."""
    x = np.arange(len(variables))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(MODES):
        means = [all_summaries[mode][v]["mean_recovery_fraction"] for v in variables]
        means_clean = [0.0 if np.isnan(m) else m for m in means]  # plot NaN as 0 with annotation
        bars = ax.bar(x + i * width, means_clean, width, label=mode)
        for j, m in enumerate(means):
            if np.isnan(m):
                ax.annotate("N/A", (x[j] + i * width, 0.02), ha="center", fontsize=8, color="red")

    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, label="Full recovery")
    ax.axhline(y=0.0, color="grey", linestyle="-", alpha=0.4)
    ax.set_xticks(x + width)
    ax.set_xticklabels(variables)
    ax.set_ylabel("Mean Recovery Fraction")
    ax.set_title("Causal Intervention Comparison Across Three Methodologies")
    ax.legend()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nComparison plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run all causal intervention modes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Pairs per variable per mode (reduced default since we now run 3x)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))
    hdf5_path = get_hdf5_path(args.env)
    variables_to_test = PHYSICS_VARIABLES if args.env == "physics" else MINIGRID_VARIABLES
    env = args.env

    print(f"\nRunning causal interventions on layer {args.layer}")
    print(f"Variables: {variables_to_test}")
    print(f"Modes: {MODES}")
    print(f"Pairs per variable per mode: {args.n_pairs}\n")

    all_summaries = {mode: {} for mode in MODES}

    for variable in variables_to_test:
        print(f"\n{'='*60}")
        print(f"VARIABLE: {variable}")
        print('='*60)

        for mode in ["last", "agent_cell"]:
            print(f"\n--- Mode: {mode} ---")
            results = run_intervention_mode_a_or_b(
                model=model, hdf5_path=hdf5_path, layer_idx=args.layer,
                variable=variable, device=device, mode=mode, env=env,
                n_pairs=args.n_pairs,
            )
            summary = summarise_intervention_results(results)
            all_summaries[mode][variable] = summary
            if summary["n_pairs"] == 0:
                print(f"  NO VALID PAIRS — see [diagnostic] line above for why "
                      f"(likely: target position's value is position-invariant)")
            else:
                print(f"  Valid pairs: {summary['n_pairs']} | "
                      f"Recovery: {summary['mean_recovery_fraction']:.3f} "
                      f"(+/- {summary['std_recovery_fraction']:.3f}) | "
                      f"Relative patch size: {summary['relative_patch_size']*100:.1f}%")

        print(f"\n--- Mode: filtered_full ---")
        results_c = run_intervention_mode_c(
            model=model, hdf5_path=hdf5_path, layer_idx=args.layer,
            variable=variable, device=device, env=env, n_pairs=args.n_pairs,
        )
        summary_c = summarise_intervention_results(results_c)
        all_summaries["filtered_full"][variable] = summary_c
        if summary_c["n_pairs"] == 0:
            print(f"  NO VALID PAIRS — see [diagnostic] line above for why")
        else:
            print(f"  Valid pairs: {summary_c['n_pairs']} | "
                  f"Recovery: {summary_c['mean_recovery_fraction']:.3f} "
                  f"(+/- {summary_c['std_recovery_fraction']:.3f})")

    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY — Recovery Fraction by Variable and Mode")
    print('='*60)
    print(f"{'Variable':16s} | {'last':>10s} | {'agent_cell':>10s} | {'filtered_full':>14s}")
    print("-" * 60)
    for var in variables_to_test:
        row = f"{var:16s} |"
        for mode in MODES:
            val = all_summaries[mode][var]["mean_recovery_fraction"]
            val_str = "N/A" if np.isnan(val) else f"{val:.3f}"
            row += f" {val_str:>10s} |" if mode != "filtered_full" else f" {val_str:>14s}"
        print(row)

    plot_path = f"results/{args.env}_layer{args.layer}_interventions_comparison.png"
    plot_comparison(all_summaries, plot_path)

    if not args.no_wandb:
        try:
            import wandb
            wandb_cfg = model_config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"interventions_v2_{args.env}_layer{args.layer}",
                tags=["track-a", "interventions", args.env],
            )
            for mode in MODES:
                for var in variables_to_test:
                    wandb.log({f"intervention/{mode}/{var}": all_summaries[mode][var]["mean_recovery_fraction"]})
            wandb.log({"interventions_comparison": wandb.Image(plot_path)})
            wandb.finish()
        except ImportError:
            print("wandb not installed — skipping logging")


if __name__ == "__main__":
    main()
