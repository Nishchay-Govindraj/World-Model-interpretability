"""
scripts/run_untrained_baseline.py

Sanity check: run the full linear probe suite on an UNTRAINED model
with random weights. If probe scores are near zero for all variables,
this confirms that the representations found in the trained model
genuinely emerge from training, not from random initialization or
architectural inductive biases.

This is a basic but essential control that any frontier lab reviewer
would require. Without it, one cannot claim the representations are
"learned" rather than "coincidentally present."

Expected result: all R² scores near zero or negative for an untrained
model. Any variable with R² > 0.05 in the untrained model would indicate
architectural inductive bias that must be accounted for in the main results.

Usage:
    python scripts/run_untrained_baseline.py --env minigrid --scale small
    python scripts/run_untrained_baseline.py --env physics --scale small
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from interpretability.probes import run_probe_suite, results_to_matrix, get_variable_types
from models.transformer import build_model


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


def main():
    parser = argparse.ArgumentParser(description="Untrained model probe baseline")
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"\nBuilding UNTRAINED model with random weights...")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)

    # Build model with fresh random weights — deliberately NOT loading any checkpoint
    model = build_model(model_config, scale=config_key).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M parameters (random initialisation, no training)")

    hdf5_path = get_hdf5_path(args.env)
    variable_types = get_variable_types(hdf5_path)

    print(f"\nRunning probe suite on UNTRAINED model...")
    print("Expected: all scores near zero (confirming representations are learned)")
    print("If any variable scores > 0.05, it indicates architectural inductive bias.\n")

    results = run_probe_suite(
        model=model,
        hdf5_path=hdf5_path,
        device=device,
        n_trajectories=300,  # fewer since this is a control
        max_steps_per_traj=15,
    )

    print(f"\n=== Untrained Baseline Summary ===")
    print(f"{'Variable':20s} | {'Best Score':>10s} | {'Best Layer':>10s} | {'Flag':>6s}")
    print("-" * 60)

    variables = list(variable_types.keys())
    matrix, matrix_variables, _ = results_to_matrix(results)

    any_suspicious = False
    import numpy as np
    for j, var in enumerate(matrix_variables):
        col = matrix[:, j]
        if np.all(np.isnan(col)) or np.all(col == 0):
            print(f"{var:20s} | {'SKIPPED':>10s} | {'':>10s} | {'':>6s}")
            continue
        best_layer = int(np.nanargmax(col))
        best_score = col[best_layer]
        flag = "BIAS?" if best_score > 0.05 else "OK"
        if best_score > 0.05:
            any_suspicious = True
        print(f"{var:20s} | {best_score:10.3f} | {best_layer:10d} | {flag:>6s}")

    if any_suspicious:
        print("\nWARNING: Some variables show non-trivial scores in the untrained model.")
        print("These must be accounted for when interpreting trained model probe results.")
        print("The trained model's score MINUS the untrained baseline score = genuine learned R².")
    else:
        print("\nAll untrained scores near zero. ✓")
        print("Confirms that probe results in the trained model reflect genuinely learned representations.")

    # Save results
    import json
    baseline_results = {
        var: {
            "best_score": float(np.nanmax(matrix[:, j])),
            "best_layer": int(np.nanargmax(matrix[:, j])),
        }
        for j, var in enumerate(matrix_variables)
        if not (np.all(np.isnan(matrix[:, j])) or np.all(matrix[:, j] == 0))
    }
    out_path = f"results/{args.env}_untrained_baseline.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\nBaseline saved to: {out_path}")


if __name__ == "__main__":
    main()
