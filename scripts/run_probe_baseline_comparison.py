"""
scripts/run_probe_baseline_comparison.py

Corrected probe analysis controlling for the untrained baseline.

CRITICAL FINDING (motivating this script): an untrained model with random
weights achieves high linear probe R² for position variables (MiniGrid
agent_x ~0.99, Physics pos_y ~0.46). This is because the observation token
encoding is linearly preserved through the residual stream via skip
connections, regardless of training. Raw probe R² therefore conflates:
  (a) trivially-decodable input structure (present even untrained)
  (b) genuinely learned representation (the quantity of interest)

This script computes the corrected metric (trained R² - untrained R²) and
runs a regularisation robustness check to distinguish two hypotheses:

  H1 (input preservation): high untrained R² reflects linear input structure
      in the residual stream. Under strong Ridge regularisation, untrained R²
      should remain high IF the structure is robust/low-dimensional, or
      collapse if it relies on high-dimensional probe overfitting.

  H2 (probe overfitting): high untrained R² is an artifact of fitting 256
      residual dimensions to a low-cardinality target. Under strong
      regularisation (alpha=100, 1000), untrained R² should collapse.

The comparison of trained vs untrained R² ACROSS regularisation strengths
tells us whether training makes position representations MORE ROBUST
(holding up under regularisation) even if not higher in raw R².

Usage:
    python scripts/run_probe_baseline_comparison.py \\
        --trained-checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --scale small --layer 5

    python scripts/run_probe_baseline_comparison.py \\
        --trained-checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small --layer 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.transformer import load_model, build_model
from interpretability.probes import ActivationCache, get_variable_types


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


POSITION_VARS = {
    "minigrid": ["agent_x", "agent_y", "goal_x", "goal_y"],
    "physics":  ["pos_x_0", "pos_y_0", "pos_x_1", "pos_y_1", "pos_x_2", "pos_y_2"],
}

ALPHAS = [1.0, 10.0, 100.0, 1000.0]


def probe_at_alpha(X, y, alpha, test_size=0.2, seed=42):
    """Fit a Ridge probe at a given regularisation strength, return test R²."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=test_size, random_state=seed)
    probe = Ridge(alpha=alpha)
    probe.fit(X_tr, y_tr)
    return r2_score(y_te, probe.predict(X_te))


def collect_activations(model, hdf5_path, layer, device, n_traj=400):
    cache = ActivationCache(model, device)
    activations, states = cache.extract(
        hdf5_path=hdf5_path, split="val",
        n_trajectories=n_traj, max_steps_per_traj=20,
    )
    return activations[layer], states


def main():
    parser = argparse.ArgumentParser(description="Probe baseline comparison + regularisation check")
    parser.add_argument("--trained-checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    hdf5_path = get_hdf5_path(args.env)

    # Load trained model
    print("Loading TRAINED model...")
    trained = load_model(args.trained_checkpoint, model_config, scale=config_key, device=str(device))
    trained.eval()

    # Build untrained model (same architecture, random weights)
    print("Building UNTRAINED model (random weights)...")
    untrained = build_model(model_config, scale=config_key).to(device)
    untrained.eval()

    variables = POSITION_VARS[args.env]

    print(f"\nCollecting activations at layer {args.layer} for both models...")
    X_trained, states = collect_activations(trained, hdf5_path, args.layer, device)
    X_untrained, _ = collect_activations(untrained, hdf5_path, args.layer, device)

    print(f"\n{'='*90}")
    print(f"REGULARISATION ROBUSTNESS CHECK — {args.env} layer {args.layer}")
    print(f"{'='*90}")
    print("For each variable: trained R² vs untrained R² across Ridge alphas.")
    print("Key question: does training make the representation more ROBUST to regularisation?\n")

    summary = {}
    for var in variables:
        y = states.get(var)
        if y is None or len(np.unique(y)) < 2:
            continue

        print(f"\n--- {var} ---")
        print(f"{'alpha':>8s} | {'trained R²':>11s} | {'untrained R²':>13s} | {'difference':>11s}")
        print("-" * 50)

        var_results = {}
        for alpha in ALPHAS:
            r2_tr = probe_at_alpha(X_trained, y, alpha)
            r2_un = probe_at_alpha(X_untrained, y, alpha)
            diff = r2_tr - r2_un
            var_results[alpha] = {"trained": r2_tr, "untrained": r2_un, "diff": diff}
            print(f"{alpha:8.0f} | {r2_tr:11.3f} | {r2_un:13.3f} | {diff:+11.3f}")

        summary[var] = var_results

    print(f"\n\n{'='*90}")
    print("INTERPRETATION")
    print(f"{'='*90}")
    print("""
At alpha=1 (weak reg): if untrained ~ trained, raw probe R² is dominated by
  input-preservation, NOT learned representation.

At alpha=1000 (strong reg): probe can only use robust, low-dimensional structure.
  - If trained R² >> untrained R² here: training built a ROBUST low-dimensional
    position code (genuine learned representation that survives regularisation).
  - If both collapse equally: the position signal is high-dimensional/fragile
    in both, consistent with input-preservation rather than a learned code.

The corrected 'genuine learned encoding' metric is (trained - untrained),
most meaningfully read at strong regularisation.
""")

    # Save
    import json
    out = {var: {str(a): summary[var][a] for a in ALPHAS} for var in summary}
    out_path = f"results/{args.env}_layer{args.layer}_baseline_comparison.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
