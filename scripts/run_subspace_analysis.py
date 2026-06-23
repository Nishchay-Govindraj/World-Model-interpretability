"""
scripts/run_subspace_analysis.py

Subspace dimensionality analysis for position representations.

The Mode B vs Mode C causal intervention gap is our central finding:
single linear direction → ~0 recovery; full residual stream → 1.000 recovery.

Two competing explanations:
  A. SUPERPOSITION: position is encoded in a high-dimensional subspace
     (many directions jointly carry the signal). A single probe direction
     captures only 1/k of the signal. The full residual preserves all k
     directions, explaining Mode C's perfect recovery.

  B. NON-LINEAR CODING: position is encoded via non-linear feature combinations.
     No linear subspace captures it well, regardless of dimensionality.

This script distinguishes A from B by measuring how many PCA dimensions
of the probe-residual-stream activations are needed to recover the position
signal. If A: 5-20 dimensions explain most variance and achieve good R².
If B: even 50+ dimensions fail to substantially outperform 1 dimension.

Also computes pairwise angles between position-encoding directions for
different variables (agent_x vs agent_y, pos_x_0 vs pos_y_0) to test
whether they are orthogonal (independent encoding) or aligned (shared subspace).

Usage:
    python scripts/run_subspace_analysis.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --scale small --layer 5

    python scripts/run_subspace_analysis.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small --layer 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.transformer import load_model
from interpretability.probes import get_variable_types, ActivationCache


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


VARIABLES_OF_INTEREST = {
    "minigrid": ["agent_x", "agent_y", "goal_x", "goal_y"],
    "physics":  ["pos_x_0", "pos_y_0", "pos_x_1", "pos_y_1"],
}


def pca_dimensionality_analysis(
    X: np.ndarray,
    y: np.ndarray,
    variable: str,
    max_components: int = 50,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    For k = 1, 2, 5, 10, 20, 50 PCA components, fit a Ridge probe and
    measure R². Shows how many dimensions are needed to capture the signal.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed
    )

    n_components_to_test = [k for k in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
                             if k <= min(X_train.shape)]

    results = {}
    for k in n_components_to_test:
        pca = PCA(n_components=k, random_state=seed)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        probe = Ridge(alpha=1.0)
        probe.fit(X_train_pca, y_train)
        score = r2_score(y_test, probe.predict(X_test_pca))
        variance_explained = pca.explained_variance_ratio_.sum()
        results[k] = {"r2": score, "var_explained": float(variance_explained)}

    # Also compute 1D probe in original space (the linear probe direction)
    probe_1d = Ridge(alpha=1.0)
    probe_1d.fit(X_train, y_train)
    score_1d = r2_score(y_test, probe_1d.predict(X_test))
    results["full_d"] = {"r2": score_1d, "var_explained": 1.0}

    return results


def compute_direction_angles(
    X: np.ndarray,
    states: dict,
    variables: list[str],
) -> np.ndarray:
    """
    Compute pairwise cosine angles between probe directions for each variable.
    Returns a (n_vars, n_vars) matrix of angles in degrees.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    directions = {}
    for var in variables:
        y = states.get(var)
        if y is None or len(np.unique(y)) < 2:
            continue
        # Use alpha=10 (not 1.0) to avoid the ill-conditioned matrix warning
        # from near-singular residual stream covariance. The direction is
        # robust to this; regularisation only stabilises the linear solve.
        probe = Ridge(alpha=10.0)
        probe.fit(X_scaled, y)
        d = probe.coef_
        directions[var] = d / (np.linalg.norm(d) + 1e-9)

    vars_with_directions = list(directions.keys())
    n = len(vars_with_directions)
    angles = np.zeros((n, n))

    for i, v1 in enumerate(vars_with_directions):
        for j, v2 in enumerate(vars_with_directions):
            cos_sim = np.clip(directions[v1] @ directions[v2], -1.0, 1.0)
            angles[i, j] = np.degrees(np.arccos(abs(cos_sim)))

    return angles, vars_with_directions


def plot_pca_curves(pca_results: dict, output_path: str, env: str, layer: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for var, results in pca_results.items():
        ks = sorted([k for k in results.keys() if isinstance(k, int)])
        r2s = [results[k]["r2"] for k in ks]
        axes[0].plot(ks, r2s, marker="o", label=var)
        var_exps = [results[k]["var_explained"] for k in ks]
        axes[1].plot(ks, var_exps, marker="o", label=var)

    axes[0].set_xlabel("Number of PCA components")
    axes[0].set_ylabel("Probe R²")
    axes[0].set_title("Probe R² vs PCA dimensionality\n(how many dimensions carry the signal?)")
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log")

    axes[1].set_xlabel("Number of PCA components")
    axes[1].set_ylabel("Variance explained")
    axes[1].set_title("PCA variance explained by k components")
    axes[1].legend(fontsize=8)
    axes[1].set_xscale("log")

    plt.suptitle(f"Subspace Dimensionality Analysis — {env} Layer {layer}", fontsize=13)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Subspace dimensionality analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))

    hdf5_path = get_hdf5_path(args.env)
    variables = VARIABLES_OF_INTEREST[args.env]

    print(f"\nCollecting activations at layer {args.layer}...")
    cache = ActivationCache(model, device)
    activations, states = cache.extract(
        hdf5_path=hdf5_path, split="val",
        n_trajectories=500, max_steps_per_traj=20,
    )
    X = activations[args.layer]  # (N, d_model)

    print(f"\nRunning PCA dimensionality analysis for {variables}...")
    pca_results = {}
    for var in variables:
        y = states.get(var)
        if y is None or len(np.unique(y)) < 2:
            continue
        print(f"  {var}...")
        pca_results[var] = pca_dimensionality_analysis(X, y, var)

    print(f"\n=== PCA Dimensionality Analysis ===")
    print(f"R² achieved at each number of PCA components:")
    header = f"{'Variable':16s}" + "".join(f" | k={k:2d}" for k in [1,2,3,5,8,10,20,50])
    print(header)
    print("-" * len(header))
    for var, res in pca_results.items():
        row = f"{var:16s}"
        for k in [1, 2, 3, 5, 8, 10, 20, 50]:
            if k in res:
                row += f" | {res[k]['r2']:5.3f}"
            else:
                row += f" |  N/A "
        print(row)

    print(f"\n=== Direction Angle Analysis ===")
    print("Pairwise angles between probe directions (degrees).")
    print("~90° = orthogonal (independent encoding)")
    print("~0°  = aligned (shared subspace)")
    angles, vars_with_dirs = compute_direction_angles(X, states, variables)
    header = f"{'':16s}" + "".join(f" | {v[:8]:>8s}" for v in vars_with_dirs)
    print(header)
    for i, v in enumerate(vars_with_dirs):
        row = f"{v:16s}" + "".join(f" | {angles[i,j]:8.1f}" for j in range(len(vars_with_dirs)))
        print(row)

    plot_path = f"results/{args.env}_layer{args.layer}_subspace_analysis.png"
    plot_pca_curves(pca_results, plot_path, args.env, args.layer)


if __name__ == "__main__":
    main()
