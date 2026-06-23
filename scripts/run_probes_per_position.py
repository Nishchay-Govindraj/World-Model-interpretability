"""
scripts/run_probes_per_position.py

Per-position linear probe pipeline for Physics Sandbox.

Addresses a known methodological limitation of the mean-pooled probe approach:
in Physics, object positions are encoded in the SPATIAL ARRANGEMENT of VQ-VAE
tokens across the 8x8 grid, not in any single token's value. Mean-pooling
the residual stream across all 64 spatial positions dilutes this signal.

This script probes the residual stream at EACH of the 64 spatial positions
independently, then reports the BEST score per variable across positions.
This tests whether position information exists strongly at specific spatial
locations (e.g. the token corresponding to where the object actually is)
even if it's invisible in the mean-pooled representation.

For MiniGrid we also run per-position probing for comparison — this tests
whether agent_x/agent_y are even more strongly encoded at the specific
cell position than in the mean-pooled representation.

Usage:
    python scripts/run_probes_per_position.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small --layer 2

    python scripts/run_probes_per_position.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --scale small --layer 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.transformer import load_model
from interpretability.probes import get_variable_types


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


def collect_per_position_activations(
    model,
    hdf5_path: str,
    layer_idx: int,
    device: torch.device,
    n_trajectories: int = 300,
    max_steps_per_traj: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Collect residual stream activations at EVERY sequence position independently.

    Returns:
        activations: (N, T, d_model) — all positions preserved, NOT pooled
        states:      dict of var_name -> (N,) ground-truth values
    """
    rng = np.random.default_rng(seed)
    all_activations = []
    all_states = {}

    model.eval()

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = min(n_trajectories, len(val_grp))
        traj_indices = rng.choice(len(val_grp), size=n_traj, replace=False)

        first_traj = val_grp["0"]
        state_var_names = list(first_traj["states"].keys())

        for traj_idx in tqdm(traj_indices, desc="Collecting per-position activations"):
            traj_grp = val_grp[str(traj_idx)]
            num_steps = int(traj_grp.attrs["length"])
            if num_steps < 2:
                continue

            steps = rng.choice(num_steps - 1, size=min(max_steps_per_traj, num_steps - 1), replace=False)

            for step in steps:
                obs = traj_grp["observations"][step].flatten().astype(np.int64)
                obs = np.clip(obs, 0, model.config.vocab_size - 1)
                tokens = torch.from_numpy(obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    residual_stream = model.get_residual_stream(tokens)

                # Store ALL positions: (T, d_model) not pooled
                layer_acts = residual_stream[layer_idx].squeeze(0).cpu().numpy()  # (T, d_model)
                all_activations.append(layer_acts)

                for var in state_var_names:
                    val = traj_grp[f"states/{var}"][step]
                    all_states.setdefault(var, []).append(val)

    activations = np.stack(all_activations)  # (N, T, d_model)
    states = {k: np.array(v) for k, v in all_states.items()}
    return activations, states


def run_per_position_probes(
    activations: np.ndarray,
    states: dict,
    variable_types: dict,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    For each variable, probe EACH sequence position independently.
    Returns the best score across positions and which position achieved it.

    Returns:
        dict of var_name -> {
            "best_score": float,
            "best_position": int,
            "mean_score": float,
            "scores_per_position": np.ndarray  # (T,)
        }
    """
    N, T, d_model = activations.shape
    results = {}

    for var_name, var_type in variable_types.items():
        y = states.get(var_name)
        if y is None:
            continue

        # Skip degenerate variables
        if len(np.unique(y)) < 2:
            results[var_name] = {
                "best_score": float("nan"),
                "best_position": -1,
                "mean_score": float("nan"),
                "scores_per_position": np.full(T, float("nan")),
            }
            continue

        scores = np.zeros(T)
        for pos in range(T):
            X = activations[:, pos, :]  # (N, d_model) at this position

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if var_type == "continuous":
                probe = Ridge(alpha=1.0)
                probe.fit(X_train_s, y_train)
                preds = probe.predict(X_test_s)
                scores[pos] = r2_score(y_test, preds)
            else:
                unique = np.unique(y_train)
                if len(unique) < 2:
                    scores[pos] = float("nan")
                    continue
                probe = LogisticRegression(max_iter=500)
                probe.fit(X_train_s, y_train)
                preds = probe.predict(X_test_s)
                scores[pos] = accuracy_score(y_test, preds)

        valid_scores = scores[~np.isnan(scores)]
        best_pos = int(np.nanargmax(scores))
        results[var_name] = {
            "best_score": float(scores[best_pos]),
            "best_position": best_pos,
            "mean_score": float(np.mean(valid_scores)) if len(valid_scores) > 0 else float("nan"),
            "scores_per_position": scores,
        }
        print(f"  {var_name:20s}: best={scores[best_pos]:.3f} at pos {best_pos} "
              f"(mean={np.mean(valid_scores):.3f})")

    return results


def plot_position_heatmap(
    results: dict,
    seq_len: int,
    output_path: str,
    env: str,
    layer: int,
) -> None:
    """
    Plot per-position probe scores as a heatmap:
    variables (rows) x sequence positions (cols).
    """
    vars_to_show = [v for v in results if not np.all(np.isnan(results[v]["scores_per_position"]))]
    if not vars_to_show:
        print("No valid results to plot.")
        return

    matrix = np.stack([results[v]["scores_per_position"] for v in vars_to_show])
    matrix_clipped = np.clip(np.nan_to_num(matrix, nan=0.0), 0, 1)

    # For Physics 8x8 grid, reshape position axis to spatial (optional label)
    fig, ax = plt.subplots(figsize=(min(seq_len // 4, 20), max(4, len(vars_to_show) * 0.5)))
    sns.heatmap(
        matrix_clipped,
        cmap="viridis", vmin=0, vmax=1,
        yticklabels=vars_to_show,
        xticklabels=[str(i) if i % 8 == 0 else "" for i in range(seq_len)],
        cbar_kws={"label": "Probe score"},
        ax=ax,
    )
    ax.set_title(f"Per-Position Probe Scores — {env} Layer {layer}")
    ax.set_xlabel("Sequence Position (token index)")
    ax.set_ylabel("State Variable")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Per-position linear probes")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-trajectories", type=int, default=300)
    parser.add_argument("--max-steps-per-traj", type=int, default=20)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))

    hdf5_path = get_hdf5_path(args.env)
    variable_types = get_variable_types(hdf5_path)

    print(f"\nCollecting per-position activations at layer {args.layer}...")
    activations, states = collect_per_position_activations(
        model=model, hdf5_path=hdf5_path, layer_idx=args.layer,
        device=device, n_trajectories=args.n_trajectories,
        max_steps_per_traj=args.max_steps_per_traj,
    )
    N, T, d_model = activations.shape
    print(f"Collected {N} samples, sequence length {T}, d_model={d_model}")

    print(f"\nRunning per-position probes ({T} positions x {len(variable_types)} variables)...")
    results = run_per_position_probes(activations, states, variable_types)

    print(f"\n=== Summary: Best Per-Position vs Mean-Pooled ===")
    print(f"{'Variable':20s} | {'Best pos score':>14s} | {'Best position':>13s} | {'Mean score':>10s}")
    print("-" * 65)
    for var, res in results.items():
        if np.isnan(res["best_score"]):
            print(f"{var:20s} | {'SKIPPED':>14s} | {'':>13s} | {'':>10s}")
        else:
            print(f"{var:20s} | {res['best_score']:14.3f} | {res['best_position']:13d} | {res['mean_score']:10.3f}")

    plot_path = f"results/{args.env}_layer{args.layer}_per_position_probes.png"
    plot_position_heatmap(results, T, plot_path, args.env, args.layer)


if __name__ == "__main__":
    main()
