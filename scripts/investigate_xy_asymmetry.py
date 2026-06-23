"""
scripts/investigate_xy_asymmetry.py

Investigates the persistent x/y probe asymmetry found in MiniGrid FourRooms:
agent_x R²=0.999 vs agent_y R²=0.791 even after extended training.

Tests three hypotheses:

HYPOTHESIS 1 — Room structure effect:
  FourRooms' cross-shaped wall divides the space into quadrants. Horizontal
  position may be more behaviourally salient because the four rooms are
  arranged in a left/right/top/bottom pattern. Test: run probes on a trained
  model but with trajectories from MiniGrid-Empty-8x8, which has no room
  structure. If asymmetry disappears, room structure is the cause.

HYPOTHESIS 2 — Action space asymmetry:
  MiniGrid's turn_left/turn_right/forward actions are more directly related
  to x-movement than y-movement under certain direction distributions.
  Test: check if the distribution of agent_x changes vs agent_y changes
  in the dataset differs systematically.

HYPOTHESIS 3 — Observation encoding asymmetry:
  The flattened observation encodes row-major (y first, then x). If the
  transformer's attention patterns are biased by token order, x-position
  tokens (later in row) might be more accessible. Test: compare probe
  scores at specific token positions corresponding to the same y-row
  vs x-column.

We implement Hypothesis 1 (most testable) and Hypothesis 2 (fast data check).
Hypothesis 3 is addressed by the per-position probe results.

Usage:
    python scripts/investigate_xy_asymmetry.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import yaml

from interpretability.probes import run_probe_suite, results_to_matrix
from models.transformer import load_model


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hypothesis_2_action_space_check(hdf5_path: str, n_trajectories: int = 2000) -> None:
    """
    Check whether x and y positions change at different rates across the dataset.
    If x changes more often than y (or vice versa), the model has an easier
    prediction target for one axis, which could explain the asymmetry.
    """
    print("\n=== Hypothesis 2: Action Space / Movement Rate Analysis ===")
    all_dx, all_dy = [], []

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = min(n_trajectories, len(val_grp))

        for i in range(n_traj):
            traj = val_grp[str(i)]
            xs = traj["states/agent_x"][:]
            ys = traj["states/agent_y"][:]
            dx = np.abs(np.diff(xs.astype(float)))
            dy = np.abs(np.diff(ys.astype(float)))
            all_dx.extend(dx.tolist())
            all_dy.extend(dy.tolist())

    all_dx = np.array(all_dx)
    all_dy = np.array(all_dy)

    print(f"Analysed {n_traj} trajectories, {len(all_dx):,} consecutive step pairs")
    print(f"\nX-axis movement:")
    print(f"  Steps where x changed: {(all_dx > 0).sum():,} / {len(all_dx):,} ({100*(all_dx>0).mean():.1f}%)")
    print(f"  Mean |Δx| per step: {all_dx.mean():.4f}")

    print(f"\nY-axis movement:")
    print(f"  Steps where y changed: {(all_dy > 0).sum():,} / {len(all_dy):,} ({100*(all_dy>0).mean():.1f}%)")
    print(f"  Mean |Δy| per step: {all_dy.mean():.4f}")

    ratio = (all_dx > 0).mean() / max((all_dy > 0).mean(), 1e-9)
    print(f"\nX/Y movement rate ratio: {ratio:.2f}")
    if ratio > 1.1:
        print("FINDING: X-axis changes more frequently — agent moves horizontally more than vertically.")
        print("         This could explain x-probe dominance: more training signal for x.")
    elif ratio < 0.9:
        print("FINDING: Y-axis changes more frequently — asymmetry is NOT explained by movement rate.")
    else:
        print("FINDING: Movement rates are roughly balanced — asymmetry not explained by movement rate.")


def hypothesis_1_room_structure_check(
    model,
    hdf5_path: str,
    device,
    layer: int = 5,
) -> None:
    """
    Test Hypothesis 1: run probes on trajectories from our existing model
    but split by which room the agent is in. If the asymmetry is due to
    room structure, we expect different x/y scores in different quadrants.

    For FourRooms (19x19): rooms are approximately:
      Top-left:     x<9,  y<9
      Top-right:    x>=9, y<9
      Bottom-left:  x<9,  y>=9
      Bottom-right: x>=9, y>=9

    If x is more predictable WITHIN rooms (ignoring which room), that
    suggests something other than room identity drives the asymmetry.
    """
    print("\n=== Hypothesis 1: Room Structure Analysis ===")
    print("Splitting dataset by room quadrant and running probes per quadrant...")

    mid = 9  # FourRooms approximate room boundary
    quadrant_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    quadrant_filters = [
        lambda x, y: (x < mid) & (y < mid),
        lambda x, y: (x >= mid) & (y < mid),
        lambda x, y: (x < mid) & (y >= mid),
        lambda x, y: (x >= mid) & (y >= mid),
    ]

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = min(500, len(val_grp))

        # Collect all activations with position labels
        import torch
        from interpretability.probes import ActivationCache

        cache = ActivationCache(model, device)
        activations_all, states_all = cache.extract(
            hdf5_path=hdf5_path,
            split="val",
            n_trajectories=n_traj,
            max_steps_per_traj=20,
        )

    X_all = activations_all[layer]  # (N, d_model) at target layer
    ax_all = states_all["agent_x"]
    ay_all = states_all["agent_y"]

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(f"\n{'Quadrant':15s} | {'n_samples':>9s} | {'agent_x R²':>10s} | {'agent_y R²':>10s} | {'Ratio x/y':>9s}")
    print("-" * 65)

    for qname, qfilt in zip(quadrant_names, quadrant_filters):
        mask = qfilt(ax_all, ay_all)
        if mask.sum() < 50:
            print(f"{qname:15s} | {'<50 samples':>9s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>9s}")
            continue

        X_q = X_all[mask]
        ax_q = ax_all[mask]
        ay_q = ay_all[mask]

        results_x, results_y = {}, {}
        for varname, y_var, store in [("x", ax_q, results_x), ("y", ay_q, results_y)]:
            X_tr, X_te, y_tr, y_te = train_test_split(X_q, y_var, test_size=0.2, random_state=42)
            sc = StandardScaler()
            probe = Ridge(alpha=1.0)
            probe.fit(sc.fit_transform(X_tr), y_tr)
            score = r2_score(y_te, probe.predict(sc.transform(X_te)))
            store["score"] = score

        ratio = results_x["score"] / max(results_y["score"], 1e-6)
        print(f"{qname:15s} | {mask.sum():9d} | {results_x['score']:10.3f} | {results_y['score']:10.3f} | {ratio:9.2f}")

    print("\nIf x/y ratio is consistently >1 across ALL quadrants, the asymmetry")
    print("is NOT caused by room structure alone — it's a more fundamental property")
    print("of how the model encodes the two spatial dimensions.")


def main():
    parser = argparse.ArgumentParser(description="Investigate MiniGrid x/y probe asymmetry")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    model = load_model(args.checkpoint, model_config, scale=args.scale, device=str(device))

    hdf5_path = "data/trajectories/minigrid/minigrid.hdf5"

    # Hypothesis 2: movement rate analysis (fast, no model needed)
    hypothesis_2_action_space_check(hdf5_path)

    # Hypothesis 1: room-structure quadrant analysis
    hypothesis_1_room_structure_check(model, hdf5_path, device, layer=args.layer)


if __name__ == "__main__":
    main()
