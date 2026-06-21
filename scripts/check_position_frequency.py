"""
scripts/check_position_frequency.py

Quick diagnostic: check how frequently agent_x=1, agent_y=1 appears
in the dataset relative to other positions. This determines whether
F867's exclusive firing at (1,1) reflects:
  (a) a rare, specific position the SAE has isolated as meaningful
      (e.g. a corner / wall-adjacent cell), or
  (b) a data collection artifact (e.g. disproportionate sampling of
      this position due to env reset behaviour or policy bias)

Usage:
    python scripts/check_position_frequency.py --env minigrid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["minigrid", "physics"], default="minigrid")
    parser.add_argument("--n-trajectories", type=int, default=2000)
    args = parser.parse_args()

    hdf5_path = f"data/trajectories/{args.env}/{args.env}.hdf5"

    all_x, all_y = [], []
    with h5py.File(hdf5_path, "r") as f:
        split_grp = f["trajectories/train"]
        n_traj = min(args.n_trajectories, len(split_grp))

        for i in range(n_traj):
            traj_grp = split_grp[str(i)]
            all_x.append(traj_grp["states/agent_x"][:])
            all_y.append(traj_grp["states/agent_y"][:])

    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)

    print(f"Total samples: {len(all_x):,}")
    print(f"\nPosition (1,1) frequency:")
    at_corner = ((all_x == 1) & (all_y == 1)).sum()
    print(f"  Count: {at_corner:,} / {len(all_x):,} ({100*at_corner/len(all_x):.3f}%)")

    print(f"\nOverall position statistics:")
    print(f"  agent_x range: [{all_x.min()}, {all_x.max()}], mean={all_x.mean():.2f}")
    print(f"  agent_y range: [{all_y.min()}, {all_y.max()}], mean={all_y.mean():.2f}")

    # Compare to a uniform expectation
    grid_cells = (all_x.max() - all_x.min() + 1) * (all_y.max() - all_y.min() + 1)
    uniform_expected = len(all_x) / grid_cells
    print(f"\nIf uniformly distributed across ~{grid_cells} cells, "
          f"expected count per cell: {uniform_expected:.1f}")
    print(f"Actual (1,1) count: {at_corner:,} "
          f"({'OVER' if at_corner > uniform_expected else 'UNDER'}-represented "
          f"by {at_corner/uniform_expected:.1f}x)" if uniform_expected > 0 else "")

    # Check first-step positions specifically (is (1,1) common as an episode start?)
    first_steps_x, first_steps_y = [], []
    with h5py.File(hdf5_path, "r") as f:
        split_grp = f["trajectories/train"]
        for i in range(n_traj):
            traj_grp = split_grp[str(i)]
            first_steps_x.append(traj_grp["states/agent_x"][0])
            first_steps_y.append(traj_grp["states/agent_y"][0])

    first_steps_x = np.array(first_steps_x)
    first_steps_y = np.array(first_steps_y)
    first_at_corner = ((first_steps_x == 1) & (first_steps_y == 1)).sum()
    print(f"\nEpisode START positions at (1,1): {first_at_corner} / {n_traj} "
          f"({100*first_at_corner/n_traj:.2f}%)")
    print(f"Unique start positions seen: {len(set(zip(first_steps_x.tolist(), first_steps_y.tolist())))}")


if __name__ == "__main__":
    main()
