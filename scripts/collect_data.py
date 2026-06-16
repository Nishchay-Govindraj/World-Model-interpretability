"""
scripts/collect_data.py

Data collection script. Runs trajectory collection for one or both environments.

Usage:
    # Validate pipeline (small run)
    python scripts/collect_data.py --env minigrid --num-trajectories 100 --validate

    # Full MiniGrid collection
    python scripts/collect_data.py --env minigrid --num-trajectories 200000

    # Full physics collection
    python scripts/collect_data.py --env physics --num-trajectories 200000

    # Both environments
    python scripts/collect_data.py --env both --num-trajectories 200000
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.collector import TrajectoryCollector
from environments.minigrid_env import MiniGridEnv
from environments.physics_env import PhysicsEnv


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def collect_minigrid(args, mg_config: dict, model_config: dict) -> None:
    print("\n=== MiniGrid Data Collection ===")
    env = MiniGridEnv(config=mg_config, seed=42)

    output_path = (
        Path(mg_config["data_collection"]["output_dir"]) / "minigrid.hdf5"
    )

    collector = TrajectoryCollector(
        env=env,
        config={**mg_config, "wandb": model_config.get("wandb", {})},
        output_path=str(output_path),
        use_wandb=not args.no_wandb,
    )

    num = args.num_trajectories if not args.validate else 200
    collector.collect(
        num_trajectories=num,
        max_steps=mg_config["data_collection"]["max_steps_per_episode"],
        validation_split=mg_config["data_collection"]["validation_split"],
        base_seed=0,
    )
    env.close()
    print(f"MiniGrid data saved to: {output_path}")


def collect_physics(args, ph_config: dict, model_config: dict) -> None:
    print("\n=== Physics Sandbox Data Collection ===")
    env = PhysicsEnv(config=ph_config, seed=42)

    output_path = (
        Path(ph_config["data_collection"]["output_dir"]) / "physics.hdf5"
    )

    collector = TrajectoryCollector(
        env=env,
        config={**ph_config, "wandb": model_config.get("wandb", {})},
        output_path=str(output_path),
        use_wandb=not args.no_wandb,
    )

    num = args.num_trajectories if not args.validate else 100
    collector.collect(
        num_trajectories=num,
        max_steps=ph_config["data_collection"]["steps_per_trajectory"],
        validation_split=ph_config["data_collection"]["validation_split"],
        base_seed=1000,  # different base seed from MiniGrid to avoid correlation
    )
    env.close()
    print(f"Physics data saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory data")
    parser.add_argument(
        "--env",
        choices=["minigrid", "physics", "both"],
        default="both",
        help="Which environment to collect data for",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=200_000,
        help="Number of trajectories to collect (default: 200,000)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a small validation collection (100-200 trajectories) to check pipeline",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing YAML config files",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    mg_config = load_config(config_dir / "minigrid_config.yaml")
    ph_config = load_config(config_dir / "physics_config.yaml")
    model_config = load_config(config_dir / "model_config.yaml")

    if args.env in ("minigrid", "both"):
        collect_minigrid(args, mg_config, model_config)

    if args.env in ("physics", "both"):
        collect_physics(args, ph_config, model_config)

    print("\nAll done.")


if __name__ == "__main__":
    main()
