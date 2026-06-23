"""
scripts/analyse_attention.py

Mechanistic attention pattern analysis for the world model transformer.

Goes beyond residual stream probing to ask: WHICH attention heads, in which
layers, preferentially attend to the agent's current position token? This is
the mechanistic interpretability question — not just "is position represented"
but "how does the model USE that representation during computation."

Three analyses:
  1. POSITION ATTENTION SCORE: for each head in each layer, what fraction of
     attention weight flows FROM other tokens TO the agent's position token?
     A high score = this head "reads from" the agent's cell.

  2. POSITION WRITING SCORE: what fraction of attention weight flows FROM the
     agent's position token TO other tokens? A high score = this head
     "broadcasts" agent position information to the rest of the sequence.

  3. HEAD CONSISTENCY: do the same heads consistently attend to the agent's
     position across different observations (not just specific grid cells)?
     Measured via standard deviation of position attention across samples.

For Physics: equivalent analysis using the VQ-VAE spatial token corresponding
to the primary object's position.

Usage:
    # MiniGrid
    python scripts/analyse_attention.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --scale small

    # Physics
    python scripts/analyse_attention.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small
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
from tqdm import tqdm

from models.transformer import load_model
from interpretability.interventions import (
    agent_position_to_flat_index,
    physics_position_to_token_index,
    CHANNELS_PER_CELL,
    VQVAE_SPATIAL_SIZE,
)


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


def get_agent_token_index(traj_grp, step: int, env: str, seq_len: int) -> int:
    """Get the flat token index corresponding to the agent/primary object position."""
    if env == "minigrid":
        ax = int(traj_grp["states/agent_x"][step])
        ay = int(traj_grp["states/agent_y"][step])
        return min(agent_position_to_flat_index(ax, ay), seq_len - 1)
    else:
        px = float(traj_grp["states/pos_x_0"][step])
        py = float(traj_grp["states/pos_y_0"][step])
        return min(physics_position_to_token_index(px, py), seq_len - 1)


def extract_attention_patterns(
    model,
    hdf5_path: str,
    device: torch.device,
    env: str,
    n_samples: int = 500,
    seed: int = 42,
) -> dict:
    """
    Extract attention weights from all heads in all layers for a sample
    of observations. For each sample, also record the agent's token position.

    Registers forward hooks on each attention block to capture weights.

    Returns:
        dict with keys:
          "attention_weights": list of (n_layers, n_heads, T, T) arrays per sample
          "agent_positions":   list of agent token indices per sample
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # Register hooks to capture attention weights from each block
    attention_weights_per_layer = {i: [] for i in range(model.config.n_layers)}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_weights, values) or just the attended output
            # We need to access the attention weights directly
            # Works with our standard CausalSelfAttention implementation
            if hasattr(module, '_last_attn_weights') and module._last_attn_weights is not None:
                attention_weights_per_layer[layer_idx].append(
                    module._last_attn_weights.detach().cpu()
                )
        return hook

    # Check if model stores attention weights (we need to enable this)
    # Our transformer stores _last_attn_weights if we pass return_attn=True
    # or if we set a flag. Let's check the architecture first.

    all_attn_weights = []
    all_agent_positions = []

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = len(val_grp)
        traj_indices = rng.choice(n_traj, size=min(200, n_traj), replace=False)
        samples_collected = 0

        for traj_idx in tqdm(traj_indices, desc="Extracting attention patterns"):
            if samples_collected >= n_samples:
                break

            traj_grp = val_grp[str(traj_idx)]
            num_steps = int(traj_grp.attrs["length"])
            if num_steps < 2:
                continue

            step = rng.integers(0, num_steps - 1)
            obs = traj_grp["observations"][step].flatten().astype(np.int64)
            obs = np.clip(obs, 0, model.config.vocab_size - 1)
            tokens = torch.from_numpy(obs).unsqueeze(0).to(device)
            seq_len = len(obs)

            agent_pos = get_agent_token_index(traj_grp, step, env, seq_len)

            with torch.no_grad():
                # Get attention weights through the model's get_residual_stream
                # which processes the full forward pass
                attn_weights = model.get_attention_weights(tokens)

            if attn_weights is not None:
                all_attn_weights.append(attn_weights)
                all_agent_positions.append(agent_pos)
                samples_collected += 1

    return {
        "attention_weights": all_attn_weights,
        "agent_positions": all_agent_positions,
    }


def compute_position_attention_scores(
    attention_data: dict,
    n_layers: int,
    n_heads: int,
) -> dict:
    """
    For each (layer, head), compute:
      - read_score:  mean attention weight pointing TO the agent token
                     (how much does this head read from the agent's position?)
      - write_score: mean attention weight pointing FROM the agent token
                     (how much does the agent's token broadcast to others?)
      - consistency: 1 - std(read_score across samples) (how consistent is this?)
    """
    attn_list = attention_data["attention_weights"]
    pos_list = attention_data["agent_positions"]

    if not attn_list:
        print("WARNING: No attention weights collected. Model may not support get_attention_weights().")
        return {}

    n_samples = len(attn_list)
    read_scores  = np.zeros((n_samples, n_layers, n_heads))
    write_scores = np.zeros((n_samples, n_layers, n_heads))

    for s_idx, (attn, agent_pos) in enumerate(zip(attn_list, pos_list)):
        # attn: (n_layers, n_heads, T, T) — attn[l, h, i, j] = weight from i to j
        for l in range(n_layers):
            for h in range(n_heads):
                A = attn[l, h]  # (T, T)
                T = A.shape[0]

                # Read score: how much attention flows TO agent_pos FROM other tokens
                # = column sum at agent_pos column, normalised by T
                read_scores[s_idx, l, h] = A[:, agent_pos].mean().item()

                # Write score: how much the agent_pos token attends to others
                # = row sum at agent_pos row (excluding self)
                write_scores[s_idx, l, h] = A[agent_pos, :].mean().item()

    return {
        "mean_read_score":  read_scores.mean(axis=0),   # (n_layers, n_heads)
        "mean_write_score": write_scores.mean(axis=0),  # (n_layers, n_heads)
        "std_read_score":   read_scores.std(axis=0),    # consistency measure
        "read_scores_all":  read_scores,                 # (n_samples, n_layers, n_heads)
    }


def plot_attention_scores(scores: dict, output_path: str, env: str) -> None:
    """
    Plot heatmaps of per-head read and write scores.
    Rows = layers, cols = heads.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Mean Read Score\n(attention TO agent position)",
              "Mean Write Score\n(attention FROM agent position)",
              "Consistency (1 - std of read score)"]
    data = [
        scores["mean_read_score"],
        scores["mean_write_score"],
        1.0 - scores["std_read_score"],
    ]

    for ax, d, title in zip(axes, data, titles):
        sns.heatmap(
            d, ax=ax, cmap="viridis", vmin=0, vmax=d.max(),
            xticklabels=[f"H{h}" for h in range(d.shape[1])],
            yticklabels=[f"L{l}" for l in range(d.shape[0])],
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")

    plt.suptitle(f"Attention Pattern Analysis — {env}", fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Attention heatmap saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyse attention patterns")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))

    # Check if model has get_attention_weights method
    if not hasattr(model, "get_attention_weights"):
        print("ERROR: Model does not have get_attention_weights() method.")
        print("This needs to be added to models/transformer.py first.")
        print("See the implementation notes in this script.")
        return

    hdf5_path = get_hdf5_path(args.env)

    print(f"\nExtracting attention patterns for {args.n_samples} samples...")
    attention_data = extract_attention_patterns(
        model, hdf5_path, device, args.env, args.n_samples
    )

    if not attention_data["attention_weights"]:
        print("No attention weights collected — model needs get_attention_weights() implementation.")
        return

    print(f"\nComputing position attention scores...")
    scores = compute_position_attention_scores(
        attention_data,
        n_layers=model.config.n_layers,
        n_heads=model.config.n_heads,
    )

    print(f"\n=== Top Heads by Read Score (attention TO agent position) ===")
    read = scores["mean_read_score"]
    top_indices = np.argwhere(read == read.max())
    for l, h in top_indices:
        print(f"  Layer {l}, Head {h}: mean read score = {read[l,h]:.4f}")

    print(f"\n=== Top Heads by Write Score (attention FROM agent position) ===")
    write = scores["mean_write_score"]
    top_indices = np.argwhere(write == write.max())
    for l, h in top_indices:
        print(f"  Layer {l}, Head {h}: mean write score = {write[l,h]:.4f}")

    print(f"\n=== Random baseline ===")
    T = attention_data["attention_weights"][0].shape[-1]
    print(f"  Expected uniform attention to any position: {1.0/T:.4f}")
    print(f"  Sequence length T = {T}")
    print(f"  Max observed read score: {read.max():.4f} ({read.max()/(1.0/T):.1f}x above uniform)")

    plot_path = f"results/{args.env}_attention_patterns.png"
    plot_attention_scores(scores, plot_path, args.env)


if __name__ == "__main__":
    main()
