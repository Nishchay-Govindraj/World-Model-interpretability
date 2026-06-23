"""
scripts/run_sae_ablation.py

SAE feature causal ablation study.

We found F867 (MiniGrid) fires monosemantically at agent position (1,1).
This script tests CAUSAL NECESSITY: does ablating F867 (setting it to zero
in the SAE decomposition, then reconstructing) actually change the model's
predictions when the agent IS at (1,1)? And does it leave predictions
unchanged when the agent is NOT at (1,1)?

This completes the interpretability chain:
  Phase 4 (probes):        position is linearly decodable from residual stream
  Phase 5 (SAEs):          F867 fires specifically at (1,1)
  Phase 5 (ablation):      ablating F867 changes predictions at (1,1) — CAUSAL
  Phase 6 (interventions): full residual stream is causally sufficient

If ablation changes predictions strongly at (1,1) but weakly elsewhere,
F867 is causally necessary for the model's corner-specific behaviour.

Also tests the top physics features for velocity encoding (from MI table).

Usage:
    # MiniGrid — ablate F867
    python scripts/run_sae_ablation.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --sae-checkpoint checkpoints/sae_minigrid_layer5.pt \\
        --env minigrid --scale small --layer 5 \\
        --features 867 --target-var agent_x --target-val 1

    # Physics — ablate top pos_x_0 feature (F858 per MI table)
    python scripts/run_sae_ablation.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --sae-checkpoint checkpoints/sae_physics_layer2.pt \\
        --env physics --scale small --layer 2 \\
        --features 858 --target-var pos_x_1 --target-val-range 200 400
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.transformer import load_model, WorldModelTransformer
from interpretability.sae import SparseAutoencoder


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


def load_sae(checkpoint_path: str, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    sae = SparseAutoencoder(config).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def ablate_features_and_measure(
    model: WorldModelTransformer,
    sae: SparseAutoencoder,
    hdf5_path: str,
    layer_idx: int,
    features_to_ablate: list[int],
    target_var: str,
    device: torch.device,
    env: str,
    n_samples_on_target: int = 100,
    n_samples_off_target: int = 100,
    target_val: float = None,
    target_val_range: tuple = None,
    seed: int = 42,
) -> dict:
    """
    For each sample:
      1. Run forward pass, capture residual stream at layer_idx
      2. Decompose via SAE encoder
      3. Zero out the ablated features
      4. Reconstruct via SAE decoder → ablated residual stream
      5. Substitute into forward pass, measure KL divergence from original logits

    Computes separately for:
      - ON-TARGET: samples where target_var matches target condition
      - OFF-TARGET: samples where target_var does NOT match

    Strong ablation effect ON-TARGET + weak effect OFF-TARGET = F is causally
    necessary specifically for the behaviour associated with its firing condition.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    sae.eval()

    def is_on_target(val: float) -> bool:
        if target_val is not None:
            return abs(val - target_val) < 0.5
        elif target_val_range is not None:
            return target_val_range[0] <= val <= target_val_range[1]
        return False

    on_target_kls = []
    off_target_kls = []
    on_target_collected = 0
    off_target_collected = 0

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = len(val_grp)
        traj_indices = rng.choice(n_traj, size=min(500, n_traj), replace=False)

        for traj_idx in tqdm(traj_indices, desc="Running ablation"):
            if (on_target_collected >= n_samples_on_target and
                    off_target_collected >= n_samples_off_target):
                break

            traj_grp = val_grp[str(traj_idx)]
            num_steps = int(traj_grp.attrs["length"])
            if num_steps < 2:
                continue

            steps = rng.choice(num_steps - 1, size=min(10, num_steps - 1), replace=False)

            for step in steps:
                val = float(traj_grp[f"states/{target_var}"][step])
                on = is_on_target(val)

                if on and on_target_collected >= n_samples_on_target:
                    continue
                if not on and off_target_collected >= n_samples_off_target:
                    continue

                obs = traj_grp["observations"][step].flatten().astype(np.int64)
                obs = np.clip(obs, 0, model.config.vocab_size - 1)
                tokens = torch.from_numpy(obs).unsqueeze(0).to(device)

                # Get original logits
                with torch.no_grad():
                    original_logits, _ = model(tokens)

                # Capture residual stream at target layer
                captured = {}
                def capture_hook(module, input, output):
                    captured["activation"] = output.detach().clone()
                    return output

                handle = model.blocks[layer_idx].register_forward_hook(capture_hook)
                with torch.no_grad():
                    model(tokens)
                handle.remove()

                activation = captured["activation"]  # (1, T, d_model)
                pooled = activation.mean(dim=1)       # (1, d_model) — SAE was trained on pooled

                # SAE decomposition
                with torch.no_grad():
                    feature_acts = sae.encode(pooled)              # (1, d_hidden)
                    ablated_acts = feature_acts.clone()
                    ablated_acts[:, features_to_ablate] = 0.0      # zero out target features
                    ablated_recon = sae.decode(ablated_acts)        # (1, d_model)
                    original_recon = sae.decode(feature_acts)       # (1, d_model)

                # Compute ablation delta in d_model space
                delta = (ablated_recon - original_recon)  # (1, d_model)

                # Apply delta to every position's activation (broadcast from pooled)
                ablated_activation = activation + delta.unsqueeze(1)  # (1, T, d_model)

                # Run forward pass with ablated residual stream
                ablation_applied = {}
                def patch_hook(module, input, output):
                    ablation_applied["used"] = True
                    return ablated_activation

                handle = model.blocks[layer_idx].register_forward_hook(patch_hook)
                with torch.no_grad():
                    ablated_logits, _ = model(tokens)
                handle.remove()

                # KL divergence: how much did predictions change?
                orig_probs = F.softmax(original_logits.view(-1, model.config.vocab_size), dim=-1)
                abla_probs = F.softmax(ablated_logits.view(-1, model.config.vocab_size), dim=-1)
                kl = F.kl_div(
                    abla_probs.log(), orig_probs, reduction="batchmean"
                ).item()

                if on:
                    on_target_kls.append(kl)
                    on_target_collected += 1
                else:
                    off_target_kls.append(kl)
                    off_target_collected += 1

    return {
        "on_target_kls": np.array(on_target_kls),
        "off_target_kls": np.array(off_target_kls),
        "mean_kl_on_target": float(np.mean(on_target_kls)) if on_target_kls else float("nan"),
        "mean_kl_off_target": float(np.mean(off_target_kls)) if off_target_kls else float("nan"),
        "n_on_target": len(on_target_kls),
        "n_off_target": len(off_target_kls),
    }


def main():
    parser = argparse.ArgumentParser(description="SAE feature causal ablation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sae-checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--features", type=int, nargs="+", required=True,
                        help="Feature indices to ablate (e.g. --features 867)")
    parser.add_argument("--target-var", type=str, required=True,
                        help="State variable to condition on (e.g. agent_x, pos_x_0)")
    parser.add_argument("--target-val", type=float, default=None,
                        help="Exact value that constitutes 'on-target' (±0.5)")
    parser.add_argument("--target-val-range", type=float, nargs=2, default=None,
                        help="Range [min max] for on-target condition")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    if args.target_val is None and args.target_val_range is None:
        parser.error("Must specify either --target-val or --target-val-range")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    model = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))
    sae = load_sae(args.sae_checkpoint, device)

    hdf5_path = get_hdf5_path(args.env)

    target_str = (f"={args.target_val}" if args.target_val is not None
                  else f"∈{args.target_val_range}")
    print(f"\nAblating SAE features {args.features} at layer {args.layer}")
    print(f"On-target condition: {args.target_var} {target_str}")

    results = ablate_features_and_measure(
        model=model, sae=sae, hdf5_path=hdf5_path,
        layer_idx=args.layer, features_to_ablate=args.features,
        target_var=args.target_var, device=device, env=args.env,
        target_val=args.target_val,
        target_val_range=tuple(args.target_val_range) if args.target_val_range else None,
    )

    print(f"\n=== Ablation Results ===")
    print(f"ON-TARGET  (n={results['n_on_target']:3d}): "
          f"mean KL = {results['mean_kl_on_target']:.6f}")
    print(f"OFF-TARGET (n={results['n_off_target']:3d}): "
          f"mean KL = {results['mean_kl_off_target']:.6f}")

    if results["n_on_target"] > 0 and results["n_off_target"] > 0:
        ratio = results["mean_kl_on_target"] / max(results["mean_kl_off_target"], 1e-9)
        print(f"ON/OFF ratio: {ratio:.2f}x")
        if ratio > 3.0:
            print("STRONG causal evidence: ablating this feature disrupts predictions "
                  "specifically when the feature's condition is active.")
        elif ratio > 1.5:
            print("MODERATE causal evidence: ablation has larger effect on-target than off-target.")
        else:
            print("WEAK causal evidence: ablation effect is similar on and off target.")


if __name__ == "__main__":
    main()
