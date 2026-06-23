"""
scripts/run_probes_nonlinear.py

Non-linear (MLP) probe pipeline.

The Mode B vs Mode C causal intervention discrepancy implies that state
information is distributed across the residual stream in a way that no
single linear direction captures. This directly predicts that non-linear
probes (small MLPs) should substantially outperform linear probes — the
MLP can exploit higher-order feature combinations.

If MLP probes substantially outperform linear probes:
  -> Confirms distributed non-linear representation
  -> Explains Mode B failure: linear probe direction ≠ the representation structure
  -> Aligns with SAE finding of weak monosemanticity in Physics

If MLP probes show similar scores to linear:
  -> The information is simply not present in the residual stream
     (or only linearly encoded, which would contradict Mode C's perfect recovery)
  -> The Mode B/C gap must have another explanation

This is the natural follow-up experiment a frontier lab reviewer would
ask about.

Usage:
    python scripts/run_probes_nonlinear.py \\
        --checkpoint checkpoints/minigrid_small_step40000.pt \\
        --env minigrid --layer 5

    python scripts/run_probes_nonlinear.py \\
        --checkpoint checkpoints/physics_physics_small_step88000.pt \\
        --env physics --scale small --layer 2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.transformer import load_model, build_model, WorldModelTransformer


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


class MLPProbe(nn.Module):
    """
    Small 2-layer MLP probe for non-linear decoding of state variables
    from residual stream activations.

    Architecture: d_model -> 256 -> 128 -> output_dim
    Uses ReLU activation and dropout for regularisation.
    Deliberately kept small to avoid trivially overfitting — the point
    is to test whether non-linearity helps, not to achieve maximum accuracy
    with an arbitrarily powerful probe.
    """
    def __init__(self, d_model: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_type: str,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> float:
    """Train an MLP probe and return test score."""
    d_model = X_train.shape[1]

    # Standardise inputs
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_tr = torch.from_numpy(X_train_s).float().to(device)
    X_te = torch.from_numpy(X_test_s).float().to(device)

    if var_type == "continuous":
        y_tr = torch.from_numpy(y_train.astype(np.float32)).to(device)
        output_dim = 1
    else:
        classes = np.unique(y_train)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_tr_idx = np.array([class_to_idx[c] for c in y_train])
        y_tr = torch.from_numpy(y_tr_idx).long().to(device)
        output_dim = len(classes)

    probe = MLPProbe(d_model, output_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    probe.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = probe(X_tr)
        if var_type == "continuous":
            loss = F.mse_loss(out.squeeze(), y_tr)
        else:
            loss = F.cross_entropy(out, y_tr)
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        out_te = probe(X_te)
        if var_type == "continuous":
            preds = out_te.squeeze().cpu().numpy()
            score = r2_score(y_test, preds)
        else:
            preds = out_te.argmax(dim=1).cpu().numpy()
            # Map back to original class labels for accuracy computation
            classes = np.unique(y_train)
            preds_labels = classes[preds]
            score = accuracy_score(y_test, preds_labels)

    return float(score)


def collect_pooled_activations(model, hdf5_path, layer, variable_types, device,
                                n_trajectories, max_steps_per_traj, seed=42):
    """Collect mean-pooled activations + states for a given model."""
    all_activations = []
    all_states = {v: [] for v in variable_types}
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = min(n_trajectories, len(val_grp))
        traj_indices = rng.choice(len(val_grp), size=n_traj, replace=False)

        with torch.no_grad():
            for traj_idx in tqdm(traj_indices, desc="Collecting activations"):
                traj_grp = val_grp[str(traj_idx)]
                num_steps = int(traj_grp.attrs["length"])
                if num_steps < 2:
                    continue
                steps = rng.choice(num_steps - 1,
                                   size=min(max_steps_per_traj, num_steps - 1),
                                   replace=False)
                for step in steps:
                    obs = traj_grp["observations"][step].flatten().astype(np.int64)
                    obs = np.clip(obs, 0, model.config.vocab_size - 1)
                    tokens = torch.from_numpy(obs).unsqueeze(0).to(device)
                    residual = model.get_residual_stream(tokens)
                    pooled = residual[layer].mean(dim=1).squeeze(0).cpu().numpy()
                    all_activations.append(pooled)
                    for var in variable_types:
                        all_states[var].append(traj_grp[f"states/{var}"][step])

    return np.stack(all_activations), all_states


def main():
    parser = argparse.ArgumentParser(description="Non-linear (MLP) probes with untrained baseline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env", choices=["minigrid", "physics"], required=True)
    parser.add_argument("--scale", choices=["small", "large"], default="small")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-trajectories", type=int, default=400)
    parser.add_argument("--max-steps-per-traj", type=int, default=20)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--config-dir", type=str, default="config")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_config = load_config(f"{args.config_dir}/model_config.yaml")
    config_key = get_config_key(args.env, args.scale)
    hdf5_path = get_hdf5_path(args.env)

    from interpretability.probes import get_variable_types
    variable_types = get_variable_types(hdf5_path)

    # Load trained and build untrained model
    print("Loading TRAINED model...")
    trained = load_model(args.checkpoint, model_config, scale=config_key, device=str(device))
    trained.eval()
    print("Building UNTRAINED model (random weights)...")
    untrained = build_model(model_config, scale=config_key).to(device)
    untrained.eval()

    print(f"\nRunning MLP probes on layer {args.layer} for {args.env}")
    print("Includes untrained baseline — the meaningful metric is (trained MLP - untrained MLP)")
    print("This tests whether training adds NON-LINEAR structure beyond input-preservation.\n")

    print("Collecting TRAINED activations...")
    X_tr, states = collect_pooled_activations(
        trained, hdf5_path, args.layer, variable_types, device,
        args.n_trajectories, args.max_steps_per_traj)
    print("Collecting UNTRAINED activations...")
    X_un, _ = collect_pooled_activations(
        untrained, hdf5_path, args.layer, variable_types, device,
        args.n_trajectories, args.max_steps_per_traj)

    print(f"\nCollected {len(X_tr):,} samples (d_model={X_tr.shape[1]})\n")

    print(f"{'Variable':18s} | {'Trained MLP':>11s} | {'Untrained MLP':>13s} | {'Learned (Δ)':>11s}")
    print("-" * 62)

    results = {}
    for var, var_type in variable_types.items():
        y = np.array(states[var], dtype=np.float64)
        if len(np.unique(y)) < 2:
            print(f"{var:18s} | {'SKIPPED':>11s} | {'SKIPPED':>13s} | {'--':>11s}")
            continue

        Xtr_tr, Xtr_te, ytr, yte = train_test_split(X_tr, y, test_size=0.2, random_state=42)
        Xun_tr, Xun_te, _, _      = train_test_split(X_un, y, test_size=0.2, random_state=42)

        mlp_trained   = train_mlp_probe(Xtr_tr, ytr, Xtr_te, yte, var_type, device, n_epochs=args.n_epochs)
        mlp_untrained = train_mlp_probe(Xun_tr, ytr, Xun_te, yte, var_type, device, n_epochs=args.n_epochs)
        learned = mlp_trained - mlp_untrained

        results[var] = {"trained_mlp": mlp_trained, "untrained_mlp": mlp_untrained, "learned": learned}
        print(f"{var:18s} | {mlp_trained:11.3f} | {mlp_untrained:13.3f} | {learned:+11.3f}")

    print(f"\n=== Interpretation ===")
    print("Learned (Δ) = trained MLP - untrained MLP = genuine non-linear learned structure.")
    print("Positive Δ where LINEAR probes showed zero/negative learned signal would confirm")
    print("that training builds NON-LINEAR representations invisible to linear probes.")

    import json
    out_path = f"results/{args.env}_layer{args.layer}_nonlinear_probes.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")

    if not args.no_wandb:
        try:
            import wandb
            wandb_cfg = model_config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "world-model-interpretability"),
                entity=wandb_cfg.get("entity"),
                name=f"probes_mlp_{args.env}_layer{args.layer}",
                tags=["track-a", "probes", "mlp", args.env],
            )
            for var, r in results.items():
                wandb.log({f"probe_mlp_learned/{var}": r["learned"]})
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
