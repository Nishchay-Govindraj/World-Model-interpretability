"""
interpretability/sae.py

Sparse Autoencoder (SAE) for decomposing transformer residual stream
activations into sparse, interpretable features.

Methodology (Cunningham et al. 2023; Bricken et al. 2023):
  - Overcomplete dictionary: d_hidden = expansion_factor * d_model
  - ReLU activation enforces non-negativity (interpretable as feature presence)
  - L1 penalty on hidden activations enforces sparsity
  - Tied or untied decoder weights (we use untied — more expressive, standard in
    recent literature per Templeton et al. 2024)

Loss = reconstruction_loss + l1_coefficient * sparsity_penalty
     = ||x - x_hat||^2 + lambda * ||features||_1

Evaluation:
  - Reconstruction MSE (lower = better fidelity)
  - L0 sparsity: average number of non-zero features per sample (lower = sparser)
  - Feature-to-variable correspondence: mutual information between each SAE
    feature's activation and each ground-truth state variable. High MI features
    are candidates for interpretable, monosemantic features.

The gap this addresses (per dissertation proposal): SAEs have been applied to
discrete game world models (OthelloGPT, ChessGPT) but not to physically richer
continuous-dynamics environments. This MiniGrid application is a stepping stone
toward the Physics sandbox SAE work.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from tqdm import tqdm

from models.transformer import WorldModelTransformer


@dataclass
class SAEConfig:
    """Hyperparameters for the sparse autoencoder."""
    d_model: int                  # input dimension (matches transformer d_model)
    expansion_factor: int = 8     # d_hidden = expansion_factor * d_model
    l1_coefficient: float = 1e-3  # sparsity penalty weight
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 50

    @property
    def d_hidden(self) -> int:
        return self.expansion_factor * self.d_model


class SparseAutoencoder(nn.Module):
    """
    Standard ReLU sparse autoencoder.

    Architecture:
        encode: x -> ReLU(W_enc @ x + b_enc)     -> features (d_hidden,)
        decode: features -> W_dec @ features + b_dec -> x_hat (d_model,)

    Decoder weights are initialised as the transpose of encoder weights
    (standard practice) but trained independently (untied).
    Decoder columns are normalised to unit norm after each step to prevent
    the trivial solution of shrinking the dictionary to reduce L1 penalty
    while inflating reconstruction via large weights.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        self.W_enc = nn.Parameter(torch.empty(config.d_hidden, config.d_model))
        self.b_enc = nn.Parameter(torch.zeros(config.d_hidden))
        self.W_dec = nn.Parameter(torch.empty(config.d_model, config.d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialise encoder with small random weights, decoder as encoder transpose.
        This is standard SAE initialisation (Bricken et al. 2023).
        """
        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self._normalise_decoder()

    def _normalise_decoder(self) -> None:
        """Normalise each decoder column (feature direction) to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.norm(dim=0, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) -> features: (B, d_hidden)"""
        return torch.relu(x @ self.W_enc.t() + self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, d_hidden) -> x_hat: (B, d_model)"""
        return features @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_hat:    (B, d_model) reconstruction
            features: (B, d_hidden) sparse feature activations
        """
        # Centre input by decoder bias (standard SAE practice — b_dec acts
        # as a learned mean to subtract before encoding)
        x_centred = x - self.b_dec
        features = torch.relu(x_centred @ self.W_enc.t() + self.b_enc)
        x_hat = self.decode(features)
        return x_hat, features

    def compute_loss(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute SAE loss: reconstruction MSE + L1 sparsity penalty.

        Returns:
            total_loss: scalar tensor for backprop
            metrics:    dict with breakdown (recon_loss, l1_loss, l0_sparsity)
        """
        x_hat, features = self.forward(x)

        recon_loss = ((x_hat - x) ** 2).sum(dim=-1).mean()
        l1_loss = features.abs().sum(dim=-1).mean()

        total_loss = recon_loss + self.config.l1_coefficient * l1_loss

        with torch.no_grad():
            l0_sparsity = (features > 0).float().sum(dim=-1).mean()

        metrics = {
            "recon_loss": recon_loss.item(),
            "l1_loss": l1_loss.item(),
            "l0_sparsity": l0_sparsity.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics


def collect_layer_activations(
    model: WorldModelTransformer,
    hdf5_path: str,
    layer_idx: int,
    device: torch.device,
    n_trajectories: int = 1000,
    max_steps_per_traj: int = 30,
    split: str = "val",
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Collect residual stream activations at a specific layer for SAE training.

    SAEs need more data than linear probes (thousands of samples per feature
    direction), so this samples more trajectories than the probe pipeline.

    Returns:
        activations: (N, d_model) array
        states:      dict of var_name -> (N,) array, for later feature analysis
    """
    rng = np.random.default_rng(seed)
    model.eval()

    state_var_names = ["agent_x", "agent_y", "agent_direction",
                       "goal_x", "goal_y", "carrying"]

    with h5py.File(hdf5_path, "r") as f:
        split_grp = f[f"trajectories/{split}"]
        available = len(split_grp)
        n_trajectories = min(n_trajectories, available)
        traj_indices = rng.choice(available, size=n_trajectories, replace=False)

        all_activations = []
        all_states: dict[str, list] = {var: [] for var in state_var_names}

        with torch.no_grad():
            for traj_idx in tqdm(traj_indices, desc=f"Collecting layer {layer_idx} activations"):
                traj_grp = split_grp[str(traj_idx)]
                num_steps = int(traj_grp.attrs["length"])

                steps_to_use = min(max_steps_per_traj, num_steps - 1)
                if steps_to_use < 1:
                    continue

                step_indices = rng.choice(num_steps - 1, size=steps_to_use, replace=False)

                for step in step_indices:
                    obs = traj_grp["observations"][step].flatten().astype(np.int64)
                    obs = np.clip(obs, 0, model.config.vocab_size - 1)

                    tokens = torch.from_numpy(obs).unsqueeze(0).to(device)
                    residual_stream = model.get_residual_stream(tokens)

                    # Mean-pool over sequence positions for this layer
                    pooled = residual_stream[layer_idx].mean(dim=1).squeeze(0).cpu().numpy()
                    all_activations.append(pooled)

                    for var in state_var_names:
                        all_states[var].append(traj_grp[f"states/{var}"][step])

    activations = np.stack(all_activations)
    states = {var: np.array(vals) for var, vals in all_states.items()}
    return activations, states


def train_sae(
    activations: np.ndarray,
    config: SAEConfig,
    device: torch.device,
    use_wandb: bool = False,
    wandb_run=None,
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on collected residual stream activations.

    Args:
        activations: (N, d_model) array of residual stream activations
        config:      SAEConfig hyperparameters
        device:      torch device
        use_wandb:   whether to log training metrics
        wandb_run:   active wandb run object (if use_wandb=True)

    Returns:
        Trained SparseAutoencoder, in eval mode.
    """
    sae = SparseAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)

    # Cosine learning rate decay — addresses the oscillation seen with constant LR
    # where reconstruction loss degrades after an initial good minimum because
    # the optimizer overshoots as L1 pressure compounds over many epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.n_epochs, eta_min=config.learning_rate * 0.01
    )

    data = torch.from_numpy(activations).float()
    n_samples = data.shape[0]

    print(f"\nTraining SAE: d_model={config.d_model} -> d_hidden={config.d_hidden} "
          f"(expansion={config.expansion_factor}x)")
    print(f"Training samples: {n_samples:,}")
    print(f"L1 coefficient: {config.l1_coefficient}")

    sae.train()
    step = 0
    best_recon_loss = float("inf")
    best_state_dict = None

    for epoch in range(config.n_epochs):
        perm = torch.randperm(n_samples)
        epoch_metrics = {"recon_loss": 0.0, "l1_loss": 0.0, "l0_sparsity": 0.0}
        n_batches = 0

        for i in range(0, n_samples, config.batch_size):
            batch_idx = perm[i:i + config.batch_size]
            x = data[batch_idx].to(device)

            loss, metrics = sae.compute_loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Renormalise decoder columns after each step — prevents the SAE
            # from gaming the L1 penalty by shrinking the dictionary
            sae._normalise_decoder()

            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            n_batches += 1
            step += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:3d}/{config.n_epochs} | "
              f"recon={epoch_metrics['recon_loss']:.4f} | "
              f"l1={epoch_metrics['l1_loss']:.4f} | "
              f"L0={epoch_metrics['l0_sparsity']:.1f}/{config.d_hidden} | "
              f"lr={current_lr:.2e}")

        # Track best checkpoint by reconstruction loss — the final epoch isn't
        # necessarily the best, since L1 pressure can degrade reconstruction
        # over long training if the optimizer doesn't fully converge
        if epoch_metrics["recon_loss"] < best_recon_loss:
            best_recon_loss = epoch_metrics["recon_loss"]
            best_state_dict = {k: v.clone() for k, v in sae.state_dict().items()}

        if use_wandb and wandb_run is not None:
            wandb_run.log({
                "sae/recon_loss": epoch_metrics["recon_loss"],
                "sae/l1_loss": epoch_metrics["l1_loss"],
                "sae/l0_sparsity": epoch_metrics["l0_sparsity"],
                "sae/lr": current_lr,
            }, step=epoch)

    # Restore best checkpoint rather than using the final (possibly degraded) epoch
    if best_state_dict is not None:
        sae.load_state_dict(best_state_dict)
        print(f"\nRestored best checkpoint (recon_loss={best_recon_loss:.4f})")

    sae.eval()
    return sae


def compute_feature_correspondence(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    states: dict[str, np.ndarray],
    device: torch.device,
    top_k: int = 10,
    max_mi_samples: int = 3000,
    seed: int = 42,
) -> dict[str, list[tuple[int, float]]]:
    """
    Compute mutual information between each SAE feature and each ground-truth
    state variable. Identifies which features (if any) correspond to
    interpretable, monosemantic concepts.

    sklearn's mutual_info_regression/classif use a k-NN estimator that scales
    poorly with both sample count and feature count (here: 2048 features).
    We subsample to max_mi_samples before computing MI — this is standard
    practice for MI estimation (the k-NN estimator's accuracy depends on local
    density, not raw sample count, so a few thousand samples is sufficient and
    avoids multi-minute runtimes per variable).

    Args:
        sae:             trained SparseAutoencoder
        activations:     (N, d_model) activations used to compute features
        states:          dict of var_name -> (N,) ground-truth values
        device:          torch device
        top_k:           number of top-correspondence features to report per variable
        max_mi_samples:  cap on samples used for MI estimation (speed/accuracy tradeoff)
        seed:            random seed for subsampling

    Returns:
        dict of var_name -> list of (feature_idx, mutual_info_score) tuples,
        sorted by MI score descending, top_k per variable.
    """
    sae.eval()
    with torch.no_grad():
        x = torch.from_numpy(activations).float().to(device)
        _, features = sae(x)
        features_np = features.cpu().numpy()  # (N, d_hidden)

    # Subsample for tractable MI computation
    n_samples = features_np.shape[0]
    if n_samples > max_mi_samples:
        rng = np.random.default_rng(seed)
        subsample_idx = rng.choice(n_samples, size=max_mi_samples, replace=False)
        features_np = features_np[subsample_idx]
        states = {k: v[subsample_idx] for k, v in states.items()}
        print(f"Subsampled {max_mi_samples:,} of {n_samples:,} activations for MI estimation")

    results = {}
    state_var_types = {
        "agent_x": "continuous", "agent_y": "continuous",
        "agent_direction": "categorical",
        "goal_x": "continuous", "goal_y": "continuous",
        "carrying": "categorical",
    }

    for var_name, var_type in state_var_types.items():
        y = states[var_name]

        # Skip degenerate variables (e.g. carrying always 0)
        if len(np.unique(y)) < 2:
            results[var_name] = []
            print(f"  {var_name:16s}: skipped (degenerate)")
            continue

        print(f"  Computing MI for {var_name}...")
        if var_type == "continuous":
            mi_scores = mutual_info_regression(
                features_np, y, random_state=42, n_neighbors=3
            )
        else:
            mi_scores = mutual_info_classif(
                features_np, y, random_state=42, n_neighbors=3
            )

        top_indices = np.argsort(mi_scores)[::-1][:top_k]
        results[var_name] = [(int(idx), float(mi_scores[idx])) for idx in top_indices]
        print(f"  {var_name:16s}: done — top feature F{top_indices[0]} "
              f"(MI={mi_scores[top_indices[0]]:.4f})")

    return results
