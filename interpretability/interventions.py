"""
interpretability/interventions.py

Causal intervention pipeline via activation patching.

Tests whether representations identified by probes/SAEs are CAUSALLY
involved in model predictions, not merely correlated with state variables.

Methodology (Conmy et al. 2023 — Automated Circuit Discovery;
Geiger et al. 2024 — Causal abstractions / interchange interventions):

  1. Run a "clean" trajectory through the model, cache residual stream
     at the target layer. The clean run has a known state value
     (e.g. agent_x=5).
  2. Run a "corrupted" trajectory with a DIFFERENT state value for the
     same variable (e.g. agent_x=10).
  3. Patch: during the corrupted run's forward pass, replace the
     residual stream activation at the target layer with the clean
     run's activation (either the full residual stream, a single SAE
     feature, or a linear probe direction).
  4. Measure the effect on the model's output (next-token logits).
     If the patched representation is causally load-bearing, the
     model's prediction should shift toward what it would predict
     for the CLEAN state, not the corrupted state.

Two intervention targets are supported:
  - SAE feature patching: zero out / set a specific SAE feature's
    activation, reconstruct, and substitute into the residual stream.
  - Linear probe direction patching: patch along the geometric
    direction the linear probe found (the probe's weight vector),
    a more direct test of whether the PROBED direction is causal.

Causal fidelity metric: correlation between the intervention's predicted
effect (based on the clean/corrupted state difference) and the
OBSERVED effect on model output. High correlation = causally involved.
"""

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from models.transformer import WorldModelTransformer
from interpretability.sae import SparseAutoencoder


@dataclass
class InterventionResult:
    """Result of one activation patching experiment."""
    variable: str
    clean_value: float
    corrupted_value: float
    clean_loss: float          # model loss predicting clean target from clean input
    corrupted_loss: float      # model loss predicting clean target from corrupted input (no patch)
    patched_loss: float        # model loss predicting clean target from corrupted input WITH patch
    recovery_fraction: float   # how much of the clean-corrupted gap the patch recovers


class ResidualStreamPatcher:
    """
    Hooks into the transformer to capture and patch residual stream
    activations at a specific layer during forward passes.
    """

    def __init__(self, model: WorldModelTransformer, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._cached_activation: Optional[torch.Tensor] = None
        self._patch_activation: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _capture_hook(self, module, input, output):
        """Forward hook that caches the block's output (residual stream)."""
        self._cached_activation = output.detach().clone()
        return output

    def _patch_hook(self, module, input, output):
        """Forward hook that replaces the block's output with a patched value."""
        if self._patch_activation is not None:
            return self._patch_activation
        return output

    def capture(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and capture the residual stream at layer_idx.

        Returns:
            (B, T, d_model) residual stream activation at the target layer.
        """
        handle = self.model.blocks[self.layer_idx].register_forward_hook(self._capture_hook)
        with torch.no_grad():
            self.model(tokens)
        handle.remove()
        return self._cached_activation

    def run_with_patch(
        self,
        tokens: torch.Tensor,
        patch_activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a forward pass with the residual stream at layer_idx replaced
        by patch_activation.

        Args:
            tokens:           (B, T) input tokens
            patch_activation: (B, T, d_model) activation to substitute

        Returns:
            logits: (B, T, vocab_size) model output with the patch applied
        """
        self._patch_activation = patch_activation
        handle = self.model.blocks[self.layer_idx].register_forward_hook(self._patch_hook)
        with torch.no_grad():
            logits, _ = self.model(tokens)
        handle.remove()
        self._patch_activation = None
        return logits


def run_linear_direction_intervention(
    model: WorldModelTransformer,
    hdf5_path: str,
    layer_idx: int,
    variable: str,
    device: torch.device,
    n_pairs: int = 100,
    seed: int = 42,
) -> list[InterventionResult]:
    """
    Test causal involvement of the LINEAR PROBE DIRECTION for a given variable.

    Procedure per (clean, corrupted) trajectory pair:
      1. Fit a quick linear probe direction for `variable` using a sample
         of activations (reuses the same Ridge regression approach as probes.py)
      2. Capture clean and corrupted residual streams
      3. Patch ONLY the component of the corrupted activation along the
         probe direction, replacing it with the clean activation's component
         along that direction (leaving orthogonal components untouched)
      4. Measure next-token prediction loss in three conditions:
         clean (baseline), corrupted (no patch), corrupted+patched

    A causally-involved direction should show patched_loss much closer
    to clean_loss than corrupted_loss is.

    Args:
        model:      trained transformer
        hdf5_path:  path to HDF5 trajectory file
        layer_idx:  which layer's residual stream to intervene on
        variable:   state variable name (e.g. "agent_x")
        device:     torch device
        n_pairs:    number of clean/corrupted trajectory pairs to test
        seed:       random seed

    Returns:
        List of InterventionResult, one per tested pair.
    """
    rng = np.random.default_rng(seed)
    patcher = ResidualStreamPatcher(model, layer_idx)

    # Step 1: Collect a sample of activations + states to fit the probe direction
    sample_activations = []
    sample_values = []

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = len(val_grp)
        sample_traj_idx = rng.choice(n_traj, size=min(300, n_traj), replace=False)

        for traj_idx in sample_traj_idx:
            traj_grp = val_grp[str(traj_idx)]
            num_steps = int(traj_grp.attrs["length"])
            if num_steps < 2:
                continue
            step = rng.integers(0, num_steps - 1)

            obs = traj_grp["observations"][step].flatten().astype(np.int64)
            obs = np.clip(obs, 0, model.config.vocab_size - 1)
            tokens = torch.from_numpy(obs).unsqueeze(0).to(device)

            activation = patcher.capture(tokens)
            pooled = activation.mean(dim=1).squeeze(0).cpu().numpy()
            sample_activations.append(pooled)
            sample_values.append(traj_grp[f"states/{variable}"][step])

    X = np.stack(sample_activations)
    y = np.array(sample_values, dtype=np.float64)

    # Fit linear probe direction (Ridge regression weight vector)
    probe = Ridge(alpha=1.0)
    probe.fit(X, y)
    direction = torch.from_numpy(probe.coef_).float().to(device)  # (d_model,)
    direction = direction / direction.norm()  # unit vector

    # Step 2: Run clean/corrupted pairs with patching
    results = []

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = len(val_grp)
        pair_indices = rng.choice(n_traj, size=(n_pairs, 2), replace=True)

        for clean_idx, corrupt_idx in pair_indices:
            clean_grp = val_grp[str(clean_idx)]
            corrupt_grp = val_grp[str(corrupt_idx)]

            clean_steps = int(clean_grp.attrs["length"])
            corrupt_steps = int(corrupt_grp.attrs["length"])
            if clean_steps < 2 or corrupt_steps < 2:
                continue

            c_step = rng.integers(0, clean_steps - 1)
            x_step = rng.integers(0, corrupt_steps - 1)

            clean_val = float(clean_grp[f"states/{variable}"][c_step])
            corrupt_val = float(corrupt_grp[f"states/{variable}"][x_step])

            # Skip pairs with no meaningful difference (patching wouldn't show anything)
            if abs(clean_val - corrupt_val) < 1.0:
                continue

            clean_obs = np.clip(clean_grp["observations"][c_step].flatten().astype(np.int64),
                                0, model.config.vocab_size - 1)
            clean_target = np.clip(clean_grp["observations"][c_step + 1].flatten().astype(np.int64),
                                   0, model.config.vocab_size - 1)
            corrupt_obs = np.clip(corrupt_grp["observations"][x_step].flatten().astype(np.int64),
                                  0, model.config.vocab_size - 1)

            clean_tokens = torch.from_numpy(clean_obs).unsqueeze(0).to(device)
            clean_targets = torch.from_numpy(clean_target).unsqueeze(0).to(device)
            corrupt_tokens = torch.from_numpy(corrupt_obs).unsqueeze(0).to(device)

            # Baseline: clean loss (model predicting clean target from clean input)
            with torch.no_grad():
                clean_logits, clean_loss = model(clean_tokens, clean_targets)

            # Corrupted (no patch): how well does corrupted input predict CLEAN target?
            with torch.no_grad():
                corrupt_logits, _ = model(corrupt_tokens)
                corrupt_loss = F.cross_entropy(
                    corrupt_logits.view(-1, model.config.vocab_size),
                    clean_targets.view(-1),
                )

            # Capture clean activation, patch its probe-direction component
            # into the corrupted run
            clean_activation = patcher.capture(clean_tokens)       # (1, T, d_model)
            corrupt_activation = patcher.capture(corrupt_tokens)   # (1, T, d_model)

            # Project both onto the probe direction, swap that component
            clean_proj = (clean_activation @ direction).unsqueeze(-1) * direction
            corrupt_proj = (corrupt_activation @ direction).unsqueeze(-1) * direction

            patched_activation = corrupt_activation - corrupt_proj + clean_proj

            patched_logits = patcher.run_with_patch(corrupt_tokens, patched_activation)
            patched_loss = F.cross_entropy(
                patched_logits.view(-1, model.config.vocab_size),
                clean_targets.view(-1),
            )

            # Recovery fraction: how much of the (corrupted - clean) loss gap
            # does the patch recover? 1.0 = full recovery (patch fully restores
            # clean-like prediction), 0.0 = no effect, negative = made it worse
            gap = corrupt_loss.item() - clean_loss.item()
            if abs(gap) < 1e-6:
                continue
            recovered = corrupt_loss.item() - patched_loss.item()
            recovery_fraction = recovered / gap

            results.append(InterventionResult(
                variable=variable,
                clean_value=clean_val,
                corrupted_value=corrupt_val,
                clean_loss=clean_loss.item(),
                corrupted_loss=corrupt_loss.item(),
                patched_loss=patched_loss.item(),
                recovery_fraction=recovery_fraction,
            ))

    return results


def summarise_intervention_results(results: list[InterventionResult]) -> dict:
    """
    Aggregate intervention results into summary statistics.

    mean_recovery_fraction close to 1.0 = strong causal evidence
    (patching the probed direction reliably restores clean-like behaviour).
    Close to 0.0 = the direction is not causally load-bearing despite
    being correlationally decodable by the probe.
    """
    if not results:
        return {"n_pairs": 0, "mean_recovery_fraction": float("nan")}

    recoveries = np.array([r.recovery_fraction for r in results])
    # Clip extreme outliers (recovery fraction can blow up when gap is tiny)
    recoveries_clipped = np.clip(recoveries, -2.0, 2.0)

    return {
        "n_pairs": len(results),
        "mean_recovery_fraction": float(np.mean(recoveries_clipped)),
        "median_recovery_fraction": float(np.median(recoveries_clipped)),
        "std_recovery_fraction": float(np.std(recoveries_clipped)),
        "mean_clean_loss": float(np.mean([r.clean_loss for r in results])),
        "mean_corrupted_loss": float(np.mean([r.corrupted_loss for r in results])),
        "mean_patched_loss": float(np.mean([r.patched_loss for r in results])),
    }
