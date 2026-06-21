"""
interpretability/interventions.py

Causal intervention pipeline via activation patching.

Tests whether representations identified by probes/SAEs are CAUSALLY
involved in model predictions, not merely correlated with state variables.

METHODOLOGICAL HISTORY (documented for dissertation transparency):
  An initial mean-pooled intervention design produced near-zero recovery
  fractions uniformly across all variables. Diagnosis revealed this was a
  genuine experimental design flaw, not a null causal finding: pooling the
  residual stream across all 1083 sequence positions (most of which are
  static wall/floor cells unrelated to agent state) diluted any real
  causal signal to well under 1% of the activation's overall magnitude,
  and patching a single pooled direction does not respect the model's
  actual position-wise causal structure (each output position's logits
  depend on that position's own residual stream via ln_final -> unembed,
  not on a sequence-wide average).

  This module implements THREE intervention designs that respect causal
  structure correctly, each testing a related but distinct hypothesis:

  MODE A — Last-position patching:
    Probe/patch the residual stream at the FINAL sequence position
    (which has attended to the entire input via causal attention).
    Tests: does accumulated sequence-level information causally
    encode state by the end of the sequence?

  MODE B — Agent-cell-position patching:
    Probe/patch the residual stream at the SPECIFIC flattened position
    corresponding to the agent's own grid cell in the observation.
    Tests: does the model causally represent state at the position
    where that information is most directly relevant?

  MODE C — Filtered-loss patching:
    Patch the residual stream at ALL positions (matching the clean
    trajectory's full activation pattern), but evaluate loss ONLY on
    output positions near the agent's cell (a local window), rather
    than across all 1083 positions where most targets are agent-position
    invariant. Tests: is a real causal effect present but diluted by
    measuring it against mostly-irrelevant prediction targets?

Methodology references: Conmy et al. 2023 (Automated Circuit Discovery);
Geiger et al. 2024 (Causal abstractions / interchange interventions).
"""

from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.transformer import WorldModelTransformer


GRID_SIZE = 19          # FourRooms grid dimension
CHANNELS_PER_CELL = 3   # MiniGrid encodes each cell as (object, colour, state)


def agent_position_to_flat_index(agent_x: int, agent_y: int, grid_size: int = GRID_SIZE) -> int:
    """
    Map (agent_x, agent_y) grid coordinates to the flat token index
    where the agent's OWN cell appears in the flattened observation.

    MiniGrid's FullyObsWrapper produces a (H, W, 3) array, flattened
    row-major (y-major, then x, then channel) by .flatten().
    The agent's cell starts at flat index: (y * grid_size + x) * 3.
    We return the index of the first channel (object type) of that cell.
    """
    return (agent_y * grid_size + agent_x) * CHANNELS_PER_CELL


@dataclass
class InterventionResult:
    """Result of one activation patching experiment."""
    mode: str
    variable: str
    clean_value: float
    corrupted_value: float
    clean_loss: float
    corrupted_loss: float
    patched_loss: float
    recovery_fraction: float
    delta_magnitude: float = 0.0
    activation_norm: float = 0.0


class ResidualStreamPatcher:
    """Hooks into the transformer to capture and patch residual stream activations."""

    def __init__(self, model: WorldModelTransformer, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._cached_activation: Optional[torch.Tensor] = None
        self._patch_activation: Optional[torch.Tensor] = None

    def _capture_hook(self, module, input, output):
        self._cached_activation = output.detach().clone()
        return output

    def _patch_hook(self, module, input, output):
        if self._patch_activation is not None:
            return self._patch_activation
        return output

    def capture(self, tokens: torch.Tensor) -> torch.Tensor:
        handle = self.model.blocks[self.layer_idx].register_forward_hook(self._capture_hook)
        with torch.no_grad():
            self.model(tokens)
        handle.remove()
        return self._cached_activation

    def run_with_patch(self, tokens: torch.Tensor, patch_activation: torch.Tensor) -> torch.Tensor:
        self._patch_activation = patch_activation
        verification = {"ok": None}

        def _verify_hook(module, input, output):
            result = self._patch_hook(module, input, output)
            verification["ok"] = torch.allclose(result, patch_activation)
            return result

        handle = self.model.blocks[self.layer_idx].register_forward_hook(_verify_hook)
        with torch.no_grad():
            logits, _ = self.model(tokens)
        handle.remove()
        self._patch_activation = None

        if not verification["ok"]:
            raise RuntimeError("Patch hook did not correctly replace block output.")
        return logits


def fit_probe_direction(
    activations_at_position: np.ndarray,
    values: np.ndarray,
) -> tuple[torch.Tensor, StandardScaler]:
    """
    Fit a Ridge regression probe direction on activations from a SPECIFIC
    position (not pooled), with standardisation to avoid ill-conditioning.

    Returns:
        direction: unit-norm direction in ORIGINAL (unscaled) activation space
        scaler:    fitted StandardScaler (kept for potential reuse/debugging)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations_at_position)

    probe = Ridge(alpha=1.0)
    probe.fit(X_scaled, values)

    direction_raw = probe.coef_ / scaler.scale_
    direction = torch.from_numpy(direction_raw).float()
    direction = direction / direction.norm()
    return direction, scaler


def collect_position_specific_samples(
    hdf5_path: str,
    layer_idx: int,
    variable: str,
    patcher: ResidualStreamPatcher,
    device: torch.device,
    mode: str,
    n_samples: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect activations at the position relevant to `mode`:
      - "last":       final sequence position (index -1)
      - "agent_cell": the flat index corresponding to the agent's own cell

    Returns:
        activations: (N, d_model) array at the target position
        values:      (N,) ground-truth values for `variable`
    """
    rng = np.random.default_rng(seed)
    sample_activations = []
    sample_values = []

    with h5py.File(hdf5_path, "r") as f:
        val_grp = f["trajectories/val"]
        n_traj = len(val_grp)
        sample_traj_idx = rng.choice(n_traj, size=min(n_samples, n_traj), replace=False)

        for traj_idx in sample_traj_idx:
            traj_grp = val_grp[str(traj_idx)]
            num_steps = int(traj_grp.attrs["length"])
            if num_steps < 2:
                continue
            step = rng.integers(0, num_steps - 1)

            obs = traj_grp["observations"][step].flatten().astype(np.int64)
            obs = np.clip(obs, 0, 31)
            tokens = torch.from_numpy(obs).unsqueeze(0).to(device)

            activation = patcher.capture(tokens)  # (1, T, d_model)

            if mode == "last":
                pos_idx = activation.shape[1] - 1
            elif mode == "agent_cell":
                ax = int(traj_grp["states/agent_x"][step])
                ay = int(traj_grp["states/agent_y"][step])
                pos_idx = min(agent_position_to_flat_index(ax, ay), activation.shape[1] - 1)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            sample_activations.append(activation[0, pos_idx].cpu().numpy())
            sample_values.append(traj_grp[f"states/{variable}"][step])

    return np.stack(sample_activations), np.array(sample_values, dtype=np.float64)


def run_intervention_mode_a_or_b(
    model: WorldModelTransformer,
    hdf5_path: str,
    layer_idx: int,
    variable: str,
    device: torch.device,
    mode: str,
    n_pairs: int = 150,
    seed: int = 42,
) -> list[InterventionResult]:
    """
    MODE A (mode="last") or MODE B (mode="agent_cell"):
    Patch a single, well-defined sequence position and evaluate loss
    AT THAT SAME POSITION ONLY. This respects the model's causal
    structure: position i's logits depend only on its own final
    residual stream value, so patching and evaluating at the same
    position is mechanically sound.
    """
    rng = np.random.default_rng(seed)
    patcher = ResidualStreamPatcher(model, layer_idx)

    X, y = collect_position_specific_samples(
        hdf5_path, layer_idx, variable, patcher, device, mode, seed=seed
    )
    direction, _ = fit_probe_direction(X, y)
    direction = direction.to(device)

    results = []
    n_skipped_small_diff = 0
    n_skipped_tiny_gap = 0
    n_attempted = 0

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
            n_attempted += 1
            if abs(clean_val - corrupt_val) < 1.0:
                n_skipped_small_diff += 1
                continue

            clean_obs = np.clip(clean_grp["observations"][c_step].flatten().astype(np.int64), 0, 31)
            clean_target = np.clip(clean_grp["observations"][c_step + 1].flatten().astype(np.int64), 0, 31)
            corrupt_obs = np.clip(corrupt_grp["observations"][x_step].flatten().astype(np.int64), 0, 31)

            clean_tokens = torch.from_numpy(clean_obs).unsqueeze(0).to(device)
            clean_targets_full = torch.from_numpy(clean_target).unsqueeze(0).to(device)
            corrupt_tokens = torch.from_numpy(corrupt_obs).unsqueeze(0).to(device)

            # Determine the target position for this mode.
            # NOTE: "last" mode evaluates a FIXED spatial grid cell (always the
            # same flat index = bottom-right of the grid in row-major flattening),
            # regardless of where the agent is. This position's target token is
            # often agent-position-INVARIANT (e.g. consistently a wall cell),
            # which produces near-zero clean/corrupted loss gaps and causes most
            # pairs to be skipped. This is documented as a finding, not silently
            # discarded — see n_skipped_tiny_gap diagnostic below.
            if mode == "last":
                pos_idx = len(clean_obs) - 1
            else:  # agent_cell — use the CLEAN trajectory's agent position
                ax = int(clean_grp["states/agent_x"][c_step])
                ay = int(clean_grp["states/agent_y"][c_step])
                pos_idx = min(agent_position_to_flat_index(ax, ay), len(clean_obs) - 1)

            with torch.no_grad():
                clean_logits, _ = model(clean_tokens)
                clean_loss = F.cross_entropy(
                    clean_logits[0, pos_idx].unsqueeze(0), clean_targets_full[0, pos_idx].unsqueeze(0)
                )

                corrupt_logits, _ = model(corrupt_tokens)
                corrupt_loss = F.cross_entropy(
                    corrupt_logits[0, pos_idx].unsqueeze(0), clean_targets_full[0, pos_idx].unsqueeze(0)
                )

            clean_activation = patcher.capture(clean_tokens)
            corrupt_activation = patcher.capture(corrupt_tokens)

            clean_proj = (clean_activation[0, pos_idx] @ direction)
            corrupt_proj = (corrupt_activation[0, pos_idx] @ direction)
            delta = (clean_proj - corrupt_proj)

            activation_norm = corrupt_activation[0, pos_idx].norm().item()
            delta_magnitude = delta.abs().item()

            patched_activation = corrupt_activation.clone()
            patched_activation[0, pos_idx] = corrupt_activation[0, pos_idx] + delta * direction

            patched_logits = patcher.run_with_patch(corrupt_tokens, patched_activation)
            patched_loss = F.cross_entropy(
                patched_logits[0, pos_idx].unsqueeze(0), clean_targets_full[0, pos_idx].unsqueeze(0)
            )

            gap = corrupt_loss.item() - clean_loss.item()
            if abs(gap) < 1e-6:
                n_skipped_tiny_gap += 1
                continue
            recovered = corrupt_loss.item() - patched_loss.item()
            recovery_fraction = recovered / gap

            results.append(InterventionResult(
                mode=mode, variable=variable, clean_value=clean_val, corrupted_value=corrupt_val,
                clean_loss=clean_loss.item(), corrupted_loss=corrupt_loss.item(),
                patched_loss=patched_loss.item(), recovery_fraction=recovery_fraction,
                delta_magnitude=delta_magnitude, activation_norm=activation_norm,
            ))

    print(f"    [diagnostic] attempted={n_attempted}, "
          f"skipped(small_diff)={n_skipped_small_diff}, "
          f"skipped(tiny_gap)={n_skipped_tiny_gap}, "
          f"valid={len(results)}")

    return results


def run_intervention_mode_c(
    model: WorldModelTransformer,
    hdf5_path: str,
    layer_idx: int,
    variable: str,
    device: torch.device,
    n_pairs: int = 150,
    window: int = 5,
    seed: int = 42,
) -> list[InterventionResult]:
    """
    MODE C: Patch the FULL residual stream at every position (substituting
    the entire clean activation pattern in place of the corrupted one),
    but evaluate loss only on positions in a local window around the
    agent's cell — where prediction targets are actually expected to
    depend on agent position, rather than averaging in 1000+ static
    wall/floor positions that dilute any real effect.
    """
    rng = np.random.default_rng(seed)
    patcher = ResidualStreamPatcher(model, layer_idx)

    # We don't need a probe direction for Mode C — we patch the FULL
    # activation (clean replaces corrupted entirely), which is the simplest
    # possible "is this layer's representation causally sufficient" test.
    results = []
    n_skipped_small_diff = 0
    n_skipped_tiny_gap = 0
    n_attempted = 0

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
            n_attempted += 1
            if abs(clean_val - corrupt_val) < 1.0:
                n_skipped_small_diff += 1
                continue

            clean_obs = np.clip(clean_grp["observations"][c_step].flatten().astype(np.int64), 0, 31)
            clean_target = np.clip(clean_grp["observations"][c_step + 1].flatten().astype(np.int64), 0, 31)
            corrupt_obs = np.clip(corrupt_grp["observations"][x_step].flatten().astype(np.int64), 0, 31)

            ax = int(clean_grp["states/agent_x"][c_step])
            ay = int(clean_grp["states/agent_y"][c_step])
            centre_idx = agent_position_to_flat_index(ax, ay)
            window_start = max(0, centre_idx - window * CHANNELS_PER_CELL)
            window_end = min(len(clean_obs), centre_idx + (window + 1) * CHANNELS_PER_CELL)
            window_slice = slice(window_start, window_end)

            clean_tokens = torch.from_numpy(clean_obs).unsqueeze(0).to(device)
            clean_targets_full = torch.from_numpy(clean_target).unsqueeze(0).to(device)
            corrupt_tokens = torch.from_numpy(corrupt_obs).unsqueeze(0).to(device)

            with torch.no_grad():
                clean_logits, _ = model(clean_tokens)
                clean_loss = F.cross_entropy(
                    clean_logits[0, window_slice], clean_targets_full[0, window_slice]
                )
                corrupt_logits, _ = model(corrupt_tokens)
                corrupt_loss = F.cross_entropy(
                    corrupt_logits[0, window_slice], clean_targets_full[0, window_slice]
                )

            # Full activation patch: substitute clean's entire residual stream
            clean_activation = patcher.capture(clean_tokens)
            activation_norm = clean_activation.norm(dim=-1).mean().item()

            patched_logits = patcher.run_with_patch(corrupt_tokens, clean_activation)
            patched_loss = F.cross_entropy(
                patched_logits[0, window_slice], clean_targets_full[0, window_slice]
            )

            gap = corrupt_loss.item() - clean_loss.item()
            if abs(gap) < 1e-6:
                n_skipped_tiny_gap += 1
                continue
            recovered = corrupt_loss.item() - patched_loss.item()
            recovery_fraction = recovered / gap

            results.append(InterventionResult(
                mode="filtered_full", variable=variable, clean_value=clean_val,
                corrupted_value=corrupt_val, clean_loss=clean_loss.item(),
                corrupted_loss=corrupt_loss.item(), patched_loss=patched_loss.item(),
                recovery_fraction=recovery_fraction, delta_magnitude=0.0,
                activation_norm=activation_norm,
            ))

    print(f"    [diagnostic] attempted={n_attempted}, "
          f"skipped(small_diff)={n_skipped_small_diff}, "
          f"skipped(tiny_gap)={n_skipped_tiny_gap}, "
          f"valid={len(results)}")

    return results


def summarise_intervention_results(results: list[InterventionResult]) -> dict:
    """Aggregate intervention results into summary statistics.

    Always returns the full set of keys, even when results is empty,
    so downstream code can safely access any field without a KeyError.
    NaN values indicate the experiment produced no valid pairs — this
    is itself diagnostic information (see the printed [diagnostic] line
    from the calling function for WHY pairs were filtered out).
    """
    if not results:
        return {
            "n_pairs": 0,
            "mean_recovery_fraction": float("nan"),
            "median_recovery_fraction": float("nan"),
            "std_recovery_fraction": float("nan"),
            "mean_clean_loss": float("nan"),
            "mean_corrupted_loss": float("nan"),
            "mean_patched_loss": float("nan"),
            "mean_delta_magnitude": float("nan"),
            "mean_activation_norm": float("nan"),
            "relative_patch_size": float("nan"),
        }

    recoveries = np.array([r.recovery_fraction for r in results])
    recoveries_clipped = np.clip(recoveries, -2.0, 2.0)
    delta_mags = np.array([r.delta_magnitude for r in results])
    act_norms = np.array([r.activation_norm for r in results])

    return {
        "n_pairs": len(results),
        "mean_recovery_fraction": float(np.mean(recoveries_clipped)),
        "median_recovery_fraction": float(np.median(recoveries_clipped)),
        "std_recovery_fraction": float(np.std(recoveries_clipped)),
        "mean_clean_loss": float(np.mean([r.clean_loss for r in results])),
        "mean_corrupted_loss": float(np.mean([r.corrupted_loss for r in results])),
        "mean_patched_loss": float(np.mean([r.patched_loss for r in results])),
        "mean_delta_magnitude": float(np.mean(delta_mags)),
        "mean_activation_norm": float(np.mean(act_norms)),
        "relative_patch_size": float(np.mean(delta_mags) / np.mean(act_norms)) if np.mean(act_norms) > 0 else 0.0,
    }
