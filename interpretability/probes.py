"""
interpretability/probes.py

Linear probe training and evaluation pipeline.

Tests whether the transformer's residual stream linearly encodes
ground-truth state variables (agent position, direction, goal location)
that were NEVER provided as model input during training.

Methodology (following Li et al. 2023 — Othello-GPT; Belinkov, 2022):
  1. Run held-out trajectories through the trained model
  2. Cache residual stream activations at every layer
  3. For each (layer, state_variable) pair, train a linear probe:
       - continuous variables (position) -> Ridge regression, report R^2
       - categorical variables (direction) -> Logistic regression, report accuracy
  4. Probes are trained on a probe-train split and evaluated on a held-out
     probe-test split, DISTINCT from the transformer's own train/val split,
     to prevent the probe from exploiting transformer-specific overfitting.

Output: a (n_layers, n_variables) matrix of probe scores.
This is the key dissertation figure showing where information is encoded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.transformer import WorldModelTransformer


# Classification of each state variable's probe type
# Determines which sklearn estimator and metric to use
VARIABLE_TYPES = {
    "agent_x":         "continuous",
    "agent_y":         "continuous",
    "agent_direction": "categorical",
    "goal_x":          "continuous",
    "goal_y":          "continuous",
    "carrying":        "categorical",
}


@dataclass
class ProbeResult:
    """Result of training one probe on one (layer, variable) pair."""
    layer: int
    variable: str
    variable_type: str
    score: float            # R^2 for continuous, accuracy for categorical
    baseline_score: float   # score of a trivial baseline (mean/majority predictor)
    n_train: int
    n_test: int


class ActivationCache:
    """
    Extracts and caches residual stream activations from a trained model
    for a set of held-out trajectories.

    Memory-conscious: processes trajectories in batches and stores
    activations as numpy arrays on CPU (not GPU) to allow probing
    on large held-out sets without exhausting VRAM.
    """

    def __init__(self, model: WorldModelTransformer, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def extract(
        self,
        hdf5_path: str,
        split: str,
        n_trajectories: int,
        max_steps_per_traj: Optional[int] = None,
        seed: int = 0,
    ) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
        """
        Extract residual stream activations and aligned state labels.

        Args:
            hdf5_path:          path to HDF5 trajectory file
            split:               "train" or "val" (HDF5 split to read from)
            n_trajectories:      number of trajectories to sample
            max_steps_per_traj:  cap steps per trajectory (None = use full trajectory)
            seed:                random seed for trajectory sampling

        Returns:
            activations: list of length n_layers, each (N, d_model) array
                         N = total number of (trajectory, step) samples collected
            states:      dict of var_name -> (N,) array of ground-truth values
        """
        rng = np.random.default_rng(seed)
        context_length = self.model.config.context_length

        with h5py.File(hdf5_path, "r") as f:
            split_grp = f[f"trajectories/{split}"]
            available = len(split_grp)
            n_trajectories = min(n_trajectories, available)
            traj_indices = rng.choice(available, size=n_trajectories, replace=False)

            n_layers = self.model.config.n_layers
            all_activations = [[] for _ in range(n_layers)]
            all_states: dict[str, list] = {}

            for traj_idx in tqdm(traj_indices, desc=f"Extracting activations [{split}]"):
                traj_grp = split_grp[str(traj_idx)]
                num_steps = int(traj_grp.attrs["length"])

                steps_to_use = num_steps - 1  # need t and t+1 for obs pairs
                if max_steps_per_traj is not None:
                    steps_to_use = min(steps_to_use, max_steps_per_traj)

                if steps_to_use < 1:
                    continue

                # Sample a subset of steps from this trajectory (avoid full redundant load)
                step_indices = rng.choice(
                    num_steps - 1,
                    size=min(steps_to_use, num_steps - 1),
                    replace=False,
                )

                for step in step_indices:
                    obs = traj_grp["observations"][step].flatten().astype(np.int64)
                    obs = np.clip(obs, 0, self.model.config.vocab_size - 1)

                    tokens = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, T)
                    residual_stream = self.model.get_residual_stream(tokens)

                    # Average over sequence positions to get one vector per layer
                    # (mean pooling — standard choice when sequence has no single
                    #  "final token" summary as in autoregressive LM probing)
                    for layer_idx, layer_acts in enumerate(residual_stream):
                        pooled = layer_acts.mean(dim=1).squeeze(0).cpu().numpy()  # (d_model,)
                        all_activations[layer_idx].append(pooled)

                    # Ground-truth state at this step
                    for var in VARIABLE_TYPES.keys():
                        value = traj_grp[f"states/{var}"][step]
                        all_states.setdefault(var, []).append(value)

        activations = [np.stack(layer_acts) for layer_acts in all_activations]
        states = {var: np.array(vals) for var, vals in all_states.items()}

        return activations, states


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    variable_type: str,
) -> tuple[float, float]:
    """
    Train and evaluate a single linear probe.

    Args:
        X_train, X_test: (N, d_model) activation arrays
        y_train, y_test: (N,) ground-truth state values
        variable_type:   "continuous" or "categorical"

    Returns:
        (score, baseline_score)
        continuous:  R^2 score; baseline = R^2 of predicting the mean (= 0.0 by definition)
        categorical: accuracy; baseline = majority-class accuracy
    """
    if variable_type == "continuous":
        probe = Ridge(alpha=1.0)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        score = r2_score(y_test, preds)
        baseline = 0.0   # R^2 of mean predictor is 0 by definition
    else:
        # Guard against degenerate variables (only one class present)
        # e.g. 'carrying' is always 0 in environments with no pickable objects
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            # No meaningful classification possible — report as undefined
            return float("nan"), float("nan")

        # Logistic regression for categorical variables
        # Handle binary (carrying) and multi-class (direction) uniformly
        # Newer sklearn versions auto-detect multiclass — no explicit param needed
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_train, y_train)
        preds = probe.predict(X_test)
        score = accuracy_score(y_test, preds)

        # Majority-class baseline
        values, counts = np.unique(y_train, return_counts=True)
        majority_class = values[np.argmax(counts)]
        majority_preds = np.full_like(y_test, majority_class)
        baseline = accuracy_score(y_test, majority_preds)

    return float(score), float(baseline)


def run_probe_suite(
    model: WorldModelTransformer,
    hdf5_path: str,
    device: torch.device,
    n_trajectories: int = 500,
    max_steps_per_traj: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
) -> list[ProbeResult]:
    """
    Run the full probe suite: every layer x every state variable.

    Args:
        model:               trained WorldModelTransformer
        hdf5_path:            path to HDF5 trajectory file
        device:               torch device
        n_trajectories:       number of held-out trajectories to sample
        max_steps_per_traj:   max steps sampled per trajectory (controls dataset size)
        test_size:            fraction of samples held out for probe evaluation
        seed:                 random seed

    Returns:
        List of ProbeResult, one per (layer, variable) pair.
    """
    cache = ActivationCache(model, device)

    # Use the VAL split — trajectories the world model itself never trained on.
    # This ensures we're testing emergent generalisation, not memorisation.
    activations, states = cache.extract(
        hdf5_path=hdf5_path,
        split="val",
        n_trajectories=n_trajectories,
        max_steps_per_traj=max_steps_per_traj,
        seed=seed,
    )

    n_layers = len(activations)
    results = []

    print(f"\nRunning probes: {n_layers} layers x {len(VARIABLE_TYPES)} variables")
    print(f"Total samples: {activations[0].shape[0]}")

    for layer_idx in range(n_layers):
        X = activations[layer_idx]  # (N, d_model)

        for var_name, var_type in VARIABLE_TYPES.items():
            y = states[var_name]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

            score, baseline = train_probe(X_train, y_train, X_test, y_test, var_type)

            results.append(ProbeResult(
                layer=layer_idx,
                variable=var_name,
                variable_type=var_type,
                score=score,
                baseline_score=baseline,
                n_train=len(X_train),
                n_test=len(X_test),
            ))

            if np.isnan(score):
                print(f"  Layer {layer_idx:2d} | {var_name:16s} | "
                      f"SKIPPED — only one class present in this variable")
            else:
                print(f"  Layer {layer_idx:2d} | {var_name:16s} | "
                      f"score={score:.3f} (baseline={baseline:.3f})")

    return results


def results_to_matrix(results: list[ProbeResult]) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Convert flat ProbeResult list into a (n_layers, n_variables) matrix
    for heatmap visualisation.

    Returns:
        matrix:    (n_layers, n_variables) array of scores
        variables: list of variable names (column labels)
        layers:    list of layer labels (row labels)
    """
    variables = sorted(set(r.variable for r in results))
    layers = sorted(set(r.layer for r in results))

    matrix = np.zeros((len(layers), len(variables)))
    for r in results:
        i = layers.index(r.layer)
        j = variables.index(r.variable)
        matrix[i, j] = r.score

    layer_labels = [f"Layer {l}" for l in layers]
    return matrix, variables, layer_labels
