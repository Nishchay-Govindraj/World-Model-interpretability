# Do Learned World Models Develop Interpretable Causal Structure?
### A Mechanistic Analysis of Latent Representations in Predictive Transformers

**MSc Artificial Intelligence Dissertation**
Nishchay Govindraj · 250059083
City, University of London · Supervisor: Esther Mondragon

---

## Overview

This repository implements the full experimental pipeline for the dissertation, which asks:

> Do transformer-based world models internally develop representations of causal structure — object positions, velocities, collision outcomes — or do they encode only statistical correlations?

Two independent tracks address this:

**Track A — Custom World Models** trains GPT-style transformers from scratch on controlled simulation environments, then applies interpretability methods (linear probes, sparse autoencoders, causal interventions) to examine the internal representations.

**Track B — Gemma 3 1B + Circuit Tracing** applies Anthropic's circuit tracer library with Gemma Scope 2 cross-layer transcoders to investigate how an existing LLM represents physical reasoning.

---

## Quick Start

```bash
# 1. Clone and set up environment (Python 3.12 required)
git clone <repo-url>
cd world-model-interpretability
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support first (RTX 3060 / CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Set up Weights & Biases (free academic account at wandb.ai)
# Then set entity in config/model_config.yaml: entity: "nishchay-govindraj-nishchay-govindraj"
wandb login

# 2. Validate pipeline (small run — catches any setup issues)
python scripts/collect_data.py --env minigrid --validate --no-wandb
python scripts/collect_data.py --env physics --validate --no-wandb

# 3. Full data collection (config target is 200K; the dataset actually
#    used for the reported results is 20K trajectories = 18K train / 2K val)
python scripts/collect_data.py --env both --num-trajectories 20000

# 4. Train transformer world model (small, local)
python scripts/train_model.py --env minigrid --scale small --no-wandb

# 5. Run linear probes
python scripts/run_probes.py --env minigrid --checkpoint checkpoints/minigrid_small.pt
```

---

## Repository Structure

```
world-model-interpretability/
├── config/                  # YAML configuration files
│   ├── minigrid_config.yaml
│   ├── physics_config.yaml
│   └── model_config.yaml
├── environments/            # Environment wrappers
│   ├── base_env.py          # Abstract interface
│   ├── minigrid_env.py      # MiniGrid wrapper + state logging
│   └── physics_env.py       # Pymunk physics sandbox + state logging
├── data/
│   ├── collector.py         # Trajectory collection → HDF5
│   ├── dataset.py           # PyTorch Dataset for transformer training
│   └── trajectories/        # HDF5 data files (git-ignored)
├── models/
│   ├── tokeniser.py         # VQ-VAE tokeniser (physics env)
│   └── transformer.py       # GPT-style transformer world model
├── interpretability/
│   ├── probes.py            # Linear probe training + evaluation
│   ├── sae.py               # Sparse autoencoder
│   └── interventions.py     # Activation patching / causal interventions
├── track_b/                # Track B — not yet implemented
│   ├── README.md           # Scope + tooling-verification checklist
│   └── circuit_tracer.py   # (planned) Gemma + circuit tracing pipeline
├── scripts/
│   ├── collect_data.py                    # Data collection CLI
│   ├── train_model.py                     # Transformer training CLI
│   ├── train_vqvae.py                     # VQ-VAE tokeniser training (physics)
│   ├── tokenise_physics_data.py           # Pre-tokenise physics frames via VQ-VAE
│   ├── train_sae.py                       # Sparse autoencoder training
│   ├── run_probes.py                      # Linear probe training + evaluation
│   ├── run_untrained_baseline.py          # Untrained-model probe control
│   ├── run_probe_baseline_comparison.py   # Trained vs untrained across Ridge alphas
│   ├── run_probes_nonlinear.py            # MLP (non-linear) probes + significance
│   ├── run_probes_per_position.py         # Per-position probing (+ untrained baseline)
│   ├── run_subspace_analysis.py           # PCA dimensionality + direction geometry
│   ├── investigate_xy_asymmetry.py        # MiniGrid x/y asymmetry investigation
│   ├── run_interventions.py               # Three-mode causal interventions
│   ├── run_intervention_layer_sweep.py    # Mode-C recovery across all layers
│   ├── run_logit_lens.py                  # Logit lens (computational depth)
│   ├── run_sae_ablation.py                # SAE feature causal ablation
│   ├── analyse_attention.py               # Head-level attention patterns
│   ├── inspect_sae_features.py            # SAE feature inspection / verification
│   └── check_position_frequency.py        # Spawn-point / frequency control checks
├── tasks/
│   ├── todo.md              # Living task list
│   └── lessons.md           # Rules learned from mistakes
└── notebooks/              # (empty — optional interactive analysis)
```

---

## Environments

### MiniGrid (Discrete)
Wraps `MiniGrid-FourRooms-v0` from the `minigrid` package. Fully observable grid world with object interactions.

**Observation prediction protocol** (following Li et al. 2023 — Othello-GPT):
- Input: flattened grid observation at step t (19×19×3 = 1,083 cell tokens)
- Target: flattened grid observation at step t+1 (next observation)
- vocab_size = 32 (MiniGrid cell values: object type, colour, state)
- The model never sees state labels — position/direction/goal encoding must emerge internally

Ground-truth state logged per step (probe targets only, never seen by model): `agent_x`, `agent_y`, `agent_direction`, `goal_x`, `goal_y`, `carrying`.

### Pymunk Physics Sandbox (Continuous)
Custom 2D Newtonian physics environment built on [Pymunk](http://www.pymunk.org/) (Chipmunk backend). N rigid body circles in a walled arena with gravity, elastic collisions, and friction. Rendered to 64x64 RGB frames via a fast numpy renderer, tokenised via VQ-VAE.

Ground-truth state logged per step per object: `pos_x`, `pos_y`, `vel_x`, `vel_y`, `angle`, `angular_vel`, `in_contact`.

---

## Interpretability Pipeline (Track A)

Track A is complete. The methodology combines five interpretability methods across two environments (MiniGrid, Physics), each with matched controls. State labels used as probe targets are ground-truth from the simulator and are **never** seen by the transformer during training.

1. **Linear Probes** — trained at each layer for each state variable, producing layer-by-layer decodability. **Always paired with an untrained-model baseline** (random weights) to separate genuinely learned representation from trivial input-preservation through the residual stream.
2. **Regularisation robustness** — probes across Ridge α ∈ {1, 10, 100, 1000} to distinguish robust low-dimensional learned codes from fragile/high-dimensional or overfit ones.
3. **Non-linear (MLP) Probes** — test whether information is encoded non-linearly; also baseline-controlled and multi-seed for significance.
4. **Sparse Autoencoders (SAEs)** — feature discovery + feature-to-variable mutual information; finds structure (e.g. velocity) that linear probes miss.
5. **Subspace / geometry analysis** — PCA dimensionality and pairwise direction angles, quantifying how distributed each representation is.
6. **Causal Interventions** — three-mode activation patching confirms representations are causally active, not merely correlated.
7. **Supporting analyses** — per-position probing (selection-bias controlled), logit lens (computational depth), and an x/y asymmetry investigation.

**Central finding:** across both environments, training transforms input features into distributed, causally-functional representations that are *less* probe-accessible than the raw input — position information becomes harder to decode linearly after training while remaining fully causally recoverable (Mode-C intervention recovery = 1.000). The correlation-causation gap (strong probe scores vs near-zero single-direction causal effect) is a robust cross-environment result.

**Full quantitative record with all controls:** see [`results/track_a_results_log.md`](results/track_a_results_log.md).

---

## Track B: Gemma-2 2B + Circuit Tracing

Track B is complete. It applies Anthropic's open-source circuit-tracer library
with GemmaScope transcoders to Gemma-2 2B, a pretrained language model, to study
how physical reasoning is represented.

The method builds an attribution graph for each physical-reasoning prompt, ranks
features by their influence on the prediction, and tests them causally by
ablation. The intervention method was first validated on the known
Dallas-to-Austin factual-recall circuit before being applied to physics, and a
random-ablation specificity control confirms the effect is specific to the
ranked features.

**Central finding:** physical reasoning in Gemma-2 2B is distributed and
redundant. Factual recall breaks after ablating about 5 features, whereas
physical predictions survive the ablation of 50 to over 300 of their most
influential features. The influential features are, individually, mostly generic
linguistic features rather than clean physical concepts. This mirrors the
distributed causal structure found in the custom world models of Track A.

The raw attribution graphs and full Neuronpedia exports are too large for the
repository and are archived separately; the notebook regenerates them. See
`track_b/README.md` for reproduction notes.
---

## Hardware Requirements

| Component | Hardware | Notes |
|-----------|----------|-------|
| Data collection | Any CPU | ~3 hours for 200K trajectories (numpy renderer) |
| Small transformer (5M) | RTX 3060 (6GB) | ~2GB VRAM |
| Large transformer (25M) | A100 (university cluster) | ~8GB VRAM |
| Track B (Gemma 3 1B) | RTX 3060 (6GB) | fits in 6GB with 4-bit quant |
| SAE training | RTX 3060 (6GB) | per-layer, sequential |

---

## Experiment Tracking

All experiments are logged to [Weights & Biases](https://wandb.ai). Set your W&B username in `config/model_config.yaml` under `wandb.entity`.

Disable W&B with `--no-wandb` flag on any script.

---

## Citation

```
Govindraj, N. (2026). Do Learned World Models Develop Interpretable Causal Structure?
A Mechanistic Analysis of Latent Representations in Predictive Transformers.
MSc Dissertation, City, University of London.
```

---

## License

MIT License — see LICENSE file.
