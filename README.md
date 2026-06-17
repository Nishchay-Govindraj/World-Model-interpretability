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

# 3. Full data collection
python scripts/collect_data.py --env both --num-trajectories 200000

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
├── track_b/
│   └── circuit_tracer.py    # Gemma 3 1B + circuit tracing pipeline
├── scripts/
│   ├── collect_data.py      # Data collection CLI
│   ├── train_model.py       # Transformer training CLI
│   └── run_probes.py        # Probe training + evaluation CLI
├── tasks/
│   ├── todo.md              # Living task list
│   └── lessons.md           # Rules learned from mistakes
└── notebooks/
    └── exploration.ipynb    # Interactive analysis
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

1. **Linear Probes** — trained at each transformer layer for each state variable (agent_x, agent_y, agent_direction, goal_x, goal_y). Produces a layer-by-layer accuracy heatmap showing where information is encoded. State labels are ground-truth from the simulator — never seen by the transformer during training.
2. **Sparse Autoencoders (SAEs)** — trained on the most probe-rich layers. Discovers features beyond those predefined by probes.
3. **Causal Interventions** — activation patching confirms representations are causally active (manipulating them changes model behaviour, not just correlated with it).

---

## Track B: Gemma 3 1B + Circuit Tracing

Applies Anthropic's open-source circuit tracer with Gemma Scope 2 cross-layer transcoders to investigate physical reasoning circuits in a general-purpose LLM.

See `track_b/README.md` for setup instructions (requires separate installation of circuit tracer and Gemma Scope 2).

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
