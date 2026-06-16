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
# 1. Clone and set up environment
git clone <repo-url>
cd world-model-interpretability
pip install -r requirements.txt

# 2. Validate pipeline (small run — catches any setup issues)
python scripts/collect_data.py --env minigrid --validate
python scripts/collect_data.py --env physics --validate

# 3. Full data collection
python scripts/collect_data.py --env both --num-trajectories 200000

# 4. Train transformer world model (small, local)
python scripts/train_model.py --env minigrid --scale small

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
│   └── physics_env.py       # PyBox2D sandbox + state logging
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
Wraps `MiniGrid-FourRooms-v0` from the `minigrid` package. Fully observable grid world with object interactions. Observations are flattened integer token sequences — no VQ-VAE needed.

Ground-truth state logged per step: `agent_x`, `agent_y`, `agent_direction`, `goal_x`, `goal_y`, `carrying`.

### PyBox2D Physics Sandbox (Continuous)
Custom 2D Newtonian physics environment with N rigid body circles in a walled arena. Rendered to 64x64 RGB frames, tokenised via VQ-VAE.

Ground-truth state logged per step per object: `pos_x`, `pos_y`, `vel_x`, `vel_y`, `angle`, `angular_vel`, `in_contact`.

---

## Interpretability Pipeline (Track A)

1. **Linear Probes** — trained at each transformer layer for each state variable. Produces a layer-by-layer accuracy heatmap showing where information is encoded.
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
| Data collection | Any CPU | ~hours for 200K trajectories |
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
