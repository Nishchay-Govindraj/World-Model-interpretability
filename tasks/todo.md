# Dissertation Task List
# "Do Learned World Models Develop Interpretable Causal Structure?"
# Updated: project initialisation

## PHASE 1 — Project Scaffold ✅
- [x] Create directory structure
- [x] Write requirements.txt
- [x] Write .gitignore
- [x] Write README.md
- [x] Write config YAMLs (minigrid, physics, model)
- [x] Write tasks/todo.md and tasks/lessons.md

## PHASE 2 — Environments
- [ ] Implement BaseEnvironment abstract class
- [ ] Implement MiniGridEnv wrapper with ground-truth state logging
- [ ] Validate MiniGrid: collect 100 trajectories, inspect state labels
- [ ] Implement PhysicsEnv (PyBox2D sandbox) with ground-truth state logging
- [ ] Validate PhysicsEnv: collect 100 trajectories, inspect state labels

## PHASE 3 — Data Pipeline
- [ ] Implement TrajectoryCollector (works for both envs)
- [ ] Implement TrajectoryDataset (PyTorch Dataset)
- [ ] Validate: DataLoader produces correct shapes
- [ ] Collect full datasets: 100K–500K trajectories per environment

## PHASE 4 — VQ-VAE Tokeniser (physics env only)
- [ ] Implement VQ-VAE encoder/decoder
- [ ] Train on physics observations
- [ ] Validate reconstruction quality
- [ ] Confirm discrete token sequences are sensible

## PHASE 5 — Transformer World Model
- [ ] Implement GPT-style transformer (configurable size)
- [ ] Train 5M param model on MiniGrid
- [ ] Train 5M param model on Physics
- [ ] Train 25M param model on both (A100 cluster)
- [ ] Validate prediction accuracy before ANY probing

## PHASE 6 — Linear Probes (Track A)
- [ ] Implement LinearProbe class (per layer, per state variable)
- [ ] Run probes across all layers for all state variables
- [ ] Generate layer-by-layer probe accuracy heatmaps
- [ ] Identify most probe-rich layers

## PHASE 7 — Sparse Autoencoders (Track A)
- [ ] Implement SAE (TopK or ReLU variant)
- [ ] Train SAE on most probe-rich layers
- [ ] Evaluate feature-to-variable correspondence (mutual information)
- [ ] Compare SAE features vs probe-identified variables

## PHASE 8 — Causal Interventions (Track A)
- [ ] Implement activation patching pipeline
- [ ] Design intervention experiments per state variable
- [ ] Run and measure causal fidelity
- [ ] Optional: partial observability environment + object permanence probing

## PHASE 9 — Track B: Gemma 3 1B + Circuit Tracer
- [ ] Set up Gemma 3 1B locally (RTX 3060)
- [ ] Install circuit tracer + Gemma Scope 2 CLTs
- [ ] Design physical reasoning prompt suite
- [ ] Generate attribution graphs
- [ ] Annotate and label circuits
- [ ] Compare with Track A representations

## PHASE 10 — Writing & Release
- [ ] Generate all figures and tables from results
- [ ] Write full dissertation draft (~10,000–12,000 words)
- [ ] Supervisor feedback + revisions
- [ ] Final submission (mid-May 2026)
- [ ] Open-source release on GitHub
- [ ] Optional: workshop paper draft
