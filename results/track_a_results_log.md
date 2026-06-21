# Track A — Results Log
**Project:** Do Learned World Models Develop Interpretable Causal Structure?
**Environment:** MiniGrid-FourRooms-v0
**Model:** Small transformer, 5.0M parameters (6 layers, 8 heads, d_model=256)
**Training protocol:** Observation prediction (next full grid observation), following Li et al. 2023 (Othello-GPT) paradigm — no state supervision, world model structure must emerge from the prediction objective alone.

This log records all quantitative results obtained during interpretability experiments, in chronological order, including methodological dead ends and their resolutions. Negative/inconclusive results are documented per the dissertation proposal's commitment to reporting findings "with equal rigour" regardless of outcome.

---

## Training Summary

| Checkpoint | Step | Train Loss | Val Loss (50-batch sample) |
|---|---|---|---|
| minigrid_small_step1000.pt | 1,000 | ~0.003 | 0.0029 |
| minigrid_small_step12000.pt | 12,000 | ~0.002 | 0.0023 |
| minigrid_small_step28000.pt | 28,000 | ~0.002 | 0.0023 |
| minigrid_small_step40000.pt | 40,000 | ~0.002 | 0.0023 |

**Note on training loss interpretation:** loss converges to near-zero rapidly (by ~step 500) because most of the 1083 flattened grid cells (walls, floor) are static between consecutive observations — the model can achieve low loss largely by learning to copy unchanged cells forward. This is a methodological limitation worth flagging: the prediction task provides only modest forcing pressure to encode dynamic state (e.g. agent position) since only a handful of the 1083 output tokens are affected by it. This may partially explain weaker encoding of less locally-salient variables (see goal_y below).

**Dataset:** 18,000 train / 2,000 val MiniGrid trajectories (20,000 total collected), mean episode length 467.8 steps, random policy.

---

## Phase 4 — Linear Probes

**Methodology:** Ridge regression (continuous variables) / Logistic regression (categorical variables) trained on mean-pooled residual stream activations at each of 6 layers, evaluated via train/test split (80/20) on 500 held-out VAL-split trajectories never seen during transformer training. Features standardised before fitting.

### Checkpoint: step 12,000 (initial pass)

| Variable | Best Layer | Score | Baseline |
|---|---|---|---|
| agent_x | 0 | 0.561 | 0.000 |
| agent_y | 0 | 0.038 | 0.000 |
| agent_direction | 0 | 0.924 | 0.254 |
| goal_x | 0 | 0.712 | 0.000 |
| goal_y | 0 | 0.018 | 0.000 |
| carrying | — | SKIPPED | degenerate (always 0 in FourRooms) |

### Checkpoint: step 12,000 (full 6-layer sweep, post sklearn-compatibility fix)

| Variable | Best Layer | Score | Baseline |
|---|---|---|---|
| agent_x | 5 | 0.999 | 0.000 |
| agent_y | 5 | 0.663 | 0.000 |
| agent_direction | 5 | 0.998 | 0.254 |
| goal_x | 4 | 0.951 | 0.000 |
| goal_y | 4 | 0.035 | 0.000 |
| carrying | — | SKIPPED | degenerate |

### Checkpoint: step 40,000 (training duration comparison)

| Variable | Best Layer | Score | Baseline |
|---|---|---|---|
| agent_x | 5 | 0.999 | 0.000 |
| agent_y | 5 | 0.791 | 0.000 |
| agent_direction | 0 | 0.999 | 0.254 |
| goal_x | 0 | 0.937 | 0.000 |
| goal_y | 0 | 0.171 | 0.000 |
| carrying | — | SKIPPED | degenerate |

### Training Duration Comparison (12K vs 40K steps)

| Variable | Step 12K | Step 40K | Δ |
|---|---|---|---|
| agent_x | 0.999 | 0.999 | 0.000 (saturated) |
| agent_y | 0.663 | 0.791 | +0.128 (improved, but plateauing below x) |
| agent_direction | 0.998 | 0.999 | +0.001 (saturated) |
| goal_x | 0.951 | 0.937 | -0.014 (stable, noise-level) |
| goal_y | 0.035 | 0.171 | +0.136 (improved, but still weak) |

**Key finding — the x/y asymmetry:** Agent and goal x-coordinates are encoded substantially more strongly than their y-coordinate counterparts, across both training durations. This asymmetry narrows somewhat with more training (y-variables improve ~0.13 from 12K to 40K steps) but does not close — even after 3.3x more training, agent_y remains 0.21 below agent_x, and goal_y remains 0.77 below goal_x. This rules out simple undertraining as a full explanation and points toward a structural cause (e.g. FourRooms' cross-shaped wall layout, which may make horizontal position more behaviourally salient than vertical, or an asymmetry in how the action/observation encoding interacts with the two axes). This asymmetry is treated as a genuine scientific finding warranting discussion, not noise.

---

## Phase 5 — Sparse Autoencoders

**Target layer:** Layer 5 (most probe-rich layer per Phase 4 results)
**Checkpoint used:** minigrid_small_step40000.pt
**Architecture:** ReLU SAE, expansion factor 8x (d_model=256 -> d_hidden=2048), L1 coefficient 1e-3, cosine LR decay over 50 epochs, decoder columns renormalised to unit norm after each step.
**Training data:** 29,773 activation samples (mean-pooled residual stream) from 1,000 held-out VAL-split trajectories.

### Training Stability — Methodological Note

An initial training run (constant learning rate, no best-checkpoint tracking) showed reconstruction loss decreasing to a minimum of 8.95 by epoch 17, then OSCILLATING upward for the remaining 33 epochs (ending at 30.38), while L1 loss continued decreasing — indicating the sparsity penalty was destabilising reconstruction quality at a fixed high learning rate. This was fixed by adding cosine learning rate decay and tracking/restoring the best checkpoint by reconstruction loss rather than using the final epoch. The corrected run converged monotonically with no oscillation.

### Final SAE Metrics (corrected training run)

| Metric | Value |
|---|---|
| Final reconstruction loss | 3.7645 (best checkpoint, epoch 50) |
| L0 sparsity | 302.2 / 2048 features (14.8% active per sample) |
| Reconstruction R² | 0.9996 |
| Dead features | 1,727 / 2048 (84.3%) |
| Alive features | 321 / 2048 (15.7%) |
| Mean activation frequency (alive features) | 94.17% of samples |

**Dead feature interpretation:** 84.3% dead features indicates the chosen expansion factor (8x) substantially exceeds the effective dimensionality the L1 penalty allows the model to use productively for this layer/dataset combination. This is a known SAE failure mode (cf. Bricken et al. 2023). Future work should test smaller expansion factors (2-4x) or feature resampling techniques to improve dictionary utilisation. The high reconstruction R² (0.9996) despite this sparsity demonstrates the 321 alive features are nonetheless sufficient to reconstruct the layer's representations almost perfectly.

### Feature-to-Variable Correspondence (Mutual Information, top feature per variable)

| Variable | Top Feature | MI Score |
|---|---|---|
| agent_x | F867 | 2.377 |
| agent_y | F867 | 1.110 |
| agent_direction | F1507 | 0.527 |
| goal_x | F1781 | 0.103 |
| goal_y | F593 | 0.076 |
| carrying | — | SKIPPED (degenerate) |

**Notable finding:** F867 is the top-MI feature for BOTH agent_x and agent_y, suggesting a feature that jointly encodes some aspect of agent spatial location rather than encoding the two axes independently.

### F867 Verification (Top-20 Activating Examples)

Initial hypothesis (untested): F867 encodes "room identity" / quadrant location, based on MI correlation alone.

**Verification procedure:** extracted the top-20 highest-activating examples for F867 across 300 held-out trajectories and inspected their ground-truth agent_x, agent_y values directly, rather than relying on MI score alone.

**Result:** ALL 20 top-activating examples had agent_x=1, agent_y=1 (activation range 28.207–28.897). This is NOT general quadrant/room encoding — it is a highly specific, monosemantic feature firing almost exclusively at the single grid cell (1,1), the literal top-left corner of the arena.

**Follow-up check — is (1,1) a fixed spawn point?** Verified against MiniGrid official documentation: FourRooms places both agent and goal randomly in any of the four rooms at every reset (no fixed default position). Confirmed empirically: only 4/2000 episodes (0.20%) start at (1,1); 278 unique start positions observed across 2,000 episodes. Overall (1,1) appears in 0.576% of all (agent_x, agent_y) samples across the dataset, a modest 1.7x over-representation relative to uniform (consistent with normal corner-bouncing dynamics under a random policy, not a data collection bug).

**Conclusion:** F867 is a verified, monosemantic "corner detector" feature — the SAE discovered a sparse feature corresponding to a single specific spatial landmark (the grid corner) without being told to look for one. This is a more precise and defensible finding than the initial "room identity" hypothesis, and illustrates why mutual information correlation requires direct verification of feature semantics before interpretation.

---

## Phase 6 — Causal Interventions (Methodological History)

**Objective:** test whether linear-probe-identified directions are CAUSALLY involved in model predictions (via activation patching / interchange interventions), not merely correlated with state variables.

### Attempt 1 — Mean-pooled direction patching (FAILED — diagnosed as design flaw)

**Method:** fit Ridge probe direction on mean-pooled (across all 1083 sequence positions) activations; patch by adding scalar delta along that direction to every position uniformly; measure full-sequence cross-entropy loss.

**Result:** Uniformly near-zero recovery fraction across all 4 variables tested (range: -0.000 to 0.008), despite large clean/corrupted loss gaps (clean ~0.0026, corrupted ~0.30).

**Diagnosis:** Patch delta magnitude was 0.03%-0.97% of the activation's overall norm (mean activation norm ~686 at layer 5; mean patch delta 0.21-6.64). The probed direction, while predictive in a correlational sense, represented a vanishingly small fraction of the activation's total magnitude once derived from pooling across 1083 mostly-static positions. This is a genuine experimental design flaw, not evidence of absent causal structure: averaging across positions diluted any real signal below a detectable threshold, and patching does not respect the model's actual position-wise causal computation (each output position's logits depend on that position's own final residual stream value via ln_final -> unembed, not a sequence-wide average).

### Attempt 2 — Three-mode corrected pipeline (current)

Redesigned to respect the model's causal structure properly:
- **Mode A (last-position):** patch and evaluate loss at the final sequence position only
- **Mode B (agent-cell-position):** patch and evaluate at the specific flattened index corresponding to the agent's own grid cell (computed via direct (x,y) -> flat-index mapping)
- **Mode C (filtered-full):** patch the entire residual stream, evaluate loss only in a local window around the agent's cell (avoiding dilution from agent-position-invariant wall/floor targets)

[Results pending — to be appended after running scripts/run_interventions.py]

---

## Outstanding Work

- [ ] Complete Phase 6 three-mode intervention results (in progress)
- [ ] Repeat Phase 4-6 on the Physics Sandbox environment (Track A second environment)
- [ ] Track B: Gemma 3 1B + circuit tracer pilot
- [ ] Partial observability environment (optional, time permitting)
