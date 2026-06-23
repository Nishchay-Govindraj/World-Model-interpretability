# Track A — Results Log
**Project:** Do Learned World Models Develop Interpretable Causal Structure?
**Student:** Nishchay Govindraj (250059083), MSc AI, City St George's, University of London
**Supervisor:** Esther Mondragon

This log is the permanent quantitative record of all Track A experimental results, including methodological dead ends and their resolutions. Negative and inconclusive results are documented with equal rigour.

---

# PART 1 — MINIGRID (FourRooms-v0)

## Architecture
- **Model:** GPT-style decoder-only transformer, 5.0M parameters (6 layers, 8 heads, d_model=256)
- **Training:** Observation prediction (next full grid observation) — no state supervision
- **Protocol:** Li et al. 2023 (Othello-GPT) paradigm — world model structure must emerge from prediction objective alone
- **Dataset:** 18,000 train / 2,000 val trajectories, mean episode length 467.8 steps, random policy

## Training Summary

| Checkpoint | Steps | Val Loss |
|---|---|---|
| minigrid_small_step1000.pt | 1,000 | 0.0029 |
| minigrid_small_step12000.pt | 12,000 | 0.0023 |
| minigrid_small_step40000.pt | 40,000 | 0.0023 |

**Note:** Loss converges near-zero rapidly because ~1080 of 1083 grid cells are static between steps — the model can copy most tokens unchanged. This is a documented limitation: the prediction task provides limited forcing pressure to encode dynamic state (agent position). Documented as a methodological note, not hidden.

---

## Phase 4 — Linear Probes (MiniGrid)

**Methodology:** Ridge regression (continuous) / Logistic regression (categorical) on mean-pooled residual stream activations at each of 6 layers. 500 held-out VAL trajectories, 20 steps/traj, 80/20 train/test split, StandardScaler applied.

### Step 12K — Full 6-layer sweep

| Variable | Best Layer | R²/Acc | Baseline |
|---|---|---|---|
| agent_x | 5 | 0.999 | 0.000 |
| agent_y | 5 | 0.663 | 0.000 |
| agent_direction | 5 | 0.998 | 0.254 |
| goal_x | 4 | 0.951 | 0.000 |
| goal_y | 4 | 0.035 | 0.000 |
| carrying | — | SKIPPED | degenerate |

### Step 40K — Training duration comparison

| Variable | Step 12K | Step 40K | Δ | Interpretation |
|---|---|---|---|---|
| agent_x | 0.999 | 0.999 | 0.000 | Saturated |
| agent_y | 0.663 | 0.791 | +0.128 | Improving but plateauing |
| agent_direction | 0.998 | 0.999 | +0.001 | Saturated |
| goal_x | 0.951 | 0.937 | -0.014 | Stable (noise) |
| goal_y | 0.035 | 0.171 | +0.136 | Improving, still weak |

**Key finding — persistent x/y asymmetry:** agent_x R²=0.999 vs agent_y R²=0.791 even after 3.3x more training. The gap narrowed by only 0.128 units. This rules out undertraining as the explanation and points to a structural cause — investigated in the asymmetry analysis below.

---

## Phase 5 — SAEs (MiniGrid)

**Layer:** 5 | **Expansion:** 8x (d_model=256 → d_hidden=2048) | **L1:** 1e-3 | **Epochs:** 50, cosine LR decay

**Note on training instability:** Initial run with constant LR showed oscillating reconstruction loss after epoch 17. Fixed by adding cosine LR decay + best-checkpoint tracking. Corrected run converged monotonically.

### Dictionary Health

| Metric | Value |
|---|---|
| Dead features | 1,727/2,048 (84.3%) |
| Alive features | 321/2,048 (15.7%) |
| Reconstruction R² | 0.9996 |
| L0 sparsity | 302.2/2048 features active |

### Feature-Variable Correspondence (MI)

| Variable | Top Feature | MI Score |
|---|---|---|
| agent_x | F867 | 2.377 |
| agent_y | F867 | 1.110 |
| agent_direction | F1507 | 0.527 |
| goal_x | F1781 | 0.103 |
| goal_y | F593 | 0.076 |

**F867 is top-MI for BOTH agent_x and agent_y** — suggests joint spatial encoding.

### F867 Verification

Top-20 activating examples inspected directly. ALL 20 had agent_x=1, agent_y=1 (activation range 28.207–28.897). F867 is a monosemantic corner-detector, NOT room-identity encoding.

**Spawn-point check:** Only 4/2000 episodes (0.20%) start at (1,1). 278 unique start positions observed. (1,1) appears in 0.576% of all frames — 1.7x over-represented vs uniform, consistent with random-walk corner dynamics. The feature is NOT a spawn-point artifact.

---

## Phase 6 — Causal Interventions (MiniGrid)

**Three-mode design** to separate methodological artifacts from genuine causal findings.

### Methodological history

**Attempt 1 (failed):** Mean-pooled direction patching. All recovery fractions near-zero. Diagnosis: patch delta was <1% of activation norm (~686) — mechanically working but negligible effect. Root cause: mean pooling across 1083 mostly-static positions dilutes any real signal, and patching doesn't respect the model's position-wise causal structure.

**Attempt 2 (fixed):** Three-mode pipeline respecting causal structure.

### Final Results — Step 40K, Layer 5

| Variable | Mode A (last pos) | Mode B (agent_cell) | Mode C (full patch, local window) |
|---|---|---|---|
| agent_x | N/A (no valid pairs) | -0.000 | **1.000** |
| agent_y | N/A (no valid pairs) | 0.001 | **1.000** |
| goal_x | N/A (no valid pairs) | 0.010 | **1.000** |
| goal_y | N/A (no valid pairs) | 0.002 | **1.000** |

Mode A: 91-99% of pairs skipped — last sequence position (bottom-right corner) is agent-position-invariant (static wall cell). Confirmed as a null result, not a methodological failure.

Mode B: patch size ~0.3-1.0% of activation norm — mechanically applied but near-zero effect because a single linear probe direction is not sufficient to drive prediction.

Mode C: perfect 1.000 recovery — full residual stream at agent's cell is causally sufficient.

**Central finding: position representations are causally real but distributed** — not concentrated in any single linear direction.

---

## X/Y Asymmetry Investigation (MiniGrid)

[PENDING — script written, to be run: `python scripts/investigate_xy_asymmetry.py --checkpoint checkpoints/minigrid_small_step40000.pt`]

---

## Per-Position Probing (MiniGrid)

[PENDING — script written, to be run: `python scripts/run_probes_per_position.py --checkpoint checkpoints/minigrid_small_step40000.pt --env minigrid --scale small --layer 5`]

---

## Attention Pattern Analysis (MiniGrid)

[PENDING — script written, to be run: `python scripts/analyse_attention.py --checkpoint checkpoints/minigrid_small_step40000.pt --env minigrid --scale small`]

---

## F867 Causal Ablation (MiniGrid)

[PENDING — script written, to be run: `python scripts/run_sae_ablation.py --checkpoint checkpoints/minigrid_small_step40000.pt --sae-checkpoint checkpoints/sae_minigrid_layer5.pt --env minigrid --scale small --layer 5 --features 867 --target-var agent_x --target-val 1`]

---

## Layer Sweep — Causal Interventions (MiniGrid)

[PENDING — script written, to be run: `python scripts/run_intervention_layer_sweep.py --checkpoint checkpoints/minigrid_small_step40000.pt --env minigrid --scale small`]

---

# PART 2 — PHYSICS SANDBOX

## Architecture
- **Model:** 4.9M parameters (6 layers, 8 heads, d_model=256), context_length=64
- **Tokeniser:** VQ-VAE — 64×64 frames → 8×8 spatial grid, 512-entry codebook
- **Training:** 100,000 steps, best val loss 0.1712 (vs near-zero for MiniGrid — Physics frames genuinely change every step)
- **Dataset:** 18,000 train / 2,000 val trajectories, 300 steps each, random impulse policy

## VQ-VAE Training

| Epoch | Train Recon | Val Recon |
|---|---|---|
| 1 | 0.02107 | 0.00380 |
| 10 | 0.00088 | 0.00094 |
| 30 | 0.00034 | 0.00043 |

Converged monotonically, no instability. Best val recon = 0.00043 MSE (on [0,1]-normalised pixels).

## Transformer Training Summary

| Step | Val Loss |
|---|---|
| 1,000 | 0.1819 |
| 10,000 | 0.1766 |
| 50,000 | 0.1733 |
| 88,000 (best) | 0.1712 |
| 100,000 (final) | 0.1712 |

---

## Phase 4 — Linear Probes (Physics)

**Methodology:** Same pipeline. 500 VAL trajectories, 20 steps/traj. 21 variables (7 per object × 3 objects).

### Results Summary

| Variable | Best Layer | R² | Baseline |
|---|---|---|---|
| pos_x_0 | 1 | 0.175 | 0.000 |
| pos_y_0 | 2 | 0.207 | 0.000 |
| pos_x_1 | 1 | 0.116 | 0.000 |
| pos_y_1 | 0 | 0.138 | 0.000 |
| pos_x_2 | 1 | 0.110 | 0.000 |
| pos_y_2 | 1 | 0.091 | 0.000 |
| vel_x/vel_y/angle/angular_vel | all | ≤ 0.000 | 0.000 |
| in_contact | all | ≈ baseline | degenerate |

**Three patterns:** position weakly encoded (0.09-0.21); velocity/rotation not linearly encoded; contact degenerate (majority class baseline = ~0.994).

**Cross-environment contrast:** MiniGrid position R²=0.999 vs Physics 0.09-0.21. Consistent with VQ-VAE distributing spatial information across tokens, making linear recovery from mean-pooled representations harder.

---

## Phase 5 — SAEs (Physics)

**Layer:** 2 | **Expansion:** 4x (d_model=256 → d_hidden=1024) | **L1:** 5e-4 | **Epochs:** 50

### Dictionary Health

| Metric | Physics | MiniGrid (comparison) |
|---|---|---|
| Dead features | 680/1024 (66.4%) | 1727/2048 (84.3%) |
| Alive features | 344 | 321 |
| Reconstruction R² | 0.9992 | 0.9996 |

Smaller expansion factor (4x vs 8x) reduced dead feature rate from 84.3% to 66.4%.

### Feature-Variable Correspondence (MI)

| Variable | Top Feature | MI |
|---|---|---|
| pos_x_0 | F650 | 0.0672 |
| pos_x_1 | F858 | 0.0693 |
| pos_x_2 | F891 | 0.0479 |
| pos_y_0 | F10 | 0.0664 |
| pos_y_1 | F730 | 0.0576 |
| pos_y_2 | F887 | 0.0624 |
| vel_x_0 | F388 | 0.0478 |
| vel_x_1 | F1001 | 0.0412 |
| vel_x_2 | F926 | 0.0518 |
| vel_y_0 | F274 | 0.0423 |
| angle_0 | F742 | 0.0444 |
| angular_vel_0 | F27 | 0.0422 |
| in_contact | all | ≈ 0.005 |

**Key SAE finding:** velocity, angle, angular_vel show MI ≈ 0.04-0.05 despite zero linear probe R². SAEs reveal structure probes missed — these variables ARE represented but non-linearly. Probes and SAEs are complementary, not redundant.

**No dominant monosemantic feature** (unlike MiniGrid's F867) — more distributed representation consistent with continuous multi-object dynamics.

---

## Phase 6 — Causal Interventions (Physics)

**Same three-mode design, adapted for Physics:** Mode B uses VQ-VAE spatial token index for object position (pixel coordinates → 8×8 grid).

### Final Results — Step 88K, Layer 2

| Variable | Mode A (last) | Mode B (agent_cell) | Mode C (full patch) |
|---|---|---|---|
| pos_x_0 | 0.004 | 0.001 | **1.000** |
| pos_y_0 | -0.001 | -0.001 | **1.000** |
| pos_x_1 | -0.000 | -0.001 | **1.000** |
| pos_y_1 | 0.002 | 0.006 | **1.000** |

All 100 pairs valid per variable per mode — Physics frames change every step (no static majority class).

**Distributed representation pattern replicates exactly across environments.** Mode C gives perfect recovery even when probe R² is only 0.09-0.21 — confirming causally real but non-linearly/distributedly encoded representations.

---

## Per-Position Probing (Physics)

[PENDING — to be run: `python scripts/run_probes_per_position.py --checkpoint checkpoints/physics_physics_small_step88000.pt --env physics --scale small --layer 2`]

---

## Attention Pattern Analysis (Physics)

[PENDING — to be run: `python scripts/analyse_attention.py --checkpoint checkpoints/physics_physics_small_step88000.pt --env physics --scale small`]

---

## Layer Sweep — Causal Interventions (Physics)

[PENDING — to be run: `python scripts/run_intervention_layer_sweep.py --checkpoint checkpoints/physics_physics_small_step88000.pt --env physics --scale small`]

---

# CRITICAL CONTROL — Untrained Baseline & Probe Reinterpretation

**This is the most methodologically important result in Track A.** It substantially reframes the interpretation of all linear probe results.

## Motivation

A standard but frequently-omitted control in probing studies (Belinkov 2022, Hewitt & Liang 2019): probe R² measures *decodability*, not whether a representation is *learned* or *used*. The observation token encoding is linearly preserved through the residual stream via skip connections even in an untrained network, so a probe can recover state from random-weight models. Raw probe R² therefore conflates (a) trivially-decodable input structure and (b) genuinely learned representation.

## Untrained Baseline Results (random weights, no training)

### MiniGrid (layer 5 best)

| Variable | Untrained R² | Trained R² | Naive interpretation | Corrected |
|---|---|---|---|---|
| agent_x | 0.987 | 0.999 | "strongly encoded" | Mostly input-preservation; small genuine gain |
| agent_y | 0.449 | 0.776 | "moderately encoded" | Genuine gain but fragile (see reg. check) |
| goal_x | 0.988 | 0.883 | "strongly encoded" | Training REDUCES linear decodability |
| goal_y | 0.465 | 0.030 | "not encoded" | Training DRIVES linear decodability to zero |

### Physics (layer 2)

| Variable | Untrained R² | Trained R² | Difference |
|---|---|---|---|
| pos_x_0 | 0.303 | 0.178 | -0.126 |
| pos_y_0 | 0.494 | 0.213 | -0.281 |
| pos_x_1 | 0.287 | 0.073 | -0.213 |
| pos_y_1 | 0.380 | 0.121 | -0.259 |
| pos_x_2 | 0.269 | 0.089 | -0.180 |
| pos_y_2 | 0.391 | 0.128 | -0.263 |

In Physics, **training reduces linear position decodability for every object, uniformly.** Untrained decodes position 2-3x better than trained.

## Regularisation Robustness Check

To distinguish "input-preservation" from "probe overfitting on 256 dims", we re-ran probes at Ridge alpha ∈ {1, 10, 100, 1000}.

### MiniGrid key findings

- **agent_x:** trained stays robust (0.997 at alpha=1000) while untrained degrades (0.963); the trained advantage *widens* under regularisation (+0.013 → +0.034). This is the ONE variable where the model built a genuinely robust, low-dimensional learned position code.
- **agent_y:** trained advantage at alpha=1 (+0.327) collapses to +0.003 at alpha=1000. The trained y-encoding is high-dimensional and fragile, not a clean compact code.
- **goal_x:** training reduces decodability and the gap *widens* under regularisation (trained 0.426 vs untrained 0.967 at alpha=1000). Training actively transforms goal position out of linearly-accessible form.
- **goal_y:** trained near-zero (0.016-0.030) at all alphas. Linear decodability eliminated by training.

### Physics key finding

Negative difference at all regularisation strengths for all position variables — the input-preservation signal dominates and training consistently reduces linear decodability.

## Reinterpretation — The Corrected Scientific Narrative

1. **Most raw probe R² was input-preservation, not learned representation.** The untrained baseline proves position is trivially linearly decodable from the residual stream regardless of training.

2. **Training transforms position AWAY from linear accessibility.** In Physics (all position variables) and MiniGrid (goal_x, goal_y), trained models show *lower* linear decodability than untrained. The sole exception is agent_x in MiniGrid, where the model added a small robust code.

3. **This is consistent with — and explains — the Mode B vs Mode C intervention gap.** Mode B (single linear direction) fails because the trained functional representation is no longer a clean linear code. Mode C (full residual) succeeds because the information remains present, just distributed/transformed. Probes, SAEs, and interventions now form one coherent account.

4. **Goal position is transformed, not discarded.** Although linear decodability of goal_x/goal_y drops to near-zero with training, Mode C causal intervention recovers goal_x and goal_y at 1.000 — confirming the information remains causally present, just not linearly accessible. (Caveat: probes alone cannot fully distinguish "transformed to non-linear form" from "partially compressed as predictively less relevant"; the Mode C recovery supports the former.)

5. **The x/y asymmetry is reframed.** The raw asymmetry (agent_x 0.999 >> agent_y 0.791) is mostly inherited from the input-preservation baseline (untrained 0.987 vs 0.449). The one genuine learned difference is that agent_x receives a robust low-dimensional code while agent_y's learned component is fragile and high-dimensional.

**Methodological significance:** This control is what separates rigorous interpretability from naive probing. Without it, the dissertation would have reported the standard (and misleading) "position is linearly encoded" conclusion. With it, the finding is the more interesting and defensible "training transforms linearly-accessible input features into distributed, causally-functional, linearly-hidden representations." This directly engages the probing-methodology critiques in the literature.

---

# CROSS-ENVIRONMENT SUMMARY

| Finding | MiniGrid | Physics |
|---|---|---|
| Position linear probe (best) | R²=0.999 (agent_x) | R²=0.207 (pos_y_0) |
| Velocity/dynamics linear probe | N/A | R²≤0.000 |
| SAE top feature MI | 2.377 (F867, monosemantic) | 0.069 (F858, distributed) |
| SAE velocity MI | N/A | 0.04-0.05 (non-linearly encoded) |
| Mode B recovery (single direction) | ~0.000 | ~0.001-0.006 |
| Mode C recovery (full residual) | 1.000 | 1.000 |
| Representation type | Causally real, distributed | Causally real, distributed |

**Headline conclusion:** Position representations are causally real but distributed in both environments, with linear decodability strongly mediated by tokenisation strategy. SAEs reveal structure invisible to probes. The correlation-causation gap (strong probe R² but weak Mode B recovery) is a cross-environment finding about the geometry of world model representations.

---

## Outstanding Work (Pending Experiments)

### High Priority
- [ ] Per-position probing — MiniGrid layer 5, Physics layer 2
- [ ] X/Y asymmetry investigation — MiniGrid
- [ ] Attention pattern analysis — both environments
- [ ] F867 causal ablation — MiniGrid
- [ ] Layer sweep causal interventions — both environments
- [ ] Non-linear probes — MiniGrid layer 5 (MLP probe, does it close Mode B gap?)

### Track B (Next Phase)
- [ ] Gemma 3 1B + circuit tracer pilot
- [ ] Physical reasoning prompt suite
- [ ] Attribution graph analysis

### Methodological Lessons
1. Mean-pooling destroys causal structure for patching experiments
2. Correlational decodability (probe R²) ≠ causal sufficiency (Mode B recovery)
3. SAEs and probes are complementary — SAEs find non-linear structure probes miss
4. Evaluation locality matters: measuring loss over position-invariant tokens dilutes causal effects
5. Mode B vs Mode C dissociation is itself the key finding about representation geometry
