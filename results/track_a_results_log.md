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

### Attempt 2 — Three-mode corrected pipeline (COMPLETE)

Redesigned to respect the model's causal structure properly. Run on checkpoint step 40,000, layer 5, 100 pairs attempted per variable per mode.

**Mode A — Last-position patching:** patch and evaluate loss at the final sequence position (flat index 1082, corresponding to the fixed bottom-right grid cell in row-major flattening) only.

**Mode B — Agent-cell-position patching:** patch the single best linear probe direction, evaluated specifically at the flat index corresponding to the agent's own grid cell.

**Mode C — Filtered-full patching:** patch the ENTIRE residual stream (full clean activation substituted for full corrupted activation), evaluate loss only in a local window (+/-5 cells) around the agent's position.

#### Results

| Variable | Mode A (last) | Mode B (agent_cell) | Mode C (filtered_full) |
|---|---|---|---|
| agent_x | N/A (no valid pairs) | -0.000 | **1.000** |
| agent_y | N/A (no valid pairs) | 0.001 | **1.000** |
| goal_x  | N/A (no valid pairs) | 0.010 | **1.000** |
| goal_y  | N/A (no valid pairs) | 0.002 | **1.000** |

#### Diagnostic breakdown (pairs attempted=100 per cell)

| Variable | Mode | Skipped (small diff) | Skipped (tiny gap) | Valid pairs |
|---|---|---|---|---|
| agent_x | last | 8 | 92 | 0 |
| agent_x | agent_cell | 8 | 42 | 50 |
| agent_x | filtered_full | 8 | 31 | 61 |
| agent_y | last | 1 | 99 | 0 |
| agent_y | agent_cell | 1 | 46 | 53 |
| agent_y | filtered_full | 1 | 35 | 64 |
| goal_x | last | 9 | 91 | 0 |
| goal_x | agent_cell | 9 | 43 | 48 |
| goal_x | filtered_full | 9 | 29 | 62 |
| goal_y | last | 4 | 96 | 0 |
| goal_y | agent_cell | 4 | 43 | 53 |
| goal_y | filtered_full | 4 | 33 | 63 |

#### Interpretation

**Mode A (last-position) — confirmed agent-position-invariant, as predicted.** 91-99% of pairs were skipped due to near-zero clean/corrupted loss gap, confirming that the model's prediction at the fixed final sequence position (the bottom-right grid corner) does not meaningfully depend on agent position. This is a correctly diagnosed null result, not a methodological failure: it would be surprising if a spatially fixed, agent-irrelevant grid cell's predicted value depended on where the agent is elsewhere on the grid.

**Mode B (single linear direction at agent's cell) — weak/no causal evidence.** Relative patch sizes were small but non-trivial (0.3-1.0% of activation norm, compared to <1% across the board in the earlier flawed mean-pooled attempt — now mechanically meaningful since evaluated locally). Recovery fractions remained near zero (-0.000 to 0.010) across all four variables. This indicates that the SINGLE linear direction identified by the probe, while strongly correlationally predictive (per Phase 4 results, e.g. agent_x R²=0.999), is not on its own sufficient to causally determine the model's local next-token prediction when isolated and patched alone.

**Mode C (full residual stream, local evaluation) — STRONG causal evidence, recovery=1.000 across all variables.** When the entire layer-5 residual stream at the agent's cell is patched (not just a single direction) and loss is evaluated only on the locally-relevant window, clean-level prediction performance is fully and perfectly restored in every case tested. This demonstrates that layer 5's residual stream, taken as a whole, is fully causally sufficient to determine the model's local predictions around the agent — the causal information genuinely exists at this layer and location.

#### Synthesis: A Coherent Three-Part Finding

Taken together, Modes A, B, and C tell a consistent and scientifically interesting story:

1. Causally load-bearing information about agent/goal state IS present and IS sufficient to drive local predictions (Mode C: perfect recovery).
2. This information is NOT concentrated in any single linear direction findable by Ridge regression (Mode B: near-zero recovery despite high correlational R² from Phase 4 probes).
3. This information is NOT present (or not relevant) at spatially fixed, agent-irrelevant positions (Mode A: no detectable signal to test).

**Conclusion:** the model's representation of agent/goal state at layer 5 is causally real but DISTRIBUTED across the residual stream — not reducible to a single interpretable linear direction. This is consistent with, and helps explain, the Phase 5 SAE finding that the most interpretable monosemantic feature discovered (F867) was a narrow, highly specific "corner detector" rather than a clean general-purpose "agent position" feature: if the true representation is distributed across many directions/features working jointly, no single SAE feature or probe direction would be expected to capture it in isolation, even though the FULL representation (Mode C) is unambiguously causally sufficient.

This is a substantive finding for the dissertation: probes and SAEs (correlational methods) found STRONG evidence of position encoding; targeted single-direction causal intervention (Mode B) found WEAK evidence of any single direction being causally sufficient alone; full-representation causal intervention (Mode C) found PERFECT evidence that the layer's complete representation is sufficient. The discrepancy between Mode B and Mode C is itself the key methodological lesson: correlational decodability (what probes measure) does not imply that the SPECIFIC decoded direction is causally privileged — the causal work may be done by a higher-dimensional, distributed combination of directions that any single linear probe only partially captures.

---

## Methodological Lessons (for Dissertation Discussion Section)

1. **Pooling strategy must match the causal structure being tested.** Mean-pooling activations across all sequence positions works for correlational probing (the goal is just "is information present anywhere in the summary") but actively breaks causal intervention, where patching must align with the specific computational pathway producing the prediction being measured.

2. **"Correlational sufficiency" (probe R²) and "individual causal sufficiency" (single-direction patching) are different claims.** A direction can be strongly probe-decodable (high R²) while being causally weak in isolation, if the true causal computation is distributed across multiple directions that the probe's single linear combination only partially captures.

3. **Evaluation locality matters as much as patch locality.** Even with technically correct patching, evaluating loss across positions that are invariant to the manipulated variable (Mode A) dilutes any real effect to undetectability. Both the intervention AND the measurement must be scoped to where the causal relationship is expected to manifest.

4. **Negative results at one level of granularity (Mode B) alongside positive results at another (Mode C) are not contradictory — they are jointly informative** about the geometry of the underlying representation (distributed vs. localised to a single direction).

---

# PART 2 — PHYSICS SANDBOX ENVIRONMENT

## Phase 3 (Physics) — VQ-VAE Tokeniser

**Purpose:** Physics Sandbox observations are continuous (64,64,3) RGB frames, unlike MiniGrid's naturally discrete integer grid. A VQ-VAE compresses each frame into a discrete token sequence (8x8=64 tokens per frame, codebook size 512), enabling the same GPT-style transformer architecture used for MiniGrid to be applied unchanged.

**Architecture:** 3-layer stride-2 conv encoder (64x64x3 -> 8x8x256), vector quantisation against a 512-entry codebook with straight-through gradient estimator, mirrored conv-transpose decoder. Trained with combined reconstruction (MSE) + codebook + commitment loss (van den Oord et al. 2017).

**Training data:** 50,000 frames sampled from 18,000 train trajectories (5.4M frames available total), 5,000 held-out validation frames. Trained 30 epochs, batch size 128.

### Results

| Epoch | Train Recon | Train VQ | Val Recon |
|---|---|---|---|
| 1 | 0.02107 | 12.77283 | 0.00380 |
| 2 | 0.00404 | 0.00008 | 0.00338 |
| 10 | 0.00088 | 0.00051 | 0.00094 |
| 20 | 0.00050 | 0.00045 | 0.00059 |
| 30 (final) | 0.00034 | 0.00037 | 0.00043 |

**Convergence behaviour:** Reconstruction loss decreased monotonically every single epoch with no oscillation or instability (contrast with the SAE training instability documented in Phase 5, which required LR decay to fix). The large initial VQ loss spike (12.77 at epoch 1) reflects normal codebook initialisation and resolved immediately by epoch 2 (0.00008), remaining stable throughout the rest of training. Validation loss tracked training loss closely with no divergence, indicating good generalisation without overfitting. Final val reconstruction MSE of 0.00043 (on [0,1]-normalised pixels) represents strong reconstruction fidelity.

**Checkpoint:** `checkpoints/vqvae_physics.pt`

---

## Phase 4 (Physics) — Linear Probes

**Methodology:** Same Ridge/Logistic probe pipeline as MiniGrid, applied to the Physics transformer trained on VQ-VAE token sequences. Mean-pooled residual stream activations across the 64-token sequence (8×8 spatial VQ-VAE grid). 500 held-out VAL trajectories, 20 steps each, 80/20 train/test split.

**State variables probed (per-object, 3 objects = 21 total):**
pos_x, pos_y, vel_x, vel_y, angle, angular_vel (continuous); in_contact (categorical)

**Checkpoint:** `physics_physics_small_step88000.pt` (best val loss 0.1712 after 100K steps)

### Results Summary

| Variable | Best Layer | Score | Baseline | Interpretation |
|---|---|---|---|---|
| pos_x_0 | 1 | 0.175 | 0.000 | Weakly encoded |
| pos_y_0 | 2 | 0.207 | 0.000 | Weakly encoded |
| pos_x_1 | 1 | 0.116 | 0.000 | Weakly encoded |
| pos_y_1 | 0 | 0.138 | 0.000 | Weakly encoded |
| pos_x_2 | 1 | 0.110 | 0.000 | Weakly encoded |
| pos_y_2 | 1 | 0.091 | 0.000 | Weakly encoded |
| vel_x_{0,1,2} | — | ≤ -0.012 | 0.000 | No linear encoding |
| vel_y_{0,1,2} | — | ≤ -0.004 | 0.000 | No linear encoding |
| angle_{0,1,2} | — | ≤ -0.003 | 0.000 | No linear encoding |
| angular_vel_{0,1,2} | — | ≤ -0.003 | 0.000 | No linear encoding |
| in_contact_{0,1,2} | — | 0.992-0.996 | 0.992-0.996 | Degenerate (majority class) |

**Three distinct patterns emerged:**

1. **in_contact — Degenerate.** Probe score matches baseline exactly — predicting "never in contact" (the overwhelming majority class under random impulse policy) achieves this score trivially. No genuine encoding.

2. **Position (pos_x, pos_y) — Weakly but genuinely encoded.** All six position variables show small positive R² scores (0.09-0.21) well above the 0.0 baseline, consistently across all layers with no clear layer progression. This is real but substantially weaker than MiniGrid position encoding (0.10-0.21 vs 0.999).

3. **Velocity, angle, angular_vel — Not linearly encoded.** All near-zero or negative R² across all layers — no evidence of linear encoding of these dynamic state variables.

**Key cross-environment finding — Physics vs MiniGrid comparison:**

| Variable type | MiniGrid best R²/acc | Physics best R² |
|---|---|---|
| Position (x-axis) | 0.999 | 0.175 |
| Position (y-axis) | 0.791 | 0.207 |
| Velocity | N/A | ≤ 0.000 |
| Direction/angle | 0.999 | ≤ 0.000 |

**Interpretation:** Position encoding is present in both environments but substantially weaker in Physics. This is consistent with a principled explanation: in MiniGrid, the agent occupies a single fixed grid cell whose object-type token changes directly when the agent moves — position information is explicit in specific token positions. In Physics, object positions are encoded implicitly in the spatial arrangement of VQ-VAE codes across the 8×8 grid — no single token directly represents "object N is at position (x,y)." The VQ-VAE compression step distributes spatial information, making linear recovery from mean-pooled representations significantly harder.

Velocity and rotation variables showing no encoding is consistent with the prediction task difficulty: predicting the next frame's VQ-VAE codes from the current frame primarily requires tracking *where* objects are (to predict what visual codes they'll occupy next step), not necessarily their velocity or rotation state in an explicitly decodable linear form.

**Methodological note:** Mean pooling across 64 spatial VQ-VAE tokens likely hurts Physics probe scores more than MiniGrid (1083 tokens), since the relevant spatial information is distributed across all 64 positions rather than concentrated at specific cell indices. Per-position probing (as developed for the causal intervention pipeline) might recover stronger signals for Physics — this is flagged as a potential follow-up investigation.

---

## Outstanding Work

- [x] Complete Phase 6 three-mode intervention results (MiniGrid)
- [x] Collect full-scale Physics Sandbox dataset (20,000 trajectories)
- [x] Train VQ-VAE tokeniser for Physics Sandbox
- [x] Pre-tokenise Physics dataset via VQ-VAE
- [x] Train transformer world model on Physics Sandbox
- [x] Phase 4 (probes) on Physics Sandbox
- [ ] Phase 5 (SAEs) on Physics Sandbox
- [ ] Phase 6 (causal interventions) on Physics Sandbox
- [ ] Track B: Gemma 3 1B + circuit tracer pilot
- [ ] Partial observability environment (optional, time permitting)
- [ ] Consider: per-position probing for Physics (may recover stronger velocity/angle signals than mean-pooled)
