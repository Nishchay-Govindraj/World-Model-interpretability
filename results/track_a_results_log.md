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

The persistent agent_x (≈0.999) >> agent_y (≈0.791) probe asymmetry was investigated across three hypotheses.

### Hypothesis 2 — Movement rate (RULED OUT)
Analysed 928,841 consecutive step pairs across 2,000 trajectories:
- X changed in 5.6% of steps (mean |Δx|=0.0562)
- Y changed in 5.7% of steps (mean |Δy|=0.0570)
- Ratio 0.99 — movement rates essentially identical.

The agent moves horizontally and vertically equally often. The asymmetry is NOT from more x-training-signal.

### Hypothesis 1 — Room structure (RULED OUT)
Per-quadrant probe scores (layer 5):

| Quadrant | n | agent_x R² | agent_y R² | x/y ratio |
|---|---|---|---|---|
| top-left | 2427 | 1.000 | 0.818 | 1.22 |
| top-right | 2377 | 0.999 | 0.780 | 1.28 |
| bottom-left | 2430 | 1.000 | 0.893 | 1.12 |
| bottom-right | 2708 | 0.999 | 0.823 | 1.21 |

Ratio is uniform (~1.1-1.3) across all quadrants. If room structure caused the asymmetry it would vary by quadrant. It does not — the asymmetry is a global property, not a consequence of the four-room layout.

### Hypothesis 3 — Row-major flattening (SUPPORTED — combining with baseline + subspace results)

The decisive explanation comes from combining three results:
1. The untrained baseline shows agent_x ≈0.99 vs agent_y ≈0.45 even with random weights — the asymmetry exists BEFORE any training.
2. The subspace analysis shows agent_x is recoverable from 1 PCA dimension while agent_y needs 50+.
3. The MiniGrid observation is flattened **row-major**: flat_index = (y · grid_width + x) · channels.

**Mechanism:** the agent's x-coordinate is a *local, within-row offset* in the flattened token sequence, while the y-coordinate is a *large-stride across-row index* (rows are ~57 tokens apart). A linear probe reads a local offset (x) far more easily than a large-stride index (y) from the residual stream. The asymmetry is therefore an artifact of the row-major observation flattening, inherited through input-preservation — NOT a learned asymmetry in spatial representation.

**Corrected conclusion:** The x/y probe asymmetry is primarily a consequence of how the observation is serialised (row-major flattening makes x a local feature, y a large-stride feature), present even in untrained models, rather than evidence that the world model represents horizontal position more strongly than vertical. The genuine *learned* component (regularisation-robust gain) is small for agent_x (+0.034) and fragile for agent_y — so learning does not explain the asymmetry either. This dissolves the apparent anomaly into a concrete encoding artifact.

**Dissertation note:** this finding is a strong example of why interpretability requires controls. The naive reading ("the model represents x more strongly than y") would have been wrong; the asymmetry is an artifact of observation serialisation, demonstrable via the untrained baseline.

---

## Per-Position Probing (MiniGrid)

Probed the residual stream at each of the 1083 token positions independently, taking the best-scoring position per variable. **Critically, an untrained model was run through the identical best-of-1083 selection** to control for selection-bias inflation.

| Variable | Best (trained) | Best (untrained) | Selection-corrected | Best pos |
|---|---|---|---|---|
| agent_x | 0.997 | 0.991 | **+0.007** | 965 |
| agent_y | 0.716 | 0.509 | **+0.207** | 1082 |
| agent_direction | 1.000 | 1.000 | +0.000 | 1023 |
| goal_x | 0.746 | 0.992 | **−0.246** | 596 |
| goal_y | 0.080 | 0.601 | **−0.521** | 175 |

**Findings (selection-corrected):**
- agent_x's raw 0.997 collapses to +0.007 once the untrained best-of-N is subtracted — confirming the high score is input-preservation, not learned structure. This vindicates the selection-bias caution: without the untrained control, this would have looked like strong evidence of learned x-encoding.
- agent_y shows the largest genuine positive learned gain (+0.207), consistent with the regularisation check.
- goal_x and goal_y show significant NEGATIVE corrected scores — training reduces decodability even at the best position, mirroring the mean-pooled, non-linear, and regularisation findings.

The per-position analysis, once baseline-corrected, reproduces the exact qualitative picture of every other method. This is treated as supplementary confirmation; the mechanistically cleanest version remains the subspace analysis.

---

## Attention Pattern Analysis (MiniGrid)

[NOT RUN — script written (`analyse_attention.py`) with a `get_attention_weights()` method added to the transformer, but deprioritised at Track A wrap-up. The core distributed-representation finding is already established by probes + subspace + interventions; head-level attention analysis would be confirmatory mechanistic colour rather than load-bearing. Candidate for future work or Track B.]

---

## F867 Causal Ablation (MiniGrid)

[NOT RUN — script written (`run_sae_ablation.py`) but deprioritised at wrap-up. F867's monosemantic corner-detector status is already established by direct activation inspection (top-20 activating examples all at cell (1,1)) and the spawn-point control. Causal ablation would strengthen the necessity claim but is not load-bearing for the central thesis. Candidate for future work.]

---

## Layer Sweep — Causal Interventions (MiniGrid)

[NOT RUN — script written (`run_intervention_layer_sweep.py`) but deprioritised at wrap-up. Mode C recovery is already established at the probe-selected layer; sweeping all layers would map where causal structure emerges across depth but is confirmatory. Candidate for future work.]

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

Probed each of the 64 VQ-VAE spatial token positions independently, with an untrained model run through the identical best-of-64 selection.

| Variable | Best (trained) | Best (untrained) | Selection-corrected |
|---|---|---|---|
| pos_x_0 | 0.158 | 0.205 | −0.046 |
| pos_y_0 | 0.230 | 0.394 | −0.164 |
| pos_x_1 | 0.106 | 0.276 | −0.170 |
| pos_y_1 | 0.179 | 0.362 | −0.183 |
| pos_x_2 | 0.116 | 0.213 | −0.097 |
| pos_y_2 | 0.151 | 0.270 | −0.118 |
| vel_x/vel_y (all) | ~0 | negative | **+0.004 to +0.026** |
| angle/angular_vel (all) | ~0 | negative | **+0.006 to +0.032** |

**Findings (selection-corrected):**
- All position variables: negative corrected scores (−0.05 to −0.18) — training reduces position decodability even at the best single token. The signal is not hiding at a specific spatial token; it is genuinely distributed and reduced by training. Confirms it was never a mean-pooling artifact.
- All velocity/angle variables: small but consistently positive corrected scores. The angle (+0.01 to +0.02) and angular_vel (+0.006 to +0.032) deltas echo the SAE MI and MLP-probe findings — weak, method-convergent evidence of learned non-linear dynamics structure.

This reproduces the same qualitative picture as the mean-pooled and MLP probes: training reduces position decodability, while adding weak non-linear dynamics structure that probes can barely detect but SAEs corroborate.

---

## Attention Pattern Analysis (Physics)

[NOT RUN — deprioritised at wrap-up, same rationale as MiniGrid attention analysis.]

---

## Layer Sweep — Causal Interventions (Physics)

[NOT RUN — deprioritised at wrap-up, same rationale as MiniGrid layer sweep.]

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

# NON-LINEAR PROBES (MLP) — Testing the "Transformed Representation" Hypothesis

**Motivation:** If training transforms position into a non-linear form (rather than discarding it), a non-linear MLP probe should recover what linear probes cannot. Both trained and untrained models probed; metric is (trained MLP − untrained MLP), averaged over 3 random splits with significance testing.

**Methodological note:** initial MLP runs gave spurious strongly-negative R² (e.g. −1.0) for wide-range position targets — a probe bug from un-standardised regression targets causing gradient explosion. Fixed by standardising targets and inverse-transforming predictions for scoring. Re-run results below are trustworthy (position scores positive and comparable to linear probe).

## MiniGrid (layer 5, 3 seeds)

| Variable | Trained MLP | Untrained MLP | Learned Δ | Significant? |
|---|---|---|---|---|
| agent_x | 0.997±0.002 | 0.997±0.000 | -0.001±0.002 | No (saturated) |
| agent_y | 0.945±0.025 | 0.921±0.008 | +0.024±0.018 | No (suggestive only) |
| agent_direction | 0.999±0.000 | 1.000±0.000 | -0.001±0.000 | No (saturated) |
| goal_x | 0.891±0.003 | 0.998±0.000 | -0.107±0.003 | **Yes (negative)** |
| goal_y | 0.010±0.022 | 0.982±0.002 | -0.972±0.022 | **Yes (negative)** |

## Physics (layer 2, single split — see note)

| Variable | Trained MLP | Untrained MLP | Learned Δ |
|---|---|---|---|
| pos_x_0 | 0.122 | 0.233 | -0.111 |
| pos_y_0 | 0.112 | 0.434 | -0.322 |
| vel_x_0 | -0.132 | -0.289 | +0.157 |
| vel_y_0 | -0.130 | -0.304 | +0.174 |
| angle_0 | -0.157 | -0.229 | +0.072 |
| angular_vel_0 | -0.189 | -0.229 | +0.040 |
| (objects 1, 2 show same pattern) | | | |

## Findings — Calibrated Claims

1. **Goal position is neither linearly nor non-linearly decodable after training, yet remains causally functional.** goal_x and goal_y show significant NEGATIVE learned deltas with the MLP (training reduces non-linear decodability too), while Mode C causal intervention recovers both at 1.000. This is the cleanest dissociation in the project: the information is unreadable by probes of either kind but fully usable by the model's own computation. This rules out "discarded as irrelevant" and confirms "transformed into a probe-inaccessible but causally-functional form."

2. **Velocity/angle (Physics): positive but weak non-linear learned signal.** Every velocity/angle variable shows a positive trained-vs-untrained MLP delta (+0.04 to +0.17), corroborating the SAE mutual information finding. HOWEVER, absolute trained MLP R² remains negative (e.g. vel_x_0 = −0.132), meaning velocity is still not robustly decodable even non-linearly. **Calibrated claim:** training moves velocity/angle representations in a decodable direction (positive delta, method-convergent with SAE MI) but they remain weakly decodable in absolute terms. The strong claim "velocity is non-linearly encoded" is NOT supported; the narrow claim "training adds weak non-linearly-accessible velocity structure" is.

3. **Position transformation is not a linearity artifact.** pos_x/pos_y show negative learned deltas with the MLP, mirroring the linear probe. The trained model's reduced position decodability holds for both linear and non-linear probes — the transformation away from probe-accessibility is genuine.

4. **agent_x / agent_direction saturate at ceiling** for both models — trivially decodable from input preservation, no headroom for a learned signal. agent_y shows a suggestive (not significant) positive delta.

**Note on Physics multi-seed:** the Physics non-linear table above is from a single train/test split; the MiniGrid table uses 3-seed averaging with significance flags. The Physics single-split deltas are directionally consistent with the linear-probe findings (position negative, velocity/angle weakly positive) and with the SAE MI, so a multi-seed re-run was judged non-essential at wrap-up. Listed as optional future work.

---

# SUBSPACE DIMENSIONALITY & GEOMETRY ANALYSIS

Tests how many residual-stream dimensions carry each variable's signal (PCA → Ridge probe at k components), and the geometric relationship between variables' probe directions (pairwise angles).

## MiniGrid (layer 5) — PCA dimensionality

| Variable | k=1 | k=3 | k=10 | k=20 | k=50 | Interpretation |
|---|---|---|---|---|---|---|
| agent_x | 0.982 | 0.989 | 0.997 | 0.997 | 0.999 | 1-dimensional — single dominant direction |
| agent_y | 0.003 | 0.003 | 0.070 | 0.131 | 0.597 | Highly distributed across many dims |
| goal_x | 0.003 | 0.003 | 0.064 | 0.451 | 0.790 | Distributed, concentrated ~k=10-20 |
| goal_y | 0.000 | 0.001 | 0.002 | 0.004 | 0.017 | Not present in any linear subspace |

## Physics (layer 2) — PCA dimensionality

| Variable | k=1 | k=8 | k=20 | k=50 |
|---|---|---|---|---|
| pos_x_0 | 0.010 | 0.075 | 0.107 | 0.136 |
| pos_y_0 | -0.004 | 0.091 | 0.124 | 0.174 |
| pos_x_1 | 0.004 | 0.042 | 0.070 | 0.084 |
| pos_y_1 | -0.000 | 0.056 | 0.083 | 0.094 |

All Physics position variables are uniformly distributed — near-zero at k=1, slowly climbing, no dominant dimension. Consistent with VQ-VAE spreading spatial information across the token grid.

## Direction Angles (degrees; ~90°=orthogonal, ~0°=aligned)

**MiniGrid:** agent_x↔agent_y = 11.9° (nearly aligned, shared subspace); agent↔goal ≈ 80-82° (near-orthogonal, separate subspaces).

**Physics:** mostly 68-89° between object positions (largely independent/orthogonal subspaces per object).

(Direction-angle computation re-run with Ridge alpha=10 to resolve an ill-conditioning warning from the initial alpha=1 fit; qualitative findings unchanged.)

## Key Mechanistic Findings

1. **agent_x is 1-dimensional; everything else is distributed.** agent_x reaches R²=0.982 with a single PCA component (the dominant input-preservation direction). agent_y, goal_x need 20-50 dimensions; goal_y isn't in any linear subspace. This quantifies the "distributed representation" claim with concrete dimensionality numbers.

2. **The PCA result resolves AND deepens the Mode B/Mode C puzzle.** agent_x is 1-dimensional and decodable from a single direction — yet Mode B (single-direction patching) gave ~0 causal recovery for agent_x. This proves a crucial point: the direction that DECODES a variable (readable) is not the direction that the model's downstream computation CAUSALLY reads from. Information can be linearly present in one dimension while the model's circuits do not causally use that direction. This is the read/write distinction (Elhage et al. framing), and it is a stronger, more precise claim than "distributed" alone.

3. **agent_x and agent_y share a near-collinear subspace (11.9° apart).** The two spatial axes are NOT independently encoded — they occupy nearly the same direction in residual space, differing in projection. Goal information is near-orthogonal to agent information (~80°), occupying a separate subspace.

4. **Physics positions are both distributed and weakly present** — no dominant dimension, uniformly low R² even at k=50, with objects encoded near-orthogonally to each other.

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

## Track A — Final Status (Wrap-Up)

### Completed experiments (both environments unless noted)
- [x] Linear probes (6-layer sweep) + training-duration comparison (MiniGrid)
- [x] **Untrained baseline control** + regularisation robustness check — the key methodological contribution
- [x] Non-linear (MLP) probes with untrained baseline + multi-seed significance (MiniGrid 3-seed; Physics single-split)
- [x] Sparse autoencoders + dictionary health + feature-variable MI
- [x] F867 monosemanticity verification (direct inspection + spawn-point control)
- [x] Causal interventions (three-mode design) — MiniGrid and Physics
- [x] Subspace dimensionality + direction-angle geometry analysis
- [x] X/Y asymmetry investigation (three hypotheses; resolved as serialisation artifact)
- [x] Per-position probing (both environments; supplementary)

### Deliberately deprioritised at wrap-up (scripts written, not run)
These were judged confirmatory rather than load-bearing — the central thesis is established without them. Listed honestly as available future work:
- [ ] Attention pattern analysis (head-level) — both environments
- [ ] F867 causal ablation
- [ ] Layer-sweep causal interventions — both environments
- [ ] Logit lens
- [ ] Physics non-linear probe multi-seed re-run (single-split result already consistent with linear)

### Stretch goal (not built)
- [ ] Partial-observability environment (third environment for robustness) — substantial new work

### Track B (next phase)
- [ ] Gemma 3 1B + circuit tracer pilot (verify current tool/model versions before starting)
- [ ] Physical reasoning prompt suite
- [ ] Attribution graph analysis

---

## Methodological Lessons (Track A)

1. **The untrained baseline is essential.** Raw probe R² conflates input-preservation (present even at random init) with learned representation. Most apparent position "encoding" is the former. This single control reframed the entire probe interpretation.
2. **Training transforms features away from probe-accessibility.** In both environments, trained models show *lower* linear (and non-linear) position decodability than untrained, while preserving causal function — world models reorganise input features rather than passively preserving them.
3. **Correlational decodability (probe R²) ≠ causal sufficiency (Mode B recovery).** A direction that decodes a variable is not necessarily the direction the model causally uses (read/write distinction).
4. **Mean-pooling destroys causal structure for patching; evaluation locality matters.** Measuring over position-invariant tokens dilutes causal effects.
5. **SAEs and probes are complementary.** SAEs find non-linear structure (velocity MI) that probes miss.
6. **Best-of-N position selection inflates scores** — requires a matched untrained baseline to interpret.
7. **The Mode B vs Mode C dissociation, replicated across two environments and tokenisation schemes, is the central finding** about world-model representation geometry: causally real but distributed.
