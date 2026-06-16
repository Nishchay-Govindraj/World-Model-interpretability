# Lessons Learned
# Reviewed at the start of every session. Updated immediately after any mistake.

## Literature & Citations
- L001: SAEs have already been applied to discrete game world models (OthelloGPT, ChessGPT).
         The genuine gap is physically richer, continuous-dynamics environments. Do not overclaim.
- L002: Circuit tracing has been extended to vision-language models.
         The genuine gap is standalone world models and physical reasoning specifically.
- L003: Geiger et al. causal interventions paper venue is CLeaR/PMLR, not AAAI.
- L004: Nanda et al. co-author is Wattenberg, not Bertsimas.
- L005: Always verify paper venues, co-authors, and arXiv IDs before including in writing.
         Do not trust training-data memory for citation details — look them up.

## Hardware & Compute
- L006: RTX 3060 has 6GB VRAM. Every model/batch decision must respect this ceiling.
         Always estimate VRAM usage before recommending a model size or batch size.
- L007: Track B (Gemma 3 1B) fits on RTX 3060. Track A large models (25M param) need the
         A100 cluster. Scripts must auto-detect and configure accordingly.

## Code Quality
- L008: Never assume a function, class, or module exists. Always verify with search/read before
         referencing. Hallucinated imports are a common source of silent bugs.
- L009: No placeholders, stubs, or TODOs in delivered code. Full implementations only.
- L010: Validate data pipeline outputs (shapes, dtypes, ranges) before training anything.
         A bug in data collection invalidates all downstream results.

## Project-Specific
- L011: Ground-truth state labels are the foundation of probing. If label logging is wrong,
         probe results are meaningless. Validate labels visually before any probe training.
- L012: Probe accuracy must be evaluated on held-out trajectories, not training data.
         Report per-layer, per-variable accuracy. Do not aggregate prematurely.
