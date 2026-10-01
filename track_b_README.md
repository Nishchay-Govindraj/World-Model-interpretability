# Track B — Circuit Tracing on Gemma-2 2B

This folder contains the Track B work: mechanistic analysis of physical
reasoning circuits in the pretrained Gemma-2 2B model using Anthropic's
circuit-tracer library with GemmaScope transcoders.

## Contents
- `track_b_circuit_tracing.ipynb` — the full pipeline: model and transcoder
  loading, attribution-graph construction, the validated feature-ablation
  method (with a degenerate-token guard), the Dallas-to-Austin method
  validation, the random-ablation specificity control, and the Neuronpedia
  feature-semantics lookup.
- `results/track_b_results.md` — verified ablation thresholds and
  feature-semantics findings.

## Large data files (not in this repo)
The raw attribution graphs (.pt files) and full Neuronpedia JSON exports are
too large for GitHub (several hundred MB each). They are archived on Google
Drive and available on request. The notebook regenerates them from the saved
graphs.

## Reproducing
Run the notebook in Google Colab with a GPU runtime. It requires a Hugging
Face token with access to the gated google/gemma-2-2b model, and installs
circuit-tracer and its dependencies in the first cells.
