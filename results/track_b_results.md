# Track B — Verified Results

## Ablation thresholds (features removed before the prediction changes)
Method validated on Dallas->Austin (flips to real words). Every flip checked
to land on a real word; garbage flips flagged and not counted. Random-100
specificity control passed for all six prompts.

| Prompt | Clean | Flip threshold | Notes |
|---|---|---|---|
| Dallas (control) | Austin | ~5 | factual-recall reference |
| melt | melt | 5 | unstable, alternates real/garbage |
| inertia | down | 50 | clean |
| tip_over | out | 100 | unstable near boundary |
| float_sink | sink | 200 | clean, flips to 'be' |
| support_remove | fall | >300 | survives all levels |
| collision | move | >300 | survives all levels |

Random-100 control: all six predictions unchanged, confirming specificity.

## Feature semantics (Neuronpedia auto-interp)
The two features shared across all six prompts are generic, not physical:
L25 f4717 (common short words near numbers) and L24 f13277 (questions and
requests). Prompt-specific top features are also largely generic (numerical/
lab data, code and file paths, prepositions), with only a few topically
physical (snow/cold for melt, space/physics for collision). No clean
physical-concept features drive the predictions.

## Interpretation
Physical reasoning in Gemma-2 2B is distributed and redundant (4/6 prompts
survive 50-300+ feature ablations vs ~5 for factual recall), carried by
features that are individually generic. This mirrors the distributed causal
structure found in the custom world models of Track A.
