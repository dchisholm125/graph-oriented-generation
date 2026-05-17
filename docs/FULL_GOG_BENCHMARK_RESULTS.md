# Full GOG Benchmark Results

These benchmark artifacts were produced using the full GOG engine before the public/private separation.

The public repository now includes **GOG-Lite**, a transparent reference implementation for reproducing the core graph-routed context thesis. The production engine continues privately as **GOG Professional**.

These results are preserved here as curated historical evidence for the stronger pre-separation engine. They should not be read as claims about the smaller public GOG-Lite implementation.

For current curated summaries from the private production research engine, see [GOG Professional Results Summary](GOG_PROFESSIONAL_RESULTS.md).

## Headline

In repeated early trials, the full GOG engine remained competitive with the best RAG settings while avoiding the extreme context noise and cost-to-pass seen at larger RAG budgets.

The strongest signal is not simply Pass@1. It is the relationship between:

- prompt-token cost
- context noise
- tokens spent per successful patch
- failure class

## Repeated-Trial Context-Dilution Results

| Mode | Pass@1 | Avg prompt tokens | Relative prompt cost vs GOG | Tokens spent/pass | Relative cost-to-pass vs GOG | Noise ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GOG | 8/9 | 7,996 | 1.00x | 9,676 | 1.00x | 0.523 |
| RAG 1000 | 8/9 | 5,759 | 0.72x | 7,158 | 0.74x | 0.306 |
| RAG 4000 | 9/9 | 12,248 | 1.53x | 12,764 | 1.32x | 0.556 |
| RAG 16000 | 6/9 | 43,609 | 5.45x | 66,216 | 6.84x | 0.902 |
| RAG 64000 | 8/9 | 73,959 | 9.25x | 83,805 | 8.66x | 0.951 |

## Failure Taxonomy Summary

| Mode | Failure class | Interpretation |
| --- | --- | --- |
| GOG | `invalid_json ×1` | Recoverable renderer-format failure |
| RAG 1000 | `missing_semantic_behavior ×1` | Unrecoverable semantic/context failure |
| RAG 16000 | `missing_semantic_behavior ×2`, `invalid_json ×1` | High-noise instability |
| RAG 64000 | `missing_semantic_behavior ×1` | More context did not guarantee structural correctness |

## Interpretation

These results do **not** prove that GOG is universally superior to RAG.

They suggest a narrower and more useful claim:

> High-budget flat retrieval can become extremely noisy and expensive, while graph-routed context can remain more controlled.

In this artifact, larger RAG budgets drove prompt cost and noise sharply upward. More retrieved context did not monotonically improve validation outcomes.

The key result is **RAG 16000**:

- 5.45× higher average prompt cost than GOG
- 6.84× higher observed cost-to-pass
- 90.2% context noise
- 6/9 Pass@1
- multiple unrecoverable semantic/context failures

This is the core context-dilution signal.

## Why This Matters

Coding agents often fail not because they lack context, but because the relevant structure is diluted inside too much unrelated context.

GOG tests a different approach:

> Route through repository structure first, then serve the model a smaller and more relevant context bundle.

The goal is not to eliminate language models from coding. The goal is to give them better-shaped context before they reason.

## Relationship to GOG-Lite

The public repository contains **GOG-Lite**, not the full pre-separation GOG engine.

GOG-Lite is intentionally simple:

- regex-based import graph construction
- keyword seeding
- bounded graph expansion
- transparent file and token caps
- no proprietary ranking or pruning

GOG-Lite exists so the public benchmark path remains inspectable and reproducible. GOG Professional continues privately as the production-grade engine.

## Curated Artifacts

- [Repeated-trial Markdown summary](../gog/results/context_dilution_20260514_141254.md)
- [Repeated-trial JSON artifact](../gog/results/context_dilution_20260514_141254.json)
- [Earlier full-engine JSON run](../gog/results/context_dilution_20260514_134443.json)
