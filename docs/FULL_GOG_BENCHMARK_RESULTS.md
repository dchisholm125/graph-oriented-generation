# Full GOG Benchmark Results

These benchmark artifacts were produced using the full GOG engine before the public/private separation. The public repo includes GOG-Lite for reference/reproducibility of the core graph-routed context thesis.

The production engine continues privately as GOG Professional. These results are preserved here as curated historical evidence for the stronger pre-separation engine, not as claims about the smaller public GOG-Lite reference implementation.

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
| GOG | invalid_json ×1 | recoverable renderer-format failure |
| RAG 1000 | missing_semantic_behavior ×1 | unrecoverable semantic/context failure |
| RAG 16000 | missing_semantic_behavior ×2, invalid_json ×1 | high-noise instability |
| RAG 64000 | missing_semantic_behavior ×1 | more context did not guarantee structural correctness |

## Interpretation

These results do not prove GOG is universally superior to RAG. They suggest that high-budget flat retrieval can become extremely noisy and expensive, while graph-routed context can remain more controlled.

The important signal is not only Pass@1. The repeated trials also track prompt cost, cost-to-pass, and noise ratio. In this artifact, larger RAG budgets drove prompt costs and noise sharply upward, and more retrieved context did not monotonically improve validation outcomes.

## Curated Artifacts

- [Repeated-trial Markdown summary](../gog/results/context_dilution_20260514_141254.md)
- [Repeated-trial JSON artifact](../gog/results/context_dilution_20260514_141254.json)
- [Earlier full-engine JSON run](../gog/results/context_dilution_20260514_134443.json)
