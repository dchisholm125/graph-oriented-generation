| Mode | Pass@1 | Avg prompt tokens | Relative prompt cost vs GOG-Lite | Tokens spent / pass | Relative cost-to-pass vs GOG-Lite | Noise ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GOG-Lite | 1/1 | 426 | 1.00x | 483 | 1.00x | 0.000 |
| RAG 1000 | 1/1 | 3,221 | 7.56x | 3,304 | 6.84x | 0.333 |
| RAG 4000 | 1/1 | 9,865 | 23.16x | 9,934 | 20.57x | 0.846 |
| RAG 16000 | 1/1 | 19,390 | 45.52x | 19,451 | 40.27x | 0.947 |
| RAG 64000 | 1/1 | 19,390 | 45.52x | 19,455 | 40.28x | 0.947 |

| Mode | Recoverable failures | Unrecoverable failures | High architectural concern | Failure classes |
| --- | ---: | ---: | ---: | --- |
| GOG-Lite | 0 | 0 | 0 | none |
| RAG 1000 | 0 | 0 | 0 | none |
| RAG 4000 | 0 | 0 | 0 | none |
| RAG 16000 | 0 | 0 | 0 | none |
| RAG 64000 | 0 | 0 | 0 | none |
