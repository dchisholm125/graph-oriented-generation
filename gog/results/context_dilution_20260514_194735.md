| Mode | Pass@1 | Avg prompt tokens | Relative prompt cost vs GOG-Lite | Tokens spent / pass | Relative cost-to-pass vs GOG-Lite | Noise ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GOG-Lite | 1/1 | 426 | 1.00x | 484 | 1.00x | 0.000 |
| RAG 1000 | 1/1 | 3,221 | 7.56x | 3,301 | 6.82x | 0.333 |
| RAG 4000 | 1/1 | 9,865 | 23.16x | 9,925 | 20.51x | 0.846 |
| RAG 16000 | 1/1 | 19,390 | 45.52x | 19,449 | 40.18x | 0.947 |
| RAG 64000 | 1/1 | 19,390 | 45.52x | 19,434 | 40.15x | 0.947 |

Failure taxonomy: no failed model-backed validation cases were observed in this run.
