| Mode | Pass@1 | Avg prompt tokens | Relative prompt cost vs GOG-Lite | Tokens spent / pass | Relative cost-to-pass vs GOG-Lite | Noise ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GOG-Lite | 1/1 | 426 | 1.00x | 482 | 1.00x | 0.000 |
| RAG 1000 | 1/1 | 3,221 | 7.56x | 3,289 | 6.82x | 0.333 |
| RAG 4000 | 1/1 | 9,865 | 23.16x | 9,931 | 20.60x | 0.846 |
| RAG 16000 | 0/1 | 19,390 | 45.52x | n/a | n/a | 0.947 |
| RAG 64000 | 1/1 | 19,390 | 45.52x | 19,458 | 40.37x | 0.947 |

| Mode | Recoverable failures | Unrecoverable failures | High architectural concern | Failure classes |
| --- | ---: | ---: | ---: | --- |
| GOG-Lite | 0 | 0 | 0 | none |
| RAG 1000 | 0 | 0 | 0 | none |
| RAG 4000 | 0 | 0 | 0 | none |
| RAG 16000 | 1 | 0 | 0 | invalid_json: 1 |
| RAG 64000 | 0 | 0 | 0 | none |
