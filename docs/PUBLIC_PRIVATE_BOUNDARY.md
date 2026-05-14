# Public / Private / SRM Lab Boundary

## What This Repository Contains

This repository is the **GOG-Lite reference implementation and benchmark lab** for Graph-Oriented Generation.

- **GOG-Lite** (`gog_engine_lite/`, `gog_cli/lite_*.py`) is a simplified graph-native context layer that demonstrates the core GOG thesis: graph-routed context can compete with flat keyword retrieval for coding tasks.
- **Benchmarks** (`gog/`, `gog_cli/gold_context.py`, `gog_cli/failure_taxonomy.py`) allow reproducibility of results against public repositories.
- **Research artifacts** (`docs/`, `GOG_PAPER.pdf`) document the methodology and failure analysis.

## What This Repository Does Not Contain

- **Production GOG Professional**: A separate private implementation with advanced context membranes, scalable onboarding, graph fusion, and enterprise integrations.
- **SRM Lab**: The long-term Symbolic Reasoning Model research direction is maintained in a separate repository to keep this one focused on GOG as a practical context layer for today's coding assistants.

## Reproducibility

Anyone should be able to run the benchmarks in this repository to verify the core claim. The benchmark harnesses are public; the production optimizations are private.

| Layer | Location | License |
|-------|----------|---------|
| GOG-Lite Reference | `gog_engine_lite/`, `gog_cli/lite_*.py` | Apache 2.0 |
| Benchmarks & Metrics | `gog/`, `gog_cli/gold_context.py`, `gog_cli/failure_taxonomy.py` | Apache 2.0 |
| System Docs | `docs/GOG_SYSTEM_MODEL.md`, `docs/REASONER_INTERFACE.md` | CC BY 4.0 |
| Production GOG | Private repo | Proprietary |
| SRM Lab | Separate repo | TBD |

## Moving the Boundary

If you are a contributor and believe a file should move across the public/private boundary, open an issue referencing this document and the file in question.
