# Public / Private / SRM Lab Boundary

## What is GOG-Lite?

`gog_engine_lite/` provides the **GOG Reference Implementation** — a simplified graph-native context layer that demonstrates the core GOG thesis:

> Graph-routed context can compete with flat keyword retrieval for coding tasks.

It does **not** include:
- Advanced scoring heuristics
- Semantic embeddings for seeding
- Persistent `.gog/` artifact onboarding
- Production-grade context membranes
- Graph fusion, runtime graphs, or change graphs
- Production repo-strata discovery
- Compact surgical prompt construction
- Structural anchor extraction
- Symbol capability reports
- Mutation-safety APIs or assistant integration contracts

### What GOG-Lite does

- **Simple import graph** (`import_graph.py`): Builds dependency graphs via regex scanning (not tree-sitter).
- **Keyword-based seeding** (`graph_search.py`): Extracts filename mentions and keyword tokens from prompts.
- **Bounded expansion**: BFS up to depth=2 from seeds.
- **Distance-based cap** (`lite_membrane.py`): Keeps closest files, no multi-factor scoring.
- **Direct serving** (`lite_serving.py`): Builds context bundles without persistent artifacts.

### What GOG-Lite does not do

- Semantic similarity (sentence-transformers) for seeding
- Multi-phase ASR or production tree-sitter parsing
- Repo fingerprinting, `.gog/` artifacts, or onboarding
- Advanced test-file / config-file penalties
- Task-specific graph seeding or runtime graph fusion
- Production integrations (e.g., OpenClaw, enterprise agents)
- Professional repo-strata discovery
- Context membrane scoring
- Compact surgical prompt rendering
- Structural anchor extraction
- Symbol capability checks
- Safe mutation application

### Running a GOG-Lite benchmark

```bash
# Public repo: compare GOG-Lite against traditional RAG
python3 gog/benchmark_executable_patch.py --mode gog_lite --mode traditional_rag --dry-run

# Context dilution benchmark (includes GOG-Lite + RAG)
python3 gog/benchmark_context_dilution.py --dry-run --task debug_query_serialization_easy
```

Dry-run mode validates retrieval and context construction only. It does not invoke a model or run validation commands, so Pass@1, cost-to-pass, and failure taxonomy are not available in dry-run summaries.

For a public model-backed benchmark, install a local Ollama model and the Vue fixture dependencies, then run without `--dry-run`:

```bash
ollama pull kimi-k2.6:cloud

cd gog/fixtures/vue3-realworld-example-app
corepack pnpm install
cd ../../..

PYTHONPATH=. python3 gog/benchmark_context_dilution.py \
  --task debug_query_serialization_easy \
  --trials 1 \
  --attempts 1 \
  --model kimi-k2.6:cloud
```

That command compares `gog_lite` against `traditional_rag` budgets using the public harness and the local Ollama HTTP API. It writes JSON and Markdown artifacts under `gog/results/` by default. The public CLI also supports `--output-dir`, repeated `--task`, repeated `--rag-budget`, `--timeout-s`, and retry controls. Full `gog` mode remains private as part of GOG Professional.

## What This Repository Contains

This repository is the **GOG-Lite reference implementation and benchmark lab** for Graph-Oriented Generation.

- **GOG-Lite** (`gog_engine_lite/`, `gog_cli/lite_*.py`) is a simplified graph-native context layer that demonstrates the core GOG thesis: graph-routed context can compete with flat keyword retrieval for coding tasks.
- **Benchmarks** (`gog/`, `gog_cli/gold_context.py`, `gog_cli/failure_taxonomy.py`) allow reproducibility of results against public repositories.
- **Curated professional summaries** (`docs/GOG_PROFESSIONAL_RESULTS.md`) publish selected GOG Professional outcomes without exposing the private production engine.
- **Research artifacts** (`docs/`, `GOG_PAPER.pdf`) document the methodology and failure analysis.

## What This Repository Does Not Contain

- **Production GOG Professional**: A separate private implementation with advanced repo strata, context membranes, compact task-specific prompts, structural anchors, symbol capability checks, mutation-safety contracts, scalable onboarding, graph fusion, and enterprise integrations.
- **SRM Lab**: The long-term Symbolic Reasoning Model research direction is maintained in a separate repository to keep this one focused on GOG as a practical context layer for today's coding assistants.

## Reproducibility

Anyone should be able to run the benchmarks in this repository to verify the core claim. The benchmark harnesses are public; the production optimizations are private.

| Layer | Location | License |
|-------|----------|---------|
| GOG-Lite Reference | `gog_engine_lite/`, `gog_cli/lite_*.py` | Apache 2.0 |
| Benchmarks & Metrics | `gog/`, `gog_cli/gold_context.py`, `gog_cli/failure_taxonomy.py` | Apache 2.0 |
| Curated GOG Professional Summaries | `docs/GOG_PROFESSIONAL_RESULTS.md` | CC BY 4.0 |
| System Docs | `docs/GOG_SYSTEM_MODEL.md`, `docs/REASONER_INTERFACE.md` | CC BY 4.0 |
| Production GOG | Private repo | Proprietary |
| SRM Lab | Separate repo | TBD |

## Public Messaging Rule

Public material should say:

> GOG Lite is the reference lab. GOG Professional is the production research engine.

It should not publish the mechanisms that make the professional engine commercially valuable. Public summaries may report benchmark setup, validation status, prompt-token totals, context precision, recall, noise, dominant package, and high-level product concepts. They should not include implementation-level ranking formulas, renderer internals, mutation APIs, private artifacts, or assistant integration contracts.

## Moving the Boundary

If you are a contributor and believe a file should move across the public/private boundary, open an issue referencing this document and the file in question.
