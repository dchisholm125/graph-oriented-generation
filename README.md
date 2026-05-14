# Graph-Oriented Generation

Graph-Oriented Generation (GOG) is a research direction for making software repositories easier for AI systems to inspect, edit, and validate. The core thesis is simple: a codebase is not a flat bag of text chunks. It is a graph of files, imports, symbols, tests, and validation surfaces, and retrieval should preserve enough of that structure to help a model act with less noise.

This public repository contains the GOG reference implementation and benchmark lab. It is meant to be runnable, inspectable, and scientifically useful. It contains GOG-Lite, public benchmark harnesses, gold-context scoring, failure taxonomy, curated result artifacts, and methodology notes.

The production-grade engine is developed separately for commercial use. SRM and SEL research are maintained separately as moonshot research. This public repo focuses on reproducible evidence: when graph-oriented retrieval helps, when it fails, and how to compare it honestly against RAG-style baselines.

Links:
- Project page: https://derekchisholm.com/gog
- GitHub repo: https://github.com/dchisholm125/graph-oriented-generation

## What Is Included

- `gog_engine_lite/`: GOG-Lite, a public reference graph builder and context selector.
- `gog_cli/lite_serving.py`: public context bundle construction.
- `gog_cli/lite_membrane.py`: transparent file and token caps for GOG-Lite.
- `gog_cli/gold_context.py`: gold-context precision, recall, and noise scoring.
- `gog_cli/failure_taxonomy.py`: executable-patch failure classification.
- `gog_cli/executable_patch_benchmark.py`: public executable-patch benchmark harness.
- `gog_cli/context_poisoning_benchmark.py`: GOG-Lite versus progressively larger RAG contexts.
- `gog/fixtures/`: small public fixtures used by dry-run benchmarks.
- `gog/results/`: curated public result artifacts.
- `docs/PUBLIC_PRIVATE_BOUNDARY.md`: boundary between this repo, GOG Professional, and SRM research.
- `docs/GOG_LITE_DESIGN.md`: public reference design notes.
- `docs/GOG_SYSTEM_MODEL.md`: system-level research framing.

## GOG-Lite

GOG-Lite is intentionally small:

- Regex-based import graph construction.
- Keyword seeding and bounded graph expansion.
- Simple integer scoring.
- File-count and token-budget limits.
- Keyword snippets for large files.
- No embeddings.
- No proprietary ranking or pruning.

That simplicity is the point. GOG-Lite is not positioned as the strongest possible GOG engine. It is a public reference implementation that makes the benchmark path auditable.

## Benchmark Lab

The public benchmark lab currently supports:

- `gog_lite`: the public reference implementation.
- `traditional_rag`: a deterministic keyword/chunk baseline.

Full production GOG mode is not part of this public package. If a benchmark menu used to expose `gog`, use `--mode gog_lite` for the public reference path.

Some historical benchmark artifacts were produced using the pre-separation/full GOG engine. The public GOG-Lite implementation is provided to make the core graph-routed context thesis inspectable and reproducible without the production engine.

Run the main dry-run checks:

```bash
python3 gog/benchmark_executable_patch.py --mode gog_lite --dry-run
python3 gog/benchmark_context_poisoning.py --dry-run --task debug_query_serialization_easy
```

Run the GOG-Lite regression tests:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gog_lite.py
```

## Current Results

Early repeated trials show that prompt-scoped graph retrieval can reduce context noise and improve target-file precision on focused code-edit tasks. Some curated historical results were produced with the pre-separation/full GOG engine and should be read as professional-engine results, not as claims about the smaller public GOG-Lite implementation.

On the debug query serialization dry-run fixture, the public GOG-Lite path currently returns the expected files:

- `src/utils/params-to-query.spec.ts`
- `src/utils/params-to-query.ts`
- Context precision: `1.0`
- Noise ratio: `0.0`

That is a focused debug dry run, not a universal claim that GOG-Lite achieves zero noise. The context-poisoning benchmark compares that focused context against RAG contexts at larger token budgets. Current research focus: when does GOG fail, and what failure modes are recoverable versus structural?

## Failure Taxonomy

The executable-patch benchmark records more than pass/fail. It tracks context precision, recall, noise ratio, patch validity, edit-surface accuracy, spurious imports, validation failure classes, and cost-to-pass metrics where model execution is enabled.

This matters because retrieval improvements can fail downstream for different reasons:

- The correct files were not retrieved.
- The correct files were retrieved but drowned in noise.
- The model edited outside the served context.
- The patch was syntactically invalid.
- The validation command exposed an actual behavioral miss.

The public goal is not to claim universal superiority. The goal is to make these failures measurable.

## Public / Private / SRM Boundary

This repo keeps the public benchmark surface and GOG-Lite reference implementation.

GOG Professional contains the production graph engine, advanced context membrane, onboarding and fingerprinting pipeline, production CLI, context serving, and commercial integration surfaces.

SRM Lab contains SEL, SRM engine work, symbolic distillation, distilled datasets, SRM papers, and symbolic mutation-planning research.

See `docs/PUBLIC_PRIVATE_BOUNDARY.md` for the maintained boundary.

## Development Notes

The repo should remain runnable without private code. Public imports should resolve using only:

- `gog_engine_lite`
- public `gog_cli` benchmark helpers
- Python standard library plus dependencies in `requirements.txt`

Before opening a public PR, run:

```bash
python3 -m py_compile $(find gog gog_cli gog_engine_lite tests -name '*.py')
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gog_lite.py
python3 gog/benchmark_executable_patch.py --mode gog_lite --dry-run
python3 gog/benchmark_context_poisoning.py --dry-run --task debug_query_serialization_easy
```
