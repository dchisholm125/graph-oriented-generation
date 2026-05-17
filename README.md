# Graph-Oriented Generation

Graph-Oriented Generation (GOG) is a research direction for making software repositories easier for AI systems to inspect, edit, and validate. The core thesis is simple: a codebase is not a flat bag of text chunks. It is a graph of files, imports, symbols, tests, and validation surfaces, and retrieval should preserve enough of that structure to help a model act with less noise.

This public repository contains the GOG reference implementation and benchmark lab. It is meant to be runnable, inspectable, and scientifically useful. It contains GOG-Lite, public benchmark harnesses, gold-context scoring, failure taxonomy, curated result artifacts, and methodology notes.

GOG Professional, the production research engine, is developed privately for commercial use. Selected GOG Professional benchmark summaries are published here for transparency, but the production engine is not included in this repository. SRM and SEL research are maintained separately as moonshot research. This public repo focuses on reproducible evidence: when graph-oriented retrieval helps, when it fails, and how to compare it honestly against RAG-style baselines.

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
- `gog_cli/context_dilution_benchmark.py`: GOG-Lite versus progressively larger RAG contexts.
- `gog/fixtures/`: small public fixtures used by dry-run benchmarks.
- `gog/results/`: curated public result artifacts.
- `docs/PUBLIC_PRIVATE_BOUNDARY.md`: boundary between this repo, GOG Professional, and SRM research.
- `docs/GOG_PROFESSIONAL_RESULTS.md`: curated public summaries from the private GOG Professional engine.
- `docs/GOG_LITE_DESIGN.md`: public reference design notes.
- `docs/GOG_SYSTEM_MODEL.md`: system-level research framing.

## GOG Lite and GOG Professional

This repository contains **GOG Lite**, the public reference implementation for Graph-Oriented Generation. It demonstrates the core thesis: codebases are structured symbolic systems, and coding agents can benefit from graph-routed context instead of flat retrieved text.

**GOG Professional** is the private production research implementation. It extends the public thesis with repo strata discovery, compact task-specific context, structural anchors, symbol capability checks, and mutation-safety contracts for coding-assistant integrations.

Selected GOG Professional benchmark summaries are published here for transparency, but the production engine is not included in this repository.

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
python3 gog/benchmark_context_dilution.py --dry-run --task debug_query_serialization_easy
```

Dry-run mode validates retrieval, context construction, context-quality metrics, and prompt-size estimates only. It does not invoke a model or run validation commands, so Pass@1, cost-to-pass, tokens spent per pass, and failure taxonomy are reported as dry-run or `n/a`. Run without `--dry-run` for model-backed pass/fail and failure-class results.

### Public Model-Backed Benchmark

The public model-backed path uses the local Ollama HTTP API and does not require private GOG Professional code. It runs GOG-Lite and `traditional_rag` against an executable patch task, applies the model's JSON patch in a disposable repo copy, and runs the task validation command. The resulting JSON/Markdown artifacts include Pass@1, tokens spent per pass, failure taxonomy, context precision/recall/noise, and prompt-token estimates.

Prerequisites:

```bash
ollama pull kimi-k2.6:cloud
cd gog/fixtures/vue3-realworld-example-app
corepack pnpm install
cd ../../..
```

Run a focused local benchmark:

```bash
PYTHONPATH=. python3 gog/benchmark_context_dilution.py \
  --task debug_query_serialization_easy \
  --trials 1 \
  --attempts 1 \
  --model kimi-k2.6:cloud
```

The context-dilution benchmark supports `--model`, `--output-dir`, repeated `--task`, `--attempts`, `--trials`, `--timeout-s`, `--retries`, and repeated `--rag-budget`. Expected runtime depends on the selected model, local hardware, and whether fixture dependencies are already installed.

Run the GOG-Lite regression tests:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gog_lite.py
```

## Current Results

Early repeated trials show that prompt-scoped graph retrieval can reduce context noise and improve target-file precision on focused code-edit tasks. Some curated historical results were produced with the pre-separation/full GOG engine and should be read as professional-engine results, not as claims about the smaller public GOG-Lite implementation.

The current lead professional result is an Orca Whirlpools case study where GOG Professional selected the correct SDK package, produced a validated Rust patch, and used 2,406 prompt tokens with 0.833 context precision and 0.167 noise. Comparable RAG runs either failed at low budgets or passed only with substantially worse context precision and higher structural noise.

### Lead GOG Professional Case Study

| Mode | Pass? | Prompt tokens | Precision | Recall | Noise | Dominant package |
| --- | :---: | ---: | ---: | ---: | ---: | --- |
| **GOG Professional** | **yes** | **2,406** | **0.833** | **1.0** | **0.167** | **`rust-sdk/core`** |
| RAG hybrid 1k | no | 578 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 4k | no | 578 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 16k | yes | 2,494 | 0.143 | 0.2 | 0.857 | `programs/whirlpool` |
| RAG hybrid 64k | yes | 4,831 | 0.2 | 0.6 | 0.8 | `programs/whirlpool` |

GOG Professional is being developed as a repo-intelligence and mutation-safety layer for coding assistants. It does not replace coding assistants; it gives them better repo context and safer mutation surfaces. For technical walkthroughs or pilot integrations, use the contact path on the project page.

## Benchmark Results

- [GOG Professional results summary](docs/GOG_PROFESSIONAL_RESULTS.md): selected public summaries from the private production research engine.
- [Full pre-separation GOG benchmark results](docs/FULL_GOG_BENCHMARK_RESULTS.md): curated historical results from the full GOG engine before the public/private separation.

On the debug query serialization dry-run fixture, the public GOG-Lite path currently returns the expected files:

- `src/utils/params-to-query.spec.ts`
- `src/utils/params-to-query.ts`
- Context precision: `1.0`
- Noise ratio: `0.0`

That is a focused debug dry run, not a universal claim that GOG-Lite achieves zero noise. The context-dilution benchmark compares that focused context against RAG contexts at larger token budgets. Current research focus: when does GOG fail, and what failure modes are recoverable versus structural?

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

GOG Professional contains the production graph engine, advanced repo-strata discovery, compact task-specific context, structural anchors, symbol capability checks, mutation-safety contracts, onboarding and fingerprinting pipeline, production CLI, context serving, and commercial integration surfaces.

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
python3 gog/benchmark_context_dilution.py --dry-run --task debug_query_serialization_easy
```
