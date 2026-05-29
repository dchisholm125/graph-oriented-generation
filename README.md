# Graph-Oriented Generation

Graph-Oriented Generation (GOG) is a repo-reasoning layer for AI coding agents.
It maps repository structure before LLM tokens are spent, chooses a context
strategy, identifies edit and evidence boundaries, surfaces semantic hazards,
and helps coding assistants reduce repo archaeology.

The core thesis is conservative: a software repository is not a flat bag of text
chunks. It is a graph of packages, files, imports, symbols, tests, generated
surfaces, validation commands, and architectural strata. Coding agents can work
with less noise when that structure is inspected before generation begins.

Links:

- Project page: https://derekchisholm.com/gog
- GitHub repo: https://github.com/dchisholm125/graph-oriented-generation

## What GOG Is / Is Not

GOG is:

- A repo-reasoning layer for AI coding agents.
- A graph/cartography-based context system.
- A way to reduce repo archaeology before LLM generation.
- A way to identify semantic hazards, edit boundaries, evidence boundaries, and
  bad paths.

GOG is not:

- A coding assistant replacement.
- A universal RAG killer.
- A guarantee of better patches.
- A claim that retrieval is unnecessary.

RAG and hybrid retrieval can solve many coding tasks. GOG's current evidence is
about context efficiency, repo locality, bad-path guidance, and reducing the
amount of repository discovery work pushed onto the assistant.

## Public and Private Split

This repository contains **GOG Lite**, the public reference implementation and
benchmark lab. It is intentionally small, inspectable, and runnable without
private code.

**GOG Professional** is the private research/product engine. It contains the
production context strategy layer, benchmark harnesses, semantic-hazard
synthesis, surgical mutation work, and assistant integration experiments.
Public docs may summarize professional-engine results, but private source code,
private implementation details, raw artifacts, local paths, and sensitive
benchmark internals do not belong in this repository.

See [docs/PUBLIC_PRIVATE_BOUNDARY.md](docs/PUBLIC_PRIVATE_BOUNDARY.md) for the
maintained boundary.

## Five-Pillar Model

1. **Repo understanding**

   Build a repository map: packages, strata, imports, generated regions,
   validation surfaces, and likely edit/evidence boundaries.

2. **Preflight reasoning**

   Decide whether a task fits a tight graph membrane, needs stratum expansion,
   should fall back to hybrid retrieval, or requires clarification before tokens
   are spent.

3. **Context serving**

   Serve bounded, provenance-aware context to an assistant instead of dumping a
   large undifferentiated file set.

4. **Assistant guidance**

   Add task-local guidance: expected edit files, evidence-only files,
   forbidden paths, semantic hazards, and known bad paths.

5. **Mutation + evaluation**

   Evaluate patches against validation commands, edit boundaries, context
   locality, file reads, token spend, and failure taxonomy.

## What Is Included

- `gog_engine_lite/`: GOG Lite, a public reference graph builder and context
  selector.
- `gog_cli/lite_serving.py`: public context bundle construction.
- `gog_cli/lite_membrane.py`: transparent file and token caps for GOG Lite.
- `gog_cli/gold_context.py`: gold-context precision, recall, and noise scoring.
- `gog_cli/failure_taxonomy.py`: executable-patch failure classification.
- `gog_cli/executable_patch_benchmark.py`: public executable-patch benchmark
  harness.
- `gog_cli/context_dilution_benchmark.py`: GOG Lite versus deterministic
  keyword/chunk retrieval baselines.
- `gog/fixtures/`: small public fixtures used by dry-run benchmarks.
- `gog/results/`: curated public result artifacts.
- `docs/`: public design, boundary, and benchmark summary notes.

## GOG Lite

GOG Lite is intentionally simple:

- Regex-based import graph construction.
- Keyword seeding and bounded graph expansion.
- Simple integer scoring.
- File-count and token-budget limits.
- Keyword snippets for large files.
- No embeddings.
- No proprietary ranking or pruning.

That simplicity is the point. GOG Lite is not positioned as the strongest
possible GOG engine. It is a public reference implementation that makes the core
graph-routed context thesis auditable.

## Benchmark Highlights

Public benchmark results should be read conservatively. Some results summarize
GOG Professional runs because the professional engine remains private. They are
not claims about the smaller GOG Lite implementation.

- **Orca Rust surgical task:** GOG Professional compact/surgical context
  produced a validated Rust patch against the real Orca Whirlpools SDK. The
  relevant signal is surgical context plus validation, not exposure of private
  implementation mechanics.
- **Orca cross-language deployment parity:** In a fair hybrid comparison, GOG
  and hybrid RAG all reached validated outcomes. GOG used 6 retrieved files and
  6,500 total tokens; hybrid RAG used 27 files / 14,132 total tokens at 16K and
  34 files / 18,714 total tokens at 64K. The result supports context-efficiency
  and repo-locality claims, not a claim that RAG cannot solve the task.
- **OpenCode Orca TypeScript type guards:** Baseline OpenCode failed validation
  and typecheck. RAG hybrid passed. GOG v1 selected the right files but lacked
  semantic-hazard guidance and failed. GOG v2 added bad-path guidance for the
  generated-vs-consolidated type trap and passed while reading fewer files than
  both baseline and RAG in that rerun. RAG was faster wall-clock in that run.

Detailed summaries:

- [GOG Professional results](docs/GOG_PROFESSIONAL_RESULTS.md)
- [OpenCode bakeoff summary](docs/OPENCODE_BAKEOFF_SUMMARY.md)
- [Benchmark results index](docs/FULL_GOG_BENCHMARK_RESULTS.md)

## Running the Public Benchmark Lab

The public benchmark lab currently supports:

- `gog_lite`: the public reference implementation.
- `traditional_rag`: a deterministic keyword/chunk baseline.

Full production GOG mode is not part of this public package. If a benchmark menu
used to expose `gog`, use `--mode gog_lite` for the public reference path.

Run the main dry-run checks:

```bash
python3 gog/benchmark_executable_patch.py --mode gog_lite --dry-run
python3 gog/benchmark_context_dilution.py --dry-run --task debug_query_serialization_easy
```

Dry-run mode validates retrieval, context construction, context-quality metrics,
and prompt-size estimates only. It does not invoke a model or run validation
commands, so Pass@1, cost-to-pass, tokens spent per pass, and failure taxonomy
are reported as dry-run or `n/a`. Run without `--dry-run` for model-backed
pass/fail and failure-class results.

Run the GOG Lite regression tests:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gog_lite.py
```

## Current Status

- GOG Lite is public for transparency, research, and reproducibility.
- GOG Professional is private and actively benchmarked.
- The project is looking for technical feedback, walkthroughs, and pilot
  conversations.
- For contact, use the existing project page: https://derekchisholm.com/gog

## Development Notes

The repo should remain runnable without private code. Public imports should
resolve using only:

- `gog_engine_lite`
- public `gog_cli` benchmark helpers
- Python standard library plus dependencies in `requirements.txt`

Before opening a public PR, run:

```bash
python3 -m py_compile $(find gog gog_cli gog_engine_lite tests -name '*.py')
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_gog_lite.py
python3 gog/benchmark_executable_patch.py --mode gog_lite --dry-run
python3 gog/benchmark_context_dilution.py --dry-run --task debug_query_serialization_easy
git diff --check
```
