# REPO_SEPARATION_MANIFEST.md — File-Level Migration Plan

> **Status:** Ready for execution review. Do not execute migration yet.
> **Derived from:** `REPO_SEPARATION_AUDIT.md` and consultant feedback.
> **Principle:** Public repo must feel alive, not a stripped-down corpse.

---

## Conventions Used

| Code | Meaning |
| :--- | :--- |
| `KEEP_PUBLIC` | Stays in public repo unchanged |
| `KEEP_PUBLIC_SANITIZE` | Stays public but review for sensitive content |
| `MOVE_PRIVATE` | Move to private commercial repo |
| `MOVE_SRM` | Move to SRM moonshot repo |
| `REVIEW_REQUIRED` | Needs more review before final classification |
| `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | Remove from public after moving (don't leave copies) |

For each file:
- **purpose**: one-sentence summary
- **classification**: per above codes
- **reason**: why it fits that classification
- **replacement_needed**: yes/no — if yes, describe the GOG-Lite/spec/stub that should replace it
- **risk**: LOW / MEDIUM / HIGH — potential exposure or credibility risk if misclassified

---

## 1. gog_engine/ — Core Graph Parsing & Search

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ast_parser.py` | AST-based import graph builder for JS/TS/Vue | `MOVE_PRIVATE` | Proprietary multi-language parsing heuristics, alias resolution, Vue `<script setup>` handling | YES: Simple glob-based import scanner for GOG-Lite | HIGH |
| `ts_parser.py` | Tree-sitter based parser for TS/Vue | `MOVE_PRIVATE` | Tree-sitter integration is advanced/commercial | YES: Stub using basic file reading | HIGH |
| `graph_search.py` | Semantic seeding + deterministic graph traversal | `MOVE_PRIVATE` | Core seeding logic is production IP | YES: Simple distance-based expansion for GOG-Lite | HIGH |
| `__init__.py` | Module exports | `MOVE_PRIVATE` | Follows engine contents | YES: Reduced exports for GOG-Lite | MEDIUM |
| `minimax_client.py` | LLM client for minimax | `KEEP_PUBLIC` | Standard LLM client wrapper; not GOG-specific | N/A | LOW |
| `opencode_client.py` | LLM client for OpenCode | `KEEP_PUBLIC` | Standard LLM client wrapper; not GOG-specific | N/A | LOW |
| `planner/intent_parser.py` | Intent parsing for mutation planning | `MOVE_SRM` | SRM-specific planning logic | N/A | MEDIUM |
| `planner/mutation_planner.py` | Mutation planning logic | `MOVE_SRM` | SRM-specific planning logic | N/A | MEDIUM |
| `planner/renderer_prompt.py` | Renderer prompt templates | `MOVE_SRM` | SRM-specific rendering | N/A | LOW |
| `planner/__init__.py` | Planner module init | `MOVE_SRM` | Follows planner contents | N/A | LOW |
| `salience_evaluator.py` | Evaluates node relevance | `MOVE_PRIVATE` | Advanced scoring heuristics | YES: Simple distance scoring for GOG-Lite | MEDIUM |
| `token_utils.py` | Token counting utilities | `KEEP_PUBLIC` | Generic utility, not commercial | N/A | LOW |

---

## 2. gog_cli/ — CLI & Context Serving

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cli.py` | Main CLI entry point | `MOVE_PRIVATE` | Production CLI surface | YES: Stub CLI with reduced commands | HIGH |
| `context_membrane.py` | Advanced scoring & pruning for context selection | `MOVE_PRIVATE` | Core commercial IP — scoring heuristics, penalties, distance calculations | YES: Simple max-files membrane for GOG-Lite | HIGH |
| `onboarding.py` | Repo onboarding, fingerprinting, `.gog/` artifact creation | `MOVE_PRIVATE` | Production onboarding logic, repo fingerprinting | YES: Minimal onboarding stub | HIGH |
| `serving.py` | Context serving to reasoners | `MOVE_PRIVATE` | Production context serving | YES: Simple context bundler for GOG-Lite | HIGH |
| `python_graph.py` | Python import graph extraction | `KEEP_PUBLIC` | Useful utility, not production-specific | N/A | LOW |
| `gold_context.py` | Gold-context metadata & scoring helpers | `KEEP_PUBLIC` | Benchmark credibility asset — essential for reproducibility | N/A | LOW |
| `failure_taxonomy.py` | Failure classification for benchmarks | `KEEP_PUBLIC` | Benchmark credibility asset — defines how failures are measured | N/A | LOW |
| `context_poisoning_benchmark.py` | Benchmark for context poisoning | `KEEP_PUBLIC_SANITIZE` | Benchmark infrastructure; review for any production-sensitive details | N/A | LOW |
| `executable_patch_benchmark.py` | Executable patch benchmark runner | `KEEP_PUBLIC` | Benchmark runner — essential for reproducibility | N/A | LOW |
| `reasoner_benchmark.py` | Reasoner benchmark utilities | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `semantic_plan_benchmark.py` | Semantic plan quality benchmark | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `__init__.py` | Module init | `KEEP_PUBLIC` | Basic exports | N/A | LOW |
| `__main__.py` | CLI module entry | `MOVE_PRIVATE` | Part of production CLI | YES: Simplified entry | MEDIUM |

---

## 3. gog/ — Benchmark Suite & Fixtures

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `benchmark_local_llm.py` | Local LLM benchmark runner | `KEEP_PUBLIC` | Benchmark infrastructure — core reproducibility | N/A | LOW |
| `benchmark_gauntlet_runner.py` | Gauntlet benchmark orchestrator | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `benchmark_public_vue.py` | Public Vue repo benchmark | `KEEP_PUBLIC` | Benchmark infrastructure — demonstrates GOG on real public repo | N/A | LOW |
| `benchmark_executable_patch.py` | Executable patch benchmark | `KEEP_PUBLIC` | Benchmark infrastructure — validates patches | N/A | LOW |
| `benchmark_context_poisoning.py` | Context poisoning benchmark | `KEEP_PUBLIC` | Benchmark infrastructure — tests context quality | N/A | LOW |
| `benchmark_reasoner_prompts.py` | Reasoner prompt comparison | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `benchmark_semantic_plan_quality.py` | Semantic plan quality benchmark | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `benchmark_cloud_api.py` | Cloud API benchmark | `KEEP_PUBLIC_SANITIZE` | Cloud-specific; review for credentials | N/A | MEDIUM |
| `benchmark_cloud_cli.py` | Cloud CLI benchmark | `KEEP_PUBLIC_SANITIZE` | Cloud-specific; review for credentials | N/A | MEDIUM |
| `benchmark_ui_full.py` | UI-full benchmark | `KEEP_PUBLIC` | Benchmark infrastructure | N/A | LOW |
| `benchmark_srm.py` | SRM-specific benchmark | `MOVE_SRM` | Benchmark for SRM research | N/A | LOW |
| `generate_dummy_repo.py` | Generates toy test repo | `KEEP_PUBLIC` | Useful for testing GOG-Lite | N/A | LOW |
| `seed_RAG_and_GOG.py` | Seeds RAG and GOG for benchmarks | `KEEP_PUBLIC` | Benchmark setup | N/A | LOW |
| `public-repos/` | Cloned public repos for benchmarks | `KEEP_PUBLIC` | Benchmark fixtures; keep small, don't vendor large repos | N/A | LOW |

---

## 4. sel/ — Symbolic Execution Layer (SRM Research)

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `core/` | SEL core modules | `MOVE_SRM` | SRM-specific reasoning machinery | N/A | MEDIUM |
| `expansion/` | Expansion engine | `MOVE_SRM` | SRM-specific expansion | N/A | MEDIUM |
| `experiments/` | SEL experiments | `MOVE_SRM` | SRM research artifacts | N/A | LOW |
| `tests/` | SEL tests | `MOVE_SRM` | Follows SEL modules | N/A | LOW |
| `data/` | SEL data | `MOVE_SRM` | SRM data | N/A | LOW |
| `README.md` | SEL documentation | `MOVE_SRM` | SRM-specific docs | N/A | LOW |
| `SEL.md` | SEL design doc | `MOVE_SRM` | SRM-specific design | N/A | LOW |
| `CLAUDE.md` | SEL agent guidance | `MOVE_SRM` | SRM-specific guidance | N/A | LOW |
| `taxonomy.json` | Failure taxonomy definitions | `KEEP_PUBLIC` | Generic benchmark taxonomy; useful for reproducibility | N/A | LOW |
| `templates.json` | Prompt templates | `MOVE_SRM` | SRM-specific templates | N/A | LOW |
| `EXAMPLE_INTERACES.py` | Example interfaces | `MOVE_SRM` | SRM-specific | N/A | LOW |
| `EXAMPLE_SCHEMA.json` | Example schemas | `MOVE_SRM` | SRM-specific | N/A | LOW |

---

## 5. srm_engine/ — SRM Planner

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `planner/` | SRM mutation planner | `MOVE_SRM` | SRM-specific planning | N/A | MEDIUM |
| All other files | SRM engine | `MOVE_SRM` | Moonshot research | N/A | MEDIUM |

---

## 6. symbol_distillation/ — Symbol Distillation Research

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `experiment_*.py` | Distillation experiments | `MOVE_SRM` | High-variance research | N/A | LOW |
| `semantic_primitives/` | Semantic primitives experiments | `MOVE_SRM` | SRM research | N/A | LOW |
| `results_*.json` / `.csv` | Experiment results | `MOVE_SRM` | Research data | N/A | LOW |
| `run_experiment.py` | Experiment runner | `MOVE_SRM` | Research runner | N/A | LOW |
| `analyze.py` / `visualize.py` | Analysis tools | `MOVE_SRM` | Research tools | N/A | LOW |
| `problems.py` | Problem definitions | `MOVE_SRM` | Research problems | N/A | LOW |

---

## 7. distilled_datasets/ — Research Datasets

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `exp_*.py` | Dataset generation scripts | `MOVE_SRM` | Research data generation | N/A | LOW |
| All other files | Dataset artifacts | `MOVE_SRM` | Research data | N/A | LOW |

---

## 8. docs/ — Documentation

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GOG_SYSTEM_MODEL.md` | GOG system model | `KEEP_PUBLIC` | Architecture documentation — research credibility | N/A | LOW |
| `REASONER_INTERFACE.md` | Reasoner interface spec | `KEEP_PUBLIC` | Interface spec — useful for integration | N/A | LOW |
| `REPO_ONBOARDING_PIPELINE.md` | Onboarding pipeline doc | `KEEP_PUBLIC_SANITIZE` | Review for production-specific details | N/A | MEDIUM |
| `SRM_PHASE_*.md` | SRM phase results | `MOVE_SRM` | SRM-specific research | N/A | LOW |
| `REPO_SEPARATION_AUDIT.md` | Separation audit | `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | Internal planning doc | N/A | LOW |
| `REPO_SEPARATION_MANIFEST.md` | This file | `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | Internal planning doc | N/A | LOW |

---

## 9. Root-Level Files

| Path | Purpose | Classification | Reason | Replacement Needed | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `README.md` | Main documentation | `KEEP_PUBLIC` (replace) | Update to clarify "GOG-Lite" positioning | YES: New README with GOG-Lite framing | LOW |
| `GOG_PAPER.pdf` | Published paper | `KEEP_PUBLIC` | Research credibility | N/A | LOW |
| `SRM_PAPER.md` | SRM paper draft | `MOVE_SRM` | Moonshot research | N/A | LOW |
| `GOG_architecture.md` | Architecture overview | `KEEP_PUBLIC_SANITIZE` | Review for production-specific details | N/A | MEDIUM |
| `requirements.txt` | Python dependencies | `KEEP_PUBLIC` | Standard dependencies | N/A | LOW |
| `AGENTS.md` | Agent guidance | `KEEP_PUBLIC` | Useful for contributors | N/A | LOW |
| `CONTRIBUTING.md` | Contributing guide | `KEEP_PUBLIC` | Community docs | N/A | LOW |
| `CHANGELOG.md` | Changelog | `KEEP_PUBLIC` | Historical record | N/A | LOW |
| `AI_COLLABORATION.md` | AI collaboration notes | `REVIEW_REQUIRED` | May contain sensitive notes | N/A | MEDIUM |
| `target_repo/` | Dummy test repo | `KEEP_PUBLIC` | Required for unit tests | N/A | LOW |
| `results/` | Benchmark results | `KEEP_PUBLIC` | Sanitized result summaries | N/A | LOW |
| `.gog/` | GOG artifacts | `KEEP_PUBLIC` | Can be regenerated | N/A | LOW |
| `vector_db/` | Vector DB artifacts | `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | Generated, not source | N/A | LOW |
| `gog_graph.pkl` | Pickled graph | `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | Generated, not source | N/A | LOW |

---

## 10. README Updates Required

### Current README
The current `README.md` presents the repo as the "main" GOG implementation with production CLI commands.

### Proposed README Update
Replace with:

```markdown
## Research and Production Boundary

This repository contains **GOG-Lite**, the public reference implementation and benchmark lab for Graph-Oriented Generation.

The goal of this repo is to make the core GOG thesis testable: codebases are structured symbolic systems, and coding agents can benefit from graph-routed context instead of flat retrieved text.

- **GOG-Lite** (`gog_engine/`, simplified `gog_cli/`) provides a reference implementation demonstrating the core thesis.
- **Benchmarks** (`gog/`) allow reproducibility of results against public repositories.
- **Research artifacts** document the methodology and failure analysis.

A separate private implementation (GOG Professional) is used for production-grade onboarding, advanced context membranes, large-scale graph fusion, and enterprise integrations.

The long-term Symbolic Reasoning Model (SRM) research direction is maintained separately to keep this repository focused on GOG as a practical context layer for today's coding assistants.
```

---

## 11. Replacement Specs for GOG-Lite

### gog_engine/lite_ast_parser.py (NEW)
```python
"""GOG-Lite: Simple glob-based import scanner."""

def build_graph(repo_root: Path) -> nx.DiGraph:
    """Build a basic import graph using simple regex scanning."""
    # Simple .js/.ts/.vue file discovery
    # Basic import regex (no alias handling, no tree-sitter)
    # Basic dependency edges
    pass
```

### gog_engine/lite_graph_search.py (NEW)
```python
"""GOG-Lite: Simple deterministic graph traversal."""

def isolate_context(graph, prompt) -> list[str]:
    """Simple distance-bounded expansion from keyword seeds."""
    # Extract filename mentions
    # Simple BFS up to depth=2
    # No semantic embeddings in base GOG-Lite
    pass
```

### gog_cli/lite_membrane.py (NEW)
```python
"""GOG-Lite: Simple max-files context membrane."""

def apply_membrane(graph, seed_nodes, max_files=10) -> list[str]:
    """Simply return top-k files by distance from seeds."""
    pass
```

### gog_cli/lite_serving.py (NEW)
```python
"""GOG-Lite: Simple context bundler."""

def serve_context(repo_root, prompt) -> dict:
    """Return a simple context bundle without advanced scoring."""
    pass
```

---

## 12. Summary Counts

| Classification | Count |
| :--- | :---: |
| `KEEP_PUBLIC` | ~25 |
| `KEEP_PUBLIC_SANITIZE` | ~5 |
| `MOVE_PRIVATE` | ~12 |
| `MOVE_SRM` | ~30 |
| `DELETE_FROM_PUBLIC_AFTER_MIGRATION` | ~5 |
| `REVIEW_REQUIRED` | ~1 |

---

## 13. Open Questions for Derek

1. **AI_COLLABORATION.md** — Contains internal collaboration notes. Should this be moved private or deleted?
2. **vector_db/** and **gog_graph.pkl** — These are generated artifacts. Is it safe to delete them from the public repo, or do they serve as useful examples?
3. **Cloud benchmark files** (`benchmark_cloud_*.py`) — Are these safe to keep public, or do they expose commercial deployment details?
4. **Public target repos** — Should we keep `public-repos/httpx` and `public-repos/vue3-realworld-example-app` or clone them on-demand via script?
5. **Results artifacts** — Should we keep full JSON results or only markdown summaries?