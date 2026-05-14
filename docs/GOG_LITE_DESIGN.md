# GOG-Lite Design Document

**GOG-Lite** is the public reference implementation for Graph-Oriented Generation (GOG). It is intentionally simple, transparent, and self-contained — no production heuristics, no proprietary scoring, and no external embeddings.

---

## Philosophy

The GOG thesis is that **graph-routed context can compete with or exceed flat retrieved context** for coding tasks. GOG-Lite proves this with the smallest possible surface area:

1. **Regex-based import graphs** — no AST parser, no tree-sitter.
2. **Keyword seeding + bounded BFS** — no semantic embeddings.
3. **Simple integer scoring** — all weights are small, documented constants.
4. **Per-file token caps** — large files are truncated or snippetized.
5. **Soft total token budget** — keeps prompt sizes reasonable.
6. **Keyword snippets** — instead of dumping entire large files.

---

## Architecture

```
repo/
├── gog_engine_lite/
│   ├── import_graph.py   ← regex-based import graph builder
│   └── graph_search.py   ← keyword seeding, BFS, scoring, snippets
├── gog_cli/
│   ├── lite_membrane.py  ← token-budgeted file cap
│   └── lite_serving.py   ← bundle builder (graph → file contents)
└── tests/
    └── test_gog_lite.py  ← 13 deterministic tests
```

---

## Scoring Rules (all public constants)

| Rule | Weight | Explanation |
|------|--------|-------------|
| Exact filename match | `100` | File name exactly matches a prompt keyword |
| Filename keyword match | `30` | File name contains a prompt keyword |
| Content keyword match | `20` per hit, capped at `60` | Prompt keyword appears in file content |
| Test file bonus | `15` | Files matching `*.spec.*`, `*.test.*`, or `/test/` |
| Direct import of seed | `10` | File is directly imported by a seed node |
| Distance penalty | `-5` per hop | Further from seed = lower score |
| Large file penalty | `-15` | Files > `80_000` bytes lose points |

All constants live at the top of `gog_engine_lite/graph_search.py` and are documented inline.

---

## Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_depth` | `1` | Most relevant files are 1 import hop away. Auto-expands to 2 if too few results. |
| `max_files` | `12` | Aggressive cap to keep noise low. |
| `max_tokens_per_file` | `1200` | Prevents giant files from bloating the prompt. |
| `total_token_budget` | `6000` | Soft cap across the entire context bundle. |

---

## What GOG-Lite Does NOT Do

- **No AST parsing** — regex only. Fast, but misses dynamic imports.
- **No semantic embeddings** — pure keyword matching.
- **No advanced graph ranking** — no PageRank, no HITS, no spectral methods.
- **No proprietary heuristics** — every decision is a documented constant.
- **No tuning to one task** — rules are generic; no special cases.

---

## Relationship to Production GOG

| | GOG-Lite | Production GOG |
|---|---|---|
| Graph building | Regex | Tree-sitter + custom parsers |
| Scoring | Integer constants | Multi-factor ranking + learned weights |
| Membrane | Distance + token cap | ContextMembrane with multi-objective optimization |
| Snippets | Keyword radius | Semantic chunking |
| Target | Public reference, reproducibility | Maximum benchmark performance |

GOG-Lite is the **control group**. If GOG-Lite already beats RAG, the production system is icing on the cake.

---

## Running Benchmarks

```bash
# Dry run (build context, no LLM calls)
python -m gog_cli.executable_patch_benchmark \
  --mode gog_lite \
  --dry-run \
  --task debug_query_serialization_easy

# Full context poisoning benchmark
python -m gog_cli.context_poisoning_benchmark \
  --dry-run \
  --task debug_query_serialization_easy
```

---

## Extending GOG-Lite

All tunables are module-level constants. To experiment:

1. Edit `gog_engine_lite/graph_search.py` constants.
2. Re-run `pytest tests/test_gog_lite.py`.
3. Re-run the dry-run benchmark.
4. Compare precision, recall, noise, and tokens.

No hidden config files, no onboarding artifacts, no magic.
