# CHANGELOG

All changes to this repository are documented here.
Format: dated entries, one line per change, design decisions noted where relevant.

---

## 2026-03-10 — SRM renderer: surgical content stripping (content poisoning fix)

- Fixed critical issue: renderer prompt was passing entire raw file content to LLM
  Problem: authStore.ts is 14KB with 96.6% noise (DUMMY_ASSETS base64 blobs + boilerplate comments)
  LLM was overwhelmed, lost in noise, echoed back garbage instead of following spec
- Implemented _extract_store_skeleton(): deterministic content sanitizer
  Strips DUMMY_ASSETS blocks (arbitrary base64 image data)
  Strips random boilerplate comment blocks (/** ... */ with 80+ char garbage lines)
  Preserves only imports + defineStore definition + actions
  Reduces authStore.ts from 14,388 chars (126 lines) to 491 chars (10 lines) — 96.6% reduction
- Updated build_renderer_prompt() to sanitize before inserting into spec
  LLM now sees clean symbolic spec + meaningful code, not raw file dump
  Still deterministic — planner's job: present clean target, not raw pollution
- Verification: no DUMMY_ASSETS in renderer prompt, no boilerplate noise
  Actual code (defineStore, imports, actions) fully preserved for context

## 2026-03-10 — SRM Phase 2 implementation: Intent Parser + Mutation Planner + Renderer

- Created `srm_engine/planner/` as a no-LLM zone (by policy, not configuration)
- Implemented `intent_parser.py`: Rule-based pattern matching for ADD_FIELD + MUTATE_ACTION operations
  - Parses natural language → structured OperationSpec dataclasses (no LLM, no exceptions unless pattern fails)
  - Regex patterns extract file paths, field names, types, action names, and values
  - Rejected: LLM-based parsing (violates SRM architecture)
- Implemented `mutation_planner.py`: Deterministic graph resolution and file I/O
  - Validates operations reference same target file
  - Resolves relative paths (e.g. `src/stores/authStore.ts`) to absolute paths via graph nodes
  - Reads file content at plan-time; raises PlannerError if target not in graph
  - Returns MutationPlan: immutable spec with all info needed by renderer
- Implemented `renderer_prompt.py`: Symbolic spec builder
  - Converts MutationPlan → human-readable symbolic specification (SYMBOLIC SPECIFICATION — DO NOT DEVIATE format)
  - LLM receives this spec + MUZZLE, NOT the original natural language prompt
  - Makes SRM falsifiable: if symbolic constraints don't improve small model correctness, hypothesis is rejected
- Implemented `benchmark_srm.py`: 3-tier benchmark with SRM pipeline
  - Tier 1 (Control): RAG + raw prompt
  - Tier 2 (Baseline): GOG + raw prompt
  - Tier 3-SRM (Hypothesis): Symbolic spec only (parser + planner + renderer prompt)
  - Runs all three tiers on Easy task, reports tokens/timing/correctness side-by-side
  - Prints renderer prompt to console for verification before LLM call
- Design decision: Dataclasses for all structured data (AddFieldOperation, MutateActionOperation, MutationPlan)
  - Rationale: Type safety, immutability, no dict-passing between components (enforces contracts)
- Design decision: Parser rejects unknown patterns rather than guessing
  - Rationale: False positives in planner are worse than false negatives (user sees error message, not silent wrong behavior)
- Design decision: MUZZLE appended to renderer prompt, not separate system directive
  - Rationale: LLM weight recency highly; placing MUZZLE last increases compliance

## 2026-03-10 — SRM Phase 2 groundwork + benchmark hardening

- Added `CLAUDE.md` with operating instructions for Claude Code on this repo
- Defined `ADD_FIELD` and `MUTATE_ACTION` operation types as the Phase 2 scope boundary
- Documented SRM pipeline architecture: Intent Parser → Mutation Planner → Language Renderer
- Established `srm_engine/planner/` as a no-LLM zone by policy
- Defined renderer prompt contract: LLM receives symbolic spec only, never raw prompt

## 2026-03-10 — GPU support + context-aware correctness rubric

- Removed hardcoded `num_gpu: 0` from `benchmark_local_llm.py` — Ollama now auto-detects GPU
- Added `NUM_GPU=0` environment variable override for CPU-only fallback
- Hard task rubric now context-aware: Tier 3 scored against its actual isolated file count
  (when GOG isolates 1 file, Tier 3 cannot be expected to produce a 3-file answer)
- Fixed forbidden import check to use `re.search` instead of plain string `in`
- Initialized `isolated_files = []` before conditional block to prevent `NameError`
- Added GPU benchmark results to README (llama3:8b, 6GB VRAM): 91.6% token reduction on Hard
- Added CPU vs GPU timing explanation to README Known Limitations

## 2026-03-10 — Correctness rubric + warmup call

- Added `score_response()` deterministic rubric to `benchmark_local_llm.py`
- Rubric uses string matching only — no second LLM call, no semantic judgement
- Easy criteria: `lastLogin` field, `2026-03-08` value, `defineStore`, `login` action, no React imports
- Medium criteria: Logout button, `@click` wiring, `useAuthStore` reference, Vue template/script
- Hard criteria: `deleteAccount`, `/delete` endpoint, `deleteUser`, Delete button, no direct api_client import in Vue
- Added `warmup_model()` call before gauntlet — prevents Tier 1 Easy timing being inflated by Ollama first-load cost
- Added `level` parameter to `print_results()` to enable per-task rubric evaluation

## 2026-03-10 — Research Roadmap + model recommendation

- Added Research Roadmap section to README formalizing 3 tracks: GOG (done), Membrane (in progress), SRM (future)
- Explicitly stated that current benchmark tests retrieval layer, not reasoning layer
- Added model size guidance to README: qwen2.5:7b recommended minimum for local benchmark
- Noted that 0.5b results illustrate the SRM motivation, not GOG failure

## 2026-03-10 — README scientific rewrite

- Removed social proof opener (stars, HN/Reddit mentions)
- Removed: "mathematically wrong", "broken", "Context Window Crisis", "hallucination-free", "zero false positives by construction"
- Removed rejection sampling loop description (architecture no longer exists in code)
- Added Microsoft GraphRAG citation (Edge et al., 2024) with precise one-paragraph differentiation
- Added Known Limitations section: semantic seeder false positives, indirect prompt degradation, self-contained benchmark caveat, tiktoken proxy caveat
- Rewrote Membrane section to accurately describe `patch()` instead of retry loop
- Updated mermaid diagram to reflect actual current architecture
- Tone: no exclamation points, no superlatives, numbers reported as numbers

## 2026-03-10 — Medium display bug fix + benchmark correctness

- Fixed Tier 3 display to always show `raw_response` — `extracted_code` was stripping fence markers and partial Vue blocks
- Added fence language detection (`vue` vs `ts`) when re-wrapping patched code
- Applied same fix to both `benchmark_cloud_cli.py` and `benchmark_local_llm.py`

## 2026-03-09 — Deterministic patch architecture (replaces rejection sampling)

- Replaced rejection sampling loop in `SalienceEvaluator` with single-call `patch()` method
- Graph now corrects LLM output deterministically — zero additional LLM calls, zero extra tokens
- `EvaluationResult` dataclass: added `violations: List[str]` field
- `patch()`: resolves correct absolute path via basename matching against `allowed_nodes`
- Uncommented hallucinated imports with `// [SRM PATCH: hallucinated import removed]` annotation
- Benchmark metric renamed: "Rejection Attempts" → "Topological Patches (Membrane)"

## 2026-03-09 — Semantic seeding (replaces keyword matching)

- Replaced three hardcoded `if/elif` keyword blocks in `graph_search.py` with `sentence-transformers` cosine similarity seeding
- Model: `all-MiniLM-L6-v2`; threshold: `SEED_SIMILARITY_THRESHOLD = 0.25`; max seeds: `MAX_SEEDS = 5`
- Added `_node_to_label()`: splits camelCase filenames before embedding (`authStore.ts` → `"auth Store"`)
- Added `build_node_embeddings()`: pre-computes embeddings once per graph to avoid re-embedding on every call
- Added lazy model loading via `_get_model()`

## 2026-03-09 — Token counting accuracy

- Replaced whitespace word-split `get_token_count()` with `tiktoken` `cl100k_base` encoding
- New module: `srm_engine/token_utils.py`
- `count_tokens_in_files(file_paths)` replaces old `get_token_count` in both benchmark files
- `count_tokens_in_string(text)` for prompt/response counting
- Added `tiktoken` and `sentence-transformers` to `requirements.txt`
- Note: old word-split undercounted code tokens by ~30-40%; all published numbers use tiktoken

## 2026-03-09 — stderr noise suppression

- Added `_STDERR_NOISE_PREFIXES` tuple to `opencode_client.py`
- Filters TF/CUDA/NumPy warnings from opencode subprocess stderr
- Real errors still surface; cosmetic noise suppressed