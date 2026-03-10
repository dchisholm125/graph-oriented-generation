# CLAUDE.md — Graph-Oriented Generation (GOG) / Symbolic Reasoning Model (SRM)

> Instructions for Claude Code operating on this repository.
> Read this entire file before making any changes. No exceptions.

---

## What This Project Is

This repository has two layers of purpose that must be held simultaneously:

**Layer 1 — GOG (Graph-Oriented Generation):** A deterministic alternative to Vector RAG for code context isolation. Proved. Benchmarked. Paper written. GOG replaces probabilistic cosine similarity retrieval with AST-parsed DAG traversal. The benchmark results are real and reproducible.

**Layer 2 — SRM (Symbolic Reasoning Model):** The larger thesis GOG exists to support. The claim is that LLMs are misused when asked to reason about structure. All reasoning should be offloaded to deterministic symbolic systems. The LLM becomes a language renderer only — it translates symbolic specifications into syntax. It does not plan, infer, or decide.

**The current codebase proves Layer 1. The active development goal is to prove Layer 2.**

Do not conflate these. Do not add features that optimize GOG vs RAG head-to-head unless explicitly asked. The research priority is building the SRM pipeline: Intent Parser → Mutation Planner → Language Renderer.

---

## Repository Structure

```
graph-oriented-generation/
├── srm_engine/                  # Core symbolic engine — handle with care
│   ├── ast_parser.py            # Builds NetworkX DiGraph from TS/Vue imports
│   ├── ts_parser.py             # tree-sitter AST parser for TypeScript/Vue
│   ├── graph_search.py          # Semantic seeding + deterministic DAG traversal
│   ├── salience_evaluator.py    # Neuro-Symbolic Membrane: extract/validate/patch imports
│   ├── token_utils.py           # tiktoken-based token counting (cl100k_base)
│   └── opencode_client.py       # Subprocess connector to opencode CLI
├── srm_engine/planner/          # NEW — SRM Phase 2 (Intent Parser + Mutation Planner)
│   ├── intent_parser.py         # Parses natural language → structured OperationSpec
│   ├── mutation_planner.py      # OperationSpec + GOG graph → MutationPlan
│   └── renderer_prompt.py       # MutationPlan → LLM renderer prompt
├── benchmark_local_llm.py       # 3-tier local benchmark (Ollama)
├── benchmark_cloud_cli.py       # 3-tier cloud benchmark (opencode CLI)
├── generate_dummy_repo.py       # Generates the 100+ file target Vue/TS maze
├── seed_RAG_and_GOG.py          # Builds ChromaDB index + serializes GOG graph
├── CHANGELOG.md                 # Updated on every commit — no exceptions
├── CLAUDE.md                    # This file
└── README.md                    # Scientific, not sales — keep it that way
```

---

## The SRM Pipeline (Phase 2 Target Architecture)

This is what you are building toward. Internalize this before touching any code.

```
Natural Language Prompt
        │
        ▼
┌─────────────────────┐
│   Intent Parser     │  Rule-based + pattern matching. No LLM.
│  (intent_parser.py) │  Output: OperationSpec (structured dataclass)
└─────────────────────┘
        │
        ▼  OperationSpec
┌─────────────────────┐
│  Mutation Planner   │  Pure graph operations. No LLM.
│ (mutation_planner.py│  Resolves target node, validates against GOG graph.
│                     │  Output: MutationPlan (structured dataclass)
└─────────────────────┘
        │
        ▼  MutationPlan
┌─────────────────────┐
│  Renderer Prompt    │  Assembles a precise symbolic spec for the LLM.
│(renderer_prompt.py) │  The LLM receives ONLY this — not the raw prompt.
└─────────────────────┘
        │
        ▼  Renderer Prompt
┌─────────────────────┐
│   LLM (any size)    │  Translation only. No reasoning. No architecture.
│                     │  Input: symbolic spec. Output: valid syntax.
└─────────────────────┘
```

**The LLM must never see the original natural language prompt in the SRM pipeline.** It receives only the renderer prompt produced by the Mutation Planner. This is the architectural constraint that makes SRM falsifiable.

---

## Operation Types (Phase 2 — Easy Scope)

Start with these two. Do not expand scope until both are proven.

### `ADD_FIELD`
Add a new field to a Pinia store's state object.

```python
@dataclass
class AddFieldOperation:
    operation: Literal["ADD_FIELD"]
    target_file: str        # e.g. "src/stores/authStore.ts"
    target_node: str        # e.g. "state"
    field_name: str         # e.g. "lastLogin"
    field_type: str         # e.g. "string"
    default_value: str      # e.g. "''"
```

### `MUTATE_ACTION`
Add or modify a statement inside a Pinia store action.

```python
@dataclass
class MutateActionOperation:
    operation: Literal["MUTATE_ACTION"]
    target_file: str        # e.g. "src/stores/authStore.ts"
    target_action: str      # e.g. "login"
    add_statement: str      # e.g. "this.lastLogin = '2026-03-08'"
```

Both together form a `MutationPlan` — an ordered list of operations that fully specifies what needs to change before the LLM is invoked.

---

## Renderer Prompt Contract

The renderer prompt handed to the LLM must follow this contract exactly:

```
SYMBOLIC SPECIFICATION — DO NOT DEVIATE

File: {target_file}
Current content provided below.

Operations to apply:
1. ADD_FIELD to state object:
   - name: {field_name}
   - type: {field_type}
   - default: {default_value}

2. MUTATE_ACTION '{target_action}':
   - add statement: {add_statement}

Render the complete updated file as valid TypeScript.
Use Pinia defineStore syntax.
Do not add imports that are not already present.
Do not add fields or actions beyond those specified above.
```

This prompt is a spec, not a request. The LLM is not being asked to think. It is being asked to apply a diff.

---

## Coding Rules

### What Claude Code must always do

**1. Update CHANGELOG.md on every commit.**
Every change — no matter how small — gets a dated entry. Format:
```
## YYYY-MM-DD — <short title>
- <what changed and why, one line per change>
- <if a design decision was made, note the alternative considered and why it was rejected>
```

**2. Documents-first on any new module.**
Before writing implementation code for a new file, write the module docstring first and get it right. The docstring is the contract. Code is the execution of that contract.

**3. Dataclasses for all structured data.**
`OperationSpec`, `MutationPlan`, and any intermediate structures must be Python dataclasses with type annotations. No dicts passing between components. Dicts are for serialization only.

**4. No LLM calls inside `srm_engine/planner/`.**
The planner directory is a no-LLM zone. If you find yourself reaching for an LLM to parse intent or plan a mutation, stop. Use pattern matching, regex, or rule-based logic. An LLM helping the planner plan is the exact anti-pattern SRM exists to replace.

**5. Keep the 3-tier benchmark intact.**
`benchmark_local_llm.py` and `benchmark_cloud_cli.py` are the ground truth for GOG results. Do not modify them unless the task explicitly requires it. SRM Phase 2 gets its own benchmark script: `benchmark_srm.py`.

**6. Preserve the srm_engine/ public API.**
`graph_search.isolate_context(G, prompt)` and `SalienceEvaluator` are the stable interfaces. Do not change their signatures. 7 forks depend on them.

### What Claude Code must never do

- **Never** call an LLM to perform reasoning that belongs in the symbolic layer.
- **Never** modify `generate_dummy_repo.py` or `seed_RAG_and_GOG.py` without an explicit instruction to do so. These are infrastructure, not research code.
- **Never** add dependencies without checking `requirements.txt` first and noting the addition in CHANGELOG.
- **Never** remove the correctness rubric from the benchmark or weaken its criteria to make scores look better.
- **Never** commit without updating CHANGELOG.md in the same commit.
- **Never** describe the benchmark results in superlatives. Report numbers. Let numbers speak.

---

## Current State Summary (as of 2026-03-10)

### What is proven
- GOG reduces token context by 88-92% vs RAG on a 100+ file Vue/TypeScript repo
- Deterministic import patching (Membrane) catches architectural hallucinations with zero additional LLM calls
- Results hold on both cloud CLI (opencode) and local GPU (llama3:8b, 6GB VRAM)

### What is not yet proven
- That SRM's symbolic mutation planner produces correct output when fed to a small (≤1B) LLM
- That the LLM-as-renderer pattern produces better correctness scores than LLM-as-reasoner
- That the approach generalizes beyond Pinia/Vue/TypeScript

### Known limitations (be honest about these)
- Semantic seeder produces false positives on keyword-overlapping files (Medium task: `mockLogoutHandler.ts`)
- Token reduction manifests as API cost savings, not local wall-clock savings on CPU inference
- Benchmark repository is procedurally generated — external validity not yet established
- llama3:8b defaults to Vuex patterns over Pinia even when given Pinia context — parametric knowledge dominates over provided context, which is the exact motivation for SRM

### Active branches
- `main` — stable, benchmarks reproducible, README accurate
- SRM Phase 2 work goes on `feature/srm-planner` — do not merge to main until `benchmark_srm.py` produces results

---

## On Tone and Communication

This project is headed toward arXiv. Every string that appears in output — benchmark panels, README, comments, docstrings — should read like a scientist wrote it, not a marketer.

Specifically:
- No "paradigm shift", "revolutionary", "groundbreaking", "mathematically proves"
- Report what the experiment showed, not what you wish it showed
- If a result is ambiguous, say so
- If a limitation exists, name it before a reviewer does

The research is strong enough. It does not need adjectives.

---

## Commit Message Format

```
<type>: <short description>

- <what changed>
- <why it changed>
- <any design decision worth noting>

CHANGELOG: updated
```

Types: `feat`, `fix`, `refactor`, `bench`, `docs`, `chore`

Example:
```
feat: add IntentParser for ADD_FIELD and MUTATE_ACTION operations

- Implements rule-based pattern matching for Easy-level prompts
- Uses dataclasses throughout — no dict passing between components
- Rejects LLM-based parsing by design (see CLAUDE.md §No LLM in planner)

CHANGELOG: updated
```