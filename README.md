# Graph-Oriented Generation (GOG)
### Persistent Symbolic State for Software Generation

> **Status:** GOG is evolving from a benchmark prototype into a repo-resident system model for graph-native software agents.

---

## The Core Thesis

GOG investigates whether software generation improves when codebases are first transformed into persistent symbolic state, that state is served to a higher-level reasoner, and language models are demoted from architects to renderers.

GOG is **not** the reasoner itself. It is the symbolic repository substrate and presentation layer that should make a codebase easier for a reasoner to comprehend, manipulate, and validate.

The project is now organized around three ideas:

1. **Graphize the repository first.** A GOG-aware assistant should onboard a repo into durable symbolic state before acting like a coding agent.
2. **Serve the reasoner better.** Today that reasoner may be a deterministic planner or a large language model constrained to structured output. The long-range hypothesis is a true symbolic reasoning model, but GOG's near-term role is to present repo state more clearly than ad hoc text retrieval.
3. **Use language models for rendering, not architectural invention.** Renderers turn typed plans into code or prose; validators check the result against repo state and local tooling.

---

## Architecture

GOG separates the software-generation loop into persistent system layers:

1. **Repository Onboarding**
   - Parse the repo, build symbolic state, record conventions, and persist artifacts for later sessions.
2. **Reasoner Slot**
   - Consume user intent plus GOG-served symbolic repo state and emit a typed plan such as a `MutationPlan`.
3. **Renderer**
   - Convert bounded plan steps into code, prose, or other user-facing outputs.
4. **Validators**
   - Check graph boundaries, syntax, constraints, and eventually build/test outcomes.
5. **Graph Synchronizer**
   - Refresh symbolic state after accepted changes so GOG stays aligned with the living repo.

See:

- [GOG system model](./docs/GOG_SYSTEM_MODEL.md)
- [Reasoner interface](./docs/REASONER_INTERFACE.md)
- [Repository onboarding pipeline](./docs/REPO_ONBOARDING_PIPELINE.md)

---

## Repository Roadmap

### Phase 1: Context Isolation
Completed early experiments on AST-derived context selection and benchmark instrumentation.
- [GOG_PAPER.pdf](./GOG_PAPER.pdf)
- Benchmark: `python3 gog/benchmark_local_llm.py`

### Phase 2: Symbolic Rendering Proof-of-Concept
The **Symbolic Reasoning Membrane (SRM)** experiments explore whether structure can control language-model output more reliably than raw prompting alone.
- [SRM_PAPER.md](./SRM_PAPER.md)
- Experiments: `/symbol_distillation`

### Phase 3: Repo-Resident GOG Engine
Active work now centers on persistent symbolic state, a formal reasoner contract, onboarding, graph refresh, reasoner-quality experiments, and public-repo benchmarks.
- [Architecture overview](./GOG_architecture.md)
- [System model](./docs/GOG_SYSTEM_MODEL.md)
- [Onboarding pipeline](./docs/REPO_ONBOARDING_PIPELINE.md)

---

## Quick Start

### 1. Build the Current Research Artifacts
```bash
pip install -r requirements.txt
python3 gog/generate_dummy_repo.py
python3 gog/seed_RAG_and_GOG.py
```

### 2. Onboard a Repository with the GOG CLI
```bash
python3 -m gog_cli onboard /path/to/repo
python3 -m gog_cli refresh /path/to/repo
python3 -m gog_cli inspect /path/to/repo
python3 -m gog_cli summarize /path/to/repo
python3 -m gog_cli context /path/to/repo --prompt "refactor the addUser function"
```

The first CLI slice persists a `.gog/` artifact set with a repo profile, structural Python/TS/Vue import graph, validation-command hints, and a freshness fingerprint. `refresh` rebuilds `.gog/` after accepted repo changes. `summarize` emits repo-level orientation. `context` emits a bounded prompt-scoped bundle for a downstream reasoner rather than serving the entire repo graph every cycle.
Context serving includes an always-on ContextMembrane that trims weak graph-neighborhood files before the reasoner handoff while preserving an audit trail of kept and rejected files.

### 3. Run the Current Benchmark Harness
The current benchmark harness remains useful for experiments on retrieval, planning, rendering, and validation behavior. The next benchmark generation should emphasize **cost to validated success**, not only first-attempt token totals.
```bash
python3 gog/benchmark_local_llm.py
python3 gog/benchmark_reasoner_prompts.py . --model kimi-k2.6:cloud
python3 gog/benchmark_reasoner_prompts.py . --model kimi-k2.6:cloud --task python_context_serving
python3 gog/benchmark_semantic_plan_quality.py . --model kimi-k2.6:cloud
python3 gog/benchmark_semantic_plan_quality.py . --model kimi-k2.6:cloud --strategy minimal_platter
python3 gog/benchmark_semantic_plan_quality.py . --model kimi-k2.6:cloud --strategy minimal_platter --strategy traditional_rag
python3 gog/benchmark_public_vue.py
python3 gog/benchmark_executable_patch.py --dry-run
python3 gog/benchmark_executable_patch.py --model kimi-k2.6:cloud --task debug_query_serialization_easy --attempts 1 --retries 2
python3 gog/benchmark_executable_patch.py --model kimi-k2.6:cloud --attempts 1 --retries 2
```

The executable patch benchmark validates patches inside disposable copies of the public Vue checkout. Before running it without `--dry-run`, install dependencies in the public repo checkout:

```bash
cd gog/public-repos/vue3-realworld-example-app
corepack pnpm install
```

`benchmark_reasoner_prompts.py` compares multiple GOG-to-reasoner prompt formats against the same onboarded context bundles and grades whether the model returns valid, grounded `MutationPlan` JSON.
Use repeated `--task` or `--strategy` flags to run focused pilots before spending on the full matrix.

`benchmark_semantic_plan_quality.py` moves one step deeper. It uses a documented known-problem fixture where the GOG-served context identifies `gog_cli/serving.py` but omits source excerpts and symbol summaries. A high-quality GOG plan should recognize that limitation, inspect before mutating, stay inside `allowed_nodes`, and emit appropriate validation. The same benchmark includes a `traditional_rag` baseline that serves keyword-retrieved unstructured source chunks without graph relations.
Both benchmark entry points support retry controls for transient Ollama/provider failures: `--retries` and `--retry-delay-s`.

`benchmark_public_vue.py` starts the public-repo track. It clones or reuses a Vue repository, runs GOG onboarding, and compares GOG context selection against a reproducible `traditional_rag` keyword-chunk baseline before any LLM mutation is attempted.

`benchmark_executable_patch.py` is the next layer. It copies the public Vue checkout into a disposable temp directory, injects benchmark-only acceptance tests or defects, asks the same model to produce full-file JSON patches from either GOG context or `traditional_rag` context, applies admissible patches, runs targeted validation commands, and records `Pass@1`, `Pass@k`, `TokensToPass`, `AttemptsToPass`, and `WallClockToPass`. It uses the local Ollama HTTP API with JSON mode to avoid CLI prompt-expansion artifacts when source chunks contain strings like `@example`.

Benchmark repositories should be treated honestly. Current public Vue work is a development benchmark for Vue/Vite/TypeScript conventions. Future holdout repositories should be run before tuning so GOG improvements remain general-purpose rather than fitted to one codebase.

### 4. Read the System Docs

Start with:

1. [GOG architecture](./GOG_architecture.md)
2. [GOG system model](./docs/GOG_SYSTEM_MODEL.md)
3. [Reasoner interface](./docs/REASONER_INTERFACE.md)
4. [Repository onboarding pipeline](./docs/REPO_ONBOARDING_PIPELINE.md)

---

## 📖 Citation
```bibtex
@misc{chisholm2026gog,
  author = {Chisholm, D. R.},
  title  = {Graph-Oriented Generation (GOG): Persistent Symbolic State for Software Generation},
  year   = {2026},
  url    = {https://github.com/dchisholm125/graph-oriented-generation}
}
```

---

*The codebase becomes symbolic state. Reasoning becomes explicit. Rendering becomes bounded.*
