# GOG System Model

## 1. Purpose

Graph-Oriented Generation (GOG) investigates whether software generation improves when codebases are first transformed into persistent symbolic state, that state is served to a higher-level reasoner, and language models are demoted from architects to renderers.

This document defines GOG as a system model rather than a single benchmark technique.

GOG is not "RAG, but with graphs." It is also not the reasoner itself. It is a repo-resident symbolic substrate and presentation layer for software agents:

1. A repository is onboarded into symbolic state before agentic work begins.
2. A reasoner receives that state in a more structured form than raw repo text.
3. The reasoner emits typed plans and constraints.
4. A language renderer turns those plans into code, prose, or other human-facing artifacts.
5. Validators check the result against the symbolic state and the local toolchain.
6. Accepted mutations update the symbolic state so GOG remains synchronized with the evolving repo.

---

## 2. Core Thesis

The central thesis is:

> Software generation should separate architectural reasoning from language rendering.

GOG itself is not that architectural reasoner. GOG's role is to organize and serve repository state so a reasoner can make better decisions with less guessing about the codebase.

In current GOG research, a large language model may temporarily occupy the reasoning slot. That is a proxy, not the end state. The long-range hypothesis is that a true symbolic reasoning model can eventually replace that proxy while preserving the same external contract.

The present research goal is therefore twofold:

1. Prove that a repo-resident symbolic substrate is useful.
2. Prove that a reasoner/render split is operationally valuable before the final form of the symbolic reasoner is known.
3. Prove that reasoners behave better when GOG serves repo state to them than when they infer repo structure from loosely assembled text context.

---

## 3. System Components

### 3.1 Graphized Repository

The graphized repository is the persistent symbolic state GOG installs over a codebase.

It should eventually include:

- Files, modules, imports, and exports
- Symbols, declarations, and references
- Functions, methods, classes, types, and interfaces
- Call relationships where they can be extracted
- Framework surfaces such as routes, stores, schemas, and configuration
- Test relationships and build metadata
- Repository conventions and architectural constraints
- Mutation history and graph deltas over time

The current codebase already implements an early version of this idea through AST-derived dependency graphs. The system model treats that as the foundation, not the endpoint.

### 3.2 Reasoner Slot

The reasoner receives:

- User intent
- Graphized repository state
- Constraints and invariants
- Relevant local evidence
- Prior accepted mutations when needed

The reasoner emits:

- Structured plans
- Target nodes and edges
- Required mutations
- Render ordering
- Validation requirements
- Explicit uncertainty or refusal when the graph does not support a safe plan

Today, this slot may be filled by:

- A deterministic planner for known task families
- A large language model constrained to structured output

Tomorrow, it may be filled by a dedicated symbolic reasoning model. GOG should be designed so that replacement does not require rewriting the rest of the system.

### 3.3 Renderer

The renderer is responsible for language, not architecture.

It converts typed reasoner outputs into:

- Source code
- Diffs
- Explanations
- Documentation
- User-facing summaries

The renderer should receive bounded, explicit instructions. In the code-generation case, that often means:

- One target file at a time
- A declared operation set
- Relevant local context only
- Explicit constraints

The renderer can be a small language model if the plan is sufficiently precise.

### 3.4 Validators

Validators keep the system honest.

They may check:

- Graph boundary compliance
- Forbidden imports or dependency directions
- Symbol existence
- Syntax and parse validity
- Typecheck or build success
- Test outcomes
- Patch applicability

Validation exists because a reasoner or renderer can still fail. GOG is not deterministic because every component is perfect. It becomes more deterministic because every stage exposes structure that can be inspected and constrained.

### 3.5 Graph Synchronizer

After accepted edits, the repository changes. GOG must therefore update its symbolic state.

The synchronizer should support:

- Incremental reparse of changed files
- Node and edge delta updates
- Invariant refresh
- Change history attachment
- Re-indexing auxiliary semantic retrieval structures when needed

Without synchronization, GOG decays into a stale analysis artifact. With synchronization, it becomes a persistent layer over the living codebase.

---

## 4. Session Lifecycle

An ideal GOG-assisted coding session follows this lifecycle:

1. **Onboard**
   - Build or refresh symbolic repo state.
2. **Interpret**
   - Preprocess user intent into explicit anchors, likely targets, and unresolved ambiguity.
3. **Reason**
   - Serve a bounded symbolic context bundle to the reasoner and receive a mutation plan.
4. **Validate Plan**
   - Reject unsupported references, impossible edge changes, or violated constraints.
5. **Render**
   - Produce code or prose from the plan.
6. **Validate Output**
   - Parse, typecheck, test, and confirm boundary compliance.
7. **Apply**
   - Accept the mutation only when validation clears the relevant bar.
8. **Synchronize**
   - Update `.gog/` symbolic state to reflect the new repository reality. This closes the loop.

---

## 5. What GOG Is Investigating

GOG is currently testing four linked hypotheses:

1. **Persistent symbolic state improves repo comprehension.**
2. **Reasoning over symbolic state yields better mutation plans than raw text inspection alone.**
3. **Small renderers become more reliable when architectural burden is removed.**
4. **A repo-resident symbolic layer creates a better substrate for future symbolic reasoners than ad hoc prompt context does.**

These claims should be tested in terms of final engineering quality, not only first-pass thrift. A system that fails cheaply has not outperformed a system that spends more but converges to a validated result.

The existing benchmark work explores parts of these hypotheses. The next research phase should test them more directly on public codebases and executable validation criteria.

---

## 6. Non-Goals

GOG is not currently claiming:

- That GOG itself is the reasoner
- That today's codebase already contains a general symbolic reasoning model
- That all retrieval should disappear
- That dependency graphs alone are sufficient symbolic state
- That small renderers can solve arbitrary software engineering tasks in isolation
- That current benchmark wins prove the final architecture at production scale
- That benchmark-specific heuristics are acceptable substitutes for general language and framework support

Those boundaries matter. They keep the research program falsifiable.

---

## 7. Anti-Overfitting Discipline

GOG should improve by learning language, framework, and tooling conventions, not by memorizing benchmark repositories.

Acceptable improvements include:

- resolving aliases from project configuration
- recognizing source roots, tests, generated files, and end-to-end test directories
- parsing framework surfaces such as Vue components, routes, stores, composables, and service layers
- recording unsupported conventions as partial coverage

Unacceptable improvements include:

- hard-coding expected benchmark answers
- adding rules for a specific repository's business domain
- tuning membranes until one repository's fixture passes while making the rule harder to justify elsewhere
- hiding benchmark failures by changing expected files after seeing results without documenting why

Benchmark repos should be split into development and holdout sets. Development repos may drive new general heuristics. Holdout repos should be run before tuning, with failures recorded as evidence about GOG's current limits.

---

## 8. Relationship to Existing Work

Traditional coding assistants often work episodically:

- gather text
- prompt a model
- generate a response

GOG changes the default sequence:

- install symbolic state
- present symbolic state to a reasoner
- render from explicit plans
- keep symbolic state synchronized

That difference is the system-level contribution.

---

## 9. Cost-to-Quality Research Frame

GOG should not be judged only by "how few tokens were used on the first attempt." The stronger benchmark question is:

> How much total reasoning, rendering, repair, and validation budget is required to reach the best acceptable solution?

That implies measuring:

- `Pass@1`
- `Pass@k`
- `TokensToPass`
- `AttemptsToPass`
- `WallClockToPass`

Executable patch benchmarks should split results by work domain because "coding assistant quality" is not one task:

- debugging: localize and fix an existing failing behavior
- new feature: introduce behavior that did not exist before
- refactor: preserve behavior while improving structure

They should also split by difficulty. Easy, medium, and hard cases are not interchangeable evidence. A GOG win on single-file utility fixes does not prove superiority on cross-cutting architectural work; a RAG win on one shallow feature does not disprove GOG's value as persistent symbolic state.
- `BestQualityUnderBudget`
- `ValidatedSuccessRate`

The near-term hypothesis is:

1. GOG may spend more upfront to present richer repo state.
2. GOG should reduce downstream waste by improving target selection, plan quality, and repair convergence.
3. The meaningful efficiency win is lower total cost to validated success, not lower cost to an invalid answer.

---

## 10. Prompt-Scoped Context Serving

GOG should not pass the entire repository graph into every reasoner cycle.

The repo-wide symbolic layer lives in `.gog/`. At task time, GOG should:

1. Preprocess the prompt for explicit anchors such as filenames, symbols, and operation hints.
2. Query the onboarded symbolic state.
3. Select the smallest structurally sufficient context slice.
4. Serve that bounded bundle to the reasoner.

Example prompt:

> `refactor the addUser() function`

The reasoner should receive a context bundle closer to:

- matched files or symbols
- graph-local dependencies
- relevant validation commands
- allowed context boundary
- ambiguity flags

not a whole-repo context flood.

The public reference implementation begins this path with:

- `gog_engine_lite/import_graph.py`
- `gog_engine_lite/graph_search.py`
- `gog_cli/lite_serving.py`
- `gog_cli/lite_membrane.py`

---

## 11. SalienceEvaluator Checkpoint

While the manipulator reasoner is still LLM-backed, GOG must assume it can hallucinate unsupported implementations.

The SalienceEvaluator therefore remains a final checkpoint between generated code and accepted repo mutation. Its job is to reject or repair output that violates the served symbolic boundary.

If a future symbolic reasoning model proves non-hallucinatory, this checkpoint may evolve. It should not be removed prematurely.

---

## 12. Current Repository Mapping

Relevant implementation areas:

- `gog_engine_lite/import_graph.py`
- `gog_engine_lite/graph_search.py`
- `gog_cli/lite_serving.py`
- `gog_cli/lite_membrane.py`
- `gog_cli/gold_context.py`
- `gog_cli/failure_taxonomy.py`

Relevant experimental evidence:

- `gog/results/`

Relevant next-step architecture docs:

- `docs/GOG_LITE_DESIGN.md`
- `docs/PUBLIC_PRIVATE_BOUNDARY.md`
