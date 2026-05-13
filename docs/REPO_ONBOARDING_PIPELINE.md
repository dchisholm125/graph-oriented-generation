# Repository Onboarding Pipeline

## 1. Why Onboarding Matters

GOG should not begin from an empty state each time a user asks for help.

For GOG to operate as a distinct paradigm, a repository must first be onboarded into persistent symbolic state. That symbolic layer becomes the substrate GOG serves to later reasoners, renderers, validators, and synchronizers.

Without onboarding, GOG collapses back toward ad hoc repo inspection. With onboarding, it becomes a system installed over the codebase.

---

## 2. Onboarding Goals

The onboarding pipeline should:

1. Discover the repo's structure.
2. Build a symbolic model suitable for reasoning.
3. Present repo state in forms a reasoner can query with less architectural guesswork.
4. Record validation commands and local conventions.
5. Persist artifacts for future sessions.
6. Support incremental updates after edits.

---

## 3. Onboarding Stages

### Stage 1: Repository Inventory

Collect:

- repository root
- language distribution
- package manifests
- build tooling
- test tooling
- source roots
- generated or vendored directories to ignore

Output examples:

- repo profile
- file inventory
- language/framework tags

### Stage 2: Parse and Normalize

Parse supported languages into normalized symbolic representations.

Current implementation now begins with Python, TypeScript, and Vue structural import graphs. Future onboarding should generalize toward richer symbol graphs and a common intermediate graph shape.

Capture:

- files
- imports
- exports
- declarations
- references
- parse failures

### Stage 3: Construct Graph Families

A production GOG layer likely needs more than one graph.

Recommended graph families:

1. **Structural graph**
   - file/module dependency edges
2. **Symbol graph**
   - declarations, references, call relationships
3. **Workflow graph**
   - routes, stores, events, schemas, persistence surfaces
4. **Evolution graph**
   - commits, PRs, and changed-together files when history is available

The first system milestone can remain narrower, but the architecture should leave room for these layers.

### Stage 4: Extract Constraints and Conventions

Examples:

- alias configuration
- module boundaries
- lint rules
- component/store/service layering
- route registration conventions
- test naming patterns
- file naming conventions

Some of these can be detected statically. Others may initially be learned from examples or configured explicitly.

Convention extraction must remain technology-level, not repository-specific. A rule is acceptable when it describes a general language, framework, or tooling convention, such as:

- TypeScript `paths` aliases
- Vite `resolve.alias`
- Vue single-file component imports
- test/spec filename patterns
- Playwright or Cypress end-to-end test directories
- generated API client surfaces
- source-root and package manifest discovery

A rule is not acceptable when it encodes a benchmark repository's expected answer, product domain, or exact business-specific filenames. If GOG cannot support a convention generically, onboarding should report partial coverage rather than silently adding a one-off special case.

### Stage 5: Build Auxiliary Retrieval Indices

Semantic retrieval can remain useful inside GOG, but it should be subordinate to symbolic state.

Useful indices:

- filename and symbol embeddings
- lightweight summaries attached to nodes
- benchmark-task matching indices

These indices help the reasoner find entry points. They do not replace the symbolic repo model.

### Stage 6: Persist the Symbolic Layer

Persist:

- graph artifacts
- node metadata
- constraints
- repo snapshot identity
- parser version
- onboarding timestamp
- validation command profile

This artifact set should be reloadable without reparsing the entire repo on every session.

### Stage 7: Baseline Validation

Run a repo health check when feasible:

- parse completeness
- typecheck command presence
- test command presence
- graph consistency checks
- alias resolution checks

The system should record what it could validate and what it could not.

---

## 4. Incremental Synchronization

Onboarding is not one-and-done. GOG must stay synchronized as the repository changes.

After accepted edits:

1. Detect changed files.
2. Reparse only affected files when possible.
3. Recompute changed graph edges and symbol facts.
4. Revalidate constraints touched by the change.
5. Update persistent artifacts inside `.gog/`.
6. Append the mutation to repository evolution history.

This is essential if GOG is meant to sit on top of an expanding codebase rather than act as a static analysis snapshot. The full loop only closes when `.gog/` contains the updated symbolic state after each accepted change.

---

## 5. Suggested Artifact Layout

An eventual repo-local GOG installation might look like:

```text
.gog/
  manifest.json
  repo_profile.json
  graphs/
    structural.graph
    symbols.graph
    workflows.graph
  constraints/
    architecture.json
    aliases.json
  indices/
    filenames.index
    symbols.index
  validation/
    commands.json
    last_baseline.json
  history/
    mutations.jsonl
```

The current repository does not yet install this structure, but this is the direction implied by the system model.

---

## 6. Session Preconditions

Before a GOG-assisted coding session begins, the system should know:

- whether onboarding artifacts exist
- whether they match the current repo snapshot closely enough
- whether a refresh is required
- whether the graph coverage is sufficient for the requested task

Possible states:

- `READY`
- `STALE_REFRESH_REQUIRED`
- `PARTIAL_COVERAGE`
- `UNSUPPORTED_REPO`

These states matter because GOG should not silently reason over stale or incomplete symbolic state.

At prompt time, the onboarded repo state should then be projected into a bounded context bundle. That means onboarding stores the whole symbolic layer, while runtime serving selects only the slice needed for the current request.

---

## 7. Near-Term Implementation Milestones

Recommended sequence:

1. Define a `.gog/manifest.json` contract.
2. Persist graph build metadata alongside current pickle artifacts.
3. Separate "seed benchmark assets" from "onboard repository."
4. Add freshness checks between repo files and graph artifacts.
5. Record parser coverage and unresolved imports.
6. Introduce incremental refresh for changed files.
7. Attach validation command discovery to onboarding.
8. Add prompt-scoped context serving over onboarded artifacts.

The current CLI exposes a coarse refresh command immediately:

```bash
python3 -m gog_cli refresh /path/to/repo
```

That rebuilds `.gog/` in place. Incremental refresh remains future work, but the explicit lifecycle boundary now exists.

---

## 8. Research Implications

The onboarding pipeline creates new benchmark questions:

- How expensive is graphization?
- How often does refresh occur in real work?
- How much symbolic coverage is needed before reasoning improves?
- Which graph layers actually affect plan quality?
- How quickly can a coding assistant regain correctness after the repo changes?
- How much total token and repair budget is saved when reasoners operate over onboarded symbolic state?

Those questions distinguish GOG from short-lived prompt-context experiments.

Benchmark repositories should be labeled as either:

- **development repositories**, where GOG heuristics may be improved after failures are inspected
- **holdout repositories**, where results should be recorded before any heuristic changes are made

When a failure leads to a new onboarding or membrane heuristic, the change should be documented as a general convention and rerun against prior benchmark repositories to check for regressions.

---

## 9. Relationship to Today's Code

Existing code that informs onboarding:

- `gog_engine/ast_parser.py`
- `gog_engine/ts_parser.py`
- `gog_engine/graph_search.py`
- `gog/seed_RAG_and_GOG.py`
- `gog/benchmark_ui_full.py`

The next architectural step is to turn one-off seeding scripts into a durable onboarding workflow that can be invoked before any assistant-style interaction begins.
