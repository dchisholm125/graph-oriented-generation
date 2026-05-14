# Reasoner Interface

## 1. Why This Interface Exists

GOG does not yet know the final form of a symbolic reasoning model. That uncertainty is acceptable if the interface around the reasoner is made explicit now.

GOG is not the reasoner. It is the symbolic repo-serving layer that prepares the reasoner's operating surface.

The purpose of this document is to define the contract that any reasoner must satisfy:

- today's deterministic planners
- today's LLM-backed architect
- tomorrow's symbolic reasoning model

If the contract is stable, GOG can improve how it presents repo state and evolve the internal reasoning mechanism without destabilizing ingestion, rendering, validation, or benchmarking.

---

## 2. Design Principle

The reasoner should not emit source code directly.

It should emit a structured, inspectable artifact that explains:

- what should change
- where it should change
- which graph facts justify the change
- which constraints must hold
- how output should be rendered and validated

The renderer is allowed to produce syntax. The reasoner is responsible for plan quality.

---

## 3. Reasoner Inputs

### 3.1 Task Intent

Examples:

- Add a reset action to a store and expose it in a view.
- Introduce a route and navigation link for a new settings page.
- Explain how authentication state reaches a dashboard component.

Intent may arrive as natural language, a structured work order, or a benchmark task object.

### 3.2 Symbolic Repository State

The reasoner should query a GOG-served repository model that can expose:

- targetable files
- symbol ownership
- imports and exports
- routes, stores, and component surfaces
- available mutation sites
- dependency paths
- graph neighborhoods
- project-level constraints

GOG should serve only the bounded context needed for the request, not the entire repository graph by default.

Before a context bundle reaches the reasoner, it should pass through an always-on ContextMembrane. The ContextMembrane scores candidate files, trims low-salience graph-neighborhood noise, and records kept/rejected file audit metadata. This is separate from the mutation membrane: the ContextMembrane controls what is served to the reasoner, while the mutation membrane controls whether a returned plan or patch is admissible.

### 3.3 Constraints

Examples:

- Do not create circular dependencies.
- Do not import service clients directly into presentation components.
- Preserve an interface shape.
- Use an existing store instead of creating a duplicate.
- Touch no more than a bounded file set unless escalation is justified.

### 3.4 Environment Evidence

Optional but useful:

- repo language and framework metadata
- available build and test commands
- previous validation failures
- changed files already present in the working tree

---

## 4. Reasoner Outputs

The canonical output should be a typed plan object. A future implementation may encode it as JSON, graph deltas, or a more compact symbolic representation, but its semantics should remain stable.

Minimum shape:

```json
{
  "plan_id": "string",
  "intent": "string",
  "repo_snapshot_id": "string",
  "evidence": [
    {
      "kind": "graph_fact",
      "subject": "src/views/DashboardView.vue",
      "predicate": "imports",
      "object": "src/stores/session.ts"
    }
  ],
  "steps": [
    {
      "op": "MUTATE_NODE",
      "target": "src/stores/session.ts",
      "arguments": {
        "mutation_kind": "ADD_ACTION",
        "name": "resetSession"
      },
      "depends_on": [],
      "constraints": ["preserve_existing_public_exports"]
    }
  ],
  "render_order": ["step_1"],
  "validation": [
    "parse_target_files",
    "respect_forbidden_imports",
    "run_repo_typecheck_if_available"
  ],
  "uncertainty": []
}
```

---

## 5. Required Capabilities

A reasoner should be able to:

1. Select target nodes from the symbolic repo state.
2. Explain why those nodes are relevant.
3. Propose graph mutations and code mutations separately where useful.
4. Sequence steps when mutations depend on each other.
5. Attach explicit constraints.
6. Refuse or escalate when the graph is insufficient.
7. Produce enough evidence for downstream auditing and benchmark scoring.

---

## 6. Refusal and Uncertainty

The reasoner must not fabricate unsupported structure.

If a user asks for work that requires missing evidence, the reasoner should emit a structured refusal or uncertainty object rather than inventing files, symbols, or APIs.

Example:

```json
{
  "uncertainty": [
    {
      "code": "TARGET_SYMBOL_NOT_FOUND",
      "message": "No existing ambient-status store was found in the current graph.",
      "recommended_action": "inspect nearby state stores or ask the user to choose a target"
    }
  ]
}
```

This is one of the clearest differences between graph-native reasoning and free-form generation.

---

## 7. Current Implementation Mapping

The repository already contains early forms of this interface:

- Intent parsing in `gog_engine/planner/intent_parser.py`
- Deterministic target resolution in `gog_engine/planner/mutation_planner.py`
- Renderer-facing structured specs in `gog_engine/planner/renderer_prompt.py`

Those components are narrow and task-specific today, but they prove the usefulness of:

- explicit operation types
- explicit targets
- explicit constraints
- explicit render instructions

---

## 8. LLM-Backed Reasoner as Transitional Architecture

Near-term GOG research should use a large language model as a reasoner proxy when the task exceeds deterministic parser coverage.

That LLM-backed reasoner should:

- receive symbolic repo state, not broad raw text dumps
- emit only the typed plan contract
- be judged on plan validity separately from rendering quality
- never be conflated with the final symbolic reasoner hypothesis

This preserves the research structure:

1. Test whether the reasoner/render boundary works.
2. Test whether GOG-served symbolic state improves planning quality versus unstructured context assembly.
3. Replace the reasoner implementation later.

---

## 9. Benchmark Implications

Future benchmarks should score the reasoner independently from the renderer, and should distinguish cheap failure from efficient success.

Reasoner metrics:

- target-file recall
- unsupported-node hallucinations
- invalid-edge proposals
- constraint coverage
- step ordering validity
- plan minimality

Renderer metrics:

- patch applicability
- syntax validity
- typecheck/build success
- test success
- fidelity to the plan

End-to-end metrics:

- accepted patch rate
- `TokensToPass`
- `AttemptsToPass`
- `WallClockToPass`
- `BestQualityUnderBudget`
- graph update success

Separating those metrics will make GOG's claims much more rigorous.

The core benchmark question should become:

> Given the same repository, task, validation rules, and success threshold, does a GOG-backed reasoner reach an acceptable solution with fewer total tokens, fewer repair cycles, or better final quality than a reasoner operating over less structured context?

In the current architecture, any rendered implementation produced from that plan should still pass through the SalienceEvaluator or an equivalent anti-hallucination checkpoint before acceptance.

The first concrete benchmark for this layer lives in:

- `gog/benchmark_reasoner_prompts.py`
- `gog_cli/reasoner_benchmark.py`
- `gog_cli/schemas/mutation_plan.schema.json`

It varies prompt presentation strategy while holding model, tasks, and context bundles fixed, then grades JSON compliance, schema compliance, strict JSON-only output, and allowed-node grounding.

The next benchmark layer is semantic plan quality:

- `gog/benchmark_semantic_plan_quality.py`
- `gog_cli/semantic_plan_benchmark.py`
- `gog_cli/fixtures/semantic_plan_quality_known_problem.json`

The initial semantic fixture intentionally does **not** fix the underlying issue. It documents that `gog_cli/serving.py` currently serves structural context but not source excerpts or symbol summaries. The benchmark asks whether different GOG platter formats cause the reasoner to produce a safer plan: target the right file, acknowledge missing implementation detail, inspect before mutation, stay within `allowed_nodes`, and choose an appropriate validation command.

The same semantic benchmark now includes a `traditional_rag` baseline. This baseline is intentionally simple and reproducible: it performs keyword retrieval over source files, serves unstructured chunks, and omits GOG graph relations, handoff metadata, and allowed-node semantics except for the retrieved file list. This is not a full production vector database, but it is a controlled RAG-style baseline suitable for incremental comparison.

The public Vue benchmark track begins with:

- `gog/benchmark_public_vue.py`

This harness intentionally starts below the LLM layer. It clones or reuses a public Vue repository, runs GOG onboarding, compares GOG context selection against `traditional_rag`, and writes a JSON artifact. Later phases should reuse the same prompts for `MutationPlan` quality and executable patch benchmarks.

The executable patch layer begins with:

- `gog/benchmark_executable_patch.py`
- `gog_cli/executable_patch_benchmark.py`

This layer evaluates coding work as validated repo mutation, not just plausible planning. Each case runs in a disposable copy of the public repository, applies benchmark-only setup patches, serves either GOG or `traditional_rag` context to the same model, requires a strict JSON full-file patch response, applies only patches inside the served file set, and runs targeted validation commands.

Transient provider failures should be retried and recorded separately from coding failures. A 503, timeout, or connection reset is infrastructure noise; it should not be counted as a failed patch unless retry budget is exhausted.

Initial task domains should be tracked separately:

- debugging: fix failing behavior in an existing feature
- new feature: add behavior covered by new acceptance tests
- refactor: preserve behavior while improving structure or maintainability

Difficulty should also be explicit. Easy tasks should usually touch one focused utility. Medium tasks should involve a component or composable plus tests. Hard tasks should cross subsystem boundaries such as router/auth, store/API, or component/service flows.

Executable patch scoring should report quality and cost independently:

- `Pass@1`: first generated patch validates
- `Pass@k`: any attempt validates within the configured attempt budget
- `TokensToPass`: cumulative prompt plus response tokens until the first passing attempt
- `AttemptsToPass`: number of model attempts until first passing validation
- `WallClockToPass`: elapsed benchmark time until first passing validation
- `ValidationFailure`: patch was syntactically admissible but failed the target command
- `AdmissibilityFailure`: patch was rejected before validation, usually for invalid JSON or out-of-context file mutation

The current public Vue repo should be treated as a development benchmark, not a final proof set. Heuristics discovered there are only valid when they generalize to Vue, Vite, TypeScript, test layout, generated-client handling, or similar technology-level conventions. Future public repos should include holdout benchmarks where GOG is run before any tuning based on their failures.
