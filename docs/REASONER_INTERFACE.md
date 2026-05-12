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
