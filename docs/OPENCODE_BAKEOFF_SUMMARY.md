# OpenCode Bakeoff Summary

OpenCode is a useful proving ground because it is an agentic coding assistant
that can inspect a repository, call tools, edit files, and run validation. That
makes it a realistic integration target for GOG: GOG is not tested as a
replacement assistant, but as a guidance and context layer underneath an
assistant that already knows how to work in a repo.

Baseline OpenCode is not necessarily "RAG." In this bakeoff, it is the coding
assistant operating with its normal tool use and no GOG brief. The comparison
tests whether a GOG-produced brief can reduce repo archaeology and guide the
assistant away from bad paths.

## Task

The task was to add TypeScript type guard helpers for decoded consolidated Orca
tick-array accounts and validate both runtime tests and TypeScript typecheck.

The task is useful because it crosses a common repo hazard: generated account
layouts exist near handwritten public helper APIs, but generated raw layouts are
not always the correct public type surface.

## Result

| Mode | Validation | Typecheck | Files read | Wrong-stratum reads | Transcript-token estimate |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline OpenCode | fail | fail | 37 | 18 | ~914,473 |
| GOG semantic hazards | pass | pass | 9 | 2 | ~444,430 |
| RAG hybrid 16K | pass | pass | 17 | 3 | ~726,962 |

RAG hybrid passed and was faster wall-clock in this run. GOG semantic hazards
passed with fewer file reads, fewer wrong-stratum reads, and a lower
transcript-token estimate. The result is not a broad claim that GOG beats RAG;
it is evidence that explicit semantic-hazard guidance can change assistant
behavior on a concrete repo-specific failure mode.

## Semantic-Hazard Lesson

GOG selected the right files and edit boundaries, but v1 did not warn about the
generated-vs-consolidated type trap. v2 added bad-path guidance and passed.

The key lesson is that context selection and edit boundaries are necessary but
not always sufficient. For coding-agent integrations, GOG briefs need to include
the relevant semantic traps when the repository has generated evidence files
near handwritten public APIs.

For this task family, useful guidance included:

- Use the handwritten consolidated helper type as the public narrowing target.
- Treat generated account files as evidence for decoder inputs, not as public
  helper output types.
- Keep generated files evidence-only.
- Make `tsc --noEmit` an explicit validation requirement for type predicates.

## What This Means For Integrations

GOG should integrate with coding assistants as a preflight and context layer:

- Map the repository before the assistant starts reading broadly.
- Serve a bounded context bundle with provenance.
- Mark edit-allowed and evidence-only files.
- Identify generated, vendored, legacy, or wrong-stratum hazards.
- Name known bad paths when the repo shape suggests a likely assistant mistake.
- Evaluate the resulting patch with validation and locality metrics.

This keeps the assistant in charge of implementation while giving it clearer
repo facts and fewer opportunities to wander into irrelevant or misleading
surfaces.
