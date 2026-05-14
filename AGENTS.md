# GOG Agent Notes

This repo is evolving toward a reproducible Graph-Oriented Generation CLI for AI coding work.

Keep these principles top of mind:

- GOG is not the reasoner. GOG is the repo-resident symbolic substrate that serves bounded state to a reasoner.
- The membrane is always on. Treat it as the deterministic admissibility boundary between a reasoner `MutationPlan` and repo mutation.
- The ContextMembrane is always on before reasoner handoff. It should reduce over-selection while preserving recall and recording kept/rejected context decisions.
- Benchmarks must be reproducible from CLI commands and JSON artifacts under `gog/results/`.
- Compare GOG against controlled baselines, especially `traditional_rag`, before making claims.
- Prefer incremental benchmarks: context selection, semantic `MutationPlan` quality, then executable patches.
- Track quality and cost separately: `Pass@1`, `Pass@k`, `TokensToPass`, `AttemptsToPass`, `WallClockToPass`, context precision/recall, membrane rejects, and validation accuracy.
- Split executable patch benchmarks by work domain and difficulty: debugging, new features, refactors; easy, medium, hard.
- Run executable patch benchmarks in disposable repository copies with benchmark-only fixtures so public checkouts stay clean.
- Do not fix a benchmark target problem while creating the benchmark unless the task explicitly asks for the implementation phase.
- Do not overfit GOG to a specific benchmark repository. Improvements should be language-, framework-, or tooling-level conventions, not hard-coded knowledge of one codebase.
- Treat benchmark repositories as either development repos or holdout repos. Tune on development repos only; record holdout failures before changing heuristics.
- If a heuristic is added after inspecting a benchmark failure, state the general convention it implements, such as alias resolution, test-file downranking, generated-file handling, route detection, store detection, or source-root discovery.
- Do not encode expected benchmark answers, exact fixture filenames, or repository-specific business concepts into GOG onboarding or membranes.
- Supported language/framework behavior should be explicit. If a repo uses unsupported conventions, report partial coverage instead of silently adding one-off special cases.
- GOG product direction is a command-line coding tool: keep interfaces scriptable, inspectable, and stable.
