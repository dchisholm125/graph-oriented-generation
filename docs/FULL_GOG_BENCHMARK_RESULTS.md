# GOG Benchmark Results Index

This page is an index for public benchmark summaries. It intentionally avoids
raw private artifacts and private implementation details.

## Current Results

- [GOG Professional results](GOG_PROFESSIONAL_RESULTS.md): conservative summary
  of three private-engine benchmark case studies.
- [OpenCode bakeoff summary](OPENCODE_BAKEOFF_SUMMARY.md): assistant
  integration result for the Orca TypeScript type-guard task.
- [Public/private boundary](PUBLIC_PRIVATE_BOUNDARY.md): what may and may not
  be published in this repository.

## How To Read These Results

The public repository contains GOG Lite, not the full GOG Professional engine.
Professional benchmark summaries are included to explain the current GOG story,
but they should not be read as claims about the smaller public reference
implementation.

The supported claim is narrow:

> In controlled benchmarks so far, GOG Professional has matched validated
> outcomes from hybrid RAG with smaller context in some tasks, and its
> semantic-hazard guidance has helped coding assistants avoid concrete
> repo-specific bad paths.

These results do not show that GOG always beats RAG, that RAG cannot solve
these tasks, or that GOG guarantees better patches.

## Historical Public Artifacts

The public benchmark harness also writes JSON and Markdown artifacts under
`gog/results/`. Some older artifacts were produced before the public/private
separation and are retained as historical evidence only. Prefer the summaries
above for the current public narrative.
