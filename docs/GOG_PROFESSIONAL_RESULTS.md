# GOG Professional Results

This document summarizes selected GOG Professional benchmark results in a
public-safe form. GOG Professional remains private; this repository contains GOG
Lite, the public reference implementation.

The summaries below are task-local evidence. They do not claim that GOG always
beats RAG, that RAG cannot solve these tasks, or that GOG guarantees better
patches.

## Result A: Orca Rust Surgical Task

**Repo:** real Orca Whirlpools SDK

**Task shape:** localized Rust SDK change with executable validation

**Public-safe result:** GOG compact/surgical context produced a validated Rust
patch.

In this benchmark, GOG Professional served a compact, surgical context for a
Rust task in the Orca Whirlpools SDK. The patch applied cleanly and validation
passed with the relevant Rust tests. The important public result is not a
private implementation detail; it is that a small, targeted context bundle plus
explicit mutation boundaries was sufficient to produce a validated patch in a
real SDK.

In the published comparison run, the low-budget hybrid RAG variants selected
the wrong structural region and failed validation. Higher-budget RAG eventually
passed, but with substantially more structural noise.

| Mode | Validation | Prompt tokens | Total tokens | Context precision | Context recall | Noise ratio | Dominant package |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |
| GOG Professional | pass | 2,406 | 3,004 | 0.8333 | 1.0 | 0.1667 | `rust-sdk/core` |
| RAG hybrid 1K | fail | 578 | 1,131 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 4K | fail | 578 | 1,329 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 16K | pass | 2,494 | 3,219 | 0.1429 | 0.2 | 0.8571 | `programs/whirlpool` |
| RAG hybrid 64K | pass | 4,831 | 5,434 | 0.2 | 0.6 | 0.8 | `programs/whirlpool` |

Conservative reading:

- GOG selected the right package and edit surface.
- The resulting patch was validated by tests.
- The result supports surgical-context and mutation-boundary claims.
- It does not prove that every localized Rust task will pass or that broad
  retrieval cannot pass.

## Result B: Orca Cross-Language Deployment Parity

**Repo:** real Orca Whirlpools SDK

**Task shape:** offline parity tests across TypeScript and Rust deployment
constants

**Public-safe result:** GOG and hybrid RAG all passed in the fair comparison;
GOG used substantially smaller context.

| Metric | GOG | RAG hybrid 16K | RAG hybrid 64K |
| --- | ---: | ---: | ---: |
| Validation | pass | pass | pass |
| Retrieved files | 6 | 27 | 34 |
| Prompt tokens | 3,768 | 11,461 | 16,187 |
| Total tokens | 6,500 | 14,132 | 18,714 |

RAG hybrid 64K produced a more exhaustive patch in one run. That nuance matters:
the fair conclusion is not that RAG failed. The conclusion is that all three
approaches reached a validated outcome, while GOG did so with fewer retrieved
files and fewer tokens.

Conservative reading:

- Same validated outcome across GOG and hybrid RAG.
- GOG's observed advantage was context size and repo locality.
- RAG hybrid remained competitive and, at 64K, produced a more exhaustive patch
  in one run.
- This benchmark supports context-efficiency claims, not universal superiority
  claims.

## Result C: OpenCode Orca TypeScript Type-Guard Bakeoff

**Repo:** real Orca Whirlpools SDK

**Task shape:** TypeScript helper type guards over generated and handwritten
surfaces

**Public-safe result:** semantic-hazard guidance changed the outcome for GOG.

| Metric | Baseline OpenCode | GOG semantic hazards | RAG hybrid 16K |
| --- | ---: | ---: | ---: |
| Validation | fail | pass | pass |
| Typecheck | fail | pass | pass |
| Files read | 37 | 9 | 17 |
| Wrong-stratum reads | 18 | 2 | 3 |
| Transcript-token estimate | ~914,473 | ~444,430 | ~726,962 |

Baseline OpenCode failed validation and TypeScript typecheck. GOG v1 also
failed because it selected the right files and edit boundaries but did not warn
about the generated-vs-consolidated type trap. RAG hybrid passed. GOG v2 added
a compact semantic-hazards / bad-path brief and passed.

RAG was faster wall-clock in the semantic-hazards rerun. The conservative GOG
result is locality and reduced repo archaeology: fewer files read, fewer
wrong-stratum reads, and a lower transcript-token estimate than both baseline
and RAG in that run.

Conservative reading:

- Selected files alone were not enough.
- Bad-path guidance mattered for the TypeScript type predicate failure mode.
- RAG hybrid also passed and was faster wall-clock.
- GOG v2's advantage in this run was semantic-hazard guidance, file locality,
  and transcript-token estimate.

## Overall Interpretation

The current professional evidence supports a practical product thesis:

> GOG is useful when it helps a coding assistant spend less effort discovering
> repository structure and more effort applying a bounded, validated change.

The current evidence does not support broad replacement claims. GOG should be
compared against strong assistants and strong retrieval baselines, including
hybrid RAG systems that can also pass.
