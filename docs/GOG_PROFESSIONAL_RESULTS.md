# GOG Professional Results Summary

This document publishes selected benchmark summaries from **GOG Professional**, the private production research implementation of Graph-Oriented Generation.

This repository contains **GOG Lite**: the public reference implementation for the core graph-routed context thesis. GOG Lite is intentionally transparent, small, and reproducible. GOG Professional is developed privately as a production repo-intelligence and mutation-safety layer for coding assistants.

The summaries below show what the professional engine can do without publishing the production implementation.

## Public / Private Boundary

Published here:

- Curated benchmark summaries
- Task setup, validation outcome, and high-level methodology
- Context-quality metrics such as prompt tokens, precision, recall, noise, and dominant package
- Public comparison framing against RAG-style baselines

Not published here:

- Production repo-strata discovery implementation
- Context membrane scoring internals
- Compact surgical prompt construction
- Structural anchor extraction details
- Symbol capability implementation
- Mutation-safety APIs and assistant integration contracts
- Full private benchmark artifacts

In short: the public repo shows the thesis and selected evidence; GOG Professional keeps the production engine private.

## Lead Case Study: Orca Whirlpools

**Repository:** Orca Whirlpools  
**Task:** add and test Rust tick-array containment behavior in the SDK math package  
**Model:** `kimi-k2.6:cloud`  
**Validation:** Rust library tests passed for the tick-array target  
**Comparison:** GOG Professional versus deterministic hybrid RAG budgets

This benchmark is useful because the target file is small but easy to miss inside a mixed repository. The repo contains multiple plausible Rust regions, including an Anchor program package and an SDK math package. A retrieval system can find Rust code and still land in the wrong structural region.

| Mode | Pass? | Prompt tokens | Total tokens | Context precision | Context recall | Noise ratio | Dominant package |
|------|:-----:|--------------:|-------------:|------------------:|---------------:|------------:|------------------|
| **GOG Professional** | **yes** | **2,406** | **3,004** | **0.8333** | **1.0** | **0.1667** | **`rust-sdk/core`** |
| RAG hybrid 1k | no | 578 | 1,131 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 4k | no | 578 | 1,329 | 0.0 | 0.0 | 1.0 | `programs/whirlpool` |
| RAG hybrid 16k | yes | 2,494 | 3,219 | 0.1429 | 0.2 | 0.8571 | `programs/whirlpool` |
| RAG hybrid 64k | yes | 4,831 | 5,434 | 0.2 | 0.6 | 0.8 | `programs/whirlpool` |

## Interpretation

The result is not simply that one mode used fewer tokens. The stronger signal is structural locality.

GOG Professional selected context from the correct package, served a compact edit surface, and produced a validated patch. Low-budget RAG centered on the wrong package and failed. Higher-budget RAG eventually passed, but only with substantially worse precision, recall, noise, and package locality.

That is the commercial direction for GOG Professional:

> GOG does not replace coding assistants. It gives them better repo context and safer mutation surfaces.

## Product Direction

GOG Professional is being developed as a private repo-intelligence layer for coding assistants. The system focuses on:

- Repo strata and package locality
- Compact task-specific context
- Structural anchors for constrained edits
- Symbol capability checks
- Mutation-safety contracts around assistant-generated patches
- Cost-per-validated-patch measurement

Those capabilities are described publicly at a product level, but their production implementation is not included in GOG Lite.

For technical walkthroughs or pilot integrations, use the contact path on the project page: <https://derekchisholm.com/gog>.
