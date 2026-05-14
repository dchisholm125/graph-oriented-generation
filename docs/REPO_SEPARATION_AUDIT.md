# REPO_SEPARATION_AUDIT: Graph-Oriented Generation (GOG)

## 1. Executive Summary

This audit evaluates the current GOG repository for separation into Public, Private (Commercial), and SRM (Moonshot) repositories. 

- **Current State:** The repository currently functions as a "monorepo" containing the core graph engine, a production-ready CLI, extensive benchmark instrumentation, and deep moonshot research (SRM).
- **Public Repo:** Should retain the research "soul" of GOG. It will host the benchmark results, baseline RAG implementation, and a "GOG-Lite" reference implementation. This preserves scientific credibility and reproducibility for the GOG paper without giving away the production "secret sauce."
- **Private Repo:** Will house the high-performance GOG engine, the production CLI, advanced context membrane heuristics, and scalable onboarding logic. This is the productized IP.
- **SRM Repo:** Will capture the high-variance reasoning experiments, symbolic distillation, and mutation planning research. This keeps the commercial GOG focus sharp as a "context wedge" rather than an "AI scientist."

## 2. File/Directory Classification Table

| Path | Current Purpose | Classification | Reason | Recommended Action | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gog_engine/` | Core graph parsing & search logic | **PRIVATE_COMMERCIAL** | Contains proprietary AST/Tree-sitter heuristics | Move to private repo; leave stubs or Lite version in public | HIGH |
| `gog_cli/` | Production CLI & onboarding | **PRIVATE_COMMERCIAL** | Commercial product interface | Move to private repo | HIGH |
| `gog/` | Benchmark suite & runners | **PUBLIC_BENCHMARK** | Essential for reproducibility | Keep public; sanitize sensitive results | LOW |
| `sel/` | Symbolic reasoning experiments | **SRM_RESEARCH** | Core moonshot IP | Move to SRM repo | MEDIUM |
| `symbol_distillation/` | Reasoning distillation research | **SRM_RESEARCH** | High-variance research | Move to SRM repo | LOW |
| `srm_engine/` | Symbolic mutation planner | **SRM_RESEARCH** | Distracts from near-term GOG | Move to SRM repo | MEDIUM |
| `docs/` | System & architecture docs | **PUBLIC_DOCS** | Essential for understanding GOG | Keep public; redact commercial internals | LOW |
| `GOG_PAPER.pdf` | Public research paper | **PUBLIC_DOCS** | Established research artifact | Keep public | LOW |
| `GOG_architecture.md` | Internal architecture details | **REVIEW_REQUIRED** | Mixed detail | Split: keep concept public, move logic private | MEDIUM |
| `target_repo/` | Dummy test codebase | **PUBLIC_CORE** | Required for unit tests | Keep public | LOW |
| `distilled_datasets/` | Research datasets | **SRM_RESEARCH** | Data for reasoning models | Move to SRM repo | LOW |

## 3. Commercial Sensitivity Audit

The following areas contain sensitive IP that should be protected:

- **`gog_cli/context_membrane.py`:** Contains the exact scoring and pruning heuristics (e.g., `_score_node`, distance calculations) that differentiate GOG from simple graph RAG.
- **`gog_engine/ast_parser.py`:** Proprietary handling of aliased imports and Vue-specific graph construction.
- **`gog_engine/graph_search.py`:** Semantic seeding strategy and depth-bounded expansion logic.
- **`gog_cli/onboarding.py`:** Logic for scalable repository fingerprinting and persistent `.gog/` artifact creation.
- **`GOG_architecture.md`:** Deep dives into how the synchronizer and validator layers work in production.

## 4. Public Reproducibility Plan

To maintain the project's reputation for honesty, the public repo must allow anyone to run:
- **`gog/benchmark_public_vue.py`** and **`gog/benchmark_local_llm.py`**.
- A **GOG-Lite** reference engine: A simplified version of `gog_engine` that handles basic imports but lacks the advanced "Context Membrane" pruning.
- **Baseline RAG:** The `traditional_rag` implementation must remain transparent so the GOG improvement delta can be verified.
- **Result Logs:** Sanitized summary JSONs of the benchmarks.

## 5. Private Commercial Repo Plan

The private repository will be the home of the "GOG Professional" engine:
- **Full CLI:** The `onboard`, `refresh`, and `context` commands.
- **Optimized Engine:** Advanced multi-phase ASR (if applicable) and the refined Context Membrane.
- **Integration Layer:** Logic for hooking into enterprise tools or specific agent harnesses (like OpenClaw).
- **Security Features:** Logic for protecting against "context poisoning" and ensuring privacy boundaries.

## 6. SRM Repo Plan

The SRM repository will focus on the "Moonshot" goal:
- **`sel/` and `srm_engine/`:** These will form the foundation of a separate research effort into "Planning as Reasoning."
- **Symbol Distillation:** All data and scripts in `symbol_distillation/` which are speculative and high-overhead for a near-term context product.
- **SRM Paper/Prototype:** This avoids confusing investors who want a "better coding context tool" (GOG) with a "new way to build models" (SRM).

## 7. Proposed Public README Positioning

The public README should be updated with a "Research vs. Production" section:

> **Research & Production**
> 
> This repository contains the public research, reference implementation, and benchmark artifacts for GOG. It is designed for scientific reproducibility of the results described in the [GOG Paper](./GOG_PAPER.pdf).
> 
> The production GOG engine, optimized CLI, and enterprise integrations are developed in a separate private repository. For enterprise use, high-scale onboarding, or production-grade context membranes, please contact [Derek].
> 
> The Symbolic Reasoning Model (SRM) research has been moved to a dedicated moonshot repository to focus on long-term symbolic mutation planning.

## 8. Risk Register

- **Risk: History Leakage.** Commits already pushed contain sensitive code. 
  - *Mitigation:* We acknowledge this and will not attempt to rewrite history (which looks suspicious to investors). We simply move forward with new development in private.
- **Risk: Reproducibility Gap.** If "GOG-Lite" is too simple, it won't beat RAG as well as the paper claims.
  - *Mitigation:* Ensure GOG-Lite includes enough topological awareness to prove the core thesis.
- **Risk: Apache 2.0 Confusion.** 
  - *Mitigation:* Clearly mark the public repo as the "Reference Implementation" under Apache 2.0, while the production engine uses a proprietary license in the private repo.

## 9. Concrete Migration Plan

- **Stage 0 (Freeze):** Tag the current repo as `v0.5-pre-separation` to preserve a snapshot.
- **Stage 1 (Classify):** Add `.separation_ignore` or similar tracking to identify which files move where.
- **Stage 2 (Creation):** Create `gog-private` and `srm-moonshot` repos.
- **Stage 3 (Migration):** Clone current repo into both new targets; delete irrelevant folders in each.
- **Stage 4 (Public Sanitization):** In the public repo, replace `gog_engine` with `gog_engine_lite` and remove the `sel/` and `srm_engine/` directories.
- **Stage 5 (Docs):** Update README and create the "Separation Report" (this document).
- **Stage 6 (Verification):** Run `benchmark_public_vue.py` on the public repo to ensure it still passes/fails in a way that matches the paper's claims.

## 10. Open Questions for Derek

1. **GOG-Lite Scope:** How much "Context Membrane" logic are you comfortable leaving in the public reference implementation? (e.g., simple graph-depth pruning vs. the full scoring heuristic).
2. **Branding:** Do we want to keep the name "GOG" for both, or call the public version "GOG-Ref" and the private one "GOG-Pro"?
3. **Target Repos:** Should we move the `public-repos/` used in benchmarks (like `httpx` or `vue-realworld`) out of this repo and into a `gog-benchmarks` submodule?
