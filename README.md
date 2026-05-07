# Graph-Oriented Generation (GOG)
### The Neuro-Symbolic Engine for Deterministic Code Generation

> **Status: Moving beyond RAG.** We have validated that structure is a more reliable control variable than content. We are now building the full GOG engine to replace stochastic "side-car" assistants with deterministic, graph-native generation.

---

## 🚀 The Core Thesis
Current AI coding tools (Vector RAG) treat codebases as collections of text strings and hope the LLM can "guess" the architecture. **They fail at scale because they ignore the topology of the code.**

**GOG treats a codebase as a finite, deterministic graph.** By offloading architectural reasoning to a symbolic graph layer and using a small language model (SLM) only for syntax rendering, we achieve higher correctness with 60-80% fewer tokens.

---

## 🏗️ The GOG Architecture
We separate **Reasoning** from **Rendering** using a three-layer Neuro-Symbolic stack:

1.  **Symbolic Reasoning Layer (The Architect):** A deterministic planner (or a large LLM constrained to structured output) that analyzes the codebase graph and outputs a `MutationPlan` JSON. It never writes raw code.
2.  **The Language Membrane (The Renderer):** A small model (e.g., Qwen 2.5 0.5B) that receives atomic mutation steps and renders them into syntax. It is "headless" and has no architectural agency.
3.  **SalienceEvaluator (The Validator):** A deterministic gate that validates every line of generated code against the graph boundary, auto-patching illegal imports in real-time.

---

## 🧬 Repository Roadmap

### ✅ Phase 1: Context Isolation (Complete)
Proven that AST-based graph traversal outperforms Vector RAG in context curation.
- [GOG_PAPER.pdf](./GOG_PAPER.pdf)
- Benchmark: `python3 gog/benchmark_local_llm.py`

### ✅ Phase 2: Symbolic Proof-of-Concept (Complete)
The **Symbolic Reasoning Membrane (SRM)** experiments proved that "Atoms of Meaning" exist and that structure controls LLM output better than language.
- [SRM_RESEARCH.md](./SRM_RESEARCH.md)
- Experiments: `/symbol_distillation`

### 🔄 Phase 3: The GOG Engine (Active Research)
Transitioning from benchmark experiments to a production-ready generation engine.
- [GOG_ARCHITECTURE.md](./GOG_architecture.md)
- **Investigation:** Multi-language AST support, graph mutation algebra, and the `MutationPlan` schema.

---

## 🛠️ Quick Start

### 1. Build the Graph
```bash
pip install -r requirements.txt
python3 gog/generate_dummy_repo.py
python3 gog/seed_RAG_and_GOG.py
```

### 2. Run the 3-Tier Benchmark
Compare Vector RAG (probabilistic) vs. GOG (deterministic) vs. GOG+Membrane (patched).
```bash
python3 gog/benchmark_local_llm.py
```

### 3. Explore the Architecture
Read [GOG_architecture.md](./GOG_architecture.md) for the deep dive into why we are separating reasoning from language rendering.

---

## 📖 Citation
```bibtex
@misc{chisholm2026gog,
  author = {Chisholm, D. R.},
  title  = {Graph-Oriented Generation (GOG): Offloading AI Reasoning to Deterministic Symbolic Graphs},
  year   = {2026},
  url    = {https://github.com/dchisholm125/graph-oriented-generation}
}
```

---

*The atoms of meaning are structural. We stop guessing. We start calculating.*
