# Graph-Oriented Generation & Symbolic Reasoning Membrane
### An empirical research program into the structure of meaning

> **Active research.** Benchmarks are reproducible, results are real,
> and the research is ongoing. Contributions, challenges, and 
> replications welcome via issues or pull requests.

---

## What this repository is

This repository documents two connected research programs.

**GOG (Graph-Oriented Generation)** replaces probabilistic vector 
retrieval with deterministic graph traversal over a codebase's actual 
import dependency structure. It is complete, benchmarked, and 
validated. The paper is available here: 
[`GOG_PAPER.pdf`](./GOG_PAPER.pdf)

**SRM (Symbolic Reasoning Membrane)** is the deeper investigation 
that GOG made possible. It asks: if structure can control a language 
model's output reliably, what is the minimum structure required? What 
are the atomic units of meaning that a language model recognizes, 
responds to, and can combine into richer concepts?

Eighteen experiments later, that question has a partial and surprising 
answer. The full research paper is here: 
[`SRM_PAPER.md`](./SRM_PAPER.md)

---

## The core finding

Across four language model architectures — Qwen 2.5 (0.5B), 
Gemma 3 (1B), LLaMA 3.2 (1B), and SmolLM2 (360M) — we found 
consistent empirical evidence for a primitive layer underlying 
language model behavior.

Anna Wierzbicka proposed in the 1970s that all human languages 
share approximately 65 irreducible semantic concepts — WANT, KNOW, 
FEEL, GOOD, BAD, DO, HAPPEN — from which all other meaning is 
constructed. Cowen and Keltner identified 27 universal emotional 
states in 2017.

We tested whether these primitives appear as measurable activation 
patterns in small language models. They do.

Specifically:

**The Layer 0a/0b distinction is real and architecture-independent.**
Scaffolding primitives — SOMEONE, TIME, PLACE — produce abstract, 
relational responses. Content primitives — FEAR, GRIEF, JOY, ANGER, 
RELIEF, NOSTALGIA — produce phenomenological, embodied responses. 
The activation gap between these two classes averaged +0.245 across 
all four models. The direction was consistent in every model tested.

**Primitive composition produces predictable Layer 1 concepts.**
Eleven operator-seed combinations matched pre-registered predictions 
in three out of four model architectures:

| Combination | Predicted | Validated |
|-------------|-----------|-----------|
| KNOW + FEAR | dread / awareness | ✓ 3/4 models |
| FEEL + GRIEF | heartbreak / sorrow | ✓ 3/4 models |
| WANT + FEAR | anxiety / avoidance | ✓ 3/4 models |
| WANT + ANGER | ambition / revenge | ✓ 3/4 models |
| TIME + GRIEF | mourning / melancholy | ✓ 3/4 models |
| TIME + NOSTALGIA | memory / reminiscence | ✓ 3/4 models |
| TIME + RELIEF | healing / recovery | ✓ 3/4 models |
| WANT + GRIEF | longing / yearning | ✓ 3/4 models |
| WANT + NOSTALGIA | longing / regret | ✓ 3/4 models |
| FEEL + JOY | delight / bliss | ✓ 3/4 models |
| KNOW + NOSTALGIA | wisdom / reflection | ✓ 3/4 models |

**The scaling pattern has an implication.**
The primitive activation gap is largest in the smallest model and 
narrows as model size increases — not because content primitives 
weaken, but because larger models develop richer phenomenological 
access to scaffolding primitives too. As language models scale, they 
appear to converge toward a more coherent internal representation of 
the primitive layer. This may partly explain why larger models reason 
better — they are closer to the atoms of meaning.

---

## Architecture

The SRM proposes a three-layer architecture:
```
Symbolic Reasoning Layer  (pure code — no LLM)
         ↓
    generates primitive combinations
         ↓
  Structure Membrane  (small LLM, role-conditioned)
         ↓
    translates structure into language
         ↓
  Language Output  (English or any modality)
```

Structure is deterministic. Language is emergent. The membrane 
carries structure into the language space and lets emergence do 
the rest.

This is not a trained system. It is a theoretical architecture 
grounded in eighteen empirical experiments. Building it is the 
next phase.

---

## GOG: The Foundation

GOG was the first indication that something deeper was possible.

It demonstrated that replacing natural language prompts with 
deterministic symbolic specifications dramatically improves 
correctness in small language models — a 0.5B parameter model 
that fails completely on a reasoning task with a natural language 
prompt succeeds completely with a symbolic spec.

The key result:

| Tier | Input | Correctness | Time |
|------|-------|-------------|------|
| RAG | 53,137-token corpus + raw prompt | FAIL 2/5 | 5.71s |
| GOG | 6,323-token context + raw prompt | PARTIAL 4/5 | 11.63s |
| SRM | 6,323-token context + symbolic spec | **PASS 5/5** | **0.94s** |

The model did not fail because it could not write correct code. 
It failed because it could not reason about what to write. 
When the reasoning was done externally and passed in as structure, 
the language capability was sufficient.

GOG is complete and documented. The full paper, benchmark code, 
and reproduction instructions are in [`/gog`](./gog).

---

## Repository Structure
```
/
├── README.md                     this file
├── GOG_PAPER.pdf                 GOG research paper
├── SRM_RESEARCH.md               SRM research paper
├── /gog                          GOG benchmark code
│   ├── generate_dummy_repo.py
│   ├── seed_RAG_and_GOG.py
│   ├── benchmark_local_llm.py    local LLM via Ollama
│   ├── benchmark_cloud_cli.py    cloud LLM via opencode CLI
│   └── benchmark_cloud_api.py    cloud LLM via MiniMax API
├── /gog_engine                   engine modules
│   ├── minimax_client.py         MiniMax Cloud API client
│   └── ...
├── /symbol_distillation          SRM experiments 1-18
│   ├── experiment_1.py
│   ├── ...
│   └── /semantic_primitives
│       ├── wierzbicka_65_primitives.json
│       ├── cowen_keltner_27_emotions.json
│       ├── experiment_17_primitive_validation.py
│       ├── experiment_17b_layer_distinction.py
│       ├── experiment_17c_cross_model.py
│       ├── experiment_18_composition.py
│       └── primitive_summary.json
└── /results                      all experiment outputs
```

---

## Reproducing the SRM experiments

All experiments run locally via Ollama. No API keys required.
```bash
# install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# pull the models used in cross-model validation
ollama pull qwen2.5:0.5b
ollama pull gemma3:1b
ollama pull llama3.2:1b
ollama pull smollm2:360m

# install Python dependencies
pip install requests

# run primitive validation (experiment 17)
cd symbol_distillation/semantic_primitives
python3 experiment_17_primitive_validation.py

# run cross-model validation (experiment 17c)
python3 experiment_17c_cross_model.py

# run composition experiment (experiment 18)
python3 experiment_18_composition.py
```

Results are saved to `semantic_primitives/results_exp_*.json` 
and `.csv`. The living summary document 
`primitive_summary.json` accumulates findings across experiments.

---

## Reproducing the GOG benchmark

### Local LLM (Ollama)
```bash
pip install -r requirements.txt
cd gog
python3 generate_dummy_repo.py
python3 seed_RAG_and_GOG.py
python3 benchmark_local_llm.py
```

### Cloud API ([MiniMax](https://www.minimaxi.com))
```bash
pip install -r requirements.txt
export MINIMAX_API_KEY="your-key-here"
cd gog
python3 generate_dummy_repo.py
python3 seed_RAG_and_GOG.py
python3 benchmark_cloud_api.py
```

Available environment variables for the Cloud API benchmark:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_API_KEY` | *(required)* | Your MiniMax API key |
| `MINIMAX_MODEL` | `MiniMax-M2.5` | Model to benchmark (`MiniMax-M2.5`, `MiniMax-M2.5-highspeed`) |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | API endpoint (OpenAI-compatible) |

Full instructions including cloud CLI benchmark are in
[`/gog`](./gog).

---

## Open questions

The primitive composition map covers 30 combinations out of 
hundreds possible. The mechanistic explanation for the Layer 0a/0b 
distinction is unknown. The full SRM pipeline has not been 
implemented. A trained membrane does not exist.

These are not gaps to apologize for. They are directions.

Specific questions we cannot pursue alone:

- Does the Layer 0a/0b distinction appear in mechanistic 
  interpretability analysis of model internals?
- Does the activation gap scale predictably beyond 1B parameters?
- Do the same primitives drift to the same languages across 
  membranes trained in different primary languages?
- What does fine-tuning a membrane on the validated composition 
  map produce?
- Can the primitive vocabulary serve as an output-level 
  interpretability framework for existing language models?

If you pursue any of these, we want to know what you find.

---

## Status

| Track | Status | Description |
|-------|--------|-------------|
| GOG | ✓ Complete | Deterministic context isolation |
| SRM Architecture | ✓ Validated | Structure > content as control variable |
| Primitive Layer | ✓ Partially mapped | 11 validated Layer 1 compositions |
| Composition Map | 🔄 In progress | 30/390+ combinations tested |
| Trained Membrane | ⬜ Not started | Next major phase |
| Full SRM Pipeline | ⬜ Not started | Requires composition map completion |

---

## Contributing

The research is open. Replications, challenges, and extensions 
are all welcome.

If you find a flaw in the methodology, open an issue. If you run 
additional experiments and get different results, open an issue. 
If you extend the composition map, submit a pull request with your 
data. If you run the cross-model validation on models we did not 
test, we want to see the results.

Please open an issue before submitting a large pull request.

---

## Development process

This research was conducted by a single human researcher working 
in close collaboration with Claude, an AI assistant developed by 
Anthropic. Experimental design, theoretical framework, core 
insights, and research direction were conceived and driven by the 
human researcher. Claude contributed to experimental 
implementation, code generation, data interpretation, and writing.

We consider transparency about AI assistance in research to be 
important as the field develops norms around this. The 
collaboration was deep and genuine. "We" in this repository means 
both of us.

---

## Citation
```bibtex
@misc{chisholm2026srm,
  author = {Chisholm, D. R.},
  title  = {Symbolic Reasoning Membrane: An Empirical Investigation 
            of Semantic Primitives in Small Language Models},
  year   = {2026},
  url    = {https://github.com/dchisholm125/graph-oriented-generation}
}

@misc{chisholm2026gog,
  author = {Chisholm, D. R.},
  title  = {Graph-Oriented Generation (GOG): Offloading AI Reasoning 
            to Deterministic Symbolic Graphs},
  year   = {2026},
  url    = {https://github.com/dchisholm125/graph-oriented-generation}
}
```

---

*The atoms of meaning are there. We have found several of them. 
The rest are waiting.*