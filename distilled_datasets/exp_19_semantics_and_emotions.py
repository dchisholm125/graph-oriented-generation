#!/usr/bin/env python3
"""
Experiment 19: Primitive-Emotion Meaning Scores
LOCATION: semantic_primitives/experiment_19_meaning_scores.py

HYPOTHESIS:
If we ask a language model to score its own internal response
to primitive-emotion combinations across multiple dimensions,
the scores will reveal a geometry of meaning that is:
  - Consistent across runs (same combination scores similarly)
  - Differentiated across combinations (not all scores the same)
  - Interpretable (high scores make intuitive sense)

SCORING DIMENSIONS:
  meaning_score      — how much unified meaning emerges (0-10)
  excitement_score   — how activating/energizing the combination is (0-10)
  emotional_elicitation — how strongly it triggers felt emotion (0-10)
  clarity_score      — how unambiguous the meaning is (0-10)
  universality_score — how cross-cultural/universal the concept feels (0-10)
  embodiment_score   — how physically/sensorially grounded it is (0-10)
  novelty_score      — how surprising or unexpected the combination is (0-10)

These scores are explicitly speculative — the model is reporting
on its own internal states, which are not directly observable.
The value is in the PATTERNS across combinations, not the
absolute numbers.

INPUT: NSM primitives × Cowen & Keltner 27 emotions
SOURCE: pre-built mapping from semantic_primitives/
"""
import requests
import json
import csv
import time
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ── NSM → emotion mappings ────────────────────────────────────────────────────
# sourced from the pre-built mapping document

NSM_EMOTION_MAPPINGS = {
    # SUBSTANTIVES
    "I":          ["Joy", "Anxiety", "Fear", "Shame"],
    "you":        ["Admiration", "Adoration", "Romance", "Sympathy"],
    "someone":    ["Admiration", "Sympathy", "Envy", "Adoration"],
    "people":     ["Admiration", "Sympathy", "Awe"],
    "something":  ["Interest", "Craving", "Awe"],
    "body":       ["Calmness", "Excitement", "Fear", "Satisfaction"],
    # EVALUATORS
    "good":       ["Joy", "Satisfaction", "Admiration", "Calmness"],
    "bad":        ["Disgust", "Sadness", "Fear", "Horror"],
    # MENTAL PREDICATES
    "think":      ["Interest", "Confusion", "Awe"],
    "know":       ["Satisfaction", "Calmness", "Anxiety"],
    "want":       ["Craving", "Excitement", "Anxiety"],
    "feel":       ["Joy", "Sadness", "Fear", "Awe", "Nostalgia"],
    "see":        ["Aesthetic Appreciation", "Interest", "Awe", "Fear"],
    "hear":       ["Aesthetic Appreciation", "Interest", "Awe", "Calmness"],
    # ACTIONS
    "do":         ["Excitement", "Triumph", "Anxiety"],
    "happen":     ["Awe", "Fear", "Surprise", "Interest"],
    "move":       ["Excitement", "Awe", "Fear", "Interest"],
    # EXISTENCE
    "live":       ["Joy", "Excitement", "Awe", "Fear"],
    "die":        ["Horror", "Fear", "Sadness", "Awe", "Nostalgia"],
    # TIME
    "time":       ["Nostalgia", "Anxiety", "Awe", "Interest"],
    "now":        ["Anxiety", "Excitement", "Calmness"],
    "before":     ["Nostalgia", "Sadness", "Pride"],
    "after":      ["Relief", "Satisfaction", "Nostalgia"],
    "moment":     ["Awe", "Excitement", "Fear", "Interest"],
    # SPACE
    "place":      ["Nostalgia", "Awe", "Anxiety", "Curiosity"],
    "here":       ["Calmness", "Nostalgia", "Anxiety"],
    "far":        ["Nostalgia", "Awe", "Anxiety"],
    "near":       ["Calmness", "Anxiety", "Romance"],
    "inside":     ["Comfort", "Curiosity", "Fear"],
    # LOGICAL
    "not":        ["Disgust", "Fear", "Anxiety", "Calmness"],
    "maybe":      ["Hope", "Anxiety", "Curiosity"],
    "if":         ["Hope", "Anxiety", "Curiosity"],
    # INTENSIFIERS
    "very":       ["Awe", "Excitement", "Fear"],
    "more":       ["Interest", "Craving", "Excitement", "Envy"],
}

# ── all 27 Cowen & Keltner emotions ──────────────────────────────────────────
CK_27_EMOTIONS = [
    "Admiration", "Adoration", "Aesthetic Appreciation", "Amusement",
    "Anxiety", "Awe", "Awkwardness", "Boredom", "Calmness", "Confusion",
    "Craving", "Disgust", "Empathic Pain", "Entrancement", "Envy",
    "Excitement", "Fear", "Horror", "Interest", "Joy", "Nostalgia",
    "Romance", "Sadness", "Satisfaction", "Sexual Desire", "Sympathy",
    "Triumph",
]

MODELS = [
    {"name": "qwen2.5:0.5b",  "short": "qwen"},
    {"name": "gemma3:1b",     "short": "gemma"},
    {"name": "llama3.2:1b",   "short": "llama"},
]

TRIALS_PER_COMBO = 3
SCORE_DIMENSIONS = [
    "meaning_score",
    "excitement_score",
    "emotional_elicitation",
    "clarity_score",
    "universality_score",
    "embodiment_score",
    "novelty_score",
]

# ── system prompt ─────────────────────────────────────────────────────────────

SYSTEM_MEMBRANE = """You are a language membrane with introspective 
scoring capability.

When given a primitive concept paired with an emotion, you will:
1. Feel into the combination — what arises internally
2. Score your response across 7 dimensions from 0-10

Be honest about your internal states. These scores are 
speculative but meaningful. Trust your first response.
Score 0 = none/absent, 10 = maximum/overwhelming.

Output ONLY valid JSON. No explanation outside the JSON."""

def build_prompt(primitive: str, emotion: str) -> str:
    return (
        f"Primitive concept: {primitive}\n"
        f"Emotion: {emotion}\n\n"
        f"Score this combination across all 7 dimensions.\n\n"
        f"Output ONLY this JSON:\n"
        f"{{\n"
        f'  "meaning_score": 0-10,\n'
        f'  "excitement_score": 0-10,\n'
        f'  "emotional_elicitation": 0-10,\n'
        f'  "clarity_score": 0-10,\n'
        f'  "universality_score": 0-10,\n'
        f'  "embodiment_score": 0-10,\n'
        f'  "novelty_score": 0-10,\n'
        f'  "dominant_sensation": "one word",\n'
        f'  "composite_concept": "2-3 words — what this combination becomes",\n'
        f'  "confidence": 0.0-1.0\n'
        f"}}"
    )

# ── HTTP ──────────────────────────────────────────────────────────────────────

def ollama_call(host, model, prompt, system=None,
                timeout=90, max_retries=3):
    for attempt in range(max_retries):
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            resp = requests.post(
                f"{host}/api/generate", json=payload, timeout=timeout
            )
            return resp.json().get("response", "")
        except requests.exceptions.ReadTimeout:
            wait = 2 ** attempt
            print(f"\n    timeout, retrying in {wait}s...",
                  end=" ", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"\n    error: {e}", end=" ", flush=True)
            time.sleep(2)
    return ""

def extract_json(raw: str) -> dict:
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except Exception:
        return {}

def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0) -> int:
    try:
        v = float(val)
        return max(0, min(10, int(round(v))))
    except (TypeError, ValueError):
        return default

# ── data structure ────────────────────────────────────────────────────────────

@dataclass
class ScoredCombo:
    model_name: str
    model_short: str
    primitive: str
    emotion: str
    combo: str
    trial: int
    # scores
    meaning_score: float
    excitement_score: float
    emotional_elicitation: float
    clarity_score: float
    universality_score: float
    embodiment_score: float
    novelty_score: float
    composite_score: float      # weighted average of all 7
    # qualitative
    dominant_sensation: str
    composite_concept: str
    confidence: float
    parse_failed: bool

# ── composite scoring ─────────────────────────────────────────────────────────

WEIGHTS = {
    "meaning_score":        0.25,
    "excitement_score":     0.10,
    "emotional_elicitation": 0.20,
    "clarity_score":        0.15,
    "universality_score":   0.15,
    "embodiment_score":     0.10,
    "novelty_score":        0.05,
}

def compute_composite(scores: dict) -> float:
    total = sum(
        scores.get(dim, 0) * weight
        for dim, weight in WEIGHTS.items()
    )
    return round(total, 3)

# ── probe ─────────────────────────────────────────────────────────────────────

def probe_combo(primitive: str, emotion: str,
                model: str, host: str, trial: int,
                model_short: str) -> ScoredCombo:
    prompt = build_prompt(primitive, emotion)
    raw    = ollama_call(host, model,
                         prompt=prompt,
                         system=SYSTEM_MEMBRANE)
    result = extract_json(raw)

    base = ScoredCombo(
        model_name=model,
        model_short=model_short,
        primitive=primitive,
        emotion=emotion,
        combo=f"{primitive} × {emotion}",
        trial=trial,
        meaning_score=0.0,
        excitement_score=0.0,
        emotional_elicitation=0.0,
        clarity_score=0.0,
        universality_score=0.0,
        embodiment_score=0.0,
        novelty_score=0.0,
        composite_score=0.0,
        dominant_sensation="",
        composite_concept="",
        confidence=0.0,
        parse_failed=True,
    )

    if not result:
        return base

    scores = {dim: safe_int(result.get(dim, 0))
              for dim in SCORE_DIMENSIONS}

    base.meaning_score         = scores["meaning_score"]
    base.excitement_score      = scores["excitement_score"]
    base.emotional_elicitation = scores["emotional_elicitation"]
    base.clarity_score         = scores["clarity_score"]
    base.universality_score    = scores["universality_score"]
    base.embodiment_score      = scores["embodiment_score"]
    base.novelty_score         = scores["novelty_score"]
    base.composite_score       = compute_composite(scores)
    base.dominant_sensation    = str(result.get("dominant_sensation", ""))
    base.composite_concept     = str(result.get("composite_concept", ""))
    base.confidence            = safe_float(result.get("confidence", 0.0))
    base.parse_failed          = False
    return base

# ── main experiment ───────────────────────────────────────────────────────────

def run_experiment(host="http://localhost:11434"):
    all_results: list[ScoredCombo] = []

    # build combo list from mappings
    combos = []
    for primitive, emotions in NSM_EMOTION_MAPPINGS.items():
        for emotion in emotions:
            combos.append((primitive, emotion))

    total_combos  = len(combos)
    total_models  = len(MODELS)
    total_calls   = total_combos * TRIALS_PER_COMBO * total_models

    print(f"SRM experiment 19 — primitive × emotion meaning scores")
    print(f"{total_models} models × {total_combos} combinations "
          f"× {TRIALS_PER_COMBO} trials = {total_calls} calls\n")
    print(f"Scoring dimensions: {', '.join(SCORE_DIMENSIONS)}\n")

    for model_def in MODELS:
        model_name  = model_def["name"]
        model_short = model_def["short"]

        print(f"\n{'═'*60}")
        print(f"MODEL: {model_name}")
        print(f"{'═'*60}")

        for primitive, emotion in combos:
            combo_str = f"{primitive} × {emotion}"
            trial_scores = []

            for trial in range(1, TRIALS_PER_COMBO + 1):
                r = probe_combo(primitive, emotion,
                                model_name, host,
                                trial, model_short)
                all_results.append(r)
                trial_scores.append(r.composite_score)
                time.sleep(0.2)

            avg = sum(trial_scores) / len(trial_scores) if trial_scores else 0
            print(f"  {combo_str:<35} "
                  f"avg:{avg:>5.2f}  "
                  f"→ {r.composite_concept[:30]}")

        # checkpoint per model
        save_results(all_results,
                     base="semantic_primitives/results_exp_19_checkpoint")

    report(all_results)
    save_results(all_results)
    return all_results

# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: list[ScoredCombo]):
    models = list(dict.fromkeys(r.model_name for r in results))

    print(f"\n{'═'*60}")
    print(f"MEANING SCORE ANALYSIS")
    print(f"{'═'*60}")

    # aggregate by combo across all models and trials
    combo_aggregates: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.parse_failed:
            key = r.combo
            for dim in SCORE_DIMENSIONS:
                combo_aggregates[key][dim].append(getattr(r, dim))
            combo_aggregates[key]["composite"].append(r.composite_score)
            combo_aggregates[key]["concepts"].append(r.composite_concept)

    # compute averages
    combo_avgs = {}
    for combo, data in combo_aggregates.items():
        combo_avgs[combo] = {
            dim: round(sum(v)/len(v), 2)
            for dim, v in data.items()
            if dim != "concepts"
        }
        combo_avgs[combo]["top_concept"] = (
            max(set(data["concepts"]), key=data["concepts"].count)
            if data["concepts"] else ""
        )

    print(f"\n── TOP 20 HIGHEST MEANING SCORE COMBINATIONS ────────────────")
    print(f"  {'Combination':<35} {'Mean':>5}  {'EE':>5}  "
          f"{'Emb':>5}  {'Uni':>5}  Concept")
    print(f"  {'───────────':<35} {'────':>5}  {'──':>5}  "
          f"{'───':>5}  {'───':>5}  ───────")

    top_meaning = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("meaning_score", 0),
        reverse=True
    )[:20]

    for combo, avgs in top_meaning:
        print(f"  {combo:<35} "
              f"{avgs.get('meaning_score',0):>5.1f}  "
              f"{avgs.get('emotional_elicitation',0):>5.1f}  "
              f"{avgs.get('embodiment_score',0):>5.1f}  "
              f"{avgs.get('universality_score',0):>5.1f}  "
              f"{avgs.get('top_concept','')[:25]}")

    print(f"\n── TOP 20 HIGHEST EXCITEMENT SCORE COMBINATIONS ─────────────")
    top_excitement = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("excitement_score", 0),
        reverse=True
    )[:20]
    for combo, avgs in top_excitement:
        print(f"  {combo:<35} "
              f"exc:{avgs.get('excitement_score',0):>4.1f}  "
              f"→ {avgs.get('top_concept','')[:30]}")

    print(f"\n── TOP 20 HIGHEST EMOTIONAL ELICITATION ─────────────────────")
    top_ee = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("emotional_elicitation", 0),
        reverse=True
    )[:20]
    for combo, avgs in top_ee:
        print(f"  {combo:<35} "
              f"ee:{avgs.get('emotional_elicitation',0):>4.1f}  "
              f"→ {avgs.get('top_concept','')[:30]}")

    print(f"\n── HIGHEST NOVELTY SCORES ────────────────────────────────────")
    print(f"  Unexpected combinations — meaning that surprises")
    top_novelty = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("novelty_score", 0),
        reverse=True
    )[:15]
    for combo, avgs in top_novelty:
        print(f"  {combo:<35} "
              f"nov:{avgs.get('novelty_score',0):>4.1f}  "
              f"→ {avgs.get('top_concept','')[:30]}")

    print(f"\n── HIGHEST UNIVERSALITY SCORES ───────────────────────────────")
    print(f"  Most cross-cultural, language-agnostic combinations")
    top_universal = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("universality_score", 0),
        reverse=True
    )[:15]
    for combo, avgs in top_universal:
        print(f"  {combo:<35} "
              f"uni:{avgs.get('universality_score',0):>4.1f}  "
              f"→ {avgs.get('top_concept','')[:30]}")

    print(f"\n── HIGHEST EMBODIMENT SCORES ─────────────────────────────────")
    print(f"  Most physically/sensorially grounded combinations")
    top_embodied = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("embodiment_score", 0),
        reverse=True
    )[:15]
    for combo, avgs in top_embodied:
        print(f"  {combo:<35} "
              f"emb:{avgs.get('embodiment_score',0):>4.1f}  "
              f"→ {avgs.get('top_concept','')[:30]}")

    print(f"\n── COMPOSITE SCORE LEADERS ───────────────────────────────────")
    print(f"  Highest weighted composite across all 7 dimensions")
    top_composite = sorted(
        combo_avgs.items(),
        key=lambda x: x[1].get("composite", 0),
        reverse=True
    )[:20]
    for i, (combo, avgs) in enumerate(top_composite):
        print(f"  {i+1:>2}. {combo:<33} "
              f"comp:{avgs.get('composite',0):>5.2f}  "
              f"→ {avgs.get('top_concept','')[:25]}")

    print(f"\n── PRIMITIVE PERFORMANCE ─────────────────────────────────────")
    print(f"  Which primitives produce the highest meaning scores?")
    prim_scores: dict[str, list] = defaultdict(list)
    for combo, avgs in combo_avgs.items():
        primitive = combo.split(" × ")[0]
        prim_scores[primitive].append(avgs.get("meaning_score", 0))
    prim_avgs = {
        p: round(sum(v)/len(v), 2)
        for p, v in prim_scores.items()
    }
    for prim, avg in sorted(prim_avgs.items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(avg) + "░" * (10 - int(avg))
        print(f"  {prim:<12} [{bar}] {avg:.2f}")

    print(f"\n── EMOTION PERFORMANCE ───────────────────────────────────────")
    print(f"  Which emotions produce the highest meaning scores?")
    emo_scores: dict[str, list] = defaultdict(list)
    for combo, avgs in combo_avgs.items():
        emotion = combo.split(" × ")[1]
        emo_scores[emotion].append(avgs.get("meaning_score", 0))
    emo_avgs = {
        e: round(sum(v)/len(v), 2)
        for e, v in emo_scores.items()
    }
    for emo, avg in sorted(emo_avgs.items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(avg) + "░" * (10 - int(avg))
        print(f"  {emo:<25} [{bar}] {avg:.2f}")

    print(f"\n── CROSS-MODEL CONSISTENCY ───────────────────────────────────")
    print(f"  Do all models agree on high-scoring combinations?")
    for combo, avgs in top_composite[:10]:
        model_scores = []
        for model_def in MODELS:
            model_results = [
                r for r in results
                if r.model_name == model_def["name"]
                and r.combo == combo
                and not r.parse_failed
            ]
            if model_results:
                avg = sum(r.composite_score
                          for r in model_results) / len(model_results)
                model_scores.append(f"{model_def['short']}:{avg:.1f}")
        print(f"  {combo:<35} {' '.join(model_scores)}")

    print(f"\n── WHAT THIS TELLS US ────────────────────────────────────────")
    print(f"  High meaning + high universality = SRM Layer 1 candidates")
    print(f"  High novelty + high meaning = surprising compounds worth")
    print(f"  investigating further")
    print(f"  High embodiment = physically grounded concepts —")
    print(f"  strongest membrane activation expected")
    print(f"  Low clarity + high meaning = ambiguous richness —")
    print(f"  multiple valid interpretations")

# ── persistence ───────────────────────────────────────────────────────────────

def save_results(results: list[ScoredCombo],
                 base="semantic_primitives/results_exp_19"):
    if not results:
        return
    Path("semantic_primitives").mkdir(exist_ok=True)

    json_path = Path(f"{base}.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = Path(f"{base}.csv")
    fieldnames = list(asdict(results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # save ranked summary
    summary_path = Path(f"{base}_summary.json")
    combo_aggregates: dict[str, dict] = defaultdict(
        lambda: defaultdict(list))
    for r in results:
        if not r.parse_failed:
            for dim in SCORE_DIMENSIONS:
                combo_aggregates[r.combo][dim].append(
                    getattr(r, dim))
            combo_aggregates[r.combo]["composite"].append(
                r.composite_score)
            combo_aggregates[r.combo]["concepts"].append(
                r.composite_concept)

    ranked = []
    for combo, data in combo_aggregates.items():
        primitive, emotion = combo.split(" × ")
        entry = {
            "combo": combo,
            "primitive": primitive,
            "emotion": emotion,
        }
        for dim in SCORE_DIMENSIONS + ["composite"]:
            if dim in data:
                entry[f"avg_{dim}"] = round(
                    sum(data[dim])/len(data[dim]), 2)
        entry["top_concept"] = (
            max(set(data["concepts"]),
                key=data["concepts"].count)
            if data["concepts"] else ""
        )
        ranked.append(entry)

    ranked.sort(key=lambda x: x.get("avg_composite", 0),
                reverse=True)

    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_combos": len(ranked),
            "top_20_by_composite": ranked[:20],
            "top_20_by_meaning": sorted(
                ranked,
                key=lambda x: x.get("avg_meaning_score", 0),
                reverse=True)[:20],
            "top_20_by_universality": sorted(
                ranked,
                key=lambda x: x.get("avg_universality_score", 0),
                reverse=True)[:20],
            "top_20_by_embodiment": sorted(
                ranked,
                key=lambda x: x.get("avg_embodiment_score", 0),
                reverse=True)[:20],
        }, f, indent=2)

    # update living primitive summary
    psummary_path = Path(
        "semantic_primitives/primitive_summary.json")
    existing = {}
    if psummary_path.exists():
        with open(psummary_path) as f:
            existing = json.load(f)

    existing["exp_19"] = {
        "timestamp":    datetime.now().isoformat(),
        "total_combos": len(ranked),
        "top_combo":    ranked[0]["combo"] if ranked else "",
        "top_concept":  ranked[0].get("top_concept", "") if ranked else "",
    }
    with open(psummary_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n── SAVED ────────────────────────────────────────────────────")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}  ← ranked summary")
    print(f"  primitive_summary.json  (updated)")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    run_experiment(host)