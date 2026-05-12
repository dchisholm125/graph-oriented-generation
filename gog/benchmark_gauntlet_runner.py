"""
benchmark_gauntlet_runner.py — Automated GOG vs RAG Benchmark Runner
─────────────────────────────────────────────────────────────────────
A headless, reproducible 3-tier benchmark that compares Vector RAG
(control), GOG Vanilla (deterministic graph isolation), and GOG +
SalienceEvaluator Membrane (topological patching).

Architecture-Aligned Metrics
----------------------------
| Metric                | Why it matters for the GOG thesis                |
|-----------------------|---------------------------------------------------|
| Topological Integrity   | % of generated local imports that exist in graph  |
| Context Density       | Required tokens / total repo tokens (%)           |
| Pass@1 Correctness   | Rubric score: PASS / PARTIAL / FAIL               |
| Closure Verification  | Membrane patches applied & violations caught       |
| Execution Time        | Local compute vs LLM generation split             |

Usage
-----
    python3 benchmark_gauntlet_runner.py [--output-dir results/]

Requires:
    - gog/target_repo  (run generate_dummy_repo.py)
    - gog/vector_db    (run seed_RAG_and_GOG.py)
    - gog/gog_graph.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure gog_engine (sibling to gog/) is on the path ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

from benchmark_local_llm import (
    client,
    console as console_instance,
    PROMPTS,
    run_control_pipeline,
    run_srm_pipeline_vanilla,
    run_srm_pipeline_membrane,
    score_response,
)
from gog_engine.token_utils import count_tokens_in_files, count_tokens_in_string
from gog_engine.salience_evaluator import SalienceEvaluator, EvaluationResult
from gog_engine import graph_search

# ─────────────────────────────────────────────────────────────────────────────
# Data structures for one task × tier observation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TierResult:
    """Raw pipeline result augmented with derived metrics."""
    tier: str                         # "rag" | "gog_vanilla" | "gog_membrane"
    task: str                         # "easy" | "medium" | "hard"
    total_time_s: float
    local_time_s: float
    api_time_s: float
    tokens_in: int
    tokens_out: int
    response: str = ""
    patches_applied: int = 0
    rubric_label: str = "N/A"
    rubric_passed: int = 0
    rubric_total: int = 0
    rubric_failures: List[str] = field(default_factory=list)
    # Derived metrics (populated after run)
    topological_integrity: Optional[float] = None   # 0.0-1.0
    closure_patched_count: int = 0
    closure_violations: List[str] = field(default_factory=list)


@dataclass
class GauntletReport:
    """Top-level report for the full gauntlet."""
    meta: Dict[str, Any]
    environment: Dict[str, Any]
    metrics: List[TierResult]
    summary: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_repo_token_count(repo_path: str) -> int:
    """Total token count across all tracked source files in a repo."""
    all_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith((".ts", ".vue", ".js", ".tsx")):
                all_files.append(os.path.join(root, file))
    return count_tokens_in_files(all_files)


def _detect_generated_imports(response_text: str) -> List[str]:
    """Extract local import specifiers from a raw LLM response (best effort)."""
    code_blocks = re.findall(r"```(?:ts|typescript|vue)?\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    code = "\n".join(block.strip() for block in code_blocks) if code_blocks else response_text

    imports: List[str] = []
    for line in code.splitlines():
        m = re.search(r"import\s+.*?from\s+['\"](.+?)['\"]", line)
        if m:
            specifier = m.group(1)
            if specifier.startswith(".") or specifier.startswith("/"):
                imports.append(specifier)
    return imports


def _check_topological_integrity(response_text: str, allowed_nodes: List[str]) -> Dict[str, Any]:
    """
    Check what % of generated local imports resolve to files inside the allowed
    graph boundary.  Returns a dict with percentage and violation list.
    """
    generated_imports = _detect_generated_imports(response_text)
    if not generated_imports:
        return {"percentage": 1.0, "violation_count": 0, "violations": []}

    allowed_basenames = {os.path.basename(p) for p in allowed_nodes}
    violations: List[str] = []

    for specifier in generated_imports:
        raw_basename = os.path.basename(specifier)
        if not raw_basename:
            continue
        candidates = [raw_basename]
        if not os.path.splitext(raw_basename)[1]:
            candidates = [f"{raw_basename}.ts", f"{raw_basename}.vue", f"{raw_basename}/index.ts"]
        if not any(c in allowed_basenames for c in candidates):
            violations.append(specifier)

    pct = (len(generated_imports) - len(violations)) / len(generated_imports) if generated_imports else 1.0
    return {"percentage": pct, "violation_count": len(violations), "violations": violations}


def _score_rubric(level: str, response_text: str, isolated_count: Optional[int] = None) -> Dict[str, Any]:
    """Wraps benchmark_local_llm.score_response into a structured dict."""
    label, passed, total, failures = score_response(level, response_text, isolated_count=isolated_count)
    return {
        "label": label,
        "passed": passed,
        "total": total,
        "failures": failures,
    }


def _pass_rate_from_label(label: str) -> int:
    """Heuristic: return numeric pass state from rich-formatted label string."""
    if "PASS" in label:
        return 1
    elif "PARTIAL" in label:
        return 0
    return -1  # FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline wrappers — one per tier
# ────────c────────────────────────────────────────────────────────────────────


def _tier_label(tier: str) -> str:
    return {"rag": "RAG", "gog_vanilla": "GOG Vanilla", "gog_membrane": "GOG Membrane"}.get(tier, tier)


def run_gauntlet_task(
    prompt_text: str,
    level: str,
    repo_path: str,
    graph,
    total_repo_tokens: int,
    tier: str,
) -> TierResult:
    """Run a single (prompt, tier) observation and return a TierResult."""
    if tier == "rag":
        raw = run_control_pipeline(prompt_text, repo_path)
    elif tier == "gog_vanilla":
        raw = run_srm_pipeline_vanilla(prompt_text, repo_path)
    elif tier == "gog_membrane":
        raw = run_srm_pipeline_membrane(prompt_text, repo_path)
    else:
        raise ValueError(f"Unknown tier: {tier}")

    response = raw.get("response", "")
    patches = raw.get("patches_applied", 0)

    # ── Rubric scoring ──────────────────────────────────────────────────────
    isolated_files = []
    if graph is not None:
        try:
            isolated_files = graph_search.isolate_context(graph, prompt_text)
        except Exception:
            pass

    isolated_count = len(isolated_files) if isolated_files else None
    rubric = _score_rubric(level, response, isolated_count=isolated_count)

    # ── Topological integrity (closure verification) ──────────────────────────
    ti_data = _check_topological_integrity(response, isolated_files)

    result = TierResult(
        tier=tier,
        task=level.lower(),
        total_time_s=raw.get("time", 0.0),
        local_time_s=raw.get("local_time", 0.0),
        api_time_s=raw.get("api_time", 0.0),
        tokens_in=raw.get("tokens_in", 0),
        tokens_out=raw.get("tokens_out", 150),
        response=response,
        patches_applied=patches,
        rubric_label=rubric["label"],
        rubric_passed=rubric["passed"],
        rubric_total=rubric["total"],
        rubric_failures=[str(f) for f in rubric["failures"]],
        topological_integrity=ti_data["percentage"],
        closure_patched_count=patches,
        closure_violations=ti_data["violations"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Summary computation
# ─────────────────────────────────────────────────────────────────────────────


def build_summary(metrics: List[TierResult], total_repo_tokens: int) -> Dict[str, Any]:
    """Aggregate raw observations into architecture-aligned high-level KPIs."""
    summary: Dict[str, Any] = {
        "by_tier": {},
        "by_task": {},
        "global": {},
    }

    for tier in ["rag", "gog_vanilla", "gog_membrane"]:
        tier_rows = [m for m in metrics if m.tier == tier]
        if not tier_rows:
            continue
        avg_time = sum(m.total_time_s for m in tier_rows) / len(tier_rows)
        avg_local = sum(m.local_time_s for m in tier_rows) / len(tier_rows)
        avg_api = sum(m.api_time_s for m in tier_rows) / len(tier_rows)
        avg_tokens = sum(m.tokens_in for m in tier_rows) / len(tier_rows)
        pass_1 = sum(1 for m in tier_rows if m.rubric_passed == m.rubric_total and m.rubric_total > 0)
        partial = sum(1 for m in tier_rows if 0 < m.rubric_passed < m.rubric_total)
        fail = len(tier_rows) - pass_1 - partial
        ti_vals = [m.topological_integrity for m in tier_rows if m.topological_integrity is not None]
        avg_ti = sum(ti_vals) / len(ti_vals) if ti_vals else None
        patches = sum(m.closure_patched_count for m in tier_rows)
        summary["by_tier"][tier] = {
            "tier_name": _tier_label(tier),
            "observations": len(tier_rows),
            "avg_total_time_s": round(avg_time, 4),
            "avg_local_time_s": round(avg_local, 4),
            "avg_api_time_s": round(avg_api, 4),
            "avg_tokens_in": round(avg_tokens, 1),
            "context_density_pct": round((avg_tokens / total_repo_tokens) * 100, 2) if total_repo_tokens else None,
            "pass_at_1_count": pass_1,
            "partial_count": partial,
            "fail_count": fail,
            "avg_topological_integrity": round(avg_ti, 4) if avg_ti is not None else None,
            "total_patches_applied": patches,
        }

    for task in ["easy", "medium", "hard"]:
        task_rows = [m for m in metrics if m.task == task]
        if not task_rows:
            continue
        # Compare tiers for this task
        task_summary = {}
        for tier in ["rag", "gog_vanilla", "gog_membrane"]:
            tier_task_rows = [m for m in task_rows if m.tier == tier]
            if not tier_task_rows:
                continue
            r = tier_task_rows[0]
            task_summary[tier] = {
                "tokens_in": r.tokens_in,
                "total_time_s": round(r.total_time_s, 4),
                "rubric_label": r.rubric_label,
                "topological_integrity": r.topological_integrity,
                "patches_applied": r.patches_applied,
            }
        summary["by_task"][task] = task_summary

    # ── Global delta: GOG Membrane vs RAG ───────────────────────────────────
    rag = summary["by_tier"].get("rag", {})
    mem = summary["by_tier"].get("gog_membrane", {})
    if rag and mem:
        token_delta = round(((rag["avg_tokens_in"] - mem["avg_tokens_in"]) / rag["avg_tokens_in"]) * 100, 1) if rag["avg_tokens_in"] else 0
        time_delta = round(((rag["avg_total_time_s"] - mem["avg_total_time_s"]) / rag["avg_total_time_s"]) * 100, 1) if rag["avg_total_time_s"] else 0
        summary["global"] = {
            "baseline_tier": "rag",
            "comparison_tier": "gog_membrane",
            "token_reduction_pct": token_delta,
            "time_change_pct": time_delta,
            "r_improvement_passes": mem.get("pass_at_1_count", 0) - rag.get("pass_at_1_count", 0),
            "r_increase_in_topological_integrity": round((mem.get("avg_topological_integrity") or 0) - (rag.get("avg_topological_integrity") or 0), 4),
        }

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────


def write_json_report(report: GauntletReport, path: Path) -> None:
    """Serialize the full report to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


def write_markdown_summary(report: GauntletReport, path: Path) -> None:
    """Write a human-readable Markdown table of the key results."""
    md: List[str] = []
    md.append("# 🧬 GOG vs RAG — Automated Gauntlet Report")
    md.append(f"\n> Generated: `{report.meta['timestamp']}`  ")
    md.append(f"> Model: `{report.meta['model']}`  ")
    md.append(f"> Total repo tokens: `{report.meta['total_repo_tokens']}`\n")

    md.append("---\n")
    md.append("## 📊 Aggregate Metrics by Tier\n")
    md.append("| Tier | Mean Time (s) | Local (s) | API (s) | Tokens In | Context Density | Pass@1 | Partial | Fail | Topo Integrity | Patches |")
    md.append("|------|--------------:|----------:|--------:|----------:|:---------------:|:------:|:-------:|:----:|:-------------:|:-------:|")
    for tier_key, data in report.summary.get("by_tier", {}).items():
        md.append(
            f"| {data['tier_name']} | {data['avg_total_time_s']} | "
            f"{data['avg_local_time_s']} | {data['avg_api_time_s']} | "
            f"{data['avg_tokens_in']} | {data['context_density_pct']}% | "
            f"{data['pass_at_1_count']} | {data['partial_count']} | "
            f"{data['fail_count']} | "
            f"{data['avg_topological_integrity'] or 'N/A'} | "
            f"{data['total_patches_applied']} |"
        )

    md.append("\n---\n")
    md.append("## 🆚 Per-Task Breakdown\n")
    for task, tiers in report.summary.get("by_task", {}).items():
        md.append(f"\n### {task.capitalize()} Task\n")
        md.append("| Tier | Tokens In | Time (s) | Rubric | Topo Integrity | Patches |")
        md.append("|------|----------:|---------:|:------:|:--------------:|:-------:|")
        for tier_key, data in tiers.items():
            md.append(
                f"| {_tier_label(tier_key)} | {data['tokens_in']} | "
                f"{data['total_time_s']} | {data['rubric_label']} | "
                f"{data['topological_integrity'] or 'N/A'} | "
                f"{data['patches_applied']} |"
            )

    glob = report.summary.get("global", {})
    if glob:
        md.append("\n---\n")
        md.append("## 🌐 Global Delta: GOG Membrane vs RAG (Control)\n")
        md.append(f"- **Token Reduction:** `{glob['token_reduction_pct']}%` (fewer tokens sent to LLM)")
        md.append(f"- **Time Change:** `{glob['time_change_pct']}%` (negative = faster)")
        md.append(f"- **Pass@1 Improvement:** `{glob['r_improvement_passes']}` additional passing tasks")
        md.append(f"- **Topological Integrity Gain:** `{glob['r_increase_in_topological_integrity']}`")

    md.append("\n---\n")
    md.append("## 🧪 Raw Observations\n")
    md.append("<details><summary>Expand JSON lines</summary>\n")
    md.append("```json")
    for obs in report.metrics:
        md.append(json.dumps({
            "tier": obs.tier,
            "task": obs.task,
            "total_time_s": obs.total_time_s,
            "tokens_in": obs.tokens_in,
            "rubric_label": obs.rubric_label,
            "topological_integrity": obs.topological_integrity,
            "patches_applied": obs.patches_applied,
            "closure_violations": obs.closure_violations,
        }))
    md.append("```")
    md.append("</details>\n")
    md.append("\n---\n*The atoms of meaning are structural.*\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Automated GOG vs RAG Benchmark Runner")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON + Markdown reports (default: results/)",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Override target repo path (default: gog/target_repo)",
    )
    args = parser.parse_args()

    repo_path = args.repo_path or os.path.join(os.path.dirname(__file__), "target_repo")
    if not os.path.exists(repo_path):
        print(f"[ERROR] target_repo not found at {repo_path}. Run generate_dummy_repo.py first.")
        sys.exit(1)

    # Load graph
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    graph = None
    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    else:
        print(f"[WARNING] Graph not found at {graph_path}. GOG tiers will return errors.")

    total_repo_tokens = _get_repo_token_count(repo_path)
    print(f"Auto-Gauntlet: repo={repo_path}  total_tokens={total_repo_tokens}  model={client.model}")

    # ── Run all tasks × tiers ────────────────────────────────────────────────
    all_metrics: List[TierResult] = []
    for level_key, data in PROMPTS.items():
        prompt_text = data["text"]
        print(f"\n>> {level_key}: {data['desc']}")
        for tier in ["rag", "gog_vanilla", "gog_membrane"]:
            print(f"   [{tier}] running …", end=" ", flush=True)
            observation = run_gauntlet_task(
                prompt_text, level_key, repo_path, graph, total_repo_tokens, tier
            )
            print(
                f"done ({observation.total_time_s:.2f}s | "
                f"tok={observation.tokens_in} | label={observation.rubric_label})"
            )
            all_metrics.append(observation)

    # ── Build report ─────────────────────────────────────────────────────────
    summary = build_summary(all_metrics, total_repo_tokens)
    report = GauntletReport(
        meta={
            "timestamp": _now_iso(),
            "model": client.model,
            "total_repo_tokens": total_repo_tokens,
            "repo_path": os.path.abspath(repo_path),
            "tiers_evaluated": ["rag", "gog_vanilla", "gog_membrane"],
            "tasks_evaluated": list(PROMPTS.keys()),
        },
        environment={
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        metrics=all_metrics,
        summary=summary,
    )

    # ── Export ─────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    json_path = output_dir / f"gog_gauntlet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    md_path = output_dir / f"gog_gauntlet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"

    write_json_report(report, json_path)
    write_markdown_summary(report, md_path)

    print(f"\n{'=' * 60}")
    print(" Gauntlet complete.")
    print(f" JSON → {json_path}")
    print(f" MD   → {md_path}")
    print(f"{'=' * 60}")

    # Print a quick terminal summary for readability
    glob = summary.get("global", {})
    if glob:
        print(f"\n🧬 GOG Membrane vs RAG Summary:")
        print(f"   Token reduction:    {glob.get('token_reduction_pct', 0):.1f}%")
        print(f"   Time change:        {glob.get('time_change_pct', 0):.1f}%")
        print(f"   Pass@1 delta:       +{glob.get('r_improvement_passes', 0)}")
        print(f"   Topo integrity Δ:   {glob.get('r_increase_in_topological_integrity', 0):.4f}")


if __name__ == "__main__":
    main()
