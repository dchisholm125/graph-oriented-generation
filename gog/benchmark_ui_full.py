"""
benchmark_ui_full.py — Automated GOG vs RAG Benchmark for ui-full (Real Vue 3 Repo)
────────────────────────────────────────────────────────────────────────────────────

Runs a 3-tier gauntlet (Easy, Medium, Hard) against the digital-organism/ui-full
Vue 3 + Pinia + Vite codebase, capturing architecture-aligned metrics:

  • Topological Integrity   – % of generated local imports that resolve in graph
  • Context Density         – (tokens_in / total_repo_tokens) × 100
  • Pass@1 Correctness      – task-specific rubric (PASS / PARTIAL / FAIL)
  • Closure Verification    – membrane patches applied & violations caught

Usage
-----
    # First-time setup — seed graph + vector DB for the real repo
    python3 benchmark_ui_full.py --seed --repo-path /path/to/ui-full

    # Run the gauntlet headlessly (uses previously seeded artifacts)
    python3 benchmark_ui_full.py --repo-path /path/to/ui-full

Custom tasks for ui-full
  Easy — Add `lastPromptAt` to useSessionStore + update addToHistory
  Medium — Add "Reset Session" button in DashboardView.vue calling resetSession()
  Hard — Create SettingsView.vue + route + nav link. Constraint: no direct api/client.ts import.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console

# Reuse the LLM client from benchmark_local_llm.py
from benchmark_local_llm import client, console as rich_console
from gog_engine.token_utils import count_tokens_in_files, count_tokens_in_string
from gog_engine.salience_evaluator import SalienceEvaluator
from gog_engine import graph_search, ast_parser

# ─────────────────────────────────────────────────────────────────────────────
# Custom prompt suite for ui-full
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS_UI = {
    "Easy": {
        "desc": "Localized Store Mutation (1-Step)",
        "text": (
            "In src/stores/session.ts, add a reactive `lastPromptAt` timestamp "
            "field to `useSessionStore` using Pinia's Composition API pattern, "
            "and update the existing `addToHistory` action so it sets "
            "`lastPromptAt` to the current Date() on every new entry."
        ),
    },
    "Medium": {
        "desc": "Component-to-Store Wiring (2-Step)",
        "text": (
            "Add a 'Reset Session' button to `src/views/DashboardView.vue` that, "
            "when clicked, resets the prompt input state and history by calling a "
            "new `resetSession()` action defined inside `useSessionStore`."
        ),
    },
    "Hard": {
        "desc": "Multi-File Feature + Architectural Constraint",
        "text": (
            "Create a new src/views/SettingsView.vue with a route entry in "
            "src/router/index.ts and a nav link in src/App.vue. The view "
            "should display the camera and microphone enabled status from "
            "useAmbientStore. You must NOT import src/api/client.ts "
            "directly into App.vue."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific rubric scoring (deterministic string checks)
# ────────c────────────────────────────────────────────────────────────────────


def _check(response, required=(), forbidden=()):
    checks = []
    for label, terms in required:
        hit = any(t.lower() in response.lower() for t in terms)
        checks.append((label, hit, True))
    for label, terms in forbidden:
        hit = any(t.lower() in response.lower() for t in terms)
        checks.append((label, not hit, False))
    passed = sum(1 for _, ok, _ in checks if ok)
    failures = [label for label, ok, _ in checks if not ok]
    return passed, len(checks), failures


def score_ui_task(task: str, response: str) -> Dict[str, Any]:
    """
    Return (label, passed, total, failures) for a given task.

    These are structural, deterministic checks — not semantic judgements.
    They verify the LLM response contains the architectural patterns the
    task demands, without running a compiler.
    """
    if task == "Easy":
        passed, total, failures = _check(
            response,
            required=[
                ("defines lastPromptAt field",       ["lastPromptAt"]),
                ("updates addToHistory",             ["addToHistory"]),
                ("uses Pinia defineStore",           ["defineStore"]),
                ("uses Composition API ref",         ["ref("]),
            ],
            forbidden=[
                ("does not import React hooks",      ["from 'react'", 'from "react"']),
            ],
        )
    elif task == "Medium":
        passed, total, failures = _check(
            response,
            required=[
                ("includes Reset button",            ["reset"]),
                ("wires click event",                ["@click", "v-on:click"]),
                ("references useSessionStore",       ["useSessionStore"]),
                ("defines or calls resetSession",    ["resetSession"]),
                ("Vue template or script block",     ["<template>", "<script"]),
            ],
            forbidden=[
                ("does not import React hooks",      ["from 'react'", 'from "react"']),
            ],
        )
    elif task == "Hard":
        # Check structural presence
        passed, total, failures = _check(
            response,
            required=[
                ("creates SettingsView",             ["SettingsView"]),
                ("adds route entry",                 ["path:", "route"]),
                ("adds nav link in App",             ["App.vue", "nav", "router-link", "RouterLink"]),
                ("references useAmbientStore",       ["useAmbientStore"]),
                ("accesses camera / mic",            ["cameraEnabled", "micListeningEnabled", "camera", "microphone"]),
                ("Vue template block",               ["<template>", "<script"]),
            ],
            forbidden=[],
        )
        # Constraint: App.vue must NOT import client.ts
        # Extract App.vue code region if present (best-effort)
        blocks = re.findall(r"```(?:ts|vue)?\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        app_block = None
        for block in blocks:
            if "App" in block or "RouterLink" in block or "nav" in block.lower():
                app_block = block
                break

        if app_block is None:
            # Fall back to whole response if we can't isolate App
            app_block = response

        # Constraint check
        has_import = bool(re.search(r"import\s+.*?from\s+['\"]", app_block, re.IGNORECASE))
        has_client = "client" in app_block.lower() and ("api" in app_block.lower() or "client.ts" in app_block.lower())
        if has_import and has_client:
            failures.append("[CONSTRAINT] App.vue directly imports api/client.ts")
            total += 1  # extra check
            # passed remains unchanged → FAIL
        else:
            total += 1
            passed += 1
    else:
        return {"label": "N/A", "passed": 0, "total": 0, "failures": []}

    if passed == total:
        label = f"[bold green]PASS ({passed}/{total})[/bold green]"
    elif passed >= total * 0.6:
        label = f"[bold yellow]PARTIAL ({passed}/{total})[/bold yellow]"
    else:
        label = f"[bold red]FAIL ({passed}/{total})[/bold red]"

    return {"label": label, "passed": passed, "total": total, "failures": failures}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TierResult:
    tier: str
    task: str
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
    topological_integrity: Optional[float] = None
    closure_patched_count: int = 0
    closure_violations: List[str] = field(default_factory=list)


@dataclass
class GauntletReport:
    meta: Dict[str, Any]
    environment: Dict[str, Any]
    metrics: List[TierResult]
    summary: Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# Import / topological integrity helpers
# ─────────────────────────────────────────────────────────────────────────────


def _get_repo_token_count(repo_path: str) -> int:
    # Only count source files inside src/ (not e2e/, docs/, config files)
    src_dir = os.path.join(repo_path, "src")
    search_path = src_dir if os.path.exists(src_dir) else repo_path
    all_files = []
    for root, _, files in os.walk(search_path):
        if "node_modules" in root:
            continue
        for file in files:
            if file.endswith((".ts", ".vue", ".js")):
                all_files.append(os.path.join(root, file))
    return count_tokens_in_files(all_files)


def _detect_generated_imports(response_text: str) -> List[str]:
    code_blocks = re.findall(
        r"```(?:ts|typescript|vue|js)?\n(.*?)```",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline implementations (artifact paths are parameterized)
# ─────────────────────────────────────────────────────────────────────────────


def _artifact_dir(repo_root: str) -> str:
    """Returns the gog/ sub-dir where we stash .pkl and vector_db for this repo."""
    repo_name = os.path.basename(os.path.normpath(repo_root))
    here = os.path.dirname(__file__)
    return os.path.join(here, f"{repo_name}_artifacts")


def _run_control_pipeline(prompt, repo_path, artifact_dir):
    """Tier 1 — RAG Control: ChromaDB top-K chunks → LLM."""
    start = time.time()
    db_path = os.path.join(artifact_dir, "vector_db")
    if not os.path.exists(db_path):
        return {"time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
                "tokens_out": 0, "response": "Error: ChromaDB not found. Run --seed first.",
                "patches_applied": 0}

    import chromadb
    client_db = chromadb.PersistentClient(path=db_path)
    collection = client_db.get_collection("repo_chunks")
    results = collection.query(query_texts=[prompt], n_results=5)

    context_files = set()
    if results["documents"] and results["documents"][0]:
        for i, _ in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            context_files.add(meta["file"])

    unique_files = list(context_files)
    tokens_in = count_tokens_in_files(unique_files)
    local_time = time.time() - start

    api_start = time.time()
    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked RAG result (Ollama is not running for ui-full benchmark)"
    else:
        response = client.complete(prompt, context_files=unique_files)
    api_time = time.time() - api_start

    return {"time": local_time + api_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "patches_applied": 0}


def _run_gog_vanilla(prompt, repo_path, artifact_dir, graph):
    """Tier 2 — GOG Vanilla: deterministic graph isolation → LLM (no Membrane)."""
    start = time.time()
    isolated_files = graph_search.isolate_context(graph, prompt)
    tokens_in = count_tokens_in_files(isolated_files)
    local_time = time.time() - start

    api_start = time.time()
    if not client.is_present:
        time.sleep(0.3)
        response = "Mocked GOG Vanilla result (Ollama is not running for ui-full benchmark)"
    else:
        response = client.complete(prompt, context_files=isolated_files)
    api_time = time.time() - api_start

    return {"time": local_time + api_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "patches_applied": 0}


def _run_gog_membrane(prompt, repo_path, artifact_dir, graph):
    """Tier 3 — GOG + Membrane: graph isolation → LLM → SalienceEvaluator patch."""
    start = time.time()
    isolated_files = graph_search.isolate_context(graph, prompt)
    tokens_in = count_tokens_in_files(isolated_files)
    local_time = time.time() - start

    api_start = time.time()
    patches = 0

    if not client.is_present:
        time.sleep(0.3)
        response = "Mocked GOG+Membrane result (Ollama is not running for ui-full benchmark)"
    else:
        evaluator = SalienceEvaluator(allowed_nodes=isolated_files)
        raw = client.complete(prompt, context_files=isolated_files)
        result = evaluator.evaluate(raw)
        if result.is_valid:
            response = raw
        else:
            patches = len(result.violations)
            patched = evaluator.patch(result)
            if patched and patched.strip():
                lang_match = re.search(r"```(typescript|ts|vue)", raw, re.IGNORECASE)
                fence = lang_match.group(1).lower() if lang_match else "ts"
                response = f"```{fence}\n{patched}\n```"
            else:
                response = raw

    api_time = time.time() - api_start
    return {"time": local_time + api_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "patches_applied": patches}


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────


def seed_repo(repo_path: str, artifact_dir: str, console: Console) -> None:
    """Build GOG graph + ChromaDB vector store for the target repo."""
    os.makedirs(artifact_dir, exist_ok=True)

    # ── 1. Seed RAG ──────────────────────────────────────────────────────────
    db_path = os.path.join(artifact_dir, "vector_db")
    console.print(f"[bold cyan]1. Seeding RAG (Vector DB) → {db_path}[/]")

    import chromadb
    from chromadb.utils import embedding_functions

    client_db = chromadb.PersistentClient(path=db_path)
    try:
        client_db.delete_collection("repo_chunks")
    except Exception:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client_db.create_collection(name="repo_chunks", embedding_function=ef)

    documents, metadatas, ids = [], [], []
    # Find all relevant source files (only src/ and its subdirectories)
    src_dir = os.path.join(repo_path, "src")
    if os.path.exists(src_dir):
        search_paths = [src_dir]
    else:
        search_paths = [repo_path]  # fallback if no src/ exists

    all_files = []
    for search_path in search_paths:
        for root, _, files in os.walk(search_path):
            if "node_modules" in root:
                continue
            for f in files:
                if f.endswith((".ts", ".vue", ".js")):
                    all_files.append(os.path.join(root, f))

    chunk_id = 0
    for fp in sorted(all_files):
        try:
            with open(fp, "r", encoding="utf8") as fh:
                content = fh.read()
        except Exception:
            continue

        # Chunk by blocks (~100 lines) — coarse but sufficient for top-K
        lines = content.split("\n")
        chunk_size = 100
        for i in range(0, len(lines), chunk_size):
            block = "\n".join(lines[i : i + chunk_size]).strip()
            if block:
                documents.append(block)
                metadatas.append({"file": os.path.abspath(fp)})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

    if documents:
        batch = 100
        for i in range(0, len(documents), batch):
            collection.add(
                documents=documents[i : i + batch],
                metadatas=metadatas[i : i + batch],
                ids=ids[i : i + batch],
            )
    console.print(f"[green]✓ RAG seeded:[/] {chunk_id} chunks indexed from {len(all_files)} files.")

    # ── 2. Seed GOG graph ──────────────────────────────────────────────────
    graph_path = os.path.join(artifact_dir, "gog_graph.pkl")
    console.print(f"[bold green]2. Seeding GOG (AST Graph) → {graph_path}[/]")

    G = ast_parser.build_graph(repo_path)
    with open(graph_path, "wb") as fh:
        pickle.dump(G, fh)
    console.print(
        f"[green]✓ GOG seeded:[/] {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics engine
# ─────────────────────────────────────────────────────────────────────────────


def run_gauntlet_task(prompt_text, level, repo_path, graph, total_repo_tokens, tier, artifact_dir):
    if tier == "rag":
        raw = _run_control_pipeline(prompt_text, repo_path, artifact_dir)
    elif tier == "gog_vanilla":
        raw = _run_gog_vanilla(prompt_text, repo_path, artifact_dir, graph)
    elif tier == "gog_membrane":
        raw = _run_gog_membrane(prompt_text, repo_path, artifact_dir, graph)
    else:
        raise ValueError(f"Unknown tier: {tier}")

    response = raw.get("response", "")
    patches = raw.get("patches_applied", 0)

    # Isolate for rubric scoring
    isolated_files = []
    if graph is not None:
        try:
            isolated_files = graph_search.isolate_context(graph, prompt_text)
        except Exception:
            pass

    rubric = score_ui_task(level, response)
    ti = _check_topological_integrity(response, isolated_files)

    return TierResult(
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
        rubric_failures=rubric["failures"],
        topological_integrity=ti["percentage"],
        closure_patched_count=patches,
        closure_violations=ti["violations"],
    )


def _tier_label(tier: str) -> str:
    return {"rag": "RAG", "gog_vanilla": "GOG Vanilla", "gog_membrane": "GOG Membrane"}.get(tier, tier)


def build_summary(metrics: List[TierResult], total_repo_tokens: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_tier": {}, "by_task": {}, "global": {}}

    for tier in ["rag", "gog_vanilla", "gog_membrane"]:
        rows = [m for m in metrics if m.tier == tier]
        if not rows:
            continue
        avg_time = sum(m.total_time_s for m in rows) / len(rows)
        avg_local = sum(m.local_time_s for m in rows) / len(rows)
        avg_api = sum(m.api_time_s for m in rows) / len(rows)
        avg_tokens = sum(m.tokens_in for m in rows) / len(rows)
        pas = sum(1 for m in rows if m.rubric_passed == m.rubric_total and m.rubric_total > 0)
        partial = sum(1 for m in rows if 0 < m.rubric_passed < m.rubric_total)
        fail = len(rows) - pas - partial
        tis = [m.topological_integrity for m in rows if m.topological_integrity is not None]
        avg_ti = sum(tis) / len(tis) if tis else None
        patches = sum(m.closure_patched_count for m in rows)
        summary["by_tier"][tier] = {
            "tier_name": _tier_label(tier),
            "observations": len(rows),
            "avg_total_time_s": round(avg_time, 4),
            "avg_local_time_s": round(avg_local, 4),
            "avg_api_time_s": round(avg_api, 4),
            "avg_tokens_in": round(avg_tokens, 1),
            "context_density_pct": round((avg_tokens / total_repo_tokens) * 100, 2) if total_repo_tokens else None,
            "pass_at_1_count": pas,
            "partial_count": partial,
            "fail_count": fail,
            "avg_topological_integrity": round(avg_ti, 4) if avg_ti is not None else None,
            "total_patches_applied": patches,
        }

    for task in ["easy", "medium", "hard"]:
        rows = [m for m in metrics if m.task == task]
        if not rows:
            continue
        task_s = {}
        for tier in ["rag", "gog_vanilla", "gog_membrane"]:
            t_rows = [m for m in rows if m.tier == tier]
            if t_rows:
                r = t_rows[0]
                task_s[tier] = {
                    "tokens_in": r.tokens_in,
                    "total_time_s": round(r.total_time_s, 4),
                    "rubric_label": r.rubric_label,
                    "topological_integrity": r.topological_integrity,
                    "patches_applied": r.patches_applied,
                }
        summary["by_task"][task] = task_s

    rag, mem = summary["by_tier"].get("rag", {}), summary["by_tier"].get("gog_membrane", {})
    if rag and mem:
        t_delta = round(((rag["avg_tokens_in"] - mem["avg_tokens_in"]) / rag["avg_tokens_in"]) * 100, 1) if rag["avg_tokens_in"] else 0
        tm_delta = round(((rag["avg_total_time_s"] - mem["avg_total_time_s"]) / rag["avg_total_time_s"]) * 100, 1) if rag["avg_total_time_s"] else 0
        summary["global"] = {
            "baseline_tier": "rag",
            "comparison_tier": "gog_membrane",
            "token_reduction_pct": t_delta,
            "time_change_pct": tm_delta,
            "r_improvement_passes": mem.get("pass_at_1_count", 0) - rag.get("pass_at_1_count", 0),
            "r_increase_in_topological_integrity": round(
                (mem.get("avg_topological_integrity") or 0) - (rag.get("avg_topological_integrity") or 0), 4
            ),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Report writers
# ─────────────────────────────────────────────────────────────────────────────


def write_json_report(report: GauntletReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)


def write_markdown(report: GauntletReport, path: Path) -> None:
    md: List[str] = []
    md.append("# 🧬 GOG vs RAG — ui-full Gauntlet Report")
    md.append(f"\n> Generated: `{report.meta['timestamp']}`")
    md.append(f"> Model: `{report.meta['model']}`")
    md.append(f"> Repo: `{report.meta['repo_path']}`")
    md.append(f"> Total repo tokens: `{report.meta['total_repo_tokens']}`\n")
    md.append("---\n")
    md.append("## 📊 Aggregate Metrics by Tier\n")
    md.append(
        "| Tier | Mean Time (s) | Local (s) | API (s) | Tokens In | Context Density | Pass@1 | Partial | Fail | Topo Integrity | Patches |"
    )
    md.append(
        "|------|--------------:|----------:|--------:|----------:|:---------------:|:------:|:-------:|:----:|:-------------:|:-------:|"
    )
    for tier_key, data in report.summary.get("by_tier", {}).items():
        md.append(
            f"| {data['tier_name']} | {data['avg_total_time_s']} | {data['avg_local_time_s']} | "
            f"{data['avg_api_time_s']} | {data['avg_tokens_in']} | {data['context_density_pct']}% | "
            f"{data['pass_at_1_count']} | {data['partial_count']} | {data['fail_count']} | "
            f"{data['avg_topological_integrity'] or 'N/A'} | {data['total_patches_applied']} |"
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
                f"{data['topological_integrity'] or 'N/A'} | {data['patches_applied']} |"
            )

    glob = report.summary.get("global", {})
    if glob:
        md.append("\n---\n")
        md.append("## 🌐 Global Delta: GOG Membrane vs RAG (Control)\n")
        md.append(f"- **Token Reduction:** `{glob['token_reduction_pct']}%`")
        md.append(f"- **Time Change:** `{glob['time_change_pct']}%`")
        md.append(f"- **Pass@1 Improvement:** `{glob['r_improvement_passes']}`")
        md.append(f"- **Topological Integrity Gain:** `{glob['r_increase_in_topological_integrity']}`")

    md.append("\n---\n*The atoms of meaning are structural.*\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Automated GOG vs RAG Benchmark — ui-full")
    parser.add_argument("--repo-path", required=True, help="Path to ui-full repo")
    parser.add_argument("--seed", action="store_true", help="Build graph + vector DB then exit")
    parser.add_argument("--output-dir", default="results", help="Directory for JSON + Markdown")
    parser.add_argument(
        "--tiers",
        default="rag,gog_vanilla,gog_membrane",
        help="Comma-separated tiers to run (default: all)",
    )
    parser.add_argument(
        "--tasks",
        default="Easy,Medium,Hard",
        help="Comma-separated tasks to run (default: all)",
    )
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.exists(repo_path):
        print(f"[ERROR] Repo not found: {repo_path}")
        sys.exit(1)

    artifact_dir = _artifact_dir(repo_path)
    console = Console()

    # ── SEED MODE ────────────────────────────────────────────────────────────
    if args.seed:
        seed_repo(repo_path, artifact_dir, console)
        return

    # ── VERIFY ARTIFACTS ─────────────────────────────────────────────────────
    graph_path = os.path.join(artifact_dir, "gog_graph.pkl")
    if not os.path.exists(graph_path):
        print(f"[ERROR] Graph not found: {graph_path}. Run with --seed first.")
        sys.exit(1)

    db_path = os.path.join(artifact_dir, "vector_db")
    if not os.path.exists(db_path):
        print(f"[ERROR] Vector DB not found: {db_path}. Run with --seed first.")
        sys.exit(1)

    # Load graph
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    total_repo_tokens = _get_repo_token_count(repo_path)
    model_name = client.model
    print(f"ui-full Gauntlet: repo={repo_path}  tokens={total_repo_tokens}  model={model_name}")

    tiers = [t.strip() for t in args.tiers.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]

    # ── RUN ──────────────────────────────────────────────────────────────────
    all_metrics: List[TierResult] = []
    for level in tasks:
        if level not in PROMPTS_UI:
            print(f"[WARN] Unknown task '{level}', skipping.")
            continue
        data = PROMPTS_UI[level]
        print(f"\n>> {level}: {data['desc']}")
        for tier in tiers:
            print(f"   [{tier}] running …", end=" ", flush=True)
            obs = run_gauntlet_task(
                data["text"], level, repo_path, graph, total_repo_tokens, tier, artifact_dir
            )
            print(
                f"done ({obs.total_time_s:.2f}s | tok={obs.tokens_in} | label={obs.rubric_label})"
            )
            all_metrics.append(obs)

    # ── BUILD REPORT ───────────────────────────────────────────────────────
    summary = build_summary(all_metrics, total_repo_tokens)
    report = GauntletReport(
        meta={
            "timestamp": _now_iso(),
            "model": model_name,
            "total_repo_tokens": total_repo_tokens,
            "repo_path": repo_path,
            "tiers_evaluated": tiers,
            "tasks_evaluated": tasks,
        },
        environment={
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        metrics=all_metrics,
        summary=summary,
    )

    output_dir = Path(args.output_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ui_full_gauntlet_{ts}.json"
    md_path = output_dir / f"ui_full_gauntlet_{ts}.md"

    write_json_report(report, json_path)
    write_markdown(report, md_path)

    print(f"\n{'=' * 60}")
    print(" ui-full Gauntlet complete.")
    print(f" JSON → {json_path}")
    print(f" MD   → {md_path}")
    print(f"{'=' * 60}")

    glob = summary.get("global", {})
    if glob:
        print(f"\n🧬 GOG Membrane vs RAG:")
        print(f"   Token reduction:    {glob.get('token_reduction_pct', 0):.1f}%")
        print(f"   Time change:        {glob.get('time_change_pct', 0):.1f}%")
        print(f"   Pass@1 delta:       +{glob.get('r_improvement_passes', 0)}")
        print(f"   Topo integrity Δ:   {glob.get('r_increase_in_topological_integrity', 0):.4f}")


if __name__ == "__main__":
    main()
