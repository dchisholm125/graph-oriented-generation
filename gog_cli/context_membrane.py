"""Deterministic context membrane for prompt-scoped GOG bundles."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MAX_FILES = 10
PROMPT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
PROMPT_STOPWORDS = {
    "file",
    "files",
    "find",
    "responsible",
    "involved",
    "when",
    "where",
    "that",
    "this",
    "from",
    "into",
}


@dataclass(frozen=True)
class ContextMembraneResult:
    selected_nodes: list[str]
    kept: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    config: dict[str, Any]


def apply_context_membrane(
    *,
    graph,
    repo_root: Path,
    prompt: str,
    candidate_nodes: list[str],
    seed_nodes: list[str],
    explicit_files: list[str],
    identifier_files: list[str],
    max_files: int = DEFAULT_MAX_FILES,
) -> ContextMembraneResult:
    """Score and trim candidate graph nodes before serving them to a reasoner."""
    prompt_terms = _prompt_terms(prompt)
    prompt_asks_for_tests = any(term in prompt_terms for term in {"test", "tests", "spec", "coverage"})
    prompt_asks_for_config = any(term in prompt_terms for term in {"config", "build", "vite", "eslint"})
    seed_set = set(seed_nodes)
    explicit_set = set(explicit_files)
    identifier_set = set(identifier_files)

    scored = []
    for node in sorted(set(candidate_nodes)):
        rel_path = _rel(repo_root, Path(node))
        distance = _nearest_seed_distance(graph, node, seed_set)
        score, reasons, penalties = _score_node(
            rel_path=rel_path,
            node=node,
            prompt_terms=prompt_terms,
            distance=distance,
            seed_set=seed_set,
            explicit_set=explicit_set,
            identifier_set=identifier_set,
            prompt_asks_for_tests=prompt_asks_for_tests,
            prompt_asks_for_config=prompt_asks_for_config,
        )
        scored.append(
            {
                "node": node,
                "path": rel_path,
                "score": score,
                "distance": distance,
                "reasons": reasons,
                "penalties": penalties,
            }
        )

    ranked = sorted(scored, key=lambda item: (-item["score"], item["distance"], item["path"]))
    kept = ranked[:max_files]
    rejected = [
        {
            **item,
            "rejected_reason": "trimmed_by_context_membrane",
        }
        for item in ranked[max_files:]
    ]

    return ContextMembraneResult(
        selected_nodes=[item["node"] for item in kept],
        kept=kept,
        rejected=rejected,
        config={
            "max_files": max_files,
            "prompt_asks_for_tests": prompt_asks_for_tests,
            "prompt_asks_for_config": prompt_asks_for_config,
        },
    )


def _score_node(
    *,
    rel_path: str,
    node: str,
    prompt_terms: set[str],
    distance: int,
    seed_set: set[str],
    explicit_set: set[str],
    identifier_set: set[str],
    prompt_asks_for_tests: bool,
    prompt_asks_for_config: bool,
) -> tuple[int, list[str], list[str]]:
    score = 0
    reasons: list[str] = []
    penalties: list[str] = []
    path_terms = _path_terms(rel_path)

    overlap = sorted(prompt_terms.intersection(path_terms))
    if node in explicit_set:
        score += 100
        reasons.append("explicit_file_match")
    if node in identifier_set:
        score += 60 if len(overlap) >= 2 else 22
        reasons.append("identifier_filename_match")
    if node in seed_set:
        score += 45 if len(overlap) >= 2 else 12
        reasons.append("graph_seed")

    if overlap:
        score += 12 * len(overlap)
        reasons.append(f"prompt_path_overlap:{','.join(overlap[:6])}")

    if distance == 0:
        score += 18
        reasons.append("seed_distance_0")
    elif distance == 1:
        score += 10
        reasons.append("seed_distance_1")
    elif distance == 2:
        score += 3
        reasons.append("seed_distance_2")
    else:
        score -= 8
        penalties.append("distant_from_seed")

    if _is_test_file(rel_path) and not prompt_asks_for_tests:
        score -= 150
        penalties.append("test_file_without_test_prompt")
    if _is_e2e_or_playwright_file(rel_path) and not prompt_asks_for_tests:
        score -= 180
        penalties.append("e2e_file_without_test_prompt")
    if _is_config_file(rel_path) and not prompt_asks_for_config:
        score -= 70
        penalties.append("config_file_without_config_prompt")
    if _is_generated_file(rel_path):
        penalties.append("generated_or_large_api_surface")

    if not reasons:
        reasons.append("graph_neighbor")

    return score, reasons, penalties


def _nearest_seed_distance(graph, node: str, seed_set: set[str]) -> int:
    if node in seed_set:
        return 0
    if not seed_set:
        return 99
    for max_depth in (1, 2, 3):
        queue = deque((seed, 0) for seed in seed_set)
        seen = set(seed_set)
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            neighbors = set(graph.successors(current)) | set(graph.predecessors(current))
            for neighbor in neighbors:
                if neighbor == node:
                    return depth + 1
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
    return 99


def _prompt_terms(prompt: str) -> set[str]:
    return {
        _normalize_term(term)
        for term in PROMPT_TOKEN_RE.findall(prompt)
        if len(term) >= 4 and _normalize_term(term) not in PROMPT_STOPWORDS
    }


def _path_terms(rel_path: str) -> set[str]:
    raw_parts = re.split(r"[/_.-]+", rel_path)
    terms = set()
    for part in raw_parts:
        for subpart in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", part):
            normalized = _normalize_term(subpart)
            if len(normalized) >= 4:
                terms.add(normalized)
    return terms


def _normalize_term(term: str) -> str:
    lowered = term.lower()
    if lowered.endswith("ing") and len(lowered) > 5:
        return lowered[:-3]
    if lowered.endswith("ed") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 4:
        return lowered[:-1]
    return lowered


def _is_test_file(rel_path: str) -> bool:
    return (
        ".spec." in rel_path
        or ".test." in rel_path
        or "/test/" in rel_path
        or "/__snapshots__/" in rel_path
    )


def _is_e2e_or_playwright_file(rel_path: str) -> bool:
    return rel_path.startswith("playwright/") or "/page-objects/" in rel_path


def _is_config_file(rel_path: str) -> bool:
    return rel_path.startswith("config/") or rel_path.endswith(".config.ts") or rel_path.endswith(".config.js")


def _is_generated_file(rel_path: str) -> bool:
    return rel_path.endswith("services/api.ts")


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
