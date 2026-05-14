"""GOG-Lite: Simple deterministic graph traversal for context isolation.

Provides keyword seeding + bounded distance expansion without
semantic embeddings or advanced scoring heuristics.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

import networkx as nx


def isolate_context(
    graph: nx.DiGraph,
    prompt: str,
    max_depth: int = 2,
    max_files: int = 20,
) -> list[str]:
    """Return a minimal set of files relevant to the prompt.

    Strategy:
    1. Extract explicit filename mentions from the prompt.
    2. If none, extract keyword tokens and find matching filenames.
    3. From each seed, do bounded BFS up to max_depth.
    4. Return union of all reached files, capped at max_files.

    Args:
        graph: Import graph (DiGraph with nodes = absolute file paths).
        prompt: Natural language query.
        max_depth: BFS depth limit from seeds.
        max_files: Maximum files to return.

    Returns:
        Sorted list of absolute file paths.
    """
    seeds = _extract_filename_mentions(prompt, graph)
    if not seeds:
        seeds = _keyword_seeds(prompt, graph)

    if not seeds:
        return []

    reachable: set[str] = set(seeds)

    for seed in seeds:
        current: set[str] = {seed}
        for depth in range(1, max_depth + 1):
            next_level: set[str] = set()
            for node in current:
                next_level.update(graph.successors(node))
            next_level -= reachable
            reachable.update(next_level)
            current = next_level
            if not current:
                break

    # Deduplicate and cap
    result = sorted(reachable)[:max_files]
    return result


def distance_from_seeds(
    graph: nx.DiGraph,
    target: str,
    seeds: list[str],
) -> int:
    """Return shortest distance from target to any seed node.

    Returns 0 if target is a seed, or the minimum hop count otherwise.
    Returns 99 if unreachable.
    """
    if target in seeds:
        return 0

    queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds)
    seen: set[str] = set(seeds)

    while queue:
        current, dist = queue.popleft()
        if dist >= 3:
            continue
        for neighbor in graph.successors(current):
            if neighbor == target:
                return dist + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))

    return 99


def _extract_filename_mentions(prompt: str, graph: nx.DiGraph) -> list[str]:
    """Extract `src/...` or `gog_engine/...` file mentions from prompt."""
    pattern = re.compile(
        r"(?:src|gog_engine|gog_cli|gog)/[A-Za-z0-9_/.-]+\.(?:py|ts|vue|js)"
    )
    mentions = list(dict.fromkeys(pattern.findall(prompt)))

    matched: list[str] = []
    seen: set[str] = set()
    for candidate in mentions:
        candidate_lower = candidate.lower().replace("/", "\\")
        for node in graph.nodes():
            node_lower = node.lower().replace("/", "\\")
            if node_lower.endswith(candidate_lower):
                if node not in seen:
                    seen.add(node)
                    matched.append(node)
                break
    return matched


def _keyword_seeds(prompt: str, graph: nx.DiGraph) -> list[str]:
    """Find seed nodes by matching prompt keywords against filenames."""
    terms = _prompt_terms(prompt)
    scored: list[tuple[str, int]] = []

    for node in graph.nodes():
        filename = node.split("/")[-1]
        path_parts = node.split("/")
        matches = sum(1 for term in terms if term in filename or term in node)
        if matches > 0:
            scored.append((node, matches))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in scored[:5]]


def _prompt_terms(prompt: str) -> set[str]:
    """Extract meaningful keywords from prompt."""
    stopwords = {
        "file", "files", "find", "responsible", "involved", "when", "where",
        "that", "this", "from", "into", "the", "and", "for", "with", "how",
        "add", "refactor", "fix", "create", "delete", "remove", "update",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", prompt)
    terms = set()
    for w in words:
        lw = w.lower()
        if lw not in stopwords and len(lw) >= 3:
            terms.add(lw)
    return terms
