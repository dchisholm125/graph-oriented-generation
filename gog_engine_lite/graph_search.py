"""GOG-Lite: Simple deterministic graph traversal for context isolation.

Provides keyword seeding + bounded distance expansion without
semantic embeddings or advanced scoring heuristics.

All constants are transparent tunables documented inline.
This is a public reference implementation — no production heuristics
are imported or copied.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

import networkx as nx

# =============================================================================
# GOG-Lite Transparent Tunables
# =============================================================================
# These constants are intentionally simple and self-documenting.
# They exist so reviewers can read the code and understand why a file
# was included or excluded without tracing through proprietary scoring.

DEFAULT_MAX_DEPTH: int = 1
"""Default BFS depth from seeds. Depth 2 is used only when context is sparse."""

DEFAULT_MAX_FILES: int = 12
"""Default hard cap on returned files. Aggressive to keep token budgets low."""

DEFAULT_MAX_TOKENS_PER_FILE: int = 1200
"""Per-file token budget. Large files are either truncated or penalized."""

DEFAULT_TOTAL_TOKEN_BUDGET: int = 6000
"""Rough total token budget across all selected files. Soft cap."""

MAX_FILE_SIZE_BYTES: int = 80_000
"""Files larger than this are penalized unless directly matched by filename."""

LARGE_FILE_PENALTY_DISTANCE: int = 3
"""Distance penalty added to large files (makes them sort lower)."""

KEYWORD_SNIPPET_RADIUS: int = 6
"""Lines around a keyword match to include when snippet mode is active."""

# Scoring weights — higher is better. Keep as small integers for clarity.
SCORE_EXACT_FILENAME_MATCH: int = 100
SCORE_FILENAME_KEYWORD_MATCH: int = 30
SCORE_CONTENT_KEYWORD_MATCH: int = 20
SCORE_TEST_FILE_BONUS: int = 15
SCORE_DIRECT_IMPORT_OF_SEED: int = 10
SCORE_DEPTH_PENALTY_PER_HOP: int = 5

# Files/directories to always exclude (generated, vendor, cache, build)
GENERATED_PATH_PATTERNS: tuple[str, ...] = (
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    ".coverage",
    "*.min.js",
    "*.min.css",
    ".lock",
)


def isolate_context(
    graph: nx.DiGraph,
    prompt: str,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_files: int = DEFAULT_MAX_FILES,
) -> list[str]:
    """Return a ranked list of files relevant to the prompt.

    Strategy (simple, transparent, no proprietary scoring):
    1. Extract explicit filename/symbol/test-name mentions from the prompt.
    2. If none, extract keyword tokens and find matching filenames + content.
    3. Score each candidate by:
       - Exact filename match (highest)
       - Filename containing keyword
       - File content containing keyword
       - Test file bonus
       - Direct import of a seed
       - Graph distance from seed (penalty)
       - Large file penalty
    4. Sort by score (descending), deduplicate, cap at max_files.
    5. If too few results (< 3), expand to depth 2 automatically.

    Args:
        graph: Import graph (DiGraph with nodes = absolute file paths).
        prompt: Natural language query.
        max_depth: BFS depth limit from seeds (default 1).
        max_files: Maximum files to return.

    Returns:
        Sorted list of absolute file paths, highest-scored first.
    """
    # ---- 1. Seed extraction ------------------------------------------------
    seed_nodes = _extract_filename_mentions(prompt, graph)
    if not seed_nodes:
        seed_nodes = _keyword_seeds(prompt, graph)

    if not seed_nodes:
        return []

    seed_set: set[str] = set(seed_nodes)

    # ---- 2. Build candidate pool via BFS -----------------------------------
    reachable: dict[str, int] = {}  # node -> shortest distance from any seed

    for seed in seed_nodes:
        current: set[str] = {seed}
        reachable[seed] = 0
        for depth in range(1, max_depth + 1):
            next_level: set[str] = set()
            for node in current:
                for neighbor in graph.successors(node):
                    if neighbor not in reachable:
                        reachable[neighbor] = depth
                        next_level.add(neighbor)
            current = next_level
            if not current:
                break

    # ---- 3. Score every candidate ------------------------------------------
    prompt_keywords = _prompt_terms(prompt)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()

    for node, distance in reachable.items():
        if node in seen:
            continue
        if _is_generated_or_vendor(_node_search_path(graph, node)):
            continue
        seen.add(node)

        score = _score_node(
            node=node,
            distance=distance,
            seed_set=seed_set,
            prompt_keywords=prompt_keywords,
            graph=graph,
        )
        scored.append((score, node))

    # ---- 4. Sort, deduplicate, cap -----------------------------------------
    scored.sort(key=lambda x: (-x[0], x[1]))
    result = [node for _, node in scored[:max_files]]

    # ---- 5. Auto-expand to depth 2 if context is too sparse ---------------
    if len(result) < 3 and max_depth < 2:
        return isolate_context(graph, prompt, max_depth=2, max_files=max_files)

    return result


def _score_node(
    node: str,
    distance: int,
    seed_set: set[str],
    prompt_keywords: set[str],
    graph: nx.DiGraph,
) -> int:
    """Score a single node using only simple, explainable rules."""
    score = 0
    search_path = _node_search_path(graph, node)
    filename = search_path.rsplit("/", 1)[-1]

    # 1. Exact filename match (highest priority)
    for kw in prompt_keywords:
        if kw == filename.replace(".py", "").replace(".ts", "").replace(".vue", "").replace(".js", ""):
            score += SCORE_EXACT_FILENAME_MATCH
        elif kw in filename:
            score += SCORE_FILENAME_KEYWORD_MATCH

    # 2. Content keyword match (lightweight scan)
    content_score = _count_keyword_hits(node, prompt_keywords)
    score += min(content_score * SCORE_CONTENT_KEYWORD_MATCH, 60)  # cap to avoid runaway

    # 3. Test file preference
    if _looks_like_test_file(search_path):
        score += SCORE_TEST_FILE_BONUS

    # 4. Direct import of a seed node
    if node not in seed_set:
        for seed in seed_set:
            if graph.has_edge(seed, node):
                score += SCORE_DIRECT_IMPORT_OF_SEED
                break

    # 5. Distance penalty (closer is better)
    score -= distance * SCORE_DEPTH_PENALTY_PER_HOP

    # 6. Large file penalty (unless it's a direct seed)
    if node not in seed_set:
        size = graph.nodes[node].get("size", 0)
        if size > MAX_FILE_SIZE_BYTES:
            score -= LARGE_FILE_PENALTY_DISTANCE * SCORE_DEPTH_PENALTY_PER_HOP

    return max(score, 0)


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
        search_path = _node_search_path(graph, node)
        filename = search_path.rsplit("/", 1)[-1]
        matches = sum(1 for term in terms if term in filename or term in search_path)
        if matches > 0:
            scored.append((node, matches))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [node for node, _ in scored[:5]]


def _node_search_path(graph: nx.DiGraph, node: str) -> str:
    """Return a stable path for matching that excludes machine-local parents."""
    rel_path = graph.nodes[node].get("rel_path")
    if isinstance(rel_path, str) and rel_path:
        return rel_path.lower()
    return node.replace("\\", "/").lower()


def _prompt_terms(prompt: str) -> set[str]:
    """Extract meaningful keywords from prompt."""
    stopwords = {
        "file", "files", "find", "responsible", "involved", "when", "where",
        "that", "this", "from", "into", "the", "and", "for", "with", "how",
        "add", "refactor", "fix", "create", "delete", "remove", "update",
        "should", "must", "will", "would", "could", "can", "may", "might",
    }
    words = re.findall(r"[A-Za-z][A-Za-z0-9]+", prompt)
    terms = set()
    for w in words:
        lw = w.lower()
        if lw not in stopwords and len(lw) >= 3:
            terms.add(lw)
    return terms


def _looks_like_test_file(path: str) -> bool:
    """Return True if the file looks like a test or spec file."""
    normalized = path.lower()
    return (
        "/test" in normalized
        or normalized.startswith("test")
        or ".spec." in normalized
        or ".test." in normalized
    )


def _is_generated_or_vendor(path: str) -> bool:
    """Return True if the file is in a generated/vendor/cache/build directory."""
    lowered = path.lower()
    for pattern in GENERATED_PATH_PATTERNS:
        if pattern.startswith("*"):
            if lowered.endswith(pattern.lstrip("*")):
                return True
        elif pattern in lowered:
            return True
    return False


def _count_keyword_hits(node_path: str, keywords: set[str]) -> int:
    """Count how many distinct keywords appear in file content.

    This is intentionally lightweight: it reads the file once and
    does a simple substring search. No AST parsing, no embeddings.
    """
    try:
        with open(node_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().lower()
    except (OSError, UnicodeDecodeError):
        return 0

    return sum(1 for kw in keywords if kw in text)


def extract_keyword_snippets(
    node_path: str,
    keywords: set[str],
    radius: int = KEYWORD_SNIPPET_RADIUS,
) -> Optional[str]:
    """Return a snippet of the file centered on keyword matches.

    If no keywords match, returns None (caller should fall back to full file).

    Args:
        node_path: Absolute path to the source file.
        keywords: Set of lowercase keywords to search for.
        radius: Lines of context around each match.

    Returns:
        Snippet string or None.
    """
    try:
        with open(node_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (OSError, UnicodeDecodeError):
        return None

    if not lines:
        return None

    matched_lines: set[int] = set()
    for i, line in enumerate(lines):
        lowered = line.lower()
        if any(kw in lowered for kw in keywords):
            start = max(0, i - radius)
            end = min(len(lines), i + radius + 1)
            matched_lines.update(range(start, end))

    if not matched_lines:
        return None

    # Merge contiguous blocks
    sorted_lines = sorted(matched_lines)
    blocks: list[list[int]] = []
    current_block: list[int] = [sorted_lines[0]]
    for idx in sorted_lines[1:]:
        if idx == current_block[-1] + 1:
            current_block.append(idx)
        else:
            blocks.append(current_block)
            current_block = [idx]
    blocks.append(current_block)

    snippet_parts: list[str] = []
    for block in blocks:
        if snippet_parts:
            snippet_parts.append("\n...\n")
        snippet_parts.append("".join(lines[i] for i in block))

    return "".join(snippet_parts)
