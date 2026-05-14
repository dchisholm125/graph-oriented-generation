"""GOG-Lite: Simple context membrane for prompt-scoped bundles.

Replaces the production ContextMembrane with a transparent,
distance-aware, token-budgeted file limiter. No advanced scoring,
no multi-factor ranking, no proprietary heuristics.

All constants are tunable integers documented inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gog_engine_lite.graph_search import distance_from_seeds, _is_generated_or_vendor

# =============================================================================
# GOG-Lite Membrane Transparent Tunables
# =============================================================================
# These are intentionally conservative defaults for a public reference.

DEFAULT_MAX_FILES: int = 10
"""Hard cap on files returned. More aggressive than production to keep noise low."""

DEFAULT_MAX_TOKENS_PER_FILE: int = 1500
"""Maximum tokens for any single file. Large files are truncated or snippetized."""

DEFAULT_TOTAL_TOKEN_BUDGET: int = 5000
"""Soft cap on total tokens across all files. If exceeded, trim from lowest-ranked."""

DEFAULT_MIN_FILES_FOR_DEPTH_2: int = 3
"""If fewer files are found at depth 1, auto-expand to depth 2."""


def apply_lite_membrane(
    graph,
    candidate_nodes: list[str],
    seed_nodes: list[str],
    max_files: int = DEFAULT_MAX_FILES,
    max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
    total_token_budget: int = DEFAULT_TOTAL_TOKEN_BUDGET,
) -> dict[str, Any]:
    """Trim candidate graph nodes to a bounded, token-aware set.

    Strategy (simple, explainable, no proprietary scoring):
    1. Always keep all seed nodes (they are explicitly matched).
    2. Rank remaining candidates by distance from seeds (closer = better).
    3. Apply per-file token cap and count tokens.
    4. If total tokens exceed the soft budget, trim from the end
       (lowest-ranked files are dropped first).
    5. Report why each file was kept or rejected.

    Args:
        graph: networkx.DiGraph with nodes = absolute file paths.
        candidate_nodes: All candidate files from graph isolation.
        seed_nodes: The seed files (must always be included).
        max_files: Maximum files to keep.
        max_tokens_per_file: Token cap for individual files.
        total_token_budget: Soft total token cap across the bundle.

    Returns:
        Dict with selected_nodes (list), kept (list of dicts),
        rejected (list of dicts), and config metadata.
    """
    seed_set = set(seed_nodes)
    candidates = [n for n in candidate_nodes if n not in seed_set]

    # Distance-based ranking (simplest possible heuristic)
    scored = []
    for node in candidates:
        dist = distance_from_seeds(graph, node, seed_nodes)
        scored.append((node, dist))

    # Sort by distance (closer first), then by name for stability
    scored.sort(key=lambda x: (x[1], x[0]))

    # Build selected list: seeds first, then ranked candidates
    ordered = list(seed_nodes) + [node for node, _ in scored]

    # Apply file cap
    capped = ordered[:max_files]

    # Apply token budget (soft cap — trim from lowest-ranked)
    selected, rejected = _apply_token_budget(
        capped,
        max_tokens_per_file=max_tokens_per_file,
        total_token_budget=total_token_budget,
        ordered_fallback=ordered,
    )

    kept = []
    for n in selected:
        if n in seed_set:
            kept.append({"node": n, "reason": "seed"})
        else:
            dist = distance_from_seeds(graph, n, seed_nodes)
            kept.append({"node": n, "reason": f"distance_rank(dist={dist})"})

    rejected_meta = [
        {"node": n, "rejected_reason": "trimmed_by_lite_membrane"}
        for n in ordered[max_files:]
    ]
    rejected_meta.extend(
        {"node": n, "rejected_reason": "token_budget_exceeded"}
        for n in rejected
    )

    return {
        "selected_nodes": selected,
        "kept": kept,
        "rejected": rejected_meta,
        "config": {
            "max_files": max_files,
            "max_tokens_per_file": max_tokens_per_file,
            "total_token_budget": total_token_budget,
        },
    }


def _apply_token_budget(
    capped_files: list[str],
    max_tokens_per_file: int,
    total_token_budget: int,
    ordered_fallback: list[str],
) -> tuple[list[str], list[str]]:
    """Return (selected, rejected) after enforcing per-file and total token caps.

    Simple algorithm:
    - Compute tokens for each file (capped at max_tokens_per_file).
    - Keep files in order until total_token_budget would be exceeded.
    - Any file that would push us over budget is rejected.
    """
    from .token_utils import count_tokens_in_string

    selected: list[str] = []
    total_tokens = 0
    rejected: list[str] = []

    for node in capped_files:
        try:
            text = Path(node).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            if total_tokens + 100 <= total_token_budget:
                selected.append(node)
                total_tokens += 100
            else:
                rejected.append(node)
            continue

        file_tokens = count_tokens_in_string(text)
        effective_tokens = min(file_tokens, max_tokens_per_file)

        if total_tokens + effective_tokens <= total_token_budget:
            selected.append(node)
            total_tokens += effective_tokens
        else:
            rejected.append(node)

    return selected, rejected
