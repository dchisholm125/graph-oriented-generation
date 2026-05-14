"""GOG-Lite: Simple context membrane for prompt-scoped bundles.

Replaces the production ContextMembrane with a distance-based
file limiter. No advanced scoring, no multi-factor ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gog_engine_lite.graph_search import distance_from_seeds


def apply_lite_membrane(
    graph,
    candidate_nodes: list[str],
    seed_nodes: list[str],
    max_files: int = 20,
) -> dict[str, Any]:
    """Trim candidate graph nodes to a bounded set before serving.

    Strategy:
    1. Keep all explicit seeds.
    2. Rank remaining candidates by distance from seeds (closer = better).
    3. Keep top max_files total.

    Args:
        graph: networkx.DiGraph with nodes = absolute file paths.
        candidate_nodes: All candidate files from graph isolation.
        seed_nodes: The seed files (must always be included).
        max_files: Maximum files to keep.

    Returns:
        Dict with selected_nodes (list), kept (list of dicts), rejected (list of dicts).
    """
    seed_set = set(seed_nodes)
    candidates = [n for n in candidate_nodes if n not in seed_set]

    scored = []
    for node in candidates:
        dist = distance_from_seeds(graph, node, seed_nodes)
        scored.append((node, dist))

    # Sort by distance (closer first), then by name for stability
    scored.sort(key=lambda x: (x[1], x[0]))

    # Cap non-seed files
    cap = max(0, max_files - len(seed_nodes))
    kept_non_seeds = [node for node, _ in scored[:cap]]
    rejected_non_seeds = [node for node, _ in scored[cap:]]

    selected_nodes = list(seed_nodes) + kept_non_seeds

    kept = [
        {"node": n, "reason": "seed" if n in seed_set else "distance_rank"}
        for n in selected_nodes
    ]
    rejected = [
        {"node": n, "rejected_reason": "trimmed_by_lite_membrane"}
        for n in rejected_non_seeds
    ]

    return {
        "selected_nodes": selected_nodes,
        "kept": kept,
        "rejected": rejected,
        "config": {"max_files": max_files},
    }
