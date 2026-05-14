"""GOG-Lite: Simple context bundler for benchmark reproducibility.

Replaces the production context serving with a minimal flow:
1. Build or load the lite import graph.
2. Isolate context via keyword/distance expansion.
3. Apply the lite membrane (distance-based cap).
4. Return a clean bundle of file contents.

This is designed to run on any repo without persistent `.gog/`
artifact onboarding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gog_engine_lite.import_graph import build_import_graph
from gog_engine_lite.graph_search import isolate_context
from .lite_membrane import apply_lite_membrane


def build_lite_context_bundle(
    repo_path: Path,
    prompt: str,
    max_files: int = 20,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Build a prompt-scoped context bundle using only GOG-Lite logic.

    Args:
        repo_path: Path to the target repository.
        prompt: Natural language coding task description.
        max_files: Maximum files to include.
        max_depth: Graph traversal depth from seeds.

    Returns:
        Dict with:
            - selected_nodes: list of absolute file paths
            - file_contents: dict mapping path -> file text
            - metadata: dict with seeding info, distance stats, membrane config
    """
    repo_root = repo_path.expanduser().resolve()

    graph = build_import_graph(repo_root)

    if graph.number_of_nodes() == 0:
        return {
            "selected_nodes": [],
            "file_contents": {},
            "metadata": {
                "error": "No parseable source files found in repo",
                "repo_root": str(repo_root),
            },
        }

    candidate_nodes = isolate_context(graph, prompt, max_depth=max_depth)

    # Seed nodes are the first few candidates (keywords or filename matches)
    seeds = candidate_nodes[:5] if len(candidate_nodes) > 5 else candidate_nodes

    membrane_result = apply_lite_membrane(
        graph=graph,
        candidate_nodes=candidate_nodes,
        seed_nodes=seeds,
        max_files=max_files,
    )

    selected = membrane_result["selected_nodes"]
    file_contents: dict[str, str] = {}

    # Also track relative paths for benchmark compatibility
    rel_paths: list[str] = []
    for node in selected:
        node_path = Path(node)
        try:
            rel = node_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel = node_path.name
        rel_paths.append(rel)
        try:
            text = node_path.read_text(encoding="utf-8", errors="replace")
            file_contents[node] = text
        except (OSError, UnicodeDecodeError):
            file_contents[node] = "# [unreadable file]"

    metadata = {
        "repo_root": str(repo_root),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "candidate_count": len(candidate_nodes),
        "selected_count": len(selected),
        "seed_nodes": seeds,
        "membrane": membrane_result["config"],
        "kept_reasons": [k["reason"] for k in membrane_result["kept"]],
        "rejected_count": len(membrane_result["rejected"]),
        "rel_paths": rel_paths,
    }

    return {
        "selected_nodes": selected,
        "file_contents": file_contents,
        "metadata": metadata,
    }
