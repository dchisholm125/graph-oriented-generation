"""GOG-Lite: Simple context bundler for benchmark reproducibility.

Replaces the production context serving with a minimal flow:
1. Build or load the lite import graph.
2. Isolate context via keyword/distance expansion with simple scoring.
3. Apply the lite membrane (distance-based cap + token budget).
4. Extract keyword snippets for large files instead of full contents.
5. Return a clean bundle of file contents.

This is designed to run on any repo without persistent `.gog/`
artifact onboarding.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gog_engine_lite.import_graph import build_import_graph
from gog_engine_lite.graph_search import (
    isolate_context,
    extract_keyword_snippets,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_TOKENS_PER_FILE,
    DEFAULT_TOTAL_TOKEN_BUDGET,
)
from .lite_membrane import apply_lite_membrane


def build_lite_context_bundle(
    repo_path: Path,
    prompt: str,
    max_files: int = DEFAULT_MAX_FILES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_tokens_per_file: int = DEFAULT_MAX_TOKENS_PER_FILE,
    total_token_budget: int = DEFAULT_TOTAL_TOKEN_BUDGET,
    use_snippets: bool = True,
) -> dict[str, Any]:
    """Build a prompt-scoped context bundle using only GOG-Lite logic.

    All parameters are transparent and self-documenting. There are no
    hidden heuristics or proprietary scoring functions.

    Args:
        repo_path: Path to the target repository.
        prompt: Natural language coding task description.
        max_files: Maximum files to include (default 12).
        max_depth: Graph traversal depth from seeds (default 1).
        max_tokens_per_file: Token cap for individual files (default 1200).
        total_token_budget: Soft total token cap across bundle (default 6000).
        use_snippets: If True, large files get keyword-centered snippets
            instead of full contents.

    Returns:
        Dict with:
            - selected_nodes: list of absolute file paths
            - file_contents: dict mapping path -> file text or snippet
            - metadata: dict with seeding info, distance stats, membrane config,
              token counts, and transparency notes
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
                "transparency_note": (
                    "GOG-Lite is intentionally simple. It builds a structural "
                    "import graph using regex (not tree-sitter) and scores "
                    "candidates with explainable rules. See gog_engine_lite/ "
                    "for full source."
                ),
            },
        }

    candidate_nodes = isolate_context(graph, prompt, max_depth=max_depth, max_files=max_files * 2)

    # Seed nodes are the first few candidates (keywords or filename matches)
    seeds = candidate_nodes[:5] if len(candidate_nodes) > 5 else candidate_nodes

    membrane_result = apply_lite_membrane(
        graph=graph,
        candidate_nodes=candidate_nodes,
        seed_nodes=seeds,
        max_files=max_files,
        max_tokens_per_file=max_tokens_per_file,
        total_token_budget=total_token_budget,
    )

    selected = membrane_result["selected_nodes"]
    file_contents: dict[str, str] = {}
    snippet_flags: dict[str, bool] = {}

    # Also track relative paths for benchmark compatibility
    rel_paths: list[str] = []
    for node in selected:
        node_path = Path(node)
        try:
            rel = node_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            rel = node_path.name
        rel_paths.append(rel)

        # Decide: full file or snippet?
        prompt_keywords = _extract_prompt_keywords(prompt)
        snippet_text = extract_keyword_snippets(node, prompt_keywords)

        if use_snippets and snippet_text is not None:
            file_contents[node] = snippet_text
            snippet_flags[node] = True
        else:
            try:
                text = node_path.read_text(encoding="utf-8", errors="replace")
                file_contents[node] = text
                snippet_flags[node] = False
            except (OSError, UnicodeDecodeError):
                file_contents[node] = "# [unreadable file]"
                snippet_flags[node] = False

    # Token accounting for transparency
    from .token_utils import count_tokens_in_string
    total_tokens = sum(count_tokens_in_string(text) for text in file_contents.values())

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
        "total_tokens_estimate": total_tokens,
        "snippet_flags": {node: flag for node, flag in snippet_flags.items()},
        "transparency_note": (
            "GOG-Lite is a public reference implementation. It uses: "
            "(1) regex-based import graphs, "
            "(2) keyword seeding + bounded BFS, "
            "(3) simple integer scoring (exact match > keyword match > test bonus > distance), "
            "(4) per-file token caps, "
            "(5) soft total token budget, "
            "(6) keyword snippets for large files. "
            "No embeddings, no AST parsing, no proprietary heuristics. "
            "See gog_engine_lite/graph_search.py for all constants."
        ),
    }

    return {
        "selected_nodes": selected,
        "file_contents": file_contents,
        "metadata": metadata,
    }


def _extract_prompt_keywords(prompt: str) -> set[str]:
    """Extract lowercase keywords from prompt for snippet matching."""
    import re
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
