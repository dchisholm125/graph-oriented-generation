"""Repository summaries and prompt-scoped context bundles."""

from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

from gog_engine import graph_search
from gog_engine.token_utils import count_tokens_in_files

from .context_membrane import apply_context_membrane
from .onboarding import inspect_onboarding


IDENTIFIER_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*)`?")
FILE_MENTION_RE = re.compile(r"(?:src|gog_engine|gog_cli|gog)/[A-Za-z0-9_/.-]+\.(?:py|ts|vue|js)")
IDENTIFIER_STOPWORDS = {
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


def summarize_repository(
    repo_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a repo-level orientation digest from onboarding artifacts."""
    inspection = inspect_onboarding(repo_path=repo_path, artifact_dir=artifact_dir)
    manifest = inspection["manifest"]
    profile_path = Path(manifest["artifacts"]["repo_profile"])
    graph_path = Path(manifest["artifacts"]["structural_graph_json"])

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    degree_summary = _degree_summary(graph_payload)

    return {
        "status": inspection["status"],
        "issues": inspection["issues"],
        "repo": {
            "root": manifest["repo"]["root"],
            "fingerprint": manifest["repo"]["fingerprint"],
        },
        "coverage": {
            "total_files": profile["files"]["total_files"],
            "supported_graph_files": profile["parser_coverage"]["supported_graph_file_count"],
            "supported_extensions": profile["parser_coverage"]["structural_graph_extensions"],
        },
        "manifests": profile["manifests"],
        "validation": manifest["validation"],
        "graph": {
            "node_count": graph_payload["node_count"],
            "edge_count": graph_payload["edge_count"],
            "high_out_degree_nodes": degree_summary["high_out_degree_nodes"],
            "high_in_degree_nodes": degree_summary["high_in_degree_nodes"],
        },
        "artifact_root": manifest["artifacts"]["root"],
    }


def build_context_bundle(
    repo_path: Path,
    prompt: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a prompt-scoped repo bundle for a downstream reasoner."""
    repo_root = repo_path.expanduser().resolve()
    inspection = inspect_onboarding(repo_path=repo_root, artifact_dir=artifact_dir)
    manifest = inspection["manifest"]
    graph = _load_graph(Path(manifest["artifacts"]["structural_graph_pickle"]))

    explicit_mentions = FILE_MENTION_RE.findall(prompt)
    explicit_files = _explicit_file_mentions(explicit_mentions, graph)
    explicit_identifiers = _explicit_identifier_mentions(prompt, repo_root, graph)
    unsupported_mentions = _unsupported_file_mentions(explicit_mentions, graph)

    if explicit_files:
        seed_nodes = explicit_files
        selected_nodes = _expand_from_seed_nodes(graph, explicit_files)
        selection_strategy = "explicit_file_mentions"
    elif unsupported_mentions:
        seed_nodes = []
        selected_nodes = []
        selection_strategy = "unsupported_explicit_file_mentions"
    elif explicit_identifiers:
        seed_nodes = explicit_identifiers
        selected_nodes = _expand_from_seed_nodes(graph, explicit_identifiers)
        selection_strategy = "identifier_filename_match"
    else:
        selected_nodes = graph_search.isolate_context(graph, prompt)
        seed_nodes = _infer_seed_nodes(graph, selected_nodes)
        selection_strategy = "graph_search_fallback"

    membrane = apply_context_membrane(
        graph=graph,
        repo_root=repo_root,
        prompt=prompt,
        candidate_nodes=selected_nodes,
        seed_nodes=seed_nodes,
        explicit_files=explicit_files,
        identifier_files=explicit_identifiers,
    )
    selected_nodes = membrane.selected_nodes
    rel_files = [_rel(repo_root, Path(path)) for path in selected_nodes]
    relations = _selected_relations(repo_root, graph, selected_nodes)
    token_estimate = count_tokens_in_files(selected_nodes)

    return {
        "status": inspection["status"],
        "issues": inspection["issues"],
        "prompt": prompt,
        "selection": {
            "strategy": selection_strategy,
            "explicit_file_matches": [_rel(repo_root, Path(path)) for path in explicit_files],
            "unsupported_explicit_file_mentions": unsupported_mentions,
            "identifier_matches": [_rel(repo_root, Path(path)) for path in explicit_identifiers],
            "selected_file_count": len(rel_files),
            "estimated_input_tokens": token_estimate,
            "pre_membrane_file_count": len(set(membrane.selected_nodes + [item["node"] for item in membrane.rejected])),
            "membrane_kept_count": len(membrane.kept),
            "membrane_rejected_count": len(membrane.rejected),
        },
        "context": {
            "files": rel_files,
            "relations": relations,
            "validation_commands": manifest["validation"]["commands"],
            "allowed_nodes": rel_files,
        },
        "context_membrane": {
            "enabled": True,
            "config": membrane.config,
            "kept": _strip_node_paths(membrane.kept),
            "rejected": _strip_node_paths(membrane.rejected),
        },
        "handoff": {
            "recommended_reasoner_input": (
                "Use this bounded repository slice as the operating surface for plan creation. "
                "Emit a typed MutationPlan; do not assume files or dependencies outside the served context."
            ),
            "salience_evaluator_required": True,
            "post_apply_refresh_required": True,
        },
    }


def _load_graph(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _degree_summary(graph_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for node in graph_payload["nodes"]:
        out_degree[node] = 0
        in_degree[node] = 0
    for edge in graph_payload["edges"]:
        out_degree[edge["source"]] = out_degree.get(edge["source"], 0) + 1
        in_degree[edge["target"]] = in_degree.get(edge["target"], 0) + 1

    def top(items: dict[str, int]) -> list[dict[str, Any]]:
        ranked = sorted(items.items(), key=lambda item: (-item[1], item[0]))
        return [{"path": path, "degree": degree} for path, degree in ranked[:10] if degree > 0]

    return {
        "high_out_degree_nodes": top(out_degree),
        "high_in_degree_nodes": top(in_degree),
    }


def _explicit_file_mentions(mentions: list[str], graph) -> list[str]:
    matches: list[str] = []
    seen = set()
    for mention in mentions:
        normalized = mention.lower().replace("/", os.sep)
        for node in graph.nodes():
            node_text = str(node).lower().replace("/", os.sep)
            if node_text.endswith(normalized) and node not in seen:
                seen.add(node)
                matches.append(node)
                break
    return matches


def _unsupported_file_mentions(mentions: list[str], graph) -> list[str]:
    unsupported: list[str] = []
    for mention in mentions:
        normalized = mention.lower().replace("/", os.sep)
        matched = any(
            str(node).lower().replace("/", os.sep).endswith(normalized)
            for node in graph.nodes()
        )
        if not matched:
            unsupported.append(mention)
    return unsupported


def _explicit_identifier_mentions(prompt: str, repo_root: Path, graph) -> list[str]:
    matches: list[str] = []
    seen = set()
    candidates = _identifier_candidates(prompt)
    if not candidates:
        return matches

    for node in graph.nodes():
        stem = Path(node).stem.lower()
        if any(candidate == stem or candidate in stem for candidate in candidates):
            if node not in seen:
                seen.add(node)
                matches.append(node)
    ranked = sorted(
        matches,
        key=lambda node: _identifier_match_rank(
            repo_root=repo_root,
            node=node,
            candidates=candidates,
        ),
    )
    strong_matches = [
        node for node in ranked
        if _identifier_overlap_count(repo_root, node, candidates) >= 2
    ]
    return strong_matches if len(strong_matches) >= 2 else ranked


def _identifier_match_rank(repo_root: Path, node: str, candidates: set[str]) -> tuple[int, int, str]:
    rel_path = _rel(repo_root, Path(node))
    overlap_count = _identifier_overlap_count(repo_root, node, candidates)
    penalty = 0
    if rel_path.startswith("playwright/"):
        penalty += 50
    if ".spec." in rel_path or ".test." in rel_path:
        penalty += 40
    if rel_path.startswith("config/"):
        penalty += 30
    return (penalty, -overlap_count, rel_path)


def _identifier_overlap_count(repo_root: Path, node: str, candidates: set[str]) -> int:
    rel_path = _rel(repo_root, Path(node))
    path_terms = _path_terms(rel_path)
    stem = Path(node).stem.lower()
    return sum(
        1 for candidate in candidates
        if candidate in path_terms or candidate == stem or candidate in stem
    )


def _identifier_candidates(prompt: str) -> set[str]:
    candidates: set[str] = set()
    for identifier in IDENTIFIER_RE.findall(prompt):
        if len(identifier) < 4:
            continue
        normalized = _normalize_identifier(identifier)
        if normalized in IDENTIFIER_STOPWORDS:
            continue
        candidates.add(normalized)
        if normalized.startswith("un") and len(normalized) > 4:
            candidates.add(normalized[2:])
    return candidates


def _path_terms(rel_path: str) -> set[str]:
    terms: set[str] = set()
    for part in re.split(r"[/_.-]+", rel_path):
        for subpart in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", part):
            normalized = _normalize_identifier(subpart)
            if len(normalized) >= 4:
                terms.add(normalized)
    return terms


def _normalize_identifier(identifier: str) -> str:
    lowered = identifier.lower()
    if lowered.endswith("ies") and len(lowered) > 5:
        return f"{lowered[:-3]}y"
    if lowered.endswith("es") and len(lowered) > 5:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 4:
        return lowered[:-1]
    return lowered


def _expand_from_seed_nodes(graph, seeds: list[str]) -> list[str]:
    selected = set()
    for seed in seeds[:5]:
        selected.add(seed)
        selected.update(graph_search._descendants_bounded(graph, seed))
    return sorted(selected)


def _infer_seed_nodes(graph, selected_nodes: list[str]) -> list[str]:
    selected = set(selected_nodes)
    inferred = [
        node for node in selected_nodes
        if any(successor in selected for successor in graph.successors(node))
    ]
    return inferred[:5] if inferred else selected_nodes[:5]


def _selected_relations(repo_root: Path, graph, selected_nodes: list[str]) -> list[dict[str, str]]:
    selected = set(selected_nodes)
    relations = []
    for source, target in graph.edges():
        if source in selected and target in selected:
            relations.append(
                {
                    "source": _rel(repo_root, Path(source)),
                    "relation": "imports",
                    "target": _rel(repo_root, Path(target)),
                }
            )
    return relations


def _strip_node_paths(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for item in items:
        copied = dict(item)
        copied.pop("node", None)
        stripped.append(copied)
    return stripped


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
