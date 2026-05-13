"""Conservative Python import graph extraction for GOG onboarding."""

from __future__ import annotations

import ast
from pathlib import Path

import networkx as nx


def build_python_graph(repo_root: Path, python_files: list[Path]) -> nx.DiGraph:
    """Build a repo-local Python import graph from known Python source files."""
    graph = nx.DiGraph()
    module_index = _build_module_index(repo_root, python_files)

    for path in python_files:
        graph.add_node(str(path.resolve()))

    for path in python_files:
        source = str(path.resolve())
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        for module_name in _extract_import_targets(tree, path, repo_root):
            target = module_index.get(module_name)
            if target is not None:
                graph.add_edge(source, str(target.resolve()))
    return graph


def _build_module_index(repo_root: Path, python_files: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in python_files:
        rel = path.resolve().relative_to(repo_root.resolve())
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        index[".".join(parts)] = path
    return index


def _extract_import_targets(tree: ast.AST, path: Path, repo_root: Path) -> set[str]:
    module_names: set[str] = set()
    current_module = _module_name_for_path(path, repo_root)
    current_package = current_module.split(".")[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_module(
                node.module,
                node.level,
                current_package,
            )
            if base:
                module_names.add(base)
                for alias in node.names:
                    if alias.name != "*":
                        module_names.add(f"{base}.{alias.name}")
    return module_names


def _module_name_for_path(path: Path, repo_root: Path) -> str:
    rel = path.resolve().relative_to(repo_root.resolve()).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_from_module(
    module: str | None,
    level: int,
    current_package: list[str],
) -> str | None:
    if level <= 0:
        return module

    drop = max(level - 1, 0)
    if drop > len(current_package):
        return module

    prefix = current_package[: len(current_package) - drop]
    suffix = module.split(".") if module else []
    parts = prefix + suffix
    return ".".join(part for part in parts if part)
