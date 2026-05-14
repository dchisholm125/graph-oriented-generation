"""GOG-Lite: Simple regex-based import graph builder.

This is the public reference implementation for building structural
import graphs. It does not use tree-sitter or advanced AST parsing.
It demonstrates that even simple graph traversal can improve context
selection over flat keyword retrieval.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import networkx as nx

# Simple regex for JS/TS/Vue imports
IMPORT_RE = re.compile(
    r"import\s+.*?from\s+['\"](.*?)['\"]|"
    r"import\s+['\"](.*?)['\"]",
    re.MULTILINE,
)

# Python import regex (very approximate, for GOG-Lite only)
PYTHON_IMPORT_RE = re.compile(
    r"^\s*from\s+(\S+)\s+import|^\s*import\s+(\S+)",
    re.MULTILINE,
)


def build_import_graph(
    repo_root: Path,
    supported_extensions: tuple[str, ...] | None = None,
    ignored_dirs: set[str] | None = None,
) -> nx.DiGraph:
    """Build a basic import graph using regex scanning.

    Args:
        repo_root: Path to the repository.
        supported_extensions: File extensions to scan.
        ignored_dirs: Directory names to skip.

    Returns:
        networkx.DiGraph with nodes = absolute file paths,
        edges = import dependencies.
    """
    if supported_extensions is None:
        supported_extensions = (".py", ".ts", ".vue", ".js")
    if ignored_dirs is None:
        ignored_dirs = {
            ".git", ".venv", "__pycache__", "node_modules", "public-repos",
        }

    graph = nx.DiGraph()
    file_index: dict[str, Path] = {}

    # Index all matching files
    for ext in supported_extensions:
        for path in repo_root.rglob(f"*{ext}"):
            if any(part in ignored_dirs for part in path.parts):
                continue
            if ext == ".py" and path.name == "__init__.py":
                continue
            abs_path = str(path.resolve())
            graph.add_node(abs_path, path=abs_path, ext=ext)
            file_index[str(path.resolve())] = path

    # Build edges via simple import scanning
    for abs_path, path in file_index.items():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        imports = _extract_imports(text, path)
        for raw_import in imports:
            target = _resolve_import(raw_import, path, repo_root, file_index)
            if target:
                graph.add_edge(abs_path, str(target.resolve()))

    return graph


def _extract_imports(text: str, path: Path) -> list[str]:
    """Extract raw import specifiers from file content."""
    imports: list[str] = []
    ext = path.suffix

    if ext in (".ts", ".vue", ".js"):
        for match in IMPORT_RE.finditer(text):
            imp = match.group(1) or match.group(2)
            if imp:
                imports.append(imp)
    elif ext == ".py":
        for match in PYTHON_IMPORT_RE.finditer(text):
            imp = match.group(1) or match.group(2)
            if imp:
                # Remove leading dots for relative imports
                clean = imp.lstrip(".")
                if clean:
                    imports.append(clean)

    return imports


def _resolve_import(
    raw_import: str,
    current_file: Path,
    repo_root: Path,
    file_index: dict[str, Path],
) -> Optional[Path]:
    """Resolve a raw import string to an absolute file path."""
    # Handle relative imports for JS/TS
    if raw_import.startswith("."):
        resolved = _resolve_relative(raw_import, current_file, repo_root)
        if resolved:
            return resolved

    # Handle bare Python-style imports by trying to match against file_index
    # This is naive but sufficient for GOG-Lite
    normalized = raw_import.replace(".", os.sep)
    for ext in (".py", ".ts", ".vue", ".js"):
        candidate = repo_root / f"{normalized}{ext}"
        if candidate.exists():
            return candidate
        candidate_index = repo_root / normalized / f"index{ext}"
        if candidate_index.exists():
            return candidate_index

    return None


def _resolve_relative(
    raw_import: str,
    current_file: Path,
    repo_root: Path,
) -> Optional[Path]:
    """Resolve a relative import path."""
    curr_dir = current_file.parent
    resolved = os.path.normpath(os.path.join(curr_dir, raw_import))
    full = Path(resolved)

    for ext in (".ts", ".vue", ".js", ".py"):
        candidate = full.with_suffix(ext)
        if candidate.exists():
            return candidate
        index_candidate = full / f"index{ext}"
        if index_candidate.exists():
            return index_candidate

    return None
