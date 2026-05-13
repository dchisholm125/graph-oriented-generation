"""Repository onboarding for persistent GOG artifact creation."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from gog_engine import ast_parser
from .python_graph import build_python_graph


SCHEMA_VERSION = "0.1.0"
SUPPORTED_GRAPH_EXTENSIONS = {".py", ".ts", ".vue"}
IGNORED_DIRS = {
    ".git",
    ".gog",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def onboard_repository(
    repo_path: Path,
    artifact_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build persistent onboarding artifacts for *repo_path*."""
    repo_root = repo_path.expanduser().resolve()
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"Repository path is not a directory: {repo_root}")

    artifacts_root = (
        artifact_dir.expanduser().resolve()
        if artifact_dir is not None
        else repo_root / ".gog"
    )
    manifest_path = artifacts_root / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Onboarding manifest already exists at {manifest_path}. "
            "Re-run with --force to refresh it."
        )

    profile = _build_repo_profile(repo_root)
    graph = _build_structural_graph(repo_root, profile)
    graph_payload = _serialize_graph(repo_root, graph)
    validation_commands = _detect_validation_commands(repo_root)
    fingerprint = _repo_fingerprint(repo_root, profile["files"]["supported_graph_files"])

    _ensure_artifact_dirs(artifacts_root)
    _write_json(artifacts_root / "repo_profile.json", profile)
    _write_json(artifacts_root / "graphs" / "structural_graph.json", graph_payload)
    with open(artifacts_root / "graphs" / "structural_graph.pkl", "wb") as fh:
        pickle.dump(graph, fh)
    _write_json(
        artifacts_root / "validation" / "commands.json",
        {"commands": validation_commands},
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "status": "READY",
        "repo": {
            "root": str(repo_root),
            "fingerprint": fingerprint,
        },
        "artifacts": {
            "root": str(artifacts_root),
            "manifest": str(manifest_path),
            "repo_profile": str(artifacts_root / "repo_profile.json"),
            "structural_graph_json": str(artifacts_root / "graphs" / "structural_graph.json"),
            "structural_graph_pickle": str(artifacts_root / "graphs" / "structural_graph.pkl"),
            "validation_commands": str(artifacts_root / "validation" / "commands.json"),
        },
        "graph": {
            "kind": "structural_import_graph",
            "supported_extensions": sorted(SUPPORTED_GRAPH_EXTENSIONS),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "coverage": {
                "supported_file_count": len(profile["files"]["supported_graph_files"]),
                "total_file_count": profile["files"]["total_files"],
            },
        },
        "validation": {
            "commands": validation_commands,
        },
        "notes": [
            "This onboarding slice persists a structural Python/TS/Vue import graph.",
            "Future onboarding layers should add symbol, workflow, constraint, and evolution graphs.",
        ],
    }
    _write_json(manifest_path, manifest)
    return manifest


def inspect_onboarding(
    repo_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Inspect previously generated onboarding artifacts."""
    repo_root = repo_path.expanduser().resolve()
    artifacts_root = (
        artifact_dir.expanduser().resolve()
        if artifact_dir is not None
        else repo_root / ".gog"
    )
    manifest_path = artifacts_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No onboarding manifest found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if Path(manifest["repo"]["root"]).resolve() != repo_root:
        issues.append("Manifest repo root does not match the requested repository path.")

    repo_profile_path = Path(manifest["artifacts"]["repo_profile"])
    if repo_profile_path.exists():
        profile = json.loads(repo_profile_path.read_text(encoding="utf-8"))
        current_fingerprint = _repo_fingerprint(
            repo_root,
            profile["files"]["supported_graph_files"],
        )
        if current_fingerprint != manifest["repo"]["fingerprint"]:
            issues.append("Repository fingerprint changed since onboarding; refresh is recommended.")
    else:
        issues.append("Repository profile artifact is missing.")

    for artifact_key in (
        "structural_graph_json",
        "structural_graph_pickle",
        "validation_commands",
    ):
        artifact_path = Path(manifest["artifacts"][artifact_key])
        if not artifact_path.exists():
            issues.append(f"Artifact missing: {artifact_path}")

    status = "READY" if not issues else "STALE_REFRESH_REQUIRED"
    return {
        "status": status,
        "issues": issues,
        "manifest": manifest,
    }


def refresh_repository(
    repo_path: Path,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """Refresh onboarding artifacts after accepted repository changes."""
    return onboard_repository(
        repo_path=repo_path,
        artifact_dir=artifact_dir,
        force=True,
    )


def _build_repo_profile(repo_root: Path) -> dict[str, Any]:
    extension_counts: Counter[str] = Counter()
    supported_graph_files: list[str] = []
    total_files = 0
    manifests = []

    for path in _iter_repo_files(repo_root):
        total_files += 1
        suffix = path.suffix.lower() or "<none>"
        extension_counts[suffix] += 1
        rel_path = _rel(repo_root, path)
        if suffix in SUPPORTED_GRAPH_EXTENSIONS:
            supported_graph_files.append(rel_path)
        if path.name in {
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "vite.config.ts",
            "vite.config.js",
            "tsconfig.json",
        }:
            manifests.append(rel_path)

    return {
        "created_at": _now_iso(),
        "root": str(repo_root),
        "files": {
            "total_files": total_files,
            "supported_graph_files": sorted(supported_graph_files),
            "extension_counts": dict(sorted(extension_counts.items())),
        },
        "manifests": sorted(manifests),
        "parser_coverage": {
            "structural_graph_extensions": sorted(SUPPORTED_GRAPH_EXTENSIONS),
            "supported_graph_file_count": len(supported_graph_files),
        },
    }


def _serialize_graph(repo_root: Path, graph) -> dict[str, Any]:
    nodes = sorted(_rel(repo_root, Path(node)) for node in graph.nodes())
    edges = sorted(
        {
            (_rel(repo_root, Path(source)), _rel(repo_root, Path(target)))
            for source, target in graph.edges()
        }
    )
    return {
        "kind": "structural_import_graph",
        "created_at": _now_iso(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": [{"source": source, "target": target} for source, target in edges],
    }


def _build_structural_graph(repo_root: Path, profile: dict[str, Any]) -> nx.DiGraph:
    ts_vue_graph = ast_parser.build_graph(str(repo_root))
    python_files = [
        repo_root / rel_path
        for rel_path in profile["files"]["supported_graph_files"]
        if rel_path.endswith(".py")
    ]
    python_graph = build_python_graph(repo_root, python_files)
    return nx.compose(ts_vue_graph, python_graph)


def _detect_validation_commands(repo_root: Path) -> list[str]:
    commands: list[str] = []
    package_json = repo_root / "package.json"
    if package_json.exists():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get("scripts", {})
            for name in ("typecheck", "test", "test:unit", "lint", "build"):
                if name in scripts:
                    commands.append(f"npm run {name}")
        except json.JSONDecodeError:
            pass

    if (repo_root / "pytest.ini").exists() or any(repo_root.glob("test_*.py")):
        commands.append("pytest")
    if (repo_root / "requirements.txt").exists():
        commands.append("python3 -m py_compile")

    return commands


def _repo_fingerprint(repo_root: Path, rel_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel_path in sorted(rel_paths):
        path = repo_root / rel_path
        digest.update(rel_path.encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        stat = path.stat()
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _ensure_artifact_dirs(artifacts_root: Path) -> None:
    (artifacts_root / "graphs").mkdir(parents=True, exist_ok=True)
    (artifacts_root / "validation").mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_repo_files(repo_root: Path):
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [directory for directory in dirs if directory not in IGNORED_DIRS]
        for file_name in files:
            yield Path(root) / file_name


def _rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
