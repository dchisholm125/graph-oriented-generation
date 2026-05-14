"""Gold-context metadata and scoring helpers for GOG benchmarks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoldContext:
    task_id: str
    gold_files: tuple[str, ...]
    gold_symbols: tuple[str, ...]
    expected_edit_files: tuple[str, ...]
    failure_mode: str


def gold_context_from_task(task: Any) -> GoldContext:
    """Return explicit gold context when present, otherwise derive a stable default."""
    gold_files = tuple(getattr(task, "gold_files", ()) or getattr(task, "expected_files", ()))
    expected_edit_files = tuple(
        getattr(task, "expected_edit_files", ())
        or _derive_expected_edit_files(getattr(task, "expected_files", ()))
    )
    failure_mode = str(getattr(task, "failure_mode", "") or getattr(task, "notes", ""))
    return GoldContext(
        task_id=str(getattr(task, "id")),
        gold_files=gold_files,
        gold_symbols=tuple(getattr(task, "gold_symbols", ()) or ()),
        expected_edit_files=expected_edit_files,
        failure_mode=failure_mode,
    )


def gold_context_to_dict(gold_context: GoldContext) -> dict[str, Any]:
    return {
        "task_id": gold_context.task_id,
        "gold_files": list(gold_context.gold_files),
        "gold_symbols": list(gold_context.gold_symbols),
        "expected_edit_files": list(gold_context.expected_edit_files),
        "failure_mode": gold_context.failure_mode,
    }


def score_context_selection(gold_context: GoldContext, selected_files: list[str]) -> dict[str, Any]:
    gold = set(gold_context.gold_files)
    selected = set(selected_files)
    hits = sorted(gold.intersection(selected))
    missing = sorted(gold - selected)
    noise = sorted(selected - gold)
    precision = len(hits) / len(selected) if selected else 0.0
    recall = len(hits) / len(gold) if gold else 1.0
    noise_ratio = len(noise) / len(selected) if selected else 0.0
    return {
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
        "noise_ratio": round(noise_ratio, 4),
        "gold_hit_count": len(hits),
        "gold_missing_count": len(missing),
        "noise_file_count": len(noise),
        "matched_gold_files": hits,
        "missing_gold_files": missing,
        "noise_files": noise,
    }


def score_edit_surface(gold_context: GoldContext, edited_files: list[str]) -> dict[str, Any]:
    expected = set(gold_context.expected_edit_files)
    edited = set(edited_files)
    return {
        "files_edited": sorted(edited),
        "expected_edit_files": sorted(expected),
        "unexpected_edit_files": sorted(edited - expected),
        "missing_expected_edit_files": sorted(expected - edited),
        "spurious_edit_file_count": len(edited - expected),
    }


def count_spurious_import_lines(file_patches: list[dict[str, Any]], expected_edit_files: tuple[str, ...]) -> int:
    expected = set(expected_edit_files)
    count = 0
    for patch in file_patches:
        path = patch.get("path")
        content = patch.get("content", "")
        if path in expected or not isinstance(content, str):
            continue
        count += len(re.findall(r"^\s*(?:import|from\s+\S+\s+import)\s+", content, flags=re.MULTILINE))
    return count


def _derive_expected_edit_files(expected_files: tuple[str, ...]) -> tuple[str, ...]:
    source_files = [
        path
        for path in expected_files
        if not _looks_like_test_file(path)
    ]
    return tuple(source_files or expected_files)


def _looks_like_test_file(path: str) -> bool:
    normalized = path.lower()
    return (
        "/test" in normalized
        or normalized.startswith("test")
        or ".spec." in normalized
        or ".test." in normalized
    )
