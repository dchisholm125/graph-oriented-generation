"""Failure taxonomy for executable patch benchmark attempts."""

from __future__ import annotations

from typing import Any


FAILURE_CLASSES: dict[str, dict[str, str | bool]] = {
    "none": {
        "label": "None",
        "recoverable_by_retry": False,
        "architectural_concern": "none",
        "description": "The attempt passed validation.",
    },
    "invalid_json": {
        "label": "Invalid JSON",
        "recoverable_by_retry": True,
        "architectural_concern": "low",
        "description": "The model did not return a parseable JSON patch payload.",
    },
    "invalid_syntax": {
        "label": "Invalid syntax",
        "recoverable_by_retry": True,
        "architectural_concern": "medium",
        "description": "The patch applied but produced a syntax or parser error.",
    },
    "missing_semantic_behavior": {
        "label": "Missing semantic behavior",
        "recoverable_by_retry": False,
        "architectural_concern": "high",
        "description": "The patch applied and compiled but failed task acceptance behavior.",
    },
    "spurious_import": {
        "label": "Spurious import",
        "recoverable_by_retry": True,
        "architectural_concern": "medium",
        "description": "The patch introduced imports outside expected edit files.",
    },
    "wrong_file_edited": {
        "label": "Wrong file edited",
        "recoverable_by_retry": False,
        "architectural_concern": "high",
        "description": "The model edited, or attempted to edit, files outside the expected edit surface.",
    },
    "patch_rejected": {
        "label": "Patch rejected",
        "recoverable_by_retry": True,
        "architectural_concern": "medium",
        "description": "The model returned JSON, but the benchmark rejected the patch contract.",
    },
    "validation_failure": {
        "label": "Validation failure",
        "recoverable_by_retry": False,
        "architectural_concern": "high",
        "description": "The patch failed validation for an unclassified reason.",
    },
}


def classify_attempt_failure(
    *,
    passed: bool,
    parsed: dict[str, Any] | None,
    patch_result: dict[str, Any],
    edit_metrics: dict[str, Any],
    spurious_imports: int,
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    if passed:
        return _classification("none")

    patch_errors = patch_result.get("errors", [])
    if not isinstance(parsed, dict) or "response is not a JSON object" in patch_errors:
        return _classification("invalid_json")

    if edit_metrics.get("spurious_edit_file_count", 0) > 0 or _attempted_file_boundary_error(patch_errors):
        return _classification("wrong_file_edited")

    if spurious_imports > 0:
        return _classification("spurious_import")

    if patch_errors:
        return _classification("patch_rejected", evidence=patch_errors)

    validation_text = _validation_failure_text(validation)
    if _looks_like_syntax_error(validation_text):
        return _classification("invalid_syntax", evidence=validation_text)

    if validation_text:
        return _classification("missing_semantic_behavior", evidence=validation_text)

    return _classification("validation_failure")


def summarize_failure_classes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_rows = [
        row
        for row in rows
        if not row.get("pass") and not row.get("dry_run") and row.get("attempts")
    ]
    counts: dict[str, int] = {}
    recoverable = 0
    unrecoverable = 0
    high_architectural_concern = 0
    for row in failed_rows:
        failure = row.get("final_failure_class") or {}
        name = failure.get("name", "validation_failure")
        counts[name] = counts.get(name, 0) + 1
        if failure.get("recoverable_by_retry"):
            recoverable += 1
        else:
            unrecoverable += 1
        if failure.get("architectural_concern") == "high":
            high_architectural_concern += 1
    return {
        "failed_cases": len(failed_rows),
        "failure_class_counts": counts,
        "recoverable_failure_count": recoverable,
        "unrecoverable_failure_count": unrecoverable,
        "high_architectural_concern_count": high_architectural_concern,
    }


def failure_taxonomy_table() -> list[dict[str, Any]]:
    rows = []
    for name, metadata in FAILURE_CLASSES.items():
        if name == "none":
            continue
        rows.append({"name": name, **metadata})
    return rows


def _classification(name: str, evidence: Any = None) -> dict[str, Any]:
    metadata = FAILURE_CLASSES[name]
    result = {"name": name, **metadata}
    if evidence:
        result["evidence_tail"] = _tail(str(evidence))
    return result


def _attempted_file_boundary_error(patch_errors: list[str]) -> bool:
    return any(
        "outside served context" in error or "path traversal" in error
        for error in patch_errors
    )


def _validation_failure_text(validation: dict[str, Any] | None) -> str:
    if not validation:
        return ""
    chunks = []
    for command in validation.get("commands", []):
        if command.get("returncode") == 0:
            continue
        chunks.append(command.get("stderr_tail", ""))
        chunks.append(command.get("stdout_tail", ""))
    return "\n".join(chunk for chunk in chunks if chunk)


def _looks_like_syntax_error(text: str) -> bool:
    normalized = text.lower()
    return any(
        marker in normalized
        for marker in (
            "syntaxerror",
            "error parsing",
            "unexpected token",
            "parse error",
            "unterminated",
            "expected",
        )
    ) and not _looks_like_assertion_failure(normalized)


def _looks_like_assertion_failure(text: str) -> bool:
    return "assertionerror" in text or "expected:" in text and "received:" in text


def _tail(text: str, max_chars: int = 1000) -> str:
    return text[-max_chars:]
