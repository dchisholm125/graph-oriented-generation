"""Small public token-count helpers used by benchmark harnesses."""

from __future__ import annotations

from pathlib import Path


def count_tokens_in_string(text: str) -> int:
    """Return a deterministic rough token estimate without external packages."""
    return len(text.split())


def count_tokens_in_files(paths: list[str]) -> int:
    total = 0
    for raw_path in paths:
        try:
            total += count_tokens_in_string(Path(raw_path).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return total
