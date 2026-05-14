"""Semantic-quality benchmark for GOG-backed reasoner plans."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reasoner_benchmark import (
    DEFAULT_MODEL,
    SCHEMA_PATH,
    extract_json_payload,
    invoke_ollama,
    score_plan_output,
)
from .token_utils import count_tokens_in_string


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "semantic_plan_quality_known_problem.json"
RESULTS_DIR = Path("gog") / "results"


@dataclass(frozen=True)
class SemanticStrategy:
    name: str
    description: str


SEMANTIC_STRATEGIES = [
    SemanticStrategy("flat_json", "Baseline flat GOG context plus MutationPlan schema."),
    SemanticStrategy("problem_framed", "Flat GOG context plus explicit known-problem statement."),
    SemanticStrategy("rubric_first", "Semantic scoring rubric before context and schema."),
    SemanticStrategy("minimal_platter", "Small handoff surface with only task, problem, files, relations, and rubric."),
    SemanticStrategy("traditional_rag", "Keyword-retrieved unstructured source chunks without graph relations."),
]


def run_semantic_benchmark(
    repo_path: Path,
    model: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    timeout_s: int = 600,
    strategy_names: set[str] | None = None,
    retries: int = 1,
    retry_delay_s: int = 15,
) -> dict[str, Any]:
    raise RuntimeError(
        "Full GOG semantic plan benchmarking depends on GOG Professional. "
        "The public benchmark lab supports gog_lite and traditional_rag executable-patch benchmarks."
    )
    repo_root = repo_path.expanduser().resolve()
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    context_bundle = build_context_bundle(repo_root, fixture["prompt"])
    rag_bundle = build_traditional_rag_bundle(repo_root, fixture["prompt"])
    strategies = [
        strategy
        for strategy in SEMANTIC_STRATEGIES
        if not strategy_names or strategy.name in strategy_names
    ]
    results: list[dict[str, Any]] = []

    for strategy in strategies:
        print(f"running semantic_fixture={fixture['id']} strategy={strategy.name}", flush=True)
        prompt = build_semantic_prompt(
            strategy=strategy.name,
            fixture=fixture,
            context_bundle=context_bundle,
            rag_bundle=rag_bundle,
            schema=schema,
        )
        response = invoke_ollama(
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            retries=retries,
            retry_delay_s=retry_delay_s,
        )
        parsed = extract_json_payload(response["stdout"])
        structural_score = score_plan_output(
            raw_text=response["stdout"],
            parsed=parsed,
            schema=schema,
            allowed_nodes=_allowed_nodes_for_strategy(strategy.name, context_bundle, rag_bundle),
        )
        semantic_score = score_semantic_plan(
            parsed=parsed,
            fixture=fixture,
            allowed_nodes=_allowed_nodes_for_strategy(strategy.name, context_bundle, rag_bundle),
            context_has_source=strategy.name == "traditional_rag",
        )
        results.append(
            {
                "fixture_id": fixture["id"],
                "fixture": fixture,
                "strategy": strategy.name,
                "strategy_description": strategy.description,
                "model": model,
                "retrieval_mode": "traditional_rag" if strategy.name == "traditional_rag" else "gog",
                "context_bundle": context_bundle,
                "rag_bundle": rag_bundle if strategy.name == "traditional_rag" else None,
                "prompt_tokens_estimate": count_tokens_in_string(prompt),
                "response_tokens_estimate": count_tokens_in_string(response["stdout"]),
                "latency_s": response["latency_s"],
                "returncode": response["returncode"],
                "attempts": response["attempts"],
                "stderr": response["stderr"],
                "raw_response": response["stdout"],
                "parsed_plan": parsed,
                "structural_score": structural_score,
                "semantic_score": semantic_score,
            }
        )

    payload = {
        "generated_at": _now_iso(),
        "model": model,
        "repo": str(repo_root),
        "fixture_path": str(FIXTURE_PATH),
        "schema_path": str(SCHEMA_PATH),
        "results": results,
        "summary": summarize_semantic_results(results),
    }
    write_results(payload, output_dir=output_dir)
    return payload


def build_semantic_prompt(
    strategy: str,
    fixture: dict[str, Any],
    context_bundle: dict[str, Any],
    rag_bundle: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    schema_json = json.dumps(schema, indent=2)
    context_json = json.dumps(context_bundle, indent=2)
    problem_json = json.dumps(fixture, indent=2)
    rubric_json = json.dumps(_semantic_rubric(fixture), indent=2)
    contract = (
        "Return ONLY one JSON object. Do not use markdown fences. Do not include prose. "
        "The JSON object must satisfy the MutationPlan schema. "
        "Do not reference files outside context.allowed_nodes. "
        "Prefer INSPECT_NODE when the served context is insufficient for concrete mutation details."
    )

    if strategy == "flat_json":
        return (
            f"{contract}\n\nTASK:\n{fixture['prompt']}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "problem_framed":
        return (
            f"{contract}\n\nKNOWN_PROBLEM_JSON:\n{problem_json}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "rubric_first":
        return (
            f"{contract}\n\nSEMANTIC_GRADING_RUBRIC_JSON:\n{rubric_json}\n\n"
            f"KNOWN_PROBLEM_JSON:\n{problem_json}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "minimal_platter":
        minimal_bundle = {
            "task": fixture["prompt"],
            "known_problem": fixture["problem_statement"],
            "expected_semantic_behavior": fixture["expected_semantic_behavior"],
            "failure_modes": fixture["failure_modes"],
            "selected_files": context_bundle["context"]["files"],
            "relations": context_bundle["context"]["relations"],
            "allowed_nodes": context_bundle["context"]["allowed_nodes"],
            "validation_commands": context_bundle["context"]["validation_commands"],
            "handoff": context_bundle["handoff"],
        }
        return (
            f"{contract}\n\nGOG_MINIMAL_SEMANTIC_PLATTER_JSON:\n"
            f"{json.dumps(minimal_bundle, indent=2)}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "traditional_rag":
        rag_contract = (
            "Return ONLY one JSON object. Do not use markdown fences. Do not include prose. "
            "The JSON object must satisfy the MutationPlan schema. "
            "Do not reference files outside RAG_CONTEXT_JSON.retrieved_files. "
            "Ground the plan only in the retrieved unstructured text chunks."
        )
        rag_payload = {
            "task": fixture["prompt"],
            "retrieval_mode": "traditional_rag_keyword_chunks",
            "retrieved_files": rag_bundle["retrieved_files"],
            "chunks": rag_bundle["chunks"],
            "validation_hint": fixture["validation_hint"],
        }
        return (
            f"{rag_contract}\n\nRAG_CONTEXT_JSON:\n"
            f"{json.dumps(rag_payload, indent=2)}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    raise ValueError(f"Unknown semantic strategy: {strategy}")


def build_traditional_rag_bundle(
    repo_root: Path,
    prompt: str,
    max_files: int = 4,
    max_chars_per_file: int = 4500,
    max_source_tokens: int | None = None,
) -> dict[str, Any]:
    """Build a deliberately simple keyword RAG baseline from unstructured source text."""
    query_terms = _query_terms(prompt)
    explicit_files = _explicit_prompt_files(repo_root, prompt)
    explicit_rel_paths = {path.relative_to(repo_root).as_posix() for path in explicit_files}
    candidates: list[tuple[int, str, str]] = []
    for path in explicit_files + [
        path for path in _iter_source_files(repo_root)
        if path.relative_to(repo_root).as_posix() not in explicit_rel_paths
    ]:
        rel_path = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        haystack = f"{rel_path}\n{text}".lower()
        score = sum(haystack.count(term) for term in query_terms)
        if rel_path in explicit_rel_paths:
            score += 10000
        if score <= 0:
            continue
        candidates.append((score, rel_path, text[:max_chars_per_file]))

    ranked_candidates = sorted(candidates, key=lambda item: (-item[0], item[1]))
    ranked = []
    source_tokens = 0
    for score, rel_path, text in ranked_candidates:
        if len(ranked) >= max_files:
            break
        if max_source_tokens is not None and ranked:
            next_tokens = count_tokens_in_string(text)
            if source_tokens + next_tokens > max_source_tokens:
                break
        ranked.append((score, rel_path, text))
        source_tokens += count_tokens_in_string(text)
    return {
        "retrieved_files": [rel_path for _, rel_path, _ in ranked],
        "chunks": [
            {
                "file": rel_path,
                "score": score,
                "text": text,
            }
            for score, rel_path, text in ranked
        ],
        "max_source_tokens": max_source_tokens,
        "source_tokens_estimate": source_tokens,
    }


def _query_terms(prompt: str) -> set[str]:
    terms = {
        term.lower()
        for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", prompt)
        if term.lower() not in {"this", "that", "with", "from", "into", "downstream"}
    }
    for file_match in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|ts|vue|js)", prompt):
        path = Path(file_match)
        terms.add(path.name.lower())
        terms.add(path.stem.lower())
    return terms


def _explicit_prompt_files(repo_root: Path, prompt: str) -> list[Path]:
    matches: list[Path] = []
    seen = set()
    for mention in re.findall(r"[A-Za-z0-9_/.-]+\.(?:py|ts|vue|js)", prompt):
        candidate = repo_root / mention
        if candidate.exists() and candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            matches.append(candidate)
    return matches


def _iter_source_files(repo_root: Path):
    ignored_dirs = {".git", ".gog", ".venv", "__pycache__", "node_modules"}
    supported_suffixes = {".py", ".ts", ".vue", ".js"}
    for path in repo_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported_suffixes:
            continue
        if any(part in ignored_dirs for part in path.parts):
            continue
        yield path


def score_semantic_plan(
    parsed: dict[str, Any] | None,
    fixture: dict[str, Any],
    allowed_nodes: list[str],
    context_has_source: bool = False,
) -> dict[str, Any]:
    if not parsed:
        return {
            "target_focus": False,
            "inspect_before_mutate": False,
            "missing_source_uncertainty": False,
            "validation_alignment": False,
            "no_unsupported_targets": False,
            "semantic_score": 0,
            "semantic_errors": ["response is not a parsed MutationPlan"],
        }

    errors: list[str] = []
    targets = [target for target in parsed.get("targets", []) if isinstance(target, str)]
    steps = [step for step in parsed.get("steps", []) if isinstance(step, dict)]
    step_targets = [step.get("target") for step in steps if isinstance(step.get("target"), str)]
    all_targets = targets + step_targets

    expected_targets = set(fixture["expected_targets"])
    target_focus = bool(expected_targets.intersection(targets))
    if not target_focus:
        errors.append("plan does not include the expected primary target")

    unsupported_targets = sorted({target for target in all_targets if target not in allowed_nodes})
    no_unsupported_targets = not unsupported_targets
    if unsupported_targets:
        errors.append(f"plan references unsupported targets: {', '.join(unsupported_targets)}")

    uncertainty_text = " ".join(
        item.lower()
        for item in parsed.get("uncertainty", [])
        if isinstance(item, str)
    )
    missing_source_uncertainty = any(
        phrase in uncertainty_text
        for phrase in (
            "source",
            "excerpt",
            "implementation",
            "function",
            "symbol",
            "not provided",
            "not visible",
            "missing",
        )
    )
    first_mutate_index = _first_step_index(steps, "MUTATE_NODE")
    first_inspect_index = _first_step_index(steps, "INSPECT_NODE")
    inspect_before_mutate = (
        context_has_source
        or first_mutate_index is None
        or (first_inspect_index is not None and first_inspect_index < first_mutate_index)
    )
    if not inspect_before_mutate:
        errors.append("plan mutates before inspecting despite missing source content")

    source_awareness = True if context_has_source else missing_source_uncertainty
    if not source_awareness:
        errors.append("uncertainty does not acknowledge missing source or implementation detail")

    validation_text = " ".join(
        item.lower()
        for item in parsed.get("validation", [])
        if isinstance(item, str)
    )
    validation_alignment = "py_compile" in validation_text and "gog_cli/serving.py" in validation_text
    if not validation_alignment:
        errors.append("validation does not include python compile check for gog_cli/serving.py")

    checks = {
        "target_focus": target_focus,
        "inspect_before_mutate": inspect_before_mutate,
        "source_awareness": source_awareness,
        "validation_alignment": validation_alignment,
        "no_unsupported_targets": no_unsupported_targets,
    }
    return {
        **checks,
        "semantic_score": sum(int(value) for value in checks.values()),
        "semantic_errors": errors,
    }


def summarize_semantic_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in SEMANTIC_STRATEGIES:
        rows = [row for row in results if row["strategy"] == strategy.name]
        if not rows:
            continue
        successful_rows = [row for row in rows if row["returncode"] == 0]
        summary[strategy.name] = {
            "observations": len(rows),
            "execution_success_count": len(successful_rows),
            "execution_failure_count": len(rows) - len(successful_rows),
            "json_valid_count": sum(int(row["structural_score"]["json_valid"]) for row in rows),
            "schema_valid_count": sum(int(row["structural_score"]["schema_valid"]) for row in rows),
            "strict_json_only_count": sum(int(row["structural_score"]["no_extra_prose"]) for row in rows),
            "avg_structural_score": round(
                sum(row["structural_score"]["composite_score"] for row in rows) / len(rows),
                3,
            ),
            "avg_semantic_score": round(
                sum(row["semantic_score"]["semantic_score"] for row in rows) / len(rows),
                3,
            ),
            "avg_success_semantic_score": _avg_successful(
                successful_rows,
                lambda row: row["semantic_score"]["semantic_score"],
            ),
            "avg_success_total_tokens_estimate": _avg_successful(
                successful_rows,
                lambda row: row["prompt_tokens_estimate"] + row["response_tokens_estimate"],
            ),
            "avg_prompt_tokens_estimate": round(
                sum(row["prompt_tokens_estimate"] for row in rows) / len(rows),
                1,
            ),
            "avg_response_tokens_estimate": round(
                sum(row["response_tokens_estimate"] for row in rows) / len(rows),
                1,
            ),
        }
    return summary


def _allowed_nodes_for_strategy(
    strategy: str,
    context_bundle: dict[str, Any],
    rag_bundle: dict[str, Any],
) -> list[str]:
    if strategy == "traditional_rag":
        return rag_bundle["retrieved_files"]
    return context_bundle["context"]["allowed_nodes"]


def _avg_successful(rows: list[dict[str, Any]], value_fn) -> float | None:
    if not rows:
        return None
    return round(sum(value_fn(row) for row in rows) / len(rows), 3)


def write_results(payload: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = output_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = f"semantic_plan_quality_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path = root / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _semantic_rubric(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_score": 5,
        "checks": [
            "target_focus: plan targets the expected primary file",
            "inspect_before_mutate: plan inspects before mutating when context lacks source, or uses retrieved source when available",
            "source_awareness: plan acknowledges missing source for GOG structural context, or relies on retrieved source for RAG context",
            "validation_alignment: validation includes the expected Python compile check",
            "no_unsupported_targets: all targets are within context.allowed_nodes",
        ],
        "expected_targets": fixture["expected_targets"],
        "known_failure_modes": fixture["failure_modes"],
    }


def _first_step_index(steps: list[dict[str, Any]], op: str) -> int | None:
    for index, step in enumerate(steps):
        if step.get("op") == op:
            return index
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark semantic quality of GOG reasoner plans.")
    parser.add_argument("repo", nargs="?", default=".", help="Repository root with onboarded .gog artifacts.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to benchmark.")
    parser.add_argument("--output-dir", help="Optional output directory for JSON results.")
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-call timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retry transient Ollama failures per case.")
    parser.add_argument("--retry-delay-s", type=int, default=15, help="Delay between transient failure retries.")
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        choices=[strategy.name for strategy in SEMANTIC_STRATEGIES],
        help="Limit the benchmark to one or more named semantic prompt strategies.",
    )
    args = parser.parse_args(argv)

    payload = run_semantic_benchmark(
        repo_path=Path(args.repo),
        model=args.model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        timeout_s=args.timeout_s,
        strategy_names=set(args.strategies or []),
        retries=args.retries,
        retry_delay_s=args.retry_delay_s,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
