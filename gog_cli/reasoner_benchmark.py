"""Prompt-strategy benchmark for GOG-backed reasoner plan generation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .token_utils import count_tokens_in_string


DEFAULT_MODEL = "kimi-k2.6:cloud"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "mutation_plan.schema.json"
RESULTS_DIR = Path("gog") / "results"
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


BENCHMARK_TASKS = [
    {
        "id": "python_onboarding_refactor",
        "prompt": "Refactor the onboarding workflow in gog_cli/onboarding.py to make artifact refresh semantics clearer without changing external behavior.",
    },
    {
        "id": "python_cli_argument_flow",
        "prompt": "Improve the CLI argument parsing flow in gog_cli/cli.py while preserving the current commands.",
    },
    {
        "id": "python_context_serving",
        "prompt": "Improve prompt-scoped context serving in gog_cli/serving.py to make the reasoner handoff easier to inspect.",
    },
]


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str


STRATEGIES = [
    Strategy("flat_json", "Raw JSON bundle followed by a strict JSON response contract."),
    Strategy("sectioned_brief", "Labeled textual sections with schema requirements at the end."),
    Strategy("evidence_first", "Emphasize citing served evidence before selecting mutations."),
    Strategy("schema_first", "Lead with the MutationPlan schema and then provide repo context."),
]


def run_benchmark(
    repo_path: Path,
    model: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    timeout_s: int = 600,
    task_ids: set[str] | None = None,
    strategy_names: set[str] | None = None,
    retries: int = 0,
    retry_delay_s: int = 15,
) -> dict[str, Any]:
    raise RuntimeError(
        "Full GOG reasoner prompt benchmarking depends on GOG Professional. "
        "Use gog/benchmark_executable_patch.py --mode gog_lite for the public reference benchmark."
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    repo_root = repo_path.expanduser().resolve()
    results: list[dict[str, Any]] = []
    tasks = [task for task in BENCHMARK_TASKS if not task_ids or task["id"] in task_ids]
    strategies = [strategy for strategy in STRATEGIES if not strategy_names or strategy.name in strategy_names]

    for task in tasks:
        context_bundle = build_context_bundle(repo_root, task["prompt"])
        for strategy in strategies:
            print(f"running task={task['id']} strategy={strategy.name}", flush=True)
            prompt = build_strategy_prompt(
                strategy=strategy.name,
                task_prompt=task["prompt"],
                context_bundle=context_bundle,
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
            score = score_plan_output(
                raw_text=response["stdout"],
                parsed=parsed,
                schema=schema,
                allowed_nodes=context_bundle["context"]["allowed_nodes"],
            )
            results.append(
                {
                    "task_id": task["id"],
                    "task_prompt": task["prompt"],
                    "strategy": strategy.name,
                    "strategy_description": strategy.description,
                    "model": model,
                    "context_bundle": context_bundle,
                    "prompt_tokens_estimate": count_tokens_in_string(prompt),
                    "response_tokens_estimate": count_tokens_in_string(response["stdout"]),
                    "latency_s": response["latency_s"],
                    "returncode": response["returncode"],
                    "attempts": response["attempts"],
                    "stderr": response["stderr"],
                    "raw_response": response["stdout"],
                    "parsed_plan": parsed,
                    "score": score,
                }
            )

    summary = summarize_results(results)
    payload = {
        "generated_at": _now_iso(),
        "model": model,
        "repo": str(repo_root),
        "schema_path": str(SCHEMA_PATH),
        "results": results,
        "summary": summary,
    }
    write_results(payload, output_dir=output_dir)
    return payload


def build_strategy_prompt(
    strategy: str,
    task_prompt: str,
    context_bundle: dict[str, Any],
    schema: dict[str, Any],
) -> str:
    context_json = json.dumps(context_bundle, indent=2)
    schema_json = json.dumps(schema, indent=2)
    contract = (
        "Return ONLY one JSON object. Do not use markdown fences. Do not include prose. "
        "The JSON object must satisfy the provided MutationPlan schema. "
        "Do not reference files outside context.allowed_nodes. "
        "If the task cannot be planned safely from the served context, keep steps minimal and explain the issue in uncertainty."
    )

    if strategy == "flat_json":
        return (
            f"{contract}\n\nTASK:\n{task_prompt}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "sectioned_brief":
        return (
            f"TASK\n{task_prompt}\n\n"
            f"SELECTED FILES\n{json.dumps(context_bundle['context']['files'], indent=2)}\n\n"
            f"GRAPH RELATIONS\n{json.dumps(context_bundle['context']['relations'], indent=2)}\n\n"
            f"ALLOWED NODES\n{json.dumps(context_bundle['context']['allowed_nodes'], indent=2)}\n\n"
            f"VALIDATION COMMANDS\n{json.dumps(context_bundle['context']['validation_commands'], indent=2)}\n\n"
            f"HANDOFF RULES\n{json.dumps(context_bundle['handoff'], indent=2)}\n\n"
            f"RESPONSE CONTRACT\n{contract}\n\n"
            f"MUTATION PLAN SCHEMA\n{schema_json}"
        )
    if strategy == "evidence_first":
        return (
            "Use only evidence from the served GOG context. Populate `evidence` with concrete strings "
            "that identify files or relations used. Ground every target and step in that evidence.\n\n"
            f"{contract}\n\nTASK:\n{task_prompt}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}\n\n"
            f"MUTATION_PLAN_SCHEMA_JSON:\n{schema_json}"
        )
    if strategy == "schema_first":
        return (
            f"{contract}\n\nMUTATION_PLAN_SCHEMA_JSON:\n{schema_json}\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"GOG_CONTEXT_BUNDLE_JSON:\n{context_json}"
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def invoke_ollama(
    model: str,
    prompt: str,
    timeout_s: int,
    retries: int = 0,
    retry_delay_s: int = 15,
) -> dict[str, Any]:
    start = time.time()
    attempts: list[dict[str, Any]] = []
    completed = None
    for attempt in range(retries + 1):
        completed = subprocess.run(
            ["ollama", "run", "--hidethinking", "--nowordwrap", model, prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "returncode": completed.returncode,
                "stderr": normalize_cli_output(completed.stderr),
            }
        )
        if completed.returncode == 0 or not _should_retry(completed.stderr):
            break
        if attempt < retries:
            time.sleep(retry_delay_s)

    if completed is None:
        raise RuntimeError("ollama invocation did not run")

    return {
        "stdout": normalize_cli_output(completed.stdout),
        "stderr": normalize_cli_output(completed.stderr),
        "returncode": completed.returncode,
        "latency_s": round(time.time() - start, 4),
        "attempts": attempts,
    }


def normalize_cli_output(text: str) -> str:
    """Remove terminal rendering artifacts from captured CLI streams."""
    without_ansi = ANSI_ESCAPE_RE.sub("", text)
    without_controls = CONTROL_CHARS_RE.sub("", without_ansi)
    return without_controls.strip()


def _should_retry(stderr: str) -> bool:
    normalized = stderr.lower()
    return any(
        marker in normalized
        for marker in (
            "503 service unavailable",
            "temporarily overloaded",
            "timeout",
            "connection reset",
        )
    )


def extract_json_payload(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None


def score_plan_output(
    raw_text: str,
    parsed: dict[str, Any] | None,
    schema: dict[str, Any],
    allowed_nodes: list[str],
) -> dict[str, Any]:
    json_valid = parsed is not None
    schema_valid, schema_errors = validate_plan_shape(parsed, schema)
    targets, step_targets = collect_targets(parsed)
    unsupported_targets = sorted(
        {
            target for target in targets + step_targets
            if target not in allowed_nodes
        }
    )
    allowed_node_compliance = not unsupported_targets and json_valid
    no_extra_prose = bool(parsed) and raw_text.strip().startswith("{") and raw_text.strip().endswith("}")

    return {
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "schema_errors": schema_errors,
        "allowed_node_compliance": allowed_node_compliance,
        "unsupported_targets": unsupported_targets,
        "no_extra_prose": no_extra_prose,
        "evidence_count": len(parsed.get("evidence", [])) if parsed else 0,
        "step_count": len(parsed.get("steps", [])) if parsed else 0,
        "composite_score": sum(
            int(value)
            for value in (
                json_valid,
                schema_valid,
                allowed_node_compliance,
                no_extra_prose,
            )
        ),
    }


def validate_plan_shape(
    parsed: dict[str, Any] | None,
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    if parsed is None:
        return False, ["response is not a JSON object"]

    errors: list[str] = []
    required = schema["required"]
    for key in required:
        if key not in parsed:
            errors.append(f"missing required key: {key}")

    extra_keys = sorted(set(parsed) - set(schema["properties"]))
    if extra_keys:
        errors.append(f"unexpected keys: {', '.join(extra_keys)}")

    if not isinstance(parsed.get("targets"), list) or not parsed.get("targets"):
        errors.append("targets must be a non-empty list")
    if not isinstance(parsed.get("steps"), list) or not parsed.get("steps"):
        errors.append("steps must be a non-empty list")

    valid_ops = set(schema["properties"]["steps"]["items"]["properties"]["op"]["enum"])
    for index, step in enumerate(parsed.get("steps", []) if isinstance(parsed.get("steps"), list) else []):
        if not isinstance(step, dict):
            errors.append(f"step[{index}] is not an object")
            continue
        for key in schema["properties"]["steps"]["items"]["required"]:
            if key not in step:
                errors.append(f"step[{index}] missing key: {key}")
        if step.get("op") not in valid_ops:
            errors.append(f"step[{index}] has invalid op: {step.get('op')}")

    return not errors, errors


def collect_targets(parsed: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not parsed:
        return [], []
    targets = [target for target in parsed.get("targets", []) if isinstance(target, str)]
    step_targets = [
        step.get("target")
        for step in parsed.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("target"), str)
    ]
    return targets, step_targets


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for strategy in STRATEGIES:
        rows = [row for row in results if row["strategy"] == strategy.name]
        if not rows:
            continue
        successful_rows = [row for row in rows if row["returncode"] == 0]
        summary[strategy.name] = {
            "observations": len(rows),
            "execution_success_count": len(successful_rows),
            "execution_failure_count": len(rows) - len(successful_rows),
            "json_valid_count": sum(int(row["score"]["json_valid"]) for row in rows),
            "schema_valid_count": sum(int(row["score"]["schema_valid"]) for row in rows),
            "allowed_node_compliance_count": sum(int(row["score"]["allowed_node_compliance"]) for row in rows),
            "strict_json_only_count": sum(int(row["score"]["no_extra_prose"]) for row in rows),
            "avg_composite_score": round(
                sum(row["score"]["composite_score"] for row in rows) / len(rows),
                3,
            ),
            "avg_success_composite_score": _avg_successful(
                successful_rows,
                lambda row: row["score"]["composite_score"],
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


def _avg_successful(rows: list[dict[str, Any]], value_fn) -> float | None:
    if not rows:
        return None
    return round(sum(value_fn(row) for row in rows) / len(rows), 3)


def write_results(payload: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = output_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = f"reasoner_prompt_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path = root / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark GOG reasoner prompt strategies via Ollama.")
    parser.add_argument("repo", nargs="?", default=".", help="Repository root with onboarded .gog artifacts.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to benchmark.")
    parser.add_argument("--output-dir", help="Optional output directory for JSON results.")
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-call timeout in seconds.")
    parser.add_argument("--retries", type=int, default=0, help="Retry transient Ollama failures per case.")
    parser.add_argument("--retry-delay-s", type=int, default=15, help="Delay between transient failure retries.")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=[task["id"] for task in BENCHMARK_TASKS],
        help="Limit the benchmark to one or more named tasks.",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        choices=[strategy.name for strategy in STRATEGIES],
        help="Limit the benchmark to one or more named prompt strategies.",
    )
    args = parser.parse_args(argv)

    payload = run_benchmark(
        repo_path=Path(args.repo),
        model=args.model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        timeout_s=args.timeout_s,
        task_ids=set(args.tasks or []),
        strategy_names=set(args.strategies or []),
        retries=args.retries,
        retry_delay_s=args.retry_delay_s,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
