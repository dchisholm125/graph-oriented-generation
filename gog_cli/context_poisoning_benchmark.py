"""Context poisoning benchmark for GOG versus progressively larger RAG contexts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .executable_patch_benchmark import (
    DEFAULT_REPO,
    RESULTS_DIR,
    TASKS,
    _assert_env_ready,
    load_tasks,
    run_task_mode,
)
from .failure_taxonomy import failure_taxonomy_table, summarize_failure_classes
from .reasoner_benchmark import DEFAULT_MODEL


DEFAULT_RAG_BUDGETS = (1000, 4000, 16000, 64000)


def run_context_poisoning_benchmark(
    repo_path: Path,
    model: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    task_ids: set[str] | None = None,
    attempts: int = 1,
    timeout_s: int = 600,
    retries: int = 1,
    retry_delay_s: int = 15,
    dry_run: bool = False,
    tasks: list[Any] | None = None,
    rag_budgets: tuple[int, ...] = DEFAULT_RAG_BUDGETS,
    trials: int = 1,
) -> dict[str, Any]:
    repo_root = repo_path.expanduser().resolve()
    _assert_env_ready(repo_root, dry_run)
    active_tasks = tasks if tasks else TASKS
    selected_tasks = [task for task in active_tasks if not task_ids or task.id in task_ids]

    results = []
    for trial_index in range(1, trials + 1):
        for task in selected_tasks:
            print(f"running context_poisoning trial={trial_index} task={task.id} mode=gog", flush=True)
            gog_row = run_task_mode(
                source_repo=repo_root,
                task=task,
                mode="gog",
                model=model,
                attempts=attempts,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
                dry_run=dry_run,
                rag_source_token_budget=None,
            )
            gog_row["trial_index"] = trial_index
            gog_row["poison_source_token_budget"] = None
            results.append(gog_row)

            for budget in rag_budgets:
                print(
                    f"running context_poisoning trial={trial_index} "
                    f"task={task.id} mode=traditional_rag budget={budget}",
                    flush=True,
                )
                rag_row = run_task_mode(
                    source_repo=repo_root,
                    task=task,
                    mode="traditional_rag",
                    model=model,
                    attempts=attempts,
                    timeout_s=timeout_s,
                    retries=retries,
                    retry_delay_s=retry_delay_s,
                    dry_run=dry_run,
                    rag_source_token_budget=budget,
                )
                rag_row["trial_index"] = trial_index
                rag_row["poison_source_token_budget"] = budget
                results.append(rag_row)

    payload = {
        "generated_at": _now_iso(),
        "benchmark": "context_poisoning_executable_patch",
        "model": model,
        "repo": str(repo_root),
        "attempts_per_case": attempts,
        "transient_retries": retries,
        "retry_delay_s": retry_delay_s,
        "dry_run": dry_run,
        "trials": trials,
        "rag_source_token_budgets": list(rag_budgets),
        "failure_taxonomy": failure_taxonomy_table(),
        "results": results,
        "summary": summarize_poisoning_results(results),
    }
    write_results(payload, output_dir=output_dir)
    return payload


def summarize_poisoning_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    gog_summary = _summarize_rows([row for row in results if row["mode"] == "gog"])
    summary: dict[str, Any] = {"gog": gog_summary}
    budgets = sorted(
        {
            row["poison_source_token_budget"]
            for row in results
            if row["mode"] == "traditional_rag" and row["poison_source_token_budget"] is not None
        }
    )
    summary["traditional_rag_by_budget"] = {}
    for budget in budgets:
        budget_summary = _summarize_rows(
            [
                row
                for row in results
                if row["mode"] == "traditional_rag" and row["poison_source_token_budget"] == budget
            ]
        )
        _add_relative_costs(budget_summary, gog_summary)
        summary["traditional_rag_by_budget"][str(budget)] = budget_summary
    _add_relative_costs(gog_summary, gog_summary)
    return summary


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    passed = [row for row in rows if row["pass"]]
    return {
        "cases": len(rows),
        "pass_at_1": sum(int(row["pass"] and row["attempts_to_pass"] == 1) for row in rows),
        "pass_at_k": len(passed),
        "pass_at_1_rate": round(
            sum(int(row["pass"] and row["attempts_to_pass"] == 1) for row in rows) / len(rows),
            4,
        ),
        "avg_context_precision": _avg(rows, lambda row: row["context_metrics"]["context_precision"]),
        "avg_context_recall": _avg(rows, lambda row: row["context_metrics"]["context_recall"]),
        "avg_noise_ratio": _avg(rows, lambda row: row["context_metrics"]["noise_ratio"]),
        "avg_context_tokens_estimate": _avg(rows, lambda row: row["context_tokens_estimate"]),
        "avg_prompt_tokens_estimate": _avg(rows, lambda row: row["prompt_tokens_estimate"]),
        "total_tokens_spent": round(sum(float(row.get("tokens_spent") or 0) for row in rows), 3),
        "tokens_spent_per_pass": _tokens_spent_per_pass(rows),
        "avg_tokens_to_pass": _avg_present(passed, lambda row: row["tokens_to_pass"]),
        "avg_wall_clock_to_pass_s": _avg_present(passed, lambda row: row["wall_clock_to_pass_s"]),
        "failures": summarize_failure_classes(rows),
    }


def _avg(rows: list[dict[str, Any]], value_fn) -> float:
    return round(sum(float(value_fn(row)) for row in rows) / len(rows), 3)


def _avg_present(rows: list[dict[str, Any]], value_fn) -> float | None:
    values = [float(value_fn(row)) for row in rows if value_fn(row) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _tokens_spent_per_pass(rows: list[dict[str, Any]]) -> float | None:
    pass_count = sum(int(row["pass"]) for row in rows)
    if pass_count == 0:
        return None
    return round(sum(float(row.get("tokens_spent") or 0) for row in rows) / pass_count, 3)


def _add_relative_costs(summary: dict[str, Any], gog_summary: dict[str, Any]) -> None:
    summary["relative_prompt_cost_vs_gog"] = _relative(
        summary.get("avg_prompt_tokens_estimate"),
        gog_summary.get("avg_prompt_tokens_estimate"),
    )
    summary["relative_cost_to_pass_vs_gog"] = _relative(
        summary.get("tokens_spent_per_pass"),
        gog_summary.get("tokens_spent_per_pass"),
    )


def _relative(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return round(float(value) / float(baseline), 3)


def summary_markdown_table(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = [("GOG", summary["gog"])]
    rows.extend(
        (f"RAG {budget}", values)
        for budget, values in summary["traditional_rag_by_budget"].items()
    )
    lines = [
        "| Mode | Pass@1 | Avg prompt tokens | Relative prompt cost vs GOG | Tokens spent / pass | Relative cost-to-pass vs GOG | Noise ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{values['pass_at_1']}/{values['cases']}",
                    _fmt_number(values["avg_prompt_tokens_estimate"]),
                    _fmt_multiplier(values["relative_prompt_cost_vs_gog"]),
                    _fmt_number(values["tokens_spent_per_pass"]),
                    _fmt_multiplier(values["relative_cost_to_pass_vs_gog"]),
                    f"{values['avg_noise_ratio']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "| Mode | Recoverable failures | Unrecoverable failures | High architectural concern | Failure classes |", "| --- | ---: | ---: | ---: | --- |"])
    for label, values in rows:
        failures = values["failures"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(failures["recoverable_failure_count"]),
                    str(failures["unrecoverable_failure_count"]),
                    str(failures["high_architectural_concern_count"]),
                    _fmt_failure_counts(failures["failure_class_counts"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.0f}"


def _fmt_multiplier(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _fmt_failure_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def write_results(payload: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = output_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = f"context_poisoning_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path = root / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(summary_markdown_table(payload), encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the context poisoning executable patch benchmark.")
    parser.add_argument("--repo", "-r", default=str(DEFAULT_REPO), help="Repository checkout path.")
    parser.add_argument("--tasks-file", help="JSON file defining PatchTasks for the target repo.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to benchmark.")
    parser.add_argument("--output-dir", help="Optional output directory for JSON results.")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per task/mode for Pass@k.")
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-model-call timeout.")
    parser.add_argument("--retries", type=int, default=1, help="Retry transient Ollama/provider failures per attempt.")
    parser.add_argument("--retry-delay-s", type=int, default=15, help="Delay between transient retries.")
    parser.add_argument("--rag-budget", action="append", type=int, dest="rag_budgets", help="RAG source token budget. Repeat for sweeps.")
    parser.add_argument("--trials", type=int, default=1, help="Independent repetitions of every task/mode/budget case.")
    parser.add_argument("--dry-run", action="store_true", help="Build contexts without invoking Ollama or tests.")
    args = parser.parse_args(argv)

    payload = run_context_poisoning_benchmark(
        repo_path=Path(args.repo),
        model=args.model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        task_ids=set(args.tasks or []),
        attempts=args.attempts,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_delay_s=args.retry_delay_s,
        dry_run=args.dry_run,
        tasks=load_tasks(args.tasks_file),
        rag_budgets=tuple(args.rag_budgets or DEFAULT_RAG_BUDGETS),
        trials=args.trials,
    )
    print(json.dumps(payload["summary"], indent=2))
    print(summary_markdown_table(payload))


if __name__ == "__main__":
    main()
