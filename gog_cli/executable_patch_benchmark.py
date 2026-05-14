"""Executable patch benchmark for GOG versus traditional RAG."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gog_engine.token_utils import count_tokens_in_string

from .onboarding import onboard_repository
from .reasoner_benchmark import DEFAULT_MODEL, extract_json_payload, normalize_cli_output
from .semantic_plan_benchmark import build_traditional_rag_bundle
from .serving import build_context_bundle


RESULTS_DIR = Path("gog") / "results"
DEFAULT_REPO = Path("gog") / "public-repos" / "vue3-realworld-example-app"
PATCH_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["files", "summary"],
    "properties": {
        "summary": {"type": "string"},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
        "uncertainty": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass(frozen=True)
class PatchTask:
    id: str
    domain: str
    difficulty: str
    prompt: str
    expected_files: tuple[str, ...]
    validation_commands: tuple[tuple[str, ...], ...]
    setup_patches: tuple[dict[str, str], ...]
    notes: str


TASKS = [
    PatchTask(
        id="debug_query_serialization_easy",
        domain="debugging",
        difficulty="easy",
        prompt=(
            "Fix the query parameter serialization helper so it URI-encodes keys and values, "
            "preserves 0 and false, and omits null or undefined values."
        ),
        expected_files=("src/utils/params-to-query.ts", "src/utils/params-to-query.spec.ts"),
        validation_commands=(("corepack", "pnpm", "vitest", "run", "src/utils/params-to-query.spec.ts"),),
        setup_patches=(
            {
                "path": "src/utils/params-to-query.ts",
                "content": """export default function params2query(params: Record<string, string | number | boolean | null | undefined>): string {
  return Object.entries(params)
    .filter(([, value]) => Boolean(value))
    .map(([key, value]) => `${key}=${String(value)}`)
    .join('&')
}
""",
            },
            {
                "path": "src/utils/params-to-query.spec.ts",
                "content": """import { describe, expect, it } from 'vitest'
import params2query from './params-to-query'

describe('# params2query', () => {
  it('should return query string given an object', () => {
    const params = {
      foo: 'bar',
      foo2: 'bar2',
    }

    const result = params2query(params)

    expect(result).toEqual('foo=bar&foo2=bar2')
  })

  it('should encode values and preserve falsy query parameters', () => {
    const result = params2query({
      page: 0,
      favorited: false,
      tag: 'vue js',
      author: 'a&b',
      empty: null,
      missing: undefined,
    })

    expect(result).toEqual('page=0&favorited=false&tag=vue%20js&author=a%26b')
  })
})
""",
            },
        ),
        notes="Small utility bug with focused acceptance test.",
    ),
    PatchTask(
        id="feature_pagination_edges_medium",
        domain="new_feature",
        difficulty="medium",
        prompt=(
            "Add previous and next pagination controls. Previous must emit the prior page unless "
            "already on page 1, and next must emit the next page unless already on the final page."
        ),
        expected_files=("src/components/AppPagination.vue", "src/components/AppPagination.spec.ts"),
        validation_commands=(("corepack", "pnpm", "vitest", "run", "src/components/AppPagination.spec.ts"),),
        setup_patches=(
            {
                "path": "src/components/AppPagination.spec.ts",
                "content": """import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render } from '@testing-library/vue'
import { renderOptions } from 'src/utils/test/test.utils'
import AppPagination from './AppPagination.vue'

describe('# AppPagination', () => {
  it('should highlight current active page', () => {
    const { getByRole } = render(AppPagination, renderOptions({
      props: { page: 1, count: 15 },
    }))

    expect(getByRole('link', { name: 'Go to page 1' }).parentNode).toHaveClass('active')
    expect(getByRole('link', { name: 'Go to page 2' }).parentNode).not.toHaveClass('active')
  })

  it('should call onPageChange when click a page item', async () => {
    const onPageChange = vi.fn()
    const { getByRole } = render(AppPagination, renderOptions({
      props: { page: 1, count: 15, onPageChange },
    }))

    await fireEvent.click(getByRole('link', { name: 'Go to page 2' }))

    expect(onPageChange).toHaveBeenCalledWith(2)
  })

  it('should emit previous and next page changes within pagination bounds', async () => {
    const onPageChange = vi.fn()
    const { getByRole } = render(AppPagination, renderOptions({
      props: { page: 2, count: 25, onPageChange },
    }))

    await fireEvent.click(getByRole('link', { name: 'Go to previous page' }))
    await fireEvent.click(getByRole('link', { name: 'Go to next page' }))

    expect(onPageChange).toHaveBeenNthCalledWith(1, 1)
    expect(onPageChange).toHaveBeenNthCalledWith(2, 3)
  })

  it('should not emit previous or next when already at the boundary', async () => {
    const onPageChange = vi.fn()
    const { getByRole, rerender } = render(AppPagination, renderOptions({
      props: { page: 1, count: 15, onPageChange },
    }))

    await fireEvent.click(getByRole('link', { name: 'Go to previous page' }))
    await rerender({ page: 2, count: 15, onPageChange })
    await fireEvent.click(getByRole('link', { name: 'Go to next page' }))

    expect(onPageChange).not.toHaveBeenCalled()
  })
})
""",
            },
        ),
        notes="New component behavior validated through UI-level tests.",
    ),
    PatchTask(
        id="debug_router_auth_hard",
        domain="debugging",
        difficulty="hard",
        prompt=(
            "Fix route authentication so unauthenticated users cannot access the authenticated feed, "
            "settings page, create article page, or edit article page. They should be redirected to login."
        ),
        expected_files=("src/router.ts", "src/router.spec.ts", "src/store/user.ts"),
        validation_commands=(("corepack", "pnpm", "vitest", "run", "src/router.spec.ts"),),
        setup_patches=(
            {
                "path": "src/router.spec.ts",
                "content": """import { describe, expect, it } from 'vitest'
import { userStorage } from 'src/store/user'
import fixtures from 'src/utils/test/fixtures'
import { createTestRouter } from 'src/utils/test/test.utils'

describe('# Router guards', () => {
  it('should redirect to home when logged-in user navigates to /login', async () => {
    userStorage.set(fixtures.user)
    const router = createTestRouter()
    await router.push('/login')
    await router.isReady()

    expect(router.currentRoute.value.name).toBe('global-feed')

    userStorage.remove()
  })

  it('should redirect to home when logged-in user navigates to /register', async () => {
    userStorage.set(fixtures.user)
    const router = createTestRouter()
    await router.push('/register')
    await router.isReady()

    expect(router.currentRoute.value.name).toBe('global-feed')

    userStorage.remove()
  })

  it('should allow unauthenticated user to access /login', async () => {
    userStorage.remove()
    const router = createTestRouter()
    await router.push('/login')
    await router.isReady()

    expect(router.currentRoute.value.name).toBe('login')
  })

  it.each(['/my-feeds', '/settings', '/article/create', '/article/example/edit'])(
    'should redirect unauthenticated users away from %s',
    async (path) => {
      userStorage.remove()
      const router = createTestRouter()
      await router.push(path)
      await router.isReady()

      expect(router.currentRoute.value.name).toBe('login')
    },
  )

  it('should match not-found route for unknown paths', async () => {
    const router = createTestRouter()
    await router.push('/some-nonexistent-path')
    await router.isReady()

    expect(router.currentRoute.value.name).toBe('not-found')
  })
})
""",
            },
        ),
        notes="Cross-cutting route/auth debugging task.",
    ),
]


def run_executable_patch_benchmark(
    repo_path: Path,
    model: str = DEFAULT_MODEL,
    output_dir: Path | None = None,
    task_ids: set[str] | None = None,
    modes: set[str] | None = None,
    attempts: int = 1,
    timeout_s: int = 600,
    retries: int = 1,
    retry_delay_s: int = 15,
    dry_run: bool = False,
    tasks: list[PatchTask] | None = None,
) -> dict[str, Any]:
    repo_root = repo_path.expanduser().resolve()
    _assert_env_ready(repo_root, dry_run)
    active_tasks = tasks if tasks else TASKS
    selected_tasks = [task for task in active_tasks if not task_ids or task.id in task_ids]
    selected_modes = [mode for mode in ("gog", "traditional_rag") if not modes or mode in modes]
    results: list[dict[str, Any]] = []

    for task in selected_tasks:
        for mode in selected_modes:
            print(f"running executable_patch task={task.id} mode={mode}", flush=True)
            results.append(
                run_task_mode(
                    source_repo=repo_root,
                    task=task,
                    mode=mode,
                    model=model,
                    attempts=attempts,
                    timeout_s=timeout_s,
                    retries=retries,
                    retry_delay_s=retry_delay_s,
                    dry_run=dry_run,
                )
            )

    payload = {
        "generated_at": _now_iso(),
        "benchmark": "public_vue_executable_patch",
        "model": model,
        "repo": str(repo_root.resolve()),
        "attempts_per_case": attempts,
        "transient_retries": retries,
        "retry_delay_s": retry_delay_s,
        "dry_run": dry_run,
        "tasks": [_task_metadata(task) for task in selected_tasks],
        "results": results,
        "summary": summarize_results(results),
    }
    write_results(payload, output_dir=output_dir)
    return payload


def run_task_mode(
    source_repo: Path,
    task: PatchTask,
    mode: str,
    model: str,
    attempts: int,
    timeout_s: int,
    retries: int,
    retry_delay_s: int,
    dry_run: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"gog-{task.id}-{mode}-") as tmp:
        work_repo = Path(tmp) / source_repo.name
        copy_repo(source_repo, work_repo)
        apply_setup_patches(work_repo, task)
        onboard_repository(work_repo, force=True)

        context = build_patch_context(work_repo, task, mode)
        prompt = build_patch_prompt(task, mode, context)
        base_result = {
            "task_id": task.id,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "mode": mode,
            "retrieved_files": context["files"],
            "context_tokens_estimate": context["tokens"],
            "prompt_tokens_estimate": count_tokens_in_string(prompt),
        }
        if dry_run:
            return {
                **base_result,
                "pass": False,
                "dry_run": True,
                "attempts": [],
                "tokens_to_pass": None,
                "attempts_to_pass": None,
                "wall_clock_to_pass_s": None,
            }

        attempts_payload = []
        cumulative_tokens = 0
        benchmark_start = time.time()
        for attempt_index in range(1, attempts + 1):
            response = invoke_ollama_stdin(
                model=model,
                prompt=prompt,
                timeout_s=timeout_s,
                retries=retries,
                retry_delay_s=retry_delay_s,
            )
            parsed = extract_json_payload(response["stdout"])
            response_tokens = count_tokens_in_string(response["stdout"])
            cumulative_tokens += count_tokens_in_string(prompt) + response_tokens
            patch_result = apply_model_patch(work_repo, parsed, allowed_files=context["files"])
            validation = run_validation(work_repo, task.validation_commands) if patch_result["applied"] else None
            passed = bool(validation and validation["passed"])
            attempts_payload.append(
                {
                    "attempt": attempt_index,
                    "returncode": response["returncode"],
                    "latency_s": response["latency_s"],
                    "transient_attempts": response.get("transient_attempts", []),
                    "response_tokens_estimate": response_tokens,
                    "json_valid": isinstance(parsed, dict),
                    "patch": patch_result,
                    "validation": validation,
                    "raw_response": response["stdout"],
                    "stderr": response["stderr"],
                }
            )
            if passed:
                return {
                    **base_result,
                    "pass": True,
                    "dry_run": False,
                    "attempts": attempts_payload,
                    "tokens_to_pass": cumulative_tokens,
                    "attempts_to_pass": attempt_index,
                    "wall_clock_to_pass_s": round(time.time() - benchmark_start, 4),
                }

        return {
            **base_result,
            "pass": False,
            "dry_run": False,
            "attempts": attempts_payload,
            "tokens_to_pass": None,
            "attempts_to_pass": None,
            "wall_clock_to_pass_s": None,
        }


def build_patch_context(repo_root: Path, task: PatchTask, mode: str) -> dict[str, Any]:
    if mode == "gog":
        bundle = build_context_bundle(repo_root, task.prompt)
        files = _merge_expected_hints(bundle["context"]["files"], task.expected_files)
        return {
            "mode": "gog",
            "files": files,
            "source_files": read_source_files(repo_root, files),
            "graph_relations": bundle["context"]["relations"],
            "context_membrane": bundle["context_membrane"],
            "validation_commands": [" ".join(command) for command in task.validation_commands],
            "tokens": count_tokens_in_string(json.dumps(bundle)),
        }
    if mode == "traditional_rag":
        bundle = build_traditional_rag_bundle(repo_root, task.prompt, max_files=6, max_chars_per_file=7000)
        files = _merge_expected_hints(bundle["retrieved_files"], task.expected_files)
        return {
            "mode": "traditional_rag",
            "files": files,
            "source_files": read_source_files(repo_root, files),
            "chunks": bundle["chunks"],
            "validation_commands": [" ".join(command) for command in task.validation_commands],
            "tokens": count_tokens_in_string(json.dumps(bundle)),
        }
    raise ValueError(f"Unknown mode: {mode}")


def build_patch_prompt(task: PatchTask, mode: str, context: dict[str, Any]) -> str:
    contract = (
        "Return ONLY a JSON object. Do not include markdown fences or prose. "
        "Return full replacement file contents only for files you change. "
        "Do not reference or modify files outside CONTEXT_JSON.files. "
        "Make the smallest executable change that satisfies the task and validation commands."
    )
    payload = {
        "task": _task_metadata(task),
        "retrieval_mode": mode,
        "context": context,
        "response_schema": PATCH_RESPONSE_SCHEMA,
    }
    return f"{contract}\n\nCONTEXT_JSON:\n{json.dumps(payload, indent=2)}"


def _merge_expected_hints(selected: list[str], expected: tuple[str, ...]) -> list[str]:
    """Keep executable benchmark fair by including acceptance-test files in both modes."""
    merged = list(dict.fromkeys([*selected, *expected]))
    return merged


def read_source_files(repo_root: Path, files: list[str]) -> list[dict[str, str]]:
    source_files = []
    for rel_path in files:
        path = repo_root / rel_path
        if not path.exists() or not path.is_file():
            continue
        source_files.append({"path": rel_path, "content": path.read_text(encoding="utf-8")})
    return source_files


def copy_repo(source_repo: Path, work_repo: Path) -> None:
    """Copy repo into a temp work directory, preserving language-specific dependency caches."""
    ignore_dirs = {".git", ".gog", "node_modules", "coverage", "dist", "__pycache__", ".pytest_cache"}
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignore_dirs or name.endswith(".egg-info")}

    shutil.copytree(source_repo, work_repo, ignore=ignore)

    # language-specific dependency symlinks
    for dep_dir in ("node_modules", ".venv", "venv"):
        source = source_repo / dep_dir
        if source.exists():
            target = work_repo / dep_dir
            if target.exists():
                target.unlink()
            target.symlink_to(source, target_is_directory=True)


def _assert_env_ready(repo_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if (repo_root / "pyproject.toml").exists() or (repo_root / "setup.py").exists():
        if not (repo_root / ".venv").exists() and not (repo_root / "venv").exists():
            print(f"[warn] No virtualenv found in {repo_root}; tests may fail if dependencies missing.")
        return
    if (repo_root / "package.json").exists():
        if not (repo_root / "node_modules").exists():
            raise RuntimeError(f"node_modules missing in {repo_root}. Run `npm install` or `pnpm install`.")
        return
    if (repo_root / "go.mod").exists():
        return


def apply_setup_patches(repo_root: Path, task: PatchTask) -> None:
    for patch in task.setup_patches:
        path = repo_root / patch["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch["content"], encoding="utf-8")


def apply_model_patch(repo_root: Path, parsed: dict[str, Any] | None, allowed_files: list[str]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"applied": False, "errors": ["response is not a JSON object"]}
    files = parsed.get("files")
    if not isinstance(files, list) or not files:
        return {"applied": False, "errors": ["response did not include files array"]}

    allowed = set(allowed_files)
    errors = []
    applied = []
    for item in files:
        if not isinstance(item, dict):
            errors.append("file entry is not an object")
            continue
        rel_path = item.get("path")
        content = item.get("content")
        if not isinstance(rel_path, str) or not isinstance(content, str):
            errors.append("file entry must include string path and content")
            continue
        if rel_path not in allowed:
            errors.append(f"attempted to modify file outside served context: {rel_path}")
            continue
        target = (repo_root / rel_path).resolve()
        if repo_root not in target.parents:
            errors.append(f"attempted path traversal: {rel_path}")
            continue
        target.write_text(content, encoding="utf-8")
        applied.append(rel_path)

    return {"applied": bool(applied) and not errors, "applied_files": applied, "errors": errors}


def run_validation(repo_root: Path, commands: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    command_results = []
    env = {**os.environ, "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for command in commands:
        start = time.time()
        completed = subprocess.run(
            list(command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        command_results.append(
            {
                "command": list(command),
                "returncode": completed.returncode,
                "wall_clock_s": round(time.time() - start, 4),
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }
        )
        if completed.returncode != 0:
            break
    return {
        "passed": all(result["returncode"] == 0 for result in command_results),
        "commands": command_results,
    }


def invoke_ollama_stdin(
    model: str,
    prompt: str,
    timeout_s: int,
    retries: int = 1,
    retry_delay_s: int = 15,
) -> dict[str, Any]:
    """Invoke Ollama through the local API so source text is never CLI-expanded as files."""
    start = time.time()
    attempts = []
    last_error = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "think": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            attempts.append({"attempt": attempt + 1, "returncode": 0, "stderr": ""})
            return {
                "stdout": normalize_cli_output(str(payload.get("response", ""))),
                "stderr": "",
                "returncode": 0,
                "latency_s": round(time.time() - start, 4),
                "transient_attempts": attempts,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = normalize_cli_output(str(error))
            attempts.append({"attempt": attempt + 1, "returncode": 1, "stderr": last_error})
            if attempt < retries and _is_transient_error(last_error):
                time.sleep(retry_delay_s)
                continue
            break

    return {
        "stdout": "",
        "stderr": last_error,
        "returncode": 1,
        "latency_s": round(time.time() - start, 4),
        "transient_attempts": attempts,
    }


def _is_transient_error(message: str) -> bool:
    normalized = message.lower()
    return any(marker in normalized for marker in ("503", "service unavailable", "timeout", "connection reset"))


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("gog", "traditional_rag"):
        rows = [row for row in results if row["mode"] == mode]
        if not rows:
            continue
        passed = [row for row in rows if row["pass"]]
        summary[mode] = {
            "cases": len(rows),
            "pass_at_1": sum(int(row["pass"] and row["attempts_to_pass"] == 1) for row in rows),
            "pass_at_k": len(passed),
            "avg_context_tokens_estimate": _avg(rows, "context_tokens_estimate"),
            "avg_prompt_tokens_estimate": _avg(rows, "prompt_tokens_estimate"),
            "avg_tokens_to_pass": _avg_present(passed, "tokens_to_pass"),
            "avg_attempts_to_pass": _avg_present(passed, "attempts_to_pass"),
            "avg_wall_clock_to_pass_s": _avg_present(passed, "wall_clock_to_pass_s"),
        }
    return summary


def write_results(payload: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = output_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = f"public_vue_executable_patch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path = root / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _task_metadata(task: PatchTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "prompt": task.prompt,
        "expected_files": list(task.expected_files),
        "validation_commands": [list(command) for command in task.validation_commands],
        "notes": task.notes,
    }


def _avg(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row[key]) for row in rows) / len(rows), 3)


def _avg_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _tail(text: str, max_chars: int = 5000) -> str:
    normalized = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
    return normalized[-max_chars:]


def load_tasks(tasks_file: str | None) -> list[PatchTask] | None:
    if not tasks_file:
        return None
    data = json.loads(Path(tasks_file).read_text(encoding="utf-8"))
    loaded = []
    for raw in data if isinstance(data, list) else [data]:
        commands = tuple(tuple(cmd) for cmd in raw["validation_commands"])
        patches = tuple(
            {"path": p["path"], "content": p["content"]}
            for p in raw.get("setup_patches", [])
        )
        loaded.append(
            PatchTask(
                id=raw["id"],
                domain=raw["domain"],
                difficulty=raw["difficulty"],
                prompt=raw["prompt"],
                expected_files=tuple(raw["expected_files"]),
                validation_commands=commands,
                setup_patches=patches,
                notes=raw.get("notes", ""),
            )
        )
    return loaded


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark executable patch quality for GOG versus RAG.")
    parser.add_argument("--repo", "-r", default=str(DEFAULT_REPO), help="Repository checkout path.")
    parser.add_argument("--tasks-file", help="JSON file defining PatchTasks for the target repo (loads custom tasks instead of built-in Vue tasks).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag to benchmark.")
    parser.add_argument("--output-dir", help="Optional output directory for JSON results.")
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--mode", action="append", dest="modes", choices=["gog", "traditional_rag"])
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per task/mode for Pass@k.")
    parser.add_argument("--timeout-s", type=int, default=600, help="Per-model-call timeout.")
    parser.add_argument("--retries", type=int, default=1, help="Retry transient Ollama/provider failures per attempt.")
    parser.add_argument("--retry-delay-s", type=int, default=15, help="Delay between transient retries.")
    parser.add_argument("--dry-run", action="store_true", help="Build contexts without invoking Ollama or tests.")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks_file)

    payload = run_executable_patch_benchmark(
        repo_path=Path(args.repo),
        model=args.model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        task_ids=set(args.tasks or []),
        modes=set(args.modes or []),
        attempts=args.attempts,
        timeout_s=args.timeout_s,
        retries=args.retries,
        retry_delay_s=args.retry_delay_s,
        dry_run=args.dry_run,
        tasks=tasks,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
