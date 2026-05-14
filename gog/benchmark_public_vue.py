"""Public Vue repository benchmark for GOG versus traditional RAG context selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gog_engine.token_utils import count_tokens_in_string

from gog_cli.onboarding import onboard_repository
from gog_cli.semantic_plan_benchmark import build_traditional_rag_bundle
from gog_cli.serving import build_context_bundle, summarize_repository


DEFAULT_REPO_URL = "https://github.com/mutoe/vue3-realworld-example-app"
DEFAULT_WORK_DIR = Path("gog") / "public-repos"
RESULTS_DIR = Path("gog") / "results"


@dataclass(frozen=True)
class ContextPrompt:
    id: str
    prompt: str
    expected_files: tuple[str, ...]
    notes: str


# First-stage benchmark prompts are intentionally context-selection tasks only.
# They avoid code mutation so public followers can reproduce results cheaply.
# Prompts are natural-language tasks, not exact filename hints, so misses are
# expected and should drive better graph/symbol onboarding rather than fixture tuning.
CONTEXT_PROMPTS = [
    ContextPrompt(
        id="auth_token_flow",
        prompt="Find the files responsible for attaching the authentication token to API requests.",
        expected_files=(
            "src/plugins/set-authorization-token.ts",
            "src/services/index.ts",
            "src/services/api.ts",
            "src/store/user.ts",
        ),
        notes="Cross-cutting auth/API request flow.",
    ),
    ContextPrompt(
        id="article_favorite_flow",
        prompt="Find the files involved when a user favorites or unfavorites an article from the article preview UI.",
        expected_files=(
            "src/components/ArticlesListArticlePreview.vue",
            "src/composable/use-favorite-article.ts",
            "src/services/index.ts",
            "src/services/api.ts",
        ),
        notes="Vue component to store/service flow.",
    ),
    ContextPrompt(
        id="router_guard_auth",
        prompt="Find the files that decide whether a route requires authentication before navigation.",
        expected_files=("src/router.ts", "src/store/user.ts"),
        notes="Routing and auth state flow.",
    ),
    ContextPrompt(
        id="profile_follow_flow",
        prompt="Find the files involved when following or unfollowing a user profile.",
        expected_files=(
            "src/pages/Profile.vue",
            "src/composable/use-follow-profile.ts",
            "src/composable/use-profile.ts",
            "src/services/index.ts",
            "src/services/api.ts",
        ),
        notes="Profile view, store, and API service flow.",
    ),
]


def run_public_vue_benchmark(
    repo_url: str = DEFAULT_REPO_URL,
    repo_path: Path | None = None,
    work_dir: Path = DEFAULT_WORK_DIR,
    output_dir: Path | None = None,
    refresh: bool = False,
    skip_clone: bool = False,
) -> dict[str, Any]:
    start = time.time()
    target_repo = resolve_target_repo(
        repo_url=repo_url,
        repo_path=repo_path,
        work_dir=work_dir,
        refresh=refresh,
        skip_clone=skip_clone,
    )

    onboard_start = time.time()
    manifest = onboard_repository(target_repo, force=True)
    onboarding_wall_clock_s = round(time.time() - onboard_start, 4)
    summary = summarize_repository(target_repo)

    prompt_results = []
    for prompt in CONTEXT_PROMPTS:
        gog_start = time.time()
        gog_bundle = build_context_bundle(target_repo, prompt.prompt)
        gog_wall_clock_s = round(time.time() - gog_start, 4)

        rag_start = time.time()
        rag_bundle = build_traditional_rag_bundle(target_repo, prompt.prompt)
        rag_wall_clock_s = round(time.time() - rag_start, 4)

        prompt_results.append(
            {
                "prompt_id": prompt.id,
                "prompt": prompt.prompt,
                "notes": prompt.notes,
                "expected_files": list(prompt.expected_files),
                "gog": score_gog_context(prompt, gog_bundle, gog_wall_clock_s),
                "traditional_rag": score_rag_context(prompt, rag_bundle, rag_wall_clock_s),
            }
        )

    payload = {
        "generated_at": _now_iso(),
        "benchmark": "public_vue_context_selection",
        "repo_url": repo_url,
        "repo_path": str(target_repo),
        "onboarding_wall_clock_s": onboarding_wall_clock_s,
        "total_wall_clock_s": round(time.time() - start, 4),
        "manifest": manifest,
        "repository_summary": summary,
        "prompts": prompt_results,
        "summary": summarize_prompt_results(prompt_results),
    }
    write_results(payload, output_dir=output_dir)
    return payload


def resolve_target_repo(
    repo_url: str,
    repo_path: Path | None,
    work_dir: Path,
    refresh: bool,
    skip_clone: bool,
) -> Path:
    if repo_path is not None:
        resolved = repo_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Repository path does not exist: {resolved}")
        return resolved

    repo_name = repo_url.rstrip("/").split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    target = work_dir.expanduser().resolve() / repo_name

    if skip_clone:
        if not target.exists():
            raise FileNotFoundError(f"Expected existing checkout at {target}")
        return target

    if target.exists() and refresh:
        run_command(["git", "-C", str(target), "pull", "--ff-only"])
    elif not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", repo_url, str(target)])

    return target


def score_gog_context(prompt: ContextPrompt, bundle: dict[str, Any], wall_clock_s: float) -> dict[str, Any]:
    selected_files = bundle["context"]["files"]
    expected_files = list(prompt.expected_files)
    return {
        "retrieval_mode": "gog",
        "selection_strategy": bundle["selection"]["strategy"],
        "selected_files": selected_files,
        "selected_file_count": len(selected_files),
        "relations": bundle["context"]["relations"],
        "relation_count": len(bundle["context"]["relations"]),
        "context_tokens_estimate": count_tokens_in_string(json.dumps(bundle)),
        "source_tokens_estimate": bundle["selection"]["estimated_input_tokens"],
        "wall_clock_s": wall_clock_s,
        **score_selection(expected_files, selected_files),
    }


def score_rag_context(prompt: ContextPrompt, bundle: dict[str, Any], wall_clock_s: float) -> dict[str, Any]:
    selected_files = bundle["retrieved_files"]
    expected_files = list(prompt.expected_files)
    return {
        "retrieval_mode": "traditional_rag_keyword_chunks",
        "selected_files": selected_files,
        "selected_file_count": len(selected_files),
        "chunk_count": len(bundle["chunks"]),
        "context_tokens_estimate": count_tokens_in_string(json.dumps(bundle)),
        "source_tokens_estimate": count_tokens_in_string("\n".join(chunk["text"] for chunk in bundle["chunks"])),
        "wall_clock_s": wall_clock_s,
        **score_selection(expected_files, selected_files),
    }


def score_selection(expected_files: list[str], selected_files: list[str]) -> dict[str, Any]:
    expected = set(expected_files)
    selected = set(selected_files)
    hits = sorted(expected.intersection(selected))
    missing = sorted(expected - selected)
    extra = sorted(selected - expected)
    precision = len(hits) / len(selected) if selected else 0.0
    recall = len(hits) / len(expected) if expected else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    noise_ratio = len(extra) / len(selected) if selected else 0.0
    return {
        "expected_hit_count": len(hits),
        "expected_missing_count": len(missing),
        "extra_file_count": len(extra),
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
        "context_f1": round(f1, 4),
        "noise_ratio": round(noise_ratio, 4),
        "pass_at_1_context": len(missing) == 0,
        "matched_expected_files": hits,
        "missing_expected_files": missing,
        "extra_files": extra,
    }


def summarize_prompt_results(prompt_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_count": len(prompt_results),
        "gog": summarize_mode(prompt_results, "gog"),
        "traditional_rag": summarize_mode(prompt_results, "traditional_rag"),
    }


def summarize_mode(prompt_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    rows = [row[key] for row in prompt_results]
    if not rows:
        return {}
    return {
        "pass_at_1_context_count": sum(int(row["pass_at_1_context"]) for row in rows),
        "avg_context_precision": round(sum(row["context_precision"] for row in rows) / len(rows), 4),
        "avg_context_recall": round(sum(row["context_recall"] for row in rows) / len(rows), 4),
        "avg_context_f1": round(sum(row["context_f1"] for row in rows) / len(rows), 4),
        "avg_noise_ratio": round(sum(row["noise_ratio"] for row in rows) / len(rows), 4),
        "avg_selected_file_count": round(sum(row["selected_file_count"] for row in rows) / len(rows), 2),
        "avg_context_tokens_estimate": round(sum(row["context_tokens_estimate"] for row in rows) / len(rows), 1),
        "avg_source_tokens_estimate": round(sum(row["source_tokens_estimate"] for row in rows) / len(rows), 1),
        "avg_wall_clock_s": round(sum(row["wall_clock_s"] for row in rows) / len(rows), 4),
    }


def run_command(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def write_results(payload: dict[str, Any], output_dir: Path | None = None) -> Path:
    root = output_dir or RESULTS_DIR
    root.mkdir(parents=True, exist_ok=True)
    filename = f"public_vue_context_selection_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    path = root / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark GOG versus RAG on a public Vue repository.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Public Git repository URL to benchmark.")
    parser.add_argument("--repo-path", help="Use an existing local checkout instead of cloning.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Directory for cloned public repos.")
    parser.add_argument("--output-dir", help="Optional output directory for JSON results.")
    parser.add_argument("--refresh", action="store_true", help="Run git pull --ff-only if the target checkout exists.")
    parser.add_argument("--skip-clone", action="store_true", help="Require an existing checkout in --work-dir.")
    args = parser.parse_args(argv)

    payload = run_public_vue_benchmark(
        repo_url=args.repo_url,
        repo_path=Path(args.repo_path) if args.repo_path else None,
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        refresh=args.refresh,
        skip_clone=args.skip_clone,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
