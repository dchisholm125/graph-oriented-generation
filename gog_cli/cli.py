"""Command-line interface for repo onboarding and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .onboarding import inspect_onboarding, onboard_repository, refresh_repository
from .serving import build_context_bundle, summarize_repository


console = Console()
error_console = Console(stderr=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gog",
        description="Graph-Oriented Generation repository onboarding CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    onboard = subparsers.add_parser(
        "onboard",
        help="Build and persist GOG artifacts for a repository.",
    )
    onboard.add_argument("repo", nargs="?", default=".", help="Repository root to onboard.")
    onboard.add_argument(
        "--artifact-dir",
        help="Optional artifact directory. Defaults to <repo>/.gog.",
    )
    onboard.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing manifest in the artifact directory.",
    )
    onboard.add_argument(
        "--json",
        action="store_true",
        help="Print the manifest payload as JSON after onboarding.",
    )

    refresh = subparsers.add_parser(
        "refresh",
        help="Rebuild GOG artifacts after accepted repository changes.",
    )
    refresh.add_argument("repo", nargs="?", default=".", help="Repository root to refresh.")
    refresh.add_argument(
        "--artifact-dir",
        help="Optional artifact directory. Defaults to <repo>/.gog.",
    )
    refresh.add_argument(
        "--json",
        action="store_true",
        help="Print the refreshed manifest payload as JSON.",
    )

    inspect = subparsers.add_parser(
        "inspect",
        help="Inspect an existing GOG onboarding artifact set.",
    )
    inspect.add_argument("repo", nargs="?", default=".", help="Repository root to inspect.")
    inspect.add_argument(
        "--artifact-dir",
        help="Optional artifact directory. Defaults to <repo>/.gog.",
    )
    inspect.add_argument(
        "--json",
        action="store_true",
        help="Print the inspection payload as JSON.",
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Emit a repo-level orientation digest from onboarding artifacts.",
    )
    summarize.add_argument("repo", nargs="?", default=".", help="Repository root to summarize.")
    summarize.add_argument(
        "--artifact-dir",
        help="Optional artifact directory. Defaults to <repo>/.gog.",
    )
    summarize.add_argument(
        "--json",
        action="store_true",
        help="Print the summary payload as JSON.",
    )

    context = subparsers.add_parser(
        "context",
        help="Serve a bounded prompt-scoped repository context bundle.",
    )
    context.add_argument("repo", nargs="?", default=".", help="Repository root to query.")
    context.add_argument(
        "--prompt",
        required=True,
        help="Prompt to preprocess into a task-scoped symbolic context bundle.",
    )
    context.add_argument(
        "--artifact-dir",
        help="Optional artifact directory. Defaults to <repo>/.gog.",
    )
    context.add_argument(
        "--json",
        action="store_true",
        help="Print the context bundle as JSON.",
    )
    return parser


def _print_onboard_summary(payload: dict) -> None:
    repo = payload["repo"]
    graph = payload["graph"]
    artifacts = payload["artifacts"]

    console.print("[bold green]GOG onboarding complete.[/bold green]")
    console.print(f"Repository: {repo['root']}")
    console.print(f"Artifacts:  {artifacts['root']}")
    console.print(
        f"Graph:      {graph['node_count']} nodes, "
        f"{graph['edge_count']} edges, "
        f"{graph['coverage']['supported_file_count']} supported files"
    )
    console.print(f"State:      {payload['status']}")


def _print_inspect_summary(payload: dict) -> None:
    manifest = payload["manifest"]
    repo = manifest["repo"]
    graph = manifest["graph"]
    commands = manifest["validation"]["commands"]

    table = Table(title="GOG Onboarding Inspection")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Repository", repo["root"])
    table.add_row("Artifacts", manifest["artifacts"]["root"])
    table.add_row("Status", payload["status"])
    table.add_row("Graph", f"{graph['node_count']} nodes / {graph['edge_count']} edges")
    table.add_row(
        "Supported Files",
        str(graph["coverage"]["supported_file_count"]),
    )
    table.add_row(
        "Validation Commands",
        ", ".join(commands) if commands else "none detected",
    )
    console.print(table)

    if payload["issues"]:
        console.print("[bold yellow]Issues:[/bold yellow]")
        for issue in payload["issues"]:
            console.print(f"- {issue}")


def _print_summary(payload: dict) -> None:
    table = Table(title="GOG Repository Summary")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Repository", payload["repo"]["root"])
    table.add_row("Status", payload["status"])
    table.add_row(
        "Coverage",
        f"{payload['coverage']['supported_graph_files']} supported graph files / "
        f"{payload['coverage']['total_files']} total files",
    )
    table.add_row(
        "Graph",
        f"{payload['graph']['node_count']} nodes / {payload['graph']['edge_count']} edges",
    )
    table.add_row(
        "Validation",
        ", ".join(payload["validation"]["commands"]) if payload["validation"]["commands"] else "none detected",
    )
    console.print(table)


def _print_context_summary(payload: dict) -> None:
    selection = payload["selection"]
    context = payload["context"]

    table = Table(title="GOG Prompt Context")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Prompt", payload["prompt"])
    table.add_row("Status", payload["status"])
    table.add_row("Strategy", selection["strategy"])
    table.add_row("Files", str(selection["selected_file_count"]))
    table.add_row("Estimated Tokens", str(selection["estimated_input_tokens"]))
    table.add_row(
        "Validation",
        ", ".join(context["validation_commands"]) if context["validation_commands"] else "none detected",
    )
    console.print(table)
    if context["files"]:
        console.print("[bold]Selected files:[/bold]")
        for path in context["files"]:
            console.print(f"- {path}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "onboard":
            payload = onboard_repository(
                repo_path=Path(args.repo),
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
                force=args.force,
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_onboard_summary(payload)
            return

        if args.command == "refresh":
            payload = refresh_repository(
                repo_path=Path(args.repo),
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_onboard_summary(payload)
            return

        if args.command == "inspect":
            payload = inspect_onboarding(
                repo_path=Path(args.repo),
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_inspect_summary(payload)
            return

        if args.command == "summarize":
            payload = summarize_repository(
                repo_path=Path(args.repo),
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_summary(payload)
            return

        if args.command == "context":
            payload = build_context_bundle(
                repo_path=Path(args.repo),
                prompt=args.prompt,
                artifact_dir=Path(args.artifact_dir) if args.artifact_dir else None,
            )
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_context_summary(payload)
            return
    except Exception as exc:
        error_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
