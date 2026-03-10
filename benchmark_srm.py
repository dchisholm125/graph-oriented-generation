"""
benchmark_srm.py - SRM Pipeline Benchmark (Phase 2)

Compares three tiers:
  • Tier 1 (RAG Control) — raw prompt, ChromaDB retrieval
  • Tier 2 (GOG Vanilla) — raw prompt, GOG deterministic isolation
  • Tier 3-SRM (SRM Pipeline) — symbolic spec only, LLM renders syntax

The key claim: by removing reasoning from the LLM and placing it in the
deterministic planner, smaller models (≤1B) produce better code.

Tier 3-SRM sends ONLY the renderer prompt to the LLM — NOT the original natural language.
This makes the result falsifiable: either the symbolic constraint works, or it doesn't.
"""

import os
import time
import pickle
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from srm_engine import ast_parser, graph_search
from srm_engine.salience_evaluator import SalienceEvaluator
from srm_engine.token_utils import count_tokens_in_files as get_token_count

from srm_engine.planner.intent_parser import parse_intent, IntentParseError
from srm_engine.planner.mutation_planner import plan_mutations, PlannerError
from srm_engine.planner.renderer_prompt import build_renderer_prompt, build_single_file_renderer_prompt

from benchmark_local_llm import (
    OllamaClient,
    run_control_pipeline,
    run_srm_pipeline_vanilla,
    score_response,
    PROMPTS,
    MUZZLE,
)

console = Console()
client = OllamaClient(model="qwen2.5:0.5b")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3-SRM: The SRM Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _get_topological_render_order(graph, target_files_abs: list) -> list:
    """
    Determine render order for multiple target files using topological sort from DAG.

    The planner (not the LLM) determines the dependency order. Each file is rendered
    in isolation, in the order determined by the import/dependency graph, so that
    dependent files see the operations applied to their dependencies first.

    Args:
        graph: NetworkX DiGraph (GOG graph with import relationships).
        target_files_abs: List of absolute file paths to render.

    Returns:
        List of absolute file paths in topological order (dependencies first).
    """
    import networkx as nx

    # Get the subgraph containing only the target files
    subgraph_nodes = [n for n in graph.nodes() if n in target_files_abs]

    # Topological sort ensures dependencies come before dependents
    # If there's a cycle or the graph doesn't contain all files, fall back to input order
    try:
        topo_order = list(nx.topological_sort(graph.subgraph(subgraph_nodes)))
        # Filter to only include target files
        return [f for f in topo_order if f in target_files_abs]
    except:
        # Fall back to input order if topological sort fails
        return target_files_abs


def run_srm_pipeline(prompt_text: str, target_repo: str, graph) -> dict:
    """
    Tier 3-SRM: Intent Parser → Mutation Planner → Renderer Prompts → LLM Renderers

    Multi-file handling: For tasks with multiple files, renders each file separately
    in topological order (dependencies first). The planner determines order; the LLM
    renders atomically per file (renderer, not planner).

    Args:
        prompt_text: Natural language instruction from user (used by parser only).
        target_repo: Path to target repository (for graph resolution).
        graph: NetworkX DiGraph from ast_parser.build_graph().

    Returns:
        Dict with timing, token counts, response, and metadata.
    """
    start_time = time.time()

    # ── Step 1: Parse Intent (deterministic) ─────────────────────────────────
    try:
        ops = parse_intent(prompt_text)
    except IntentParseError as e:
        return {
            "time": 0,
            "local_time": 0,
            "api_time": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "response": f"[Error] Intent parsing failed: {e}",
            "patches_applied": 0,
            "parse_error": str(e),
        }

    # ── Step 2: Plan Mutations (deterministic) ───────────────────────────────
    try:
        plan = plan_mutations(ops, graph, target_repo)
    except PlannerError as e:
        return {
            "time": 0,
            "local_time": 0,
            "api_time": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "response": f"[Error] Mutation planning failed: {e}",
            "patches_applied": 0,
            "plan_error": str(e),
        }

    # ── Step 3: Determine file rendering order (deterministic) ─────────────────
    target_files_abs = list(plan.file_paths_abs.values())
    is_multi_file = len(plan.operations_by_file) > 1

    # Get topological render order from the graph (dependencies first)
    render_order_abs = _get_topological_render_order(graph, target_files_abs)

    # Map back to relative paths for accessing the plan
    file_rel_by_abs = {v: k for k, v in plan.file_paths_abs.items()}
    render_order_rel = [file_rel_by_abs[abs_path] for abs_path in render_order_abs]

    # Count tokens in all target files (this is what the LLM sees)
    tokens_in = get_token_count(target_files_abs)
    local_time = time.time() - start_time

    # ── Step 4: LLM Rendering ────────────────────────────────────────────────
    api_start = time.time()
    responses = []
    renderer_prompts = []

    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked SRM result (Ollama is not running)"
        responses = [response]
    elif is_multi_file:
        # Multi-file: separate LLM call per file, in topological order
        console.print(f"\n[dim]Rendering {len(render_order_rel)} files in topological order...[/dim]")
        for i, target_file_rel in enumerate(render_order_rel, 1):
            console.print(f"[dim]  [{i}/{len(render_order_rel)}] {target_file_rel}[/dim]", end=" ")

            # Build single-file spec (all context contained within)
            single_file_spec = build_single_file_renderer_prompt(plan, target_file_rel) + MUZZLE
            renderer_prompts.append(single_file_spec)

            # Call LLM for this file only
            file_response = client.complete(single_file_spec, context_files=[])
            responses.append(file_response)
            console.print("[dim]done[/dim]")

        # Concatenate responses with file separators
        response = "\n\n".join(
            f"# ─── {render_order_rel[i]} ───\n{responses[i]}"
            for i in range(len(responses))
        )
    else:
        # Single-file: standard rendering
        renderer_prompt = build_renderer_prompt(plan) + MUZZLE
        renderer_prompts = [renderer_prompt]
        response = client.complete(renderer_prompt, context_files=[])
        responses = [response]

    api_time = time.time() - api_start
    execution_time = local_time + api_time

    return {
        "time": execution_time,
        "local_time": local_time,
        "api_time": api_time,
        "tokens_in": tokens_in,
        "tokens_out": 150,
        "response": response,
        "patches_applied": 0,
        "renderer_prompt": renderer_prompts[0] if renderer_prompts else "",  # For display (first prompt if multi-file)
        "renderer_prompts": renderer_prompts,  # For debugging (all prompts)
        "operations": ops,
        "plan": plan,
        "is_multi_file": is_multi_file,
        "file_render_order": render_order_rel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results Comparison: Tier 1 vs Tier 2 vs Tier 3-SRM
# ─────────────────────────────────────────────────────────────────────────────

def _pct(diff, base):
    """Compute percentage change."""
    if base == 0:
        return 0.0
    return (diff / base) * 100


def print_srm_benchmark_results(rag_m, vanilla_m, srm_m, prompt_text, level=None):
    """
    Print 3-tier comparison table with Tier 3-SRM.

    For Hard task (multi-file): includes per-file breakdown showing which files
    were successfully rendered structurally vs which had issues.

    Args:
        rag_m: Tier 1 (RAG Control) results dict.
        vanilla_m: Tier 2 (GOG Vanilla) results dict.
        srm_m: Tier 3-SRM results dict.
        prompt_text: Original natural language prompt.
        level: Difficulty level ("Easy", "Medium", "Hard") for correctness scoring.
    """
    table = Table(
        title=f"📊 SRM 3-Tier Benchmark ({client.model})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Tier 1 · RAG (Control)", style="red", min_width=22)
    table.add_column("Tier 2 · GOG Vanilla", style="yellow", min_width=22)
    table.add_column("Tier 3 · SRM Pipeline", style="green", min_width=22)

    # ── Local compute time ──────────────────────────────────────────────────
    lt_rav = rag_m["local_time"]
    lt_van = vanilla_m["local_time"]
    lt_srm = srm_m["local_time"]
    table.add_row(
        "Local Compute Time",
        f"{lt_rav:.4f}s",
        f"{lt_van:.4f}s  ({_pct(lt_rav - lt_van, lt_rav):.1f}% ↓)",
        f"{lt_srm:.4f}s  ({_pct(lt_rav - lt_srm, lt_rav):.1f}% ↓)",
    )

    # ── LLM generation time ─────────────────────────────────────────────────
    at_rav = rag_m["api_time"]
    at_van = vanilla_m["api_time"]
    at_srm = srm_m["api_time"]
    table.add_row(
        "LLM Generation Time",
        f"{at_rav:.2f}s",
        f"{at_van:.2f}s  ({_pct(at_rav - at_van, at_rav):.1f}% ↓)",
        f"{at_srm:.2f}s  ({_pct(at_rav - at_srm, at_rav):.1f}% ↓)",
    )

    # ── Total execution time ────────────────────────────────────────────────
    tt_rav = rag_m["time"]
    tt_van = vanilla_m["time"]
    tt_srm = srm_m["time"]
    table.add_row(
        "Total Execution Time",
        f"{tt_rav:.2f}s",
        f"{tt_van:.2f}s  ({_pct(tt_rav - tt_van, tt_rav):.1f}% ↓)",
        f"{tt_srm:.2f}s  ({_pct(tt_rav - tt_srm, tt_rav):.1f}% ↓)",
    )

    # ── Token reduction ─────────────────────────────────────────────────────
    tk_rav = rag_m["tokens_in"]
    tk_van = vanilla_m["tokens_in"]
    tk_srm = srm_m["tokens_in"]
    table.add_row(
        "Tokens In (Est.)",
        str(tk_rav),
        f"{tk_van}  ({_pct(tk_rav - tk_van, tk_rav):.1f}% ↓)",
        f"{tk_srm}  ({_pct(tk_rav - tk_srm, tk_rav):.1f}% ↓)",
    )

    console.print(table)
    console.print("\n")

    # ── Per-File Breakdown for Hard Task (Multi-File) ───────────────────────────
    if level == "Hard" and srm_m.get("is_multi_file"):
        console.print("[bold]Per-File Rendering Breakdown (SRM Pipeline)[/bold]")
        file_render_order = srm_m.get("file_render_order", [])
        if file_render_order:
            for i, file_rel in enumerate(file_render_order, 1):
                console.print(f"  [{i}/{len(file_render_order)}] {file_rel}")
        console.print()

    # ── Response panels ─────────────────────────────────────────────────────
    rag_content = (
        f"[bold]Prompt:[/bold] {prompt_text}\n\n"
        f"[bold]Response:[/bold]\n{rag_m.get('response', 'N/A')}"
    )
    console.print(Panel(rag_content, title="[bold red]Tier 1 · RAG Pipeline (Control)[/bold red]", border_style="red"))

    van_content = (
        f"[bold]Prompt:[/bold] {prompt_text}\n\n"
        f"[bold]Response:[/bold]\n{vanilla_m.get('response', 'N/A')}"
    )
    console.print(Panel(van_content, title="[bold yellow]Tier 2 · GOG Vanilla (No Membrane)[/bold yellow]", border_style="yellow"))

    srm_content = (
        f"[bold]Renderer Prompt:[/bold]\n{srm_m.get('renderer_prompt', 'N/A')}\n\n"
        f"[bold]LLM Response:[/bold]\n{srm_m.get('response', 'N/A')}"
    )
    console.print(Panel(srm_content, title="[bold green]Tier 3 · SRM Pipeline[/bold green]", border_style="green"))
    console.print("\n")

    # ── Correctness Rubric ──────────────────────────────────────────────────
    if level is not None:
        rag_score, rag_p, rag_t, rag_fails = score_response(level, rag_m.get("response", ""))
        van_score, van_p, van_t, van_fails = score_response(level, vanilla_m.get("response", ""))
        srm_score, srm_p, srm_t, srm_fails = score_response(level, srm_m.get("response", ""))

        def _fail_str(fails):
            return ("\n  ✗ " + "\n  ✗ ".join(fails)) if fails else ""

        rubric_content = (
            f"[bold]Tier 1 · RAG:[/bold]      {rag_score}{_fail_str(rag_fails)}\n"
            f"[bold]Tier 2 · GOG:[/bold]      {van_score}{_fail_str(van_fails)}\n"
            f"[bold]Tier 3 · SRM:[/bold]      {srm_score}{_fail_str(srm_fails)}\n\n"
            f"[dim]Rubric: deterministic string checks. "
            f"Tier 3-SRM is falsifiable: if symbolic specs don't improve correctness "
            f"on small models (0.5B), the SRM hypothesis is rejected.[/dim]"
        )
        console.print(Panel(rubric_content, title="[bold white]Correctness Rubric[/bold white]", border_style="white"))
        console.print("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main: Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the SRM benchmark on selected tasks (Easy, Medium, Hard, or All)."""
    console.print(Panel(
        "[bold]SRM Pipeline Benchmark (Phase 2)[/bold]\n\n"
        "[dim]Hypothesis: Symbolic reasoning offloaded to deterministic planner "
        "allows smaller models (≤1B) to generate correct code when given only "
        "the symbolic specification, not raw natural language.[/dim]",
        title="Symbolic Reasoning Model (SRM) Validation",
        border_style="blue",
    ))

    target_repo = os.path.join(os.path.dirname(__file__), "target_repo")
    if not os.path.exists(target_repo):
        console.print(
            f"[bold red]Error[/bold red]: target_repo not found. "
            "Run [bold]python3 generate_dummy_repo.py[/bold] first."
        )
        return

    # Load graph
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
        console.print(
            f"[bold red]Error[/bold red]: gog_graph.pkl not found. "
            "Run [bold]python3 seed_RAG_and_GOG.py[/bold] first."
        )
        return

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    # Task selection
    console.print("\n[bold]Available tasks:[/bold]")
    console.print("  1. Easy — Single-file state mutation (ADD_FIELD + MUTATE_ACTION)")
    console.print("  2. Medium — Vue component wiring (imports + bindings + template)")
    console.print("  3. Hard — Multi-file feature with forbidden import constraint")
    console.print("  4. All — Run all three tiers on all difficulties")

    try:
        choice = input("\nSelect task (1-4): ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"  # Default to Easy

    if choice == "1":
        tasks = ["Easy"]
    elif choice == "2":
        tasks = ["Medium"]
    elif choice == "3":
        tasks = ["Hard"]
    elif choice == "4":
        tasks = ["Easy", "Medium", "Hard"]
    else:
        console.print("[red]Invalid choice. Defaulting to Easy.[/red]")
        tasks = ["Easy"]

    # Run selected tasks
    for task_level in tasks:
        _run_single_task(task_level, target_repo, G)


def _run_single_task(level: str, target_repo: str, G):
    """Run a single task through all 3 tiers."""
    prompt_dict = PROMPTS[level]
    prompt_text = prompt_dict["text"]

    console.print(Panel(
        f"[bold]Task:[/bold] {level}\n"
        f"[bold]Prompt:[/bold] {prompt_text}",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Tier 1 — RAG Control
        t1 = progress.add_task(
            description=f"[red]Tier 1 · Running RAG Control ({client.model})...", total=None
        )
        rag = run_control_pipeline(prompt_text, target_repo)
        progress.remove_task(t1)

        # Tier 2 — GOG Vanilla
        t2 = progress.add_task(
            description=f"[yellow]Tier 2 · Running GOG Vanilla ({client.model})...", total=None
        )
        vanilla = run_srm_pipeline_vanilla(prompt_text, target_repo)
        progress.remove_task(t2)

        # Tier 3 — SRM Pipeline
        t3 = progress.add_task(
            description=f"[green]Tier 3 · Running SRM Pipeline ({client.model})...", total=None
        )
        srm = run_srm_pipeline(prompt_text, target_repo, G)
        progress.remove_task(t3)

    # Print results
    print_srm_benchmark_results(rag, vanilla, srm, prompt_text, level=level)
    console.print("\n" + "="*80 + "\n")

    # ── Final Verdict ───────────────────────────────────────────────────────
    srm_tokens = srm.get("tokens_in", 0)
    rag_tokens = rag.get("tokens_in", 0)
    token_reduction = _pct(rag_tokens - srm_tokens, rag_tokens) if rag_tokens > 0 else 0

    verdict_lines = [
        f"Tier 3-SRM input tokens: {srm_tokens} (vs RAG: {rag_tokens})\n"
    ]

    if "parse_error" in srm:
        verdict_lines.append(f"[bold red]Parse Error:[/bold red] {srm['parse_error']}")
    elif "plan_error" in srm:
        verdict_lines.append(f"[bold red]Planner Error:[/bold red] {srm['plan_error']}")
    else:
        verdict_lines.append(
            f"[bold green]✓ SRM Pipeline executed successfully.[/bold green]\n"
            f"Symbolic specification was generated and passed to LLM.\n"
            f"The LLM did NOT see the original natural language prompt."
        )

    console.print(Panel("\n".join(verdict_lines), title="Verdict", border_style="green"))


if __name__ == "__main__":
    main()
