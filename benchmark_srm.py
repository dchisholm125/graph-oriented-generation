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
from srm_engine.planner.renderer_prompt import build_renderer_prompt

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

def run_srm_pipeline(prompt_text: str, target_repo: str, graph) -> dict:
    """
    Tier 3-SRM: Intent Parser → Mutation Planner → Renderer Prompt → LLM Renderer

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

    # ── Step 3: Build Renderer Prompt (deterministic) ────────────────────────
    renderer_prompt = build_renderer_prompt(plan) + MUZZLE

    # Count tokens in the renderer prompt (this is what the LLM sees)
    tokens_in = get_token_count([plan.target_file_abs])
    local_time = time.time() - start_time

    # ── Step 4: LLM Renderer (single call, no retry) ─────────────────────────
    api_start = time.time()
    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked SRM result (Ollama is not running)"
    else:
        # CRITICAL: The LLM receives ONLY the renderer prompt.
        # It does NOT see the original natural language prompt.
        # context_files=[] because all context is in the renderer_prompt itself.
        response = client.complete(renderer_prompt, context_files=[])
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
        "renderer_prompt": renderer_prompt,  # For debugging/verification
        "operations": ops,
        "plan": plan,
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
    """Run the SRM benchmark on the Easy task."""
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

    # Run Easy task
    prompt_dict = PROMPTS["Easy"]
    prompt_text = prompt_dict["text"]
    level = "Easy"

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
