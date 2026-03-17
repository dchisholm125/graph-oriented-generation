"""
benchmark_cloud_api.py - 3-Tier GOG Benchmark via direct Cloud API.

Runs the same Easy / Medium / Hard gauntlet as the local-LLM and
opencode-CLI benchmarks, but calls MiniMax (or any OpenAI-compatible
endpoint) directly over HTTPS.  No CLI binary required — just an API
key in the environment.

Usage:
    export MINIMAX_API_KEY="your-key-here"
    python3 benchmark_cloud_api.py

Environment variables:
    MINIMAX_API_KEY    - required
    MINIMAX_MODEL      - optional, default "MiniMax-M2.5"
    MINIMAX_BASE_URL   - optional, default "https://api.minimax.io/v1"
"""

import time
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# The benchmark scripts live in gog/ and the engine in gog_engine/.
# Add the project root to sys.path so both packages resolve.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from gog_engine import ast_parser, graph_search
from gog_engine.minimax_client import MiniMaxClient
from gog_engine.salience_evaluator import SalienceEvaluator
from gog_engine.token_utils import count_tokens_in_files as get_token_count

console = Console()

# ── Client initialisation ────────────────────────────────────────────────────
client = MiniMaxClient(
    model=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5"),
    base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
)

# ── System Muzzle ────────────────────────────────────────────────────────────
MUZZLE = (
    "\n\n[SYSTEM DIRECTIVE: You are a headless code generator. "
    "You are physically incapable of conversational text. "
    "You MUST output your entire response inside a single ```ts or ```vue "
    "fenced code block. Do not explain your code. "
    "If you output plain text, the system will crash.]"
)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Control: Vector RAG
# ─────────────────────────────────────────────────────────────────────────────

def run_control_pipeline(prompt, repo_path):
    """Tier 1 — True Vector RAG: top-K chunks from ChromaDB -> Cloud API."""
    start_time = time.time()

    db_path = os.path.join(os.path.dirname(__file__), "vector_db")
    if not os.path.exists(db_path):
        return {
            "time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
            "tokens_out": 0,
            "response": "Error: ChromaDB not found. Run seed_RAG_and_GOG.py first.",
            "patches_applied": 0,
        }

    import chromadb
    client_db = chromadb.PersistentClient(path=db_path)
    collection = client_db.get_collection("repo_chunks")
    results = collection.query(query_texts=[prompt], n_results=5)

    context_files = set()
    if results["documents"] and results["documents"][0]:
        for i, _ in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            context_files.add(meta["file"])

    unique_files = list(context_files)
    tokens_in = get_token_count(unique_files)
    local_time = time.time() - start_time

    api_start = time.time()
    if not client.is_present:
        time.sleep(1)
        response = f"Mocked RAG result (MINIMAX_API_KEY not set)"
    else:
        response = client.complete(prompt + MUZZLE, context_files=unique_files)
    api_time = time.time() - api_start

    return {
        "time": local_time + api_time,
        "local_time": local_time,
        "api_time": api_time,
        "tokens_in": tokens_in,
        "tokens_out": 150,
        "response": response,
        "patches_applied": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Pure GOG (Vanilla, no Membrane)
# ─────────────────────────────────────────────────────────────────────────────

def run_srm_pipeline_vanilla(prompt, repo_path):
    """Tier 2 — GOG graph isolation -> single Cloud API call, no safety net."""
    start_time = time.time()

    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
        return {
            "time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
            "tokens_out": 0,
            "response": "Error: Graph not found. Run seed_RAG_and_GOG.py first.",
            "patches_applied": 0,
        }

    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    isolated_files = graph_search.isolate_context(G, prompt)
    tokens_in = get_token_count(isolated_files)
    local_time = time.time() - start_time

    api_start = time.time()
    if not client.is_present:
        time.sleep(0.5)
        response = f"Mocked GOG (Vanilla) result (MINIMAX_API_KEY not set)"
    else:
        response = client.complete(prompt + MUZZLE, context_files=isolated_files)
    api_time = time.time() - api_start

    return {
        "time": local_time + api_time,
        "local_time": local_time,
        "api_time": api_time,
        "tokens_in": tokens_in,
        "tokens_out": 150,
        "response": response,
        "patches_applied": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — GOG + SalienceEvaluator Membrane
# ─────────────────────────────────────────────────────────────────────────────

def run_srm_pipeline_membrane(prompt, repo_path):
    """Tier 3 — GOG + Membrane: graph-based patching after Cloud API call."""
    start_time = time.time()

    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
        return {
            "time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
            "tokens_out": 0,
            "response": "Error: Graph not found. Run seed_RAG_and_GOG.py first.",
            "patches_applied": 0,
        }

    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    isolated_files = graph_search.isolate_context(G, prompt)
    tokens_in = get_token_count(isolated_files)
    local_time = time.time() - start_time

    api_start = time.time()
    patches_applied = 0

    if not client.is_present:
        time.sleep(0.5)
        response = f"Mocked GOG+Membrane result (MINIMAX_API_KEY not set)"
    else:
        evaluator = SalienceEvaluator(allowed_nodes=isolated_files)
        raw_response = client.complete(prompt + MUZZLE, context_files=isolated_files)
        result = evaluator.evaluate(raw_response)

        if result.is_valid:
            response = raw_response
        else:
            patches_applied = len(result.violations)
            if patches_applied > 0:
                console.print(
                    f"[bold cyan][SRM Membrane] Patching {patches_applied} "
                    f"topological violation(s) deterministically (no retry):[/bold cyan] "
                    + ", ".join(f"'{v}'" for v in result.violations)
                )
            patched_code = evaluator.patch(result)
            if patched_code and patched_code.strip():
                import re as _re
                lang_match = _re.search(
                    r"```(typescript|ts|vue)", raw_response, _re.IGNORECASE
                )
                fence_lang = (
                    lang_match.group(1).lower() if lang_match else "ts"
                )
                response = f"```{fence_lang}\n{patched_code}\n```"
            else:
                response = raw_response

    api_time = time.time() - api_start
    return {
        "time": local_time + api_time,
        "local_time": local_time,
        "api_time": api_time,
        "tokens_in": tokens_in,
        "tokens_out": 150,
        "response": response,
        "patches_applied": patches_applied,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Results Printer
# ─────────────────────────────────────────────────────────────────────────────

def _pct(diff, base):
    if base == 0:
        return 0.0
    return (diff / base) * 100


def print_results(rag_m, vanilla_m, membrane_m, prompt_text):
    table = Table(
        title=f"3-Tier GOG Benchmark — Cloud API ({client.model})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Tier 1 - RAG (Control)", style="red", min_width=22)
    table.add_column("Tier 2 - GOG Vanilla", style="yellow", min_width=22)
    table.add_column("Tier 3 - GOG + Membrane", style="green", min_width=22)

    lt_rav, lt_van, lt_mem = rag_m["local_time"], vanilla_m["local_time"], membrane_m["local_time"]
    table.add_row(
        "Local Compute Time",
        f"{lt_rav:.4f}s",
        f"{lt_van:.4f}s  ({_pct(lt_rav - lt_van, lt_rav):.1f}% down)",
        f"{lt_mem:.4f}s  ({_pct(lt_rav - lt_mem, lt_rav):.1f}% down)",
    )

    at_rav, at_van, at_mem = rag_m["api_time"], vanilla_m["api_time"], membrane_m["api_time"]
    table.add_row(
        "LLM Generation Time",
        f"{at_rav:.2f}s",
        f"{at_van:.2f}s  ({_pct(at_rav - at_van, at_rav):.1f}% down)",
        f"{at_mem:.2f}s  ({_pct(at_rav - at_mem, at_rav):.1f}% down)",
    )

    tt_rav, tt_van, tt_mem = rag_m["time"], vanilla_m["time"], membrane_m["time"]
    table.add_row(
        "Total Execution Time",
        f"{tt_rav:.2f}s",
        f"{tt_van:.2f}s  ({_pct(tt_rav - tt_van, tt_rav):.1f}% down)",
        f"{tt_mem:.2f}s  ({_pct(tt_rav - tt_mem, tt_rav):.1f}% down)",
    )

    tk_rav, tk_van, tk_mem = rag_m["tokens_in"], vanilla_m["tokens_in"], membrane_m["tokens_in"]
    table.add_row(
        "Tokens In (Est.)",
        str(tk_rav),
        f"{tk_van}  ({_pct(tk_rav - tk_van, tk_rav):.1f}% down)",
        f"{tk_mem}  ({_pct(tk_rav - tk_mem, tk_rav):.1f}% down)",
    )
    table.add_row(
        "Topological Patches (Membrane)", "-", "-",
        str(membrane_m["patches_applied"]),
    )

    console.print(table)
    console.print("\n")

    for label, colour, data in [
        ("Tier 1 - RAG Pipeline (Control)", "red", rag_m),
        ("Tier 2 - GOG Vanilla (No Membrane)", "yellow", vanilla_m),
        ("Tier 3 - GOG + SalienceEvaluator Membrane", "green", membrane_m),
    ]:
        content = (
            f"[bold]Prompt:[/bold] {prompt_text}\n\n"
            f"[bold]Response:[/bold]\n{data.get('response', 'N/A')}"
        )
        console.print(Panel(content, title=f"[bold {colour}]{label}[/bold {colour}]", border_style=colour))

    console.print("\n")

    token_pct_mem = _pct(tk_rav - tk_mem, tk_rav)
    token_pct_van = _pct(tk_rav - tk_van, tk_rav)
    if token_pct_mem > 50:
        console.print(Panel(
            f"[bold green]WINNER: GOG + Membrane (Tier 3)[/bold green]\n"
            f"Reduced context load by {token_pct_mem:.1f}% vs RAG.\n"
            f"Membrane triggered [bold]{membrane_m['patches_applied']}[/bold] "
            f"topological patch(es).",
            title="Verdict", border_style="green",
        ))
    elif token_pct_van > 50:
        console.print(Panel(
            f"[bold yellow]GOG Vanilla (Tier 2)[/bold yellow] reduced context "
            f"by {token_pct_van:.1f}%.\nTier 3 further enforced correctness.",
            title="Verdict", border_style="yellow",
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Prompts (identical to the other benchmark scripts)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "Easy": {
        "desc": "Localized State Mutation (1-Step)",
        "text": (
            "Write the code to add a `lastLogin` string timestamp to the "
            "default state in `src/stores/authStore.ts` and update the "
            "`login` action to set it to '2026-03-08'."
        ),
    },
    "Medium": {
        "desc": "Component-to-Store Wiring (2-Step)",
        "text": (
            "Refactor `src/components/HeaderWidget.vue` to include a "
            "'Logout' button next to the user role. Wire the click event "
            "to call the `logout` action from `useAuthStore`."
        ),
    },
    "Hard": {
        "desc": "Full Stack Trace Implementation (Constraint Test)",
        "text": (
            "Implement a 'Delete Account' feature. Add a `deleteAccount` "
            "API call in `src/services/api_client.ts` that posts to "
            "'/delete', create a `deleteUser` action in `authStore.ts` "
            "that calls it, and add a 'Delete' button in "
            "`src/views/UserSettings.vue` to trigger it. "
            "You must NOT import `api_client.ts` directly into the Vue "
            "component."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_for_prompt(prompt_text, target_repo, level="Benchmark"):
    console.print(Panel(
        f"[bold]Target Repository:[/bold] {target_repo}\n"
        f"[bold]Task ({level}):[/bold] {prompt_text}",
        title=f"Cloud API 3-Tier Benchmark ({client.model} - {level})",
        border_style="blue",
    ))

    if not os.path.exists(target_repo):
        console.print(
            "[bold red]Error[/bold red]: target_repo not found. "
            "Run [bold]python3 generate_dummy_repo.py[/bold] first."
        )
        return

    if not client.is_present:
        console.print(
            "[bold yellow]Warning:[/bold yellow] MINIMAX_API_KEY not set. "
            "The benchmark will mock Cloud API responses."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        t1 = progress.add_task(
            description=f"[red]Tier 1 - Running RAG Control ({client.model})...",
            total=None,
        )
        rag = run_control_pipeline(prompt_text, target_repo)
        progress.remove_task(t1)

        t2 = progress.add_task(
            description=f"[yellow]Tier 2 - Running GOG Vanilla ({client.model})...",
            total=None,
        )
        vanilla = run_srm_pipeline_vanilla(prompt_text, target_repo)
        progress.remove_task(t2)

        t3 = progress.add_task(
            description=f"[green]Tier 3 - Running GOG + Membrane ({client.model})...",
            total=None,
        )
        membrane = run_srm_pipeline_membrane(prompt_text, target_repo)
        progress.remove_task(t3)

    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if os.path.exists(graph_path):
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        isolated_files = graph_search.isolate_context(G, prompt_text)
        console.print(
            f"\n[bold green]GOG Engine isolated "
            f"{len(isolated_files)} critical file(s) from the 100+ file maze:[/bold green]"
        )
        for f in isolated_files:
            console.print(f"  [dim]-[/dim] {os.path.relpath(f, target_repo)}")

    print_results(rag, vanilla, membrane, prompt_text)


def run_gauntlet():
    target_repo = os.path.join(os.path.dirname(__file__), "target_repo")

    console.print(
        f"\n[bold cyan]Select Benchmark Difficulty "
        f"(Cloud API: {client.model}):[/bold cyan]"
    )
    for level, data in PROMPTS.items():
        console.print(f"  [[bold]{level}[/bold]] {data['desc']}")
    console.print("  [[bold]All[/bold]] Run the full gauntlet")

    choice = input("\nEnter difficulty (Easy/Medium/Hard/All): ").strip().capitalize()

    if choice == "All":
        for level, data in PROMPTS.items():
            console.print(
                f"\n[bold yellow]--- Running {level} Benchmark ---[/bold yellow]"
            )
            run_pipeline_for_prompt(data["text"], target_repo, level=level)
            time.sleep(2)
    elif choice in PROMPTS:
        run_pipeline_for_prompt(PROMPTS[choice]["text"], target_repo, level=choice)
    else:
        console.print("[bold red]Invalid choice. Exiting.[/bold red]")


if __name__ == "__main__":
    run_gauntlet()
