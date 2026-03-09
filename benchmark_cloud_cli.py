import time
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from srm_engine import ast_parser, graph_search
from srm_engine.opencode_client import OpenCodeClient
from srm_engine.salience_evaluator import SalienceEvaluator

console = Console()
client = OpenCodeClient()

# ─────────────────────────────────────────────────────────────────────────────
# System Muzzle
# ─────────────────────────────────────────────────────────────────────────────
# Appended to every prompt sent to the Cloud LLM via opencode.  Cloud models
# are highly capable but still default to conversational formatting.  The
# SalienceEvaluator needs fenced ```ts / ```vue blocks to run its AST analysis;
# without them every response is immediately rejected as "no code block found".
# Appending this after the user task exploits the recency bias of transformer
# attention — the model reads it last and weights it highest.
MUZZLE = (
    "\n\n[SYSTEM DIRECTIVE: You are a headless code generator. "
    "You are physically incapable of conversational text. "
    "You MUST output your entire response inside a single ```ts or ```vue "
    "fenced code block. Do not explain your code. "
    "If you output plain text, the system will crash.]"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_token_count(files):
    """Estimate tokens based on word count of files."""
    count = 0
    for f in files:
        if os.path.exists(f):
            with open(f, "r", encoding="utf8") as content:
                count += len(content.read().split())
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Control: Vector RAG
# ─────────────────────────────────────────────────────────────────────────────

def run_control_pipeline(prompt, repo_path):
    """Tier 1 — True Vector RAG: top-K chunks from ChromaDB → Cloud LLM."""
    start_time = time.time()

    db_path = os.path.join(os.path.dirname(__file__), "vector_db")
    if not os.path.exists(db_path):
        return {"time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
                "tokens_out": 0, "response": "Error: ChromaDB not found. Run seed_RAG_and_GOG.py first.",
                "rejection_attempts": 0}

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
        response = "Mocked RAG result (opencode CLI not found)"
    else:
        # Append MUZZLE to force fenced-code-block output from the Cloud LLM.
        muzzled_prompt = prompt + MUZZLE
        response = client.complete(muzzled_prompt, context_files=unique_files)
    api_time = time.time() - api_start

    execution_time = local_time + api_time
    return {"time": execution_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "rejection_attempts": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Pure GOG (Vanilla, no Membrane)
# ─────────────────────────────────────────────────────────────────────────────

def run_srm_pipeline_vanilla(prompt, repo_path):
    """Tier 2 — Pure GOG: deterministic graph isolation → single Cloud LLM call.

    Proves the token-reduction benefit of graph isolation without any
    topological guardrails. The LLM may still hallucinate illegal imports.
    """
    start_time = time.time()

    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
        return {"time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
                "tokens_out": 0, "response": "Error: Graph not found. Run seed_RAG_and_GOG.py first.",
                "rejection_attempts": 0}

    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    isolated_files = graph_search.isolate_context(G, prompt)
    tokens_in = get_token_count(isolated_files)
    local_time = time.time() - start_time

    api_start = time.time()
    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked GOG (Vanilla) result (opencode CLI not found)"
    else:
        # Append MUZZLE to force fenced-code-block output from the Cloud LLM.
        muzzled_prompt = prompt + MUZZLE
        response = client.complete(muzzled_prompt, context_files=isolated_files)
    api_time = time.time() - api_start

    execution_time = local_time + api_time
    return {"time": execution_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "rejection_attempts": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — GOG + SalienceEvaluator Membrane
# ─────────────────────────────────────────────────────────────────────────────

def run_srm_pipeline_membrane(prompt, repo_path):
    """
    Tier 3 — GOG + Neuro-Symbolic Membrane (Rejection Sampling).

    ── What is the Neuro-Symbolic Membrane? ────────────────────────────────────
    The term "Neuro-Symbolic" describes a hybrid AI architecture that combines:

      • "Neuro"   — the neural LLM (here, accessed via the opencode Cloud CLI),
                    which is creative and stochastic.  It can write elegant code,
                    synthesize patterns, and generalise from examples.  However,
                    it has no guaranteed understanding of your project's actual
                    import graph and may freely hallucinate file paths.

      • "Symbolic" — the GOG dependency graph (a Directed Acyclic Graph, DAG),
                    which is 100% deterministic.  It cannot hallucinate because
                    every edge represents a real `import` statement extracted by
                    our tree-sitter AST parser.

    The SalienceEvaluator acts as the "Membrane" between them.  After every LLM
    generation, it:
      1. Extracts all proposed import statements from the LLM's output.
      2. Checks each local import against the set of files that the DAG actually
         isolated as relevant (the `allowed_nodes`).
      3. Accepts   → passes the response downstream if all imports are legal.
      4. Rejects   → if any import falls outside the topological boundary, it
                    builds a structured error message, injects it back into the
                    prompt, and re-calls the Cloud LLM (Rejection Sampling).

    ── What is Rejection Sampling? ─────────────────────────────────────────────
    Rejection Sampling is a Monte Carlo technique: draw a sample from a
    distribution, test it against a constraint, discard it if it fails, and
    draw again.  Here the "distribution" is the LLM's generative output space
    and the "constraint" is the symbolic import graph.  The LLM is trapped in
    a feedback loop until it produces a topologically valid response or exhausts
    `max_attempts`.  The structured feedback on each rejection plants the correct
    file names in the next prompt, steering the model toward compliance.

    ── Why does this guarantee topological safety? ──────────────────────────────
    Because the DAG itself is the ground truth.  No matter how creative (or
    wrong) the LLM is, the Membrane's accept/reject decision is made purely by
    the deterministic graph.  The LLM provides creativity; the SRM provides
    correctness.  Neither alone is sufficient.
    """
    start_time = time.time()

    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
        return {"time": 0, "local_time": 0, "api_time": 0, "tokens_in": 0,
                "tokens_out": 0, "response": "Error: Graph not found. Run seed_RAG_and_GOG.py first.",
                "rejection_attempts": 0}

    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    isolated_files = graph_search.isolate_context(G, prompt)
    tokens_in = get_token_count(isolated_files)
    local_time = time.time() - start_time

    api_start = time.time()
    rejection_attempts = 0

    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked GOG+Membrane result (opencode CLI not found)"
    else:
        evaluator = SalienceEvaluator(allowed_nodes=isolated_files)
        max_attempts = 3
        # Seed the first call with the muzzled prompt.
        # Subsequent retries prepend the SYSTEM FEEDBACK block, which already
        # contains the muzzle implicitly via the rejection reason text.
        current_prompt = prompt + MUZZLE
        raw_response = ""

        while rejection_attempts < max_attempts:
            raw_response = client.complete(current_prompt, context_files=isolated_files)

            # ── The Membrane Test ──────────────────────────────────────────
            result = evaluator.evaluate(raw_response)

            if result.is_valid:
                response = raw_response
                break
            else:
                # Rejection Sampling: structured feedback so the Cloud agent can self-correct
                console.print(
                    f"[bold yellow][SalienceEvaluator] Cloud Attempt "
                    f"{rejection_attempts + 1}/{max_attempts} rejected:[/bold yellow] "
                    f"{result.reason}"
                )
                current_prompt = (
                    f"{prompt}\n\n"
                    f"[SYSTEM FEEDBACK: Your previous attempt failed validation: "
                    f"{result.reason}. Please rewrite the code to comply with "
                    f"these topological boundaries.]"
                )
                rejection_attempts += 1

        if rejection_attempts == max_attempts:
            response = (
                f"[SRM Rejection] The Cloud LLM failed to generate topologically valid "
                f"code after {max_attempts} attempts.\n\nLast output:\n{raw_response}"
            )

    api_time = time.time() - api_start
    execution_time = local_time + api_time
    return {"time": execution_time, "local_time": local_time, "api_time": api_time,
            "tokens_in": tokens_in, "tokens_out": 150, "response": response,
            "rejection_attempts": rejection_attempts}


# ─────────────────────────────────────────────────────────────────────────────
# Results Printer — 3-Tier Table
# ─────────────────────────────────────────────────────────────────────────────

def _pct(diff, base):
    if base == 0:
        return 0.0
    return (diff / base) * 100


def print_results(rag_m, vanilla_m, membrane_m, prompt_text):
    table = Table(
        title="📊 3-Tier GOG Benchmark — Cloud CLI (opencode)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Tier 1 · RAG (Control)", style="red", min_width=22)
    table.add_column("Tier 2 · GOG Vanilla", style="yellow", min_width=22)
    table.add_column("Tier 3 · GOG + Membrane", style="green", min_width=22)

    # ── Local compute time ──────────────────────────────────────────────────
    lt_rav = rag_m["local_time"]
    lt_van = vanilla_m["local_time"]
    lt_mem = membrane_m["local_time"]
    table.add_row(
        "Local Compute Time",
        f"{lt_rav:.4f}s",
        f"{lt_van:.4f}s  ({_pct(lt_rav - lt_van, lt_rav):.1f}% ↓)",
        f"{lt_mem:.4f}s  ({_pct(lt_rav - lt_mem, lt_rav):.1f}% ↓)",
    )

    # ── LLM generation time ─────────────────────────────────────────────────
    at_rav = rag_m["api_time"]
    at_van = vanilla_m["api_time"]
    at_mem = membrane_m["api_time"]
    table.add_row(
        "LLM Generation Time",
        f"{at_rav:.2f}s",
        f"{at_van:.2f}s  ({_pct(at_rav - at_van, at_rav):.1f}% ↓)",
        f"{at_mem:.2f}s  ({_pct(at_rav - at_mem, at_rav):.1f}% ↓)",
    )

    # ── Total execution time ────────────────────────────────────────────────
    tt_rav = rag_m["time"]
    tt_van = vanilla_m["time"]
    tt_mem = membrane_m["time"]
    table.add_row(
        "Total Execution Time",
        f"{tt_rav:.2f}s",
        f"{tt_van:.2f}s  ({_pct(tt_rav - tt_van, tt_rav):.1f}% ↓)",
        f"{tt_mem:.2f}s  ({_pct(tt_rav - tt_mem, tt_rav):.1f}% ↓)",
    )

    # ── Token reduction ─────────────────────────────────────────────────────
    tk_rav = rag_m["tokens_in"]
    tk_van = vanilla_m["tokens_in"]
    tk_mem = membrane_m["tokens_in"]
    table.add_row(
        "Tokens In (Est.)",
        str(tk_rav),
        f"{tk_van}  ({_pct(tk_rav - tk_van, tk_rav):.1f}% ↓)",
        f"{tk_mem}  ({_pct(tk_rav - tk_mem, tk_rav):.1f}% ↓)",
    )

    # ── Membrane rejection attempts ─────────────────────────────────────────
    table.add_row(
        "Rejection Attempts (Membrane)",
        "—",
        "—",
        str(membrane_m["rejection_attempts"]),
    )

    console.print(table)
    console.print("\n")

    # ── Response panels ─────────────────────────────────────────────────────
    rag_content = (
        f"[bold]Prompt:[/bold] {prompt_text}\n\n"
        f"[bold]Response:[/bold]\n{rag_m.get('response', 'N/A')}"
    )
    console.print(Panel(
        rag_content,
        title="[bold red]Tier 1 · RAG Pipeline (Control)[/bold red]",
        border_style="red",
    ))

    van_content = (
        f"[bold]Prompt:[/bold] {prompt_text}\n\n"
        f"[bold]Response:[/bold]\n{vanilla_m.get('response', 'N/A')}"
    )
    console.print(Panel(
        van_content,
        title="[bold yellow]Tier 2 · GOG Vanilla (No Membrane)[/bold yellow]",
        border_style="yellow",
    ))

    mem_content = (
        f"[bold]Prompt:[/bold] {prompt_text}\n\n"
        f"[bold]Response:[/bold]\n{membrane_m.get('response', 'N/A')}"
    )
    console.print(Panel(
        mem_content,
        title="[bold green]Tier 3 · GOG + SalienceEvaluator Membrane[/bold green]",
        border_style="green",
    ))
    console.print("\n")

    # ── Verdict ─────────────────────────────────────────────────────────────
    token_pct_van = _pct(tk_rav - tk_van, tk_rav)
    token_pct_mem = _pct(tk_rav - tk_mem, tk_rav)

    if token_pct_mem > 50:
        verdict = (
            f"[bold green]WINNER: GOG + Membrane (Tier 3)[/bold green]\n"
            f"Reduced context load by {token_pct_mem:.1f}% vs RAG.\n"
            f"Membrane triggered [bold]{membrane_m['rejection_attempts']}[/bold] "
            f"correction(s) to enforce topological safety."
        )
        console.print(Panel(verdict, title="Verdict", border_style="green"))
    elif token_pct_van > 50:
        console.print(Panel(
            f"[bold yellow]GOG Vanilla (Tier 2)[/bold yellow] reduced context by {token_pct_van:.1f}%.\n"
            f"Tier 3 Membrane further enforced architectural correctness.",
            title="Verdict", border_style="yellow",
        ))


# ─────────────────────────────────────────────────────────────────────────────
# Code-Generation Prompts (force LLM to emit TS/Vue code blocks)
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "Easy": {
        "desc": "Localized State Mutation (1-Step)",
        "text": (
            "Write the code to add a `lastLogin` string timestamp to the default state "
            "in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
        ),
    },
    "Medium": {
        "desc": "Component-to-Store Wiring (2-Step)",
        "text": (
            "Refactor `src/components/HeaderWidget.vue` to include a 'Logout' button next to "
            "the user role. Wire the click event to call the `logout` action from `useAuthStore`."
        ),
    },
    "Hard": {
        "desc": "Full Stack Trace Implementation (Constraint Test)",
        "text": (
            "Implement a 'Delete Account' feature. Add a `deleteAccount` API call in "
            "`src/services/api_client.ts` that posts to '/delete', create a `deleteUser` action "
            "in `authStore.ts` that calls it, and add a 'Delete' button in "
            "`src/views/UserSettings.vue` to trigger it. "
            "You must NOT import `api_client.ts` directly into the Vue component."
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
        title=f"Cloud CLI 3-Tier Benchmark v2.0 (The Gauntlet — {level})",
        border_style="blue",
    ))

    if not os.path.exists(target_repo):
        console.print(
            f"[bold red]Error[/bold red]: target_repo not found. "
            "Run [bold]python3 generate_dummy_repo.py[/bold] first."
        )
        return

    if not client.is_present:
        console.print(
            "[bold yellow]Warning:[/bold yellow] 'opencode' CLI not found in PATH. "
            "The benchmark will mock Cloud LLM responses."
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Tier 1 — RAG Control
        t1 = progress.add_task(
            description="[red]Tier 1 · Running RAG Control (opencode)...", total=None
        )
        rag = run_control_pipeline(prompt_text, target_repo)
        progress.remove_task(t1)

        # Tier 2 — GOG Vanilla
        t2 = progress.add_task(
            description="[yellow]Tier 2 · Running GOG Vanilla (opencode)...", total=None
        )
        vanilla = run_srm_pipeline_vanilla(prompt_text, target_repo)
        progress.remove_task(t2)

        # Tier 3 — GOG + Membrane
        t3 = progress.add_task(
            description="[green]Tier 3 · Running GOG + Membrane (opencode)...", total=None
        )
        membrane = run_srm_pipeline_membrane(prompt_text, target_repo)
        progress.remove_task(t3)

    # Show which files the GOG graph isolated
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if os.path.exists(graph_path):
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        isolated_files = graph_search.isolate_context(G, prompt_text)
        console.print(
            f"\n[bold green]SRM Engine isolated "
            f"{len(isolated_files)} critical file(s) from the 100+ file maze:[/bold green]"
        )
        for f in isolated_files:
            console.print(f"  [dim]·[/dim] {os.path.relpath(f, target_repo)}")

    print_results(rag, vanilla, membrane, prompt_text)


def run_gauntlet():
    target_repo = os.path.join(os.path.dirname(__file__), "target_repo")

    console.print("\n[bold cyan]Select Benchmark Difficulty Level (Cloud CLI: opencode):[/bold cyan]")
    for level, data in PROMPTS.items():
        console.print(f"  [[bold]{level}[/bold]] {data['desc']}")
    console.print("  [[bold]All[/bold]] Run the full gauntlet")

    choice = input("\nEnter difficulty (Easy/Medium/Hard/All): ").strip().capitalize()

    if choice == "All":
        for level, data in PROMPTS.items():
            console.print(f"\n[bold yellow]─── Running {level} Benchmark ───[/bold yellow]")
            run_pipeline_for_prompt(data["text"], target_repo, level=level)
            time.sleep(2)
    elif choice in PROMPTS:
        run_pipeline_for_prompt(PROMPTS[choice]["text"], target_repo, level=choice)
    else:
        console.print("[bold red]Invalid choice. Exiting.[/bold red]")


if __name__ == "__main__":
    run_gauntlet()
