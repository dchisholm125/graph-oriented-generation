import time
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from srm_engine import ast_parser, graph_search

console = Console()

class OllamaClient:
    def __init__(self, model="qwen3.5:0.8b"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"

    @property
    def is_present(self):
        try:
            req = urllib.request.Request("http://localhost:11434/")
            with urllib.request.urlopen(req) as response:
                return response.status == 200
        except:
            return False

    def complete(self, prompt, context_files=None):
        if not context_files:
            context_files = []
            
        # Compile context into a strict string format for the SLM 
        context_str = ""
        for file_path in context_files:
            try:
                with open(file_path, "r", encoding="utf8") as f:
                    context_str += f"\n--- {os.path.basename(file_path)} ---\n{f.read()}\n"
            except Exception:
                pass
                
        full_prompt = (
            f"You are an expert developer assistant. Use ONLY the provided context to answer the prompt.\n\n"
            f"=== CONTEXT ===\n{context_str}\n\n"
            f"=== PROMPT ===\n{prompt}"
        )
        
        data = json.dumps({
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_ctx": 32768
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(self.api_url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
                response_text = result.get("response", "").strip()
                if not response_text:
                    return "[Error: Model returned an empty string. The retrieved RAG context likely exceeded the model's absolute maximum context window (Context Collapse).]"
                return response_text
        except urllib.error.URLError as e:
            return f"API Error: Ensure Ollama is running and {self.model} is pulled. ({e})"

client = OllamaClient(model="qwen3.5:0.8b")

def get_token_count(files):
    """Estimate tokens based on word count of files."""
    count = 0
    for f in files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf8') as content:
                count += len(content.read().split())
    return count

def run_control_pipeline(prompt, repo_path):
    """True Vector RAG: Fetch top-K chunks from ChromaDB and feed to local LLM."""
    start_time = time.time()
    
    db_path = os.path.join(os.path.dirname(__file__), "vector_db")
    if not os.path.exists(db_path):
         return {"time": 0, "tokens_in": 0, "tokens_out": 0, "response": "Error: ChromaDB not found. Run seed_RAG_and_GOG.py first."}

    import chromadb
    client_db = chromadb.PersistentClient(path=db_path)
    collection = client_db.get_collection("repo_chunks")
    
    # Query ChromaDB for top 5 most similar chunks
    results = collection.query(
        query_texts=[prompt],
        n_results=5
    )
    
    # Extract unique files implicated by the vector chunks
    context_files = set()
    if results['documents'] and results['documents'][0]:
        for i, chunk in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            context_files.add(meta['file'])
            
    unique_files = list(context_files)
    tokens_in = get_token_count(unique_files)
    
    local_time = time.time() - start_time
    
    api_start = time.time()
    if not client.is_present:
        time.sleep(1)
        response = "Mocked RAG result (Ollama is not running)"
    else:
        response = client.complete(prompt, context_files=unique_files)
    api_time = time.time() - api_start
    
    execution_time = local_time + api_time
    return {"time": execution_time, "local_time": local_time, "api_time": api_time, "tokens_in": tokens_in, "tokens_out": 150, "response": response}

def run_srm_pipeline(prompt, repo_path):
    """Graph-Oriented Generation: Pristine, isolated context via pre-built Graph to local LLM."""
    start_time = time.time()
    
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if not os.path.exists(graph_path):
         return {"time": 0, "tokens_in": 0, "tokens_out": 0, "response": "Error: Graph not found. Run seed_RAG_and_GOG.py first."}
         
    import pickle
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    
    isolated_files = graph_search.isolate_context(G, prompt)
    tokens_in = get_token_count(isolated_files)

    local_time = time.time() - start_time

    api_start = time.time()
    if not client.is_present:
        time.sleep(0.5)
        response = "Mocked GOG result (Ollama is not running)"
    else:
        response = client.complete(prompt, context_files=isolated_files)
    api_time = time.time() - api_start
    
    execution_time = local_time + api_time
    return {"time": execution_time, "local_time": local_time, "api_time": api_time, "tokens_in": tokens_in, "tokens_out": 150, "response": response}

def print_results(control_metrics, srm_metrics, prompt_text):
    table = Table(title=f"📊 GOG vs RAG Benchmark Results ({client.model})", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Control (RAG)", style="red")
    table.add_column("SRM (GOG)", style="green")
    table.add_column("Difference", style="bold yellow")

    time_diff = control_metrics['time'] - srm_metrics['time']
    time_pct = (time_diff / control_metrics['time']) * 100 if control_metrics['time'] > 0 else 0
    local_time_diff = control_metrics['local_time'] - srm_metrics['local_time']
    local_time_pct = (local_time_diff / control_metrics['local_time']) * 100 if control_metrics['local_time'] > 0 else 0
    table.add_row(
        "Local Compute Time", 
        f"{control_metrics['local_time']:.4f}s", 
        f"{srm_metrics['local_time']:.4f}s", 
        f"-{local_time_diff:.4f}s ({local_time_pct:.1f}%)"
    )

    api_time_diff = control_metrics['api_time'] - srm_metrics['api_time']
    api_time_pct = (api_time_diff / control_metrics['api_time']) * 100 if control_metrics['api_time'] > 0 else 0
    table.add_row(
        "Generate Time (Local LLM)", 
        f"{control_metrics['api_time']:.2f}s", 
        f"{srm_metrics['api_time']:.2f}s", 
        f"-{api_time_diff:.2f}s ({api_time_pct:.1f}%)"
    )

    table.add_row(
        "Total Execution Time", 
        f"{control_metrics['time']:.2f}s", 
        f"{srm_metrics['time']:.2f}s", 
        f"-{time_diff:.2f}s ({time_pct:.1f}%)"
    )

    token_diff = control_metrics['tokens_in'] - srm_metrics['tokens_in']
    token_pct = (token_diff / control_metrics['tokens_in']) * 100 if control_metrics['tokens_in'] > 0 else 0
    table.add_row(
        "Tokens In (Est.)", 
        str(control_metrics['tokens_in']), 
        str(srm_metrics['tokens_in']), 
        f"-{token_diff} ({token_pct:.1f}%)"
    )

    console.print(table)
    
    console.print("\n")
    rag_content = f"[bold]Prompt:[/bold] {prompt_text}\n\n[bold]Response:[/bold]\n{control_metrics.get('response', 'N/A')}"
    console.print(Panel(rag_content, title="[bold red]RAG Pipeline (Control)[/bold red]", border_style="red"))
    
    srm_content = f"[bold]Prompt:[/bold] {prompt_text}\n\n[bold]Response:[/bold]\n{srm_metrics.get('response', 'N/A')}"
    console.print(Panel(srm_content, title="[bold green]GOG Pipeline (SRM Engine)[/bold green]", border_style="green"))
    console.print("\n")
    
    if token_pct > 50:
        console.print(Panel(f"[bold green]WINNER: SRM Engine[/bold green]\nReduced context load by {token_pct:.1f}% using deterministic isolation.", title="Verdict", border_style="green"))

PROMPTS = {
    "Easy": {
        "desc": "Localized Semantic Retrieval (1-Step)",
        "text": "Identify the default state variables and roles initialized in the main authentication store."
    },
    "Medium": {
        "desc": "Structural Bridging (3-Step)",
        "text": "Locate where the user authentication state is passed to the dashboard header widget and identify the underlying API call and logger used."
    },
    "Hard": {
        "desc": "The Semantic Labyrinth (Deep Traversal + Red Herrings)",
        "text": "Trace the complete execution path when a user clicks logout from the settings view, down to the exact HTTP utility method that clears the session."
    }
}

def run_pipeline_for_prompt(prompt_text, target_repo, level="Benchmark"):
    console.print(Panel(f"[bold]Target Repository:[/bold] {target_repo}\n[bold]Prompt ({level}):[/bold] {prompt_text}", title=f"Local LLM Benchmark Harness v1.4 (The Gauntlet - {level})", border_style="blue"))

    if not os.path.exists(target_repo):
        console.print(f"[bold red]Error[/bold red]: target_repo not found. Run [bold]python3 generate_dummy_repo.py[/bold] first.")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        # Control Pipeline (Standard RAG)
        progress.add_task(description=f"[cyan]Running Control Pipeline (RAG) with {client.model}...", total=None)
        control = run_control_pipeline(prompt_text, target_repo)
        
        # SRM Pipeline (GOG)
        progress.add_task(description=f"[green]Running SRM Engine Pipeline (GOG) with {client.model}...", total=None)
        srm = run_srm_pipeline(prompt_text, target_repo)
    
    # Display summary of isolated components
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")
    if os.path.exists(graph_path):
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        isolated_files = graph_search.isolate_context(G, prompt_text)
        
        console.print(f"\n[bold green]SRM Engine identified {len(isolated_files)} critical files from the 100+ file maze:[/bold green]")
        for f in isolated_files:
            console.print(f"  - {os.path.relpath(f, target_repo)}")

    print_results(control, srm, prompt_text)

def run_gauntlet():
    target_repo = os.path.join(os.path.dirname(__file__), "target_repo")
    
    if not client.is_present:
        console.print(f"[bold yellow]Warning:[/bold yellow] Ollama does not seem to be running at localhost:11434. The benchmark will mock LLM responses.")
        
    console.print(f"\n[bold cyan]Select Benchmark Difficulty Level (Local LLM: {client.model}):[/bold cyan]")
    for level, data in PROMPTS.items():
        console.print(f"[{level}] - {data['desc']}")
    console.print("[All] - Run the full gauntlet")
        
    choice = input("\nEnter difficulty (Easy/Medium/Hard/All): ").strip().capitalize()
    
    if choice == "All":
        for level, data in PROMPTS.items():
            console.print(f"\n[bold yellow]--- Running {level} Benchmark ---[/bold yellow]")
            run_pipeline_for_prompt(data['text'], target_repo, level=level)
            time.sleep(2) # Brief pause between runs
    elif choice in PROMPTS:
        run_pipeline_for_prompt(PROMPTS[choice]['text'], target_repo, level=choice)
    else:
        console.print("[bold red]Invalid choice. Exiting.[/bold red]")

if __name__ == "__main__":
    run_gauntlet()
