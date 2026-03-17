import os
import pickle
import sys
from pathlib import Path
from rich.console import Console

console = Console()

def seed_everything():
    target_repo = os.path.join(os.path.dirname(__file__), "target_repo")
    db_path = os.path.join(os.path.dirname(__file__), "vector_db")
    graph_path = os.path.join(os.path.dirname(__file__), "gog_graph.pkl")

    if not os.path.exists(target_repo):
        console.print("[bold red]Error:[/] target_repo not found. Run generate_dummy_repo.py first.")
        sys.exit(1)

    import chromadb
    from chromadb.utils import embedding_functions

    try:
        from gog_engine import ast_parser
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from gog_engine import ast_parser

    # 1. Seed RAG (ChromaDB Vector Store)
    console.print("\n[bold cyan]1. Seeding RAG Environment (Vector DB)[/]")
    console.print("Initialize ChromaDB and compute embeddings for all fragments...")
    
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        client.delete_collection(name="repo_chunks")
    except Exception:
        pass
        
    # Use default embedding function
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name="repo_chunks",
        embedding_function=ef
    )

    documents = []
    metadatas = []
    ids = []
    
    files_processed = 0
    all_files = []
    for root, _, files in os.walk(target_repo):
        for file in files:
            if file.endswith(('.ts', '.vue')):
                all_files.append(os.path.join(root, file))

    chunk_id = 0
    for file_path in all_files:
        with open(file_path, 'r', encoding='utf8') as f:
            content = f.read()
        
        # Simple chunking logic: ~50 lines per chunk
        lines = content.split('\n')
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if chunk.strip():
                documents.append(chunk)
                metadatas.append({"file": file_path})
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        files_processed += 1
        
    # Add to Chroma in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
    console.print(f"[green]✓ RAG Seeded:[/] {chunk_id} chunks indexed across {files_processed} files.")

    # 2. Seed GOG (NetworkX AST Graph)
    console.print("\n[bold green]2. Seeding GOG Environment (AST Graph)[/]")
    console.print("Parsing ASTs and mapping dependency edges...")
    
    G = ast_parser.build_graph(target_repo)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
        
    console.print(f"[green]✓ GOG Seeded:[/] Graph serialized with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.\n")

if __name__ == "__main__":
    seed_everything()
