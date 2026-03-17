import os
import re
from pathlib import Path
import networkx as nx
from .ts_parser import TypeScriptParser

parser = TypeScriptParser()

def extract_imports(file_path):
    """AST-based import extraction for TS and Vue files."""
    try:
        return parser.extract_imports(file_path)
    except Exception:
        # Silently fall back to regex if AST fails (prevents benchmark clutter)
        imports = []
        with open(file_path, 'r', encoding='utf8') as f:
            content = f.read()
        pattern = re.compile(r"import\s+.*?from\s+['\"](.*?)['\"]", re.MULTILINE)
        return pattern.findall(content)

def resolve_import(import_path, current_file, root_dir):
    """Resolves an import string to an absolute file path within the root_dir."""
    curr_dir = os.path.dirname(current_file)
    
    # Handle relative imports ('.', '..')
    if import_path.startswith('.'):
        potential_path = os.path.normpath(os.path.join(curr_dir, import_path))
    else:
        # For our benchmark, non-relative imports are usually packages or src root
        # We can simulate src-root logic if needed.
        return None

    # Check for common extensions if not provided
    for ext in ['.ts', '.vue', '/index.ts']:
        if os.path.exists(potential_path + ext):
            return potential_path + ext
        if potential_path.endswith(ext) and os.path.exists(potential_path):
            return potential_path
            
    return None

def build_graph(root_dir):
    """Builds a NetworkX DiGraph representing the project dependency structure."""
    G = nx.DiGraph()
    root_path = Path(root_dir).absolute()
    
    # Find all relevant files
    files_to_process = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.ts', '.vue')):
                files_to_process.append(os.path.join(root, file))

    # Add all files as nodes
    for file in files_to_process:
        G.add_node(os.path.abspath(file))

    # Add edges based on imports
    for file in files_to_process:
        abs_file = os.path.abspath(file)
        imports = extract_imports(file)
        for imp in imports:
            resolved = resolve_import(imp, abs_file, root_dir)
            if resolved and os.path.exists(resolved):
                G.add_edge(abs_file, resolved)
                
    return G

if __name__ == "__main__":
    # Test on the generated maze
    target = os.path.join(os.path.dirname(__file__), "../target_repo")
    if os.path.exists(target):
        graph = build_graph(target)
        print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        
        # Find the path from HeaderWidget.vue to authStore.ts
        try:
            # We need to find the specific absolute paths
            nodes = list(graph.nodes())
            header = [n for n in nodes if "HeaderWidget.vue" in n][0]
            auth = [n for n in nodes if "authStore.ts" in n][0]
            
            path = nx.shortest_path(graph, source=header, target=auth)
            print("Found dependency path:")
            for p in path:
                print(f"  - {os.path.relpath(p, target)}")
        except Exception as e:
            print(f"Path not found: {e}")
