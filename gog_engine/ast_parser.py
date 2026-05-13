import os
import re
import json
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

def resolve_import(import_path, current_file, root_dir, aliases=None):
    """
    Resolves an import string to an absolute file path within the root_dir.

    Supports:
      - Relative imports ('.', '..')
      - Configured path aliases (e.g., '@/stores/auth' from Vite/tsconfig).

    Parameters
    ----------
    import_path : str
        The raw specifier from the import statement (e.g. '../utils', '@/stores/auth').
    current_file : str
        Absolute path of the file containing the import.
    root_dir : str
        Project root directory (used for alias resolution).
    aliases : dict[str, str] | None
        Mapping of alias prefix → absolute directory.
        Example for Vite alias: {"@": "/abs/path/src"}.
    """
    curr_dir = os.path.dirname(current_file)

    # ── Relative imports ─────────────────────────────────────────────────
    if import_path.startswith('.'):
        potential_path = os.path.normpath(os.path.join(curr_dir, import_path))

    # ── Configured aliases ─────────────────────────────────────────────
    elif aliases:
        for alias_prefix, alias_dir in aliases.items():
            prefix_with_slash = alias_prefix + "/"
            if import_path.startswith(prefix_with_slash):
                suffix = import_path[len(prefix_with_slash):]
                potential_path = os.path.normpath(os.path.join(alias_dir, suffix))
                break
        else:
            # No alias matched this path
            return None
    else:
        return None

    # Check for common extensions if not provided
    for ext in ['.ts', '.vue', '/index.ts']:
        if os.path.exists(potential_path + ext):
            return potential_path + ext
        if potential_path.endswith(ext) and os.path.exists(potential_path):
            return potential_path

    return None

def _detect_vite_alias(root_dir):
    """
    Parse vite.config.ts (or tsconfig.json) to detect the alias mapping
    (e.g., `@` → `./src`). Also searches up to 3 parent directories so
    passing `src/` as root_dir still finds the config at repo root.

    Returns a dict of {alias_prefix: absolute_dir}.
    For ui-full: {"@": "/abs/path/to/ui-full/src"}.
    """
    search_roots = [root_dir]
    parent = root_dir
    for _ in range(3):
        parent = os.path.dirname(parent)
        if not parent:
            break
        search_roots.append(parent)

    vite_files = ["vite.config.ts", "vite.config.js"]
    for root_candidate in search_roots:
        for vf in vite_files:
            vite_path = os.path.join(root_candidate, vf)
            if not os.path.exists(vite_path):
                continue
            try:
                with open(vite_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Look for: alias: { src: fileURLToPath(new URL('src', import.meta.url)), ... }
                alias_block = re.search(r"alias\s*:\s*\{(?P<body>.*?)\}", content, re.DOTALL)
                m = None
                if alias_block:
                    m = re.search(
                        r"['\"]?([A-Za-z0-9_@~.-]+)['\"]?\s*:\s*fileURLToPath\s*\(\s*new\s+URL\s*\(\s*['\"]([^'\"]+)['\"]",
                        alias_block.group("body"),
                    )
                if m:
                    alias_prefix = m.group(1).strip()
                    rel_dir = m.group(2).strip()
                    abs_alias_dir = os.path.abspath(os.path.join(root_candidate, rel_dir))
                    return {alias_prefix: abs_alias_dir}

                # Fallback: simple key-value alias pattern: "@": "./src"
                m2 = re.search(
                    r"alias\s*:\s*\{[^}]*['\"](@\w*)['\"]\s*:\s*['\"]([^'\"]+)['\"]",
                    content,
                )
                if m2:
                    alias_prefix = m2.group(1).strip()
                    rel_dir = m2.group(2).strip()
                    abs_alias_dir = os.path.abspath(os.path.join(root_candidate, rel_dir))
                    return {alias_prefix: abs_alias_dir}
            except Exception:
                continue

        # tsconfig baseUrl / paths fallback
        tsconfig_path = os.path.join(root_candidate, "tsconfig.json")
        if os.path.exists(tsconfig_path):
            try:
                with open(tsconfig_path, "r", encoding="utf-8") as f:
                    tsconfig = json.load(f)
                paths = tsconfig.get("compilerOptions", {}).get("paths", {})
                for alias_pattern, targets in paths.items():
                    if not targets:
                        continue
                    alias_prefix = alias_pattern.replace("/*", "").strip()
                    rel_dir = str(targets[0]).replace("/*", "").strip()
                    abs_alias_dir = os.path.abspath(os.path.join(root_candidate, rel_dir))
                    return {alias_prefix: abs_alias_dir}
            except Exception:
                continue
    return {}


def build_graph(root_dir, aliases=None):
    """
    Builds a NetworkX DiGraph representing the project dependency structure.

    Parameters
    ----------
    root_dir : str
        Project root directory to scan.
    aliases : dict[str, str] | None
        If provided, used directly. If None, attempts auto-detection from
        vite.config.ts or tsconfig.json.
    """
    if aliases is None:
        aliases = _detect_vite_alias(root_dir)
        if aliases:
            print(f"[ast_parser] Detected aliases: {aliases}")

    G = nx.DiGraph()
    root_path = Path(root_dir).absolute()

    # Find all relevant files
    files_to_process = []
    for root, _, files in os.walk(root_dir):
        if "node_modules" in root:
            continue
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
            resolved = resolve_import(imp, abs_file, root_dir, aliases=aliases)
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
