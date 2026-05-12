"""
graph_search.py - SRM Engine: Semantic Seeding & Deterministic Traversal

Phase 1: Semantic Seeding
    Maps a natural language prompt to graph entry points using cosine similarity
    over sentence embeddings (all-MiniLM-L6-v2). This replaces brittle keyword
    matching and handles prompts that lack explicit architectural vocabulary —
    the primary failure mode identified in the Phase 1 evaluation (see paper §4.5).

Phase 2: Deterministic Traversal
    Once seed nodes are established, all context isolation is performed via
    mathematically exact graph operations (shortest path, descendant traversal).
    No probabilistic inference occurs after the seeding step.
"""

import os
# ── Silence TF/CUDA noise before sentence-transformers triggers the import chain
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import re
import numpy as np
import networkx as nx
from typing import Optional

# Lazy-loaded to avoid import cost when module is imported without seeding
_embedding_model = None

# Similarity threshold: nodes with cosine similarity below this are excluded as seeds.
# Tunable — lower values increase recall at the cost of precision.
SEED_SIMILARITY_THRESHOLD = 0.25

# Maximum number of seed nodes to select from the similarity ranking.
# Prevents over-seeding in large graphs with many semantically similar filenames.
MAX_SEEDS = 5


def _get_model():
    """Lazy-loads the sentence embedding model on first use."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic seeding. "
                "Install via: pip install sentence-transformers"
            )
    return _embedding_model


def _node_to_label(node_path: str) -> str:
    """
    Converts an absolute file path to a human-readable label for embedding.

    Strips extension and converts camelCase/PascalCase to space-separated tokens,
    improving semantic alignment between filenames and natural language prompts.

    Example:
        /project/src/stores/authStore.ts -> "auth Store"
        /project/src/components/HeaderWidget.vue -> "Header Widget"
    """
    name = os.path.splitext(os.path.basename(node_path))[0]
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    label = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", label)
    return label.replace("_", " ").replace("-", " ")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def seed_graph_from_prompt(
    graph: nx.DiGraph,
    prompt: str,
    threshold: float = SEED_SIMILARITY_THRESHOLD,
    max_seeds: int = MAX_SEEDS,
    node_embeddings: Optional[dict] = None,
) -> list:
    """
    Identifies seed nodes in the graph whose filenames are semantically
    similar to the given prompt.

    Uses sentence-transformer embeddings (all-MiniLM-L6-v2) to compute
    cosine similarity between the prompt and each node label. Nodes exceeding
    the similarity threshold are returned as entry points for graph traversal.

    Args:
        graph:           The dependency graph built by ast_parser.build_graph().
        prompt:          Natural language query from the user.
        threshold:       Minimum cosine similarity to qualify as a seed node.
        max_seeds:       Maximum number of seed nodes to return.
        node_embeddings: Optional pre-computed {node_path: embedding} dict.
                         Pass this when calling repeatedly on the same graph
                         to avoid redundant embedding computation.

    Returns:
        List of absolute file paths selected as seed nodes.
    """
    model = _get_model()
    nodes = list(graph.nodes())

    if not nodes:
        return []

    prompt_embedding = model.encode(prompt, convert_to_numpy=True)

    similarities = []
    for node in nodes:
        if node_embeddings and node in node_embeddings:
            node_emb = node_embeddings[node]
        else:
            label = _node_to_label(node)
            node_emb = model.encode(label, convert_to_numpy=True)

        sim = _cosine_similarity(prompt_embedding, node_emb)
        similarities.append((node, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    seeds = [
        node for node, sim in similarities
        if sim >= threshold
    ][:max_seeds]

    return seeds


def build_node_embeddings(graph: nx.DiGraph) -> dict:
    """
    Pre-computes and returns embeddings for all node labels in the graph.

    Call this once after graph construction and pass the result to
    seed_graph_from_prompt() to avoid re-embedding on every query.
    Particularly valuable in benchmarks running multiple prompts
    against the same graph.

    Returns:
        Dict mapping absolute node path -> numpy embedding vector.
    """
    model = _get_model()
    nodes = list(graph.nodes())
    labels = [_node_to_label(n) for n in nodes]
    embeddings = model.encode(labels, convert_to_numpy=True, show_progress_bar=False)
    return dict(zip(nodes, embeddings))


def _extract_filename_mentions(prompt: str, graph: nx.DiGraph) -> list:
    """
    Extract full file paths mentioned in the prompt and match them against
    graph nodes (case-insensitive suffix match).

    Uses ONLY full relative paths (e.g. `src/router/index.ts`).
    Never falls back to bare basenames like `index.ts` to avoid collision
    between `router/index.ts` and `types/index.ts`.
    """
    m1 = re.findall(r'src/[A-Za-z0-9_/.-]+\.(?:ts|vue|js)', prompt)
    mentions = list(dict.fromkeys(m1))  # preserve order, de-duplicate

    matched = []
    seen = set()
    for candidate in mentions:
        candidate_lower = candidate.lower().replace("/", os.sep)
        for node in graph.nodes():
            node_lower = node.lower().replace("/", os.sep)
            if node_lower.endswith(candidate_lower):
                if node not in seen:
                    seen.add(node)
                    matched.append(node)
                break
    return matched


def isolate_context(
    graph: nx.DiGraph,
    prompt: str,
    node_embeddings: Optional[dict] = None,
) -> list:
    """
    Returns the minimal set of files required to answer the given prompt,
    determined by deterministic graph traversal.

    Traversal strategy:
        - Explicit filenames (e.g. `src/router/index.ts`) take absolute priority
          over semantic seeding because they are user-supplied ground truth.
        - Semantic seeding (sentence-transformer similarity) is a fallback.
        - Once seeds are chosen, descendants are bounded to max_depth=2 to
          avoid explosion from highly-connected entry-point nodes.

    Args:
        graph:           Dependency graph from ast_parser.build_graph().
        prompt:          Natural language query.
        node_embeddings: Optional pre-computed embeddings (see build_node_embeddings).

    Returns:
        Sorted list of absolute file paths forming the isolated context.
    """
    # ── Priority 1: explicit file-name mentions in prompt ────────────────────
    keyword_seeds = _extract_filename_mentions(prompt, graph)
    if keyword_seeds:
        seeds = keyword_seeds[:5]
    else:
        # ── Priority 2: semantic seeding ───────────────────────────────────
        seeds = seed_graph_from_prompt(graph, prompt, node_embeddings=node_embeddings)

    if not seeds:
        # Last resort: return only the highest-degree nodes (most connected / central)
        # rather than every file in the repo.
        degrees = {n: graph.degree(n) for n in graph.nodes()}
        top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:8]
        return sorted([n for n, _ in top])

    subgraph_nodes = set()

    if len(seeds) == 1:
        subgraph_nodes.add(seeds[0])
        subgraph_nodes.update(_descendants_bounded(graph, seeds[0]))
    else:
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                source, target = seeds[i], seeds[j]

                try:
                    path = nx.shortest_path(graph, source=source, target=target)
                    subgraph_nodes.update(path)
                    subgraph_nodes.update(_descendants_bounded(graph, target))
                except nx.NetworkXNoPath:
                    pass

                try:
                    path = nx.shortest_path(graph, source=target, target=source)
                    subgraph_nodes.update(path)
                    subgraph_nodes.update(_descendants_bounded(graph, source))
                except nx.NetworkXNoPath:
                    pass

        if not subgraph_nodes:
            for seed in seeds:
                subgraph_nodes.add(seed)
                subgraph_nodes.update(_descendants_bounded(graph, seed))

    return sorted(list(subgraph_nodes))


def _descendants_bounded(graph, source, max_depth=2):
    """
    Return descendants of *source* within *max_depth* hops.

    This replaces `nx.descendants()` which can explode when a seed node
    (e.g. a dashboard view) imports a large surface of the codebase.
    A depth of 2 is usually sufficient: the node itself, its direct
    imports, and their direct imports.
    """
    if max_depth <= 0:
        return set()
    visited = set()
    current = {source}
    for _ in range(max_depth):
        next_level = set()
        for node in current:
            for succ in graph.successors(node):
                if succ not in visited:
                    visited.add(succ)
                    next_level.add(succ)
        current = next_level
        if not current:
            break
    visited.discard(source)
    return visited


if __name__ == "__main__":
    from srm_engine import ast_parser

    target = os.path.join(os.path.dirname(__file__), "../target_repo")
    if not os.path.exists(target):
        print("target_repo not found. Run generate_dummy_repo.py first.")
    else:
        print("Building dependency graph...")
        G = ast_parser.build_graph(target)

        print("Pre-computing node embeddings...")
        embeddings = build_node_embeddings(G)

        test_prompts = [
            "Identify the default state variables initialized in the main authentication store.",
            "Locate where user authentication state is passed to the dashboard header widget.",
            "Trace the execution path when a user clicks logout from the settings view.",
        ]

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            context_files = isolate_context(G, prompt, node_embeddings=embeddings)
            print(f"Isolated {len(context_files)} files:")
            for f in context_files:
                print(f"  - {os.path.relpath(f, target)}")
