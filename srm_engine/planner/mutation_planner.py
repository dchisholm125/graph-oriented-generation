"""
mutation_planner.py - Deterministic Mutation Planning

Module contract: Takes List[OperationSpec] + nx.DiGraph (GOG graph) + repo_root,
resolves relative paths to absolute graph nodes, validates presence, and returns
a MutationPlan with the target file's content ready for the renderer.

All operations are deterministic graph lookups — no LLM calls.
"""

import os
from dataclasses import dataclass
from typing import List
import networkx as nx

from .intent_parser import OperationSpec, AddFieldOperation, MutateActionOperation


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MutationPlan:
    """
    Complete specification of mutations to apply to a target file.

    All path resolution and validation happens at plan-time (here),
    not at render-time. The plan is immutable once created.
    """
    target_file_rel: str           # Relative path as parsed ("src/stores/authStore.ts")
    target_file_abs: str           # Absolute path resolved via graph
    operations: List[OperationSpec] # Ordered list of operations to apply
    file_content: str              # Raw file content read at plan time


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PlannerError(Exception):
    """Raised when mutation planning fails (file not in graph, etc)."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plan_mutations(
    ops: List[OperationSpec],
    graph: nx.DiGraph,
    repo_root: str,
) -> MutationPlan:
    """
    Resolves target file in graph, reads content, returns validated MutationPlan.

    Args:
        ops: List of OperationSpec from intent_parser.parse_intent().
        graph: NetworkX DiGraph from ast_parser.build_graph().
        repo_root: Absolute path to repository root.

    Returns:
        MutationPlan with target_file_abs resolved and file_content loaded.

    Raises:
        PlannerError: If target file not found in graph or I/O fails.
    """
    if not ops:
        raise PlannerError("No operations provided.")

    # All operations must reference the same target file
    target_file_rel = ops[0].target_file
    for op in ops[1:]:
        if op.target_file != target_file_rel:
            raise PlannerError(
                f"All operations must reference the same file. "
                f"Found '{op.target_file}' but expected '{target_file_rel}'."
            )

    # ── Resolve relative path to absolute ────────────────────────────────────
    target_file_abs = os.path.abspath(os.path.join(repo_root, target_file_rel))

    # ── Verify file exists in graph ──────────────────────────────────────────
    # Graph nodes are absolute paths. We've computed target_file_abs,
    # so it should match a node exactly.
    if target_file_abs not in graph.nodes():
        # Try to find a close match (in case of path normalization issues)
        matching_nodes = [
            n for n in graph.nodes()
            if os.path.normpath(n) == os.path.normpath(target_file_abs)
        ]
        if not matching_nodes:
            raise PlannerError(
                f"Target file not found in graph: {target_file_abs}\n"
                f"Available nodes: {list(graph.nodes())[:3]}... "
                f"(showing first 3 of {graph.number_of_nodes()} total)"
            )
        target_file_abs = matching_nodes[0]

    # ── Read file content ────────────────────────────────────────────────────
    try:
        with open(target_file_abs, "r", encoding="utf-8") as f:
            file_content = f.read()
    except IOError as e:
        raise PlannerError(f"Failed to read file {target_file_abs}: {e}")

    return MutationPlan(
        target_file_rel=target_file_rel,
        target_file_abs=target_file_abs,
        operations=ops,
        file_content=file_content,
    )


if __name__ == "__main__":
    # Test the planner (requires graph.pkl and target_repo)
    import pickle
    from srm_engine import ast_parser

    graph_path = os.path.join(os.path.dirname(__file__), "../../gog_graph.pkl")
    target_repo = os.path.join(os.path.dirname(__file__), "../../target_repo")

    if not os.path.exists(graph_path):
        print(f"Graph file not found: {graph_path}")
        print("Run seed_RAG_and_GOG.py first.")
    else:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        from .intent_parser import parse_intent

        test_prompt = (
            "Write the code to add a `lastLogin` string timestamp to the default state "
            "in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
        )

        try:
            ops = parse_intent(test_prompt)
            print(f"✓ Parsed {len(ops)} operations")

            plan = plan_mutations(ops, G, target_repo)
            print(f"✓ Mutation plan created:")
            print(f"  Target: {plan.target_file_rel} (abs: {plan.target_file_abs})")
            print(f"  Operations: {len(plan.operations)}")
            print(f"  File content length: {len(plan.file_content)} chars")
        except Exception as e:
            print(f"✗ Error: {e}")
