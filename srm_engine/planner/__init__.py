"""
SRM Planner — Phase 2: Intent Parser + Mutation Planner + Renderer Prompt

This module implements the symbolic planning layer for the SRM (Symbolic Reasoning
Model) pipeline. It replaces LLM-based reasoning with deterministic graph operations.

Public API:
    - intent_parser.parse_intent(prompt: str) -> List[OperationSpec]
    - mutation_planner.plan_mutations(ops, graph, repo_root) -> MutationPlan
    - renderer_prompt.build_renderer_prompt(plan: MutationPlan) -> str
"""
