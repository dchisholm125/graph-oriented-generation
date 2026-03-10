"""
renderer_prompt.py - Symbolic Specification Builder

Module contract: Takes MutationPlan, assembles the symbolic spec string.
This is the prompt that the LLM renderer receives — NOT the original natural language.

The contract is strict: SYMBOLIC SPECIFICATION → DO NOT DEVIATE
The LLM receives only this spec, not the raw natural language prompt.
This is what makes SRM falsifiable: the LLM's reasoning is constrained to syntax only.
"""

from .mutation_planner import MutationPlan
from .intent_parser import AddFieldOperation, MutateActionOperation


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_renderer_prompt(plan: MutationPlan) -> str:
    """
    Assembles symbolic spec from MutationPlan.

    The output is a strict specification — not a creative prompt.
    The LLM is being asked to apply a diff, not to think about architecture.

    Args:
        plan: MutationPlan from mutation_planner.plan_mutations().

    Returns:
        String ready to be appended with MUZZLE and sent to LLM.
    """
    operations_text = ""

    for i, op in enumerate(plan.operations, 1):
        if isinstance(op, AddFieldOperation):
            operations_text += (
                f"{i}. ADD_FIELD to state object:\n"
                f"   - name: {op.field_name}\n"
                f"   - type: {op.field_type}\n"
                f"   - default: {op.default_value}\n"
            )
        elif isinstance(op, MutateActionOperation):
            operations_text += (
                f"{i}. MUTATE_ACTION '{op.target_action}':\n"
                f"   - add statement: {op.add_statement}\n"
            )

    spec = f"""SYMBOLIC SPECIFICATION — DO NOT DEVIATE

File: {plan.target_file_rel}
Current content provided below.

Operations to apply:
{operations_text}
Render the complete updated file as valid TypeScript.
Use Pinia defineStore syntax.
Do not add imports that are not already present.
Do not add fields or actions beyond those specified above.

=== CURRENT FILE CONTENT ===
{plan.file_content}"""

    return spec


if __name__ == "__main__":
    # Test the renderer (requires mutation plan)
    from .intent_parser import AddFieldOperation, MutateActionOperation
    from .mutation_planner import MutationPlan

    # Create a test plan
    test_plan = MutationPlan(
        target_file_rel="src/stores/authStore.ts",
        target_file_abs="/tmp/authStore.ts",
        operations=[
            AddFieldOperation(
                operation="ADD_FIELD",
                target_file="src/stores/authStore.ts",
                target_node="state",
                field_name="lastLogin",
                field_type="string",
                default_value="''",
            ),
            MutateActionOperation(
                operation="MUTATE_ACTION",
                target_file="src/stores/authStore.ts",
                target_action="login",
                add_statement="this.lastLogin = '2026-03-08'",
            ),
        ],
        file_content="// Test content\nstate: () => ({}),",
    )

    spec = build_renderer_prompt(test_plan)
    print("=== RENDERER PROMPT ===")
    print(spec)
    print("\n=== SPEC LENGTH ===")
    print(f"{len(spec)} characters")
