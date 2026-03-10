"""
renderer_prompt.py - Symbolic Specification Builder

Module contract: Takes MutationPlan, assembles the symbolic spec string.
This is the prompt that the LLM renderer receives — NOT the original natural language.

The contract is strict: SYMBOLIC SPECIFICATION → DO NOT DEVIATE
The LLM receives only this spec, not the raw natural language prompt.
This is what makes SRM falsifiable: the LLM's reasoning is constrained to syntax only.
"""

import re
from .mutation_planner import MutationPlan
from .intent_parser import AddFieldOperation, MutateActionOperation


# ─────────────────────────────────────────────────────────────────────────────
# Content Sanitization (Noise Stripping)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_store_skeleton(content: str) -> str:
    """
    Strip noise from file content to avoid overwhelming the LLM.

    The benchmark target files (from generate_dummy_repo.py) contain:
      1. DUMMY_ASSETS block — massive base64 SVG strings
      2. Random boilerplate comments — hundreds of lines of garbage
      3. Other structural noise — inflates token count, obscures actual code

    This extractor returns only:
      - Import lines (preserve)
      - defineStore definition (preserve, up to its closing brace)
      - Everything else is noise and dropped

    Reduces file from ~200+ lines to ~15 lines of meaningful code.
    Still deterministic — no LLM involved. Planner's job: present clean spec.

    Args:
        content: Raw file content from mutation_planner.

    Returns:
        Sanitized content with noise removed.
    """
    lines = content.split('\n')
    skeleton = []
    in_dummy_block = False
    in_boilerplate_comment = False
    brace_depth = 0
    found_define_store = False

    for line in lines:
        stripped = line.strip()

        # ── Skip DUMMY_ASSETS block ─────────────────────────────────────────
        if 'DUMMY_ASSETS' in line and '=' in line:
            in_dummy_block = True
        if in_dummy_block:
            if stripped.endswith('};'):
                in_dummy_block = False
            continue

        # ── Skip random boilerplate comment block ────────────────────────────
        # These are /** ... */ blocks with very long lines (80+ chars of gibberish)
        if '/**' in line:
            in_boilerplate_comment = True
        if in_boilerplate_comment:
            # Skip this line entirely
            if '*/' in line:
                in_boilerplate_comment = False
            continue

        # ── Collect import lines ────────────────────────────────────────────
        if not found_define_store and stripped.startswith('import '):
            skeleton.append(line)
            continue

        # ── Look for and collect defineStore definition ──────────────────────
        if 'defineStore' in line:
            found_define_store = True
            brace_depth = line.count('{') - line.count('}')
            skeleton.append(line)
            # If the entire store fits on one line, we're done
            if brace_depth == 0:
                break
            continue

        # ── Collect lines inside defineStore ────────────────────────────────
        if found_define_store:
            brace_depth += line.count('{') - line.count('}')
            skeleton.append(line)
            # Stop when store closes
            if brace_depth == 0:
                break
            continue

        # ── Skip everything else (noise, comments outside of imports) ───────
        # But only if we haven't found defineStore yet
        if not found_define_store:
            continue

    return '\n'.join(skeleton).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_renderer_prompt(plan: MutationPlan) -> str:
    """
    Assembles symbolic spec from MutationPlan, stripping noise.

    The output is a strict specification — not a creative prompt.
    The LLM is being asked to apply a diff, not to think about architecture.

    Content is sanitized before presentation: DUMMY_ASSETS blocks, random
    boilerplate comments, and other structural noise are removed. The LLM
    receives only imports + defineStore definition (~15 lines), not 200+ lines
    of noise. This prevents content poisoning where the LLM gets lost in
    deliberately injected red herrings (from generate_dummy_repo.py).

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

    # Strip noise from file content before passing to LLM
    clean_content = _extract_store_skeleton(plan.file_content)

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
{clean_content}"""

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
