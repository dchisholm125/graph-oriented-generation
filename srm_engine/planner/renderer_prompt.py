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
    Assembles symbolic spec from multi-file MutationPlan, stripping noise.

    The output is a strict specification — not a creative prompt.
    The LLM is being asked to apply a diff, not to think about architecture.

    For single-file plans: produces a single-file spec.
    For multi-file plans: produces a section per file with operations and content.

    Content is sanitized before presentation: DUMMY_ASSETS blocks, random
    boilerplate comments, and other structural noise are removed.

    Args:
        plan: MutationPlan from mutation_planner.plan_mutations().

    Returns:
        String ready to be appended with MUZZLE and sent to LLM.
    """
    # Check if this is a single-file or multi-file plan
    is_multi_file = len(plan.operations_by_file) > 1

    if is_multi_file:
        return _build_multifile_spec(plan)
    else:
        return _build_singlefile_spec(plan)


def build_single_file_renderer_prompt(plan: MutationPlan, target_file_rel: str) -> str:
    """
    Renders a single file from a multi-file MutationPlan in isolation.

    This is used for per-file LLM calls (separate rendering pass per file).
    The renderer receives only the operations and content for one file,
    eliminating the need for the LLM to parse multi-file structure.

    Args:
        plan: MutationPlan from mutation_planner.plan_mutations().
        target_file_rel: Relative path of the file to render (e.g., "src/stores/authStore.ts").

    Returns:
        Single-file symbolic spec ready for LLM (with MUZZLE to be appended).

    Raises:
        ValueError: If target_file_rel not in plan.operations_by_file.
    """
    if target_file_rel not in plan.operations_by_file:
        raise ValueError(f"File {target_file_rel} not found in mutation plan")

    operations = plan.operations_by_file[target_file_rel]
    file_content = plan.file_contents[target_file_rel]

    from .intent_parser import (
        AddFieldOperation, MutateActionOperation, AddImportOperation,
        AddTemplateElementOperation, AddSetupBindingOperation,
        AddMethodOperation, AddActionOperation
    )

    # Build operations text for this file
    operations_text = ""
    for i, op in enumerate(operations, 1):
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
        elif isinstance(op, AddImportOperation):
            operations_text += f"{i}. ADD_IMPORT:\n   {op.import_statement}\n"
        elif isinstance(op, AddTemplateElementOperation):
            operations_text += (
                f"{i}. ADD_TEMPLATE_ELEMENT (next to '{op.insert_adjacent_to}'):\n"
                f"   {op.element_html}\n"
            )
        elif isinstance(op, AddSetupBindingOperation):
            operations_text += f"{i}. ADD_SETUP_BINDING:\n   {op.binding_statement}\n"
        elif isinstance(op, AddMethodOperation):
            operations_text += (
                f"{i}. ADD_METHOD '{op.method_name}':\n"
                f"   {op.method_body}\n"
            )
        elif isinstance(op, AddActionOperation):
            operations_text += (
                f"{i}. ADD_ACTION '{op.action_name}':\n"
                f"   {op.action_body}\n"
            )

    # Determine file type and choose appropriate stripper
    is_vue = target_file_rel.endswith('.vue')
    clean_content = _extract_vue_skeleton(file_content) if is_vue else _extract_store_skeleton(file_content)

    # Use appropriate instructions based on file type
    file_type = "Vue component" if is_vue else "TypeScript/Pinia store"
    instructions = (
        f"Render the complete updated {file_type}.\n"
        f"Do not add imports that are not already present (except ADD_IMPORT operations).\n"
        f"Do not add fields or actions beyond those specified above."
        if not is_vue
        else f"Render the complete updated {file_type}.\n"
             f"Preserve all existing template blocks, script sections, and style blocks.\n"
             f"Only add the elements and bindings specified in the operations above."
    )

    spec = f"""SYMBOLIC SPECIFICATION — DO NOT DEVIATE

File: {target_file_rel}
Current content provided below.

Operations to apply:
{operations_text}
{instructions}

=== CURRENT FILE CONTENT ===
{clean_content}"""

    return spec


def _build_singlefile_spec(plan: MutationPlan) -> str:
    """Build symbolic spec for single-file tasks (Easy, Medium)."""
    from .intent_parser import (
        AddFieldOperation, MutateActionOperation, AddImportOperation,
        AddTemplateElementOperation, AddSetupBindingOperation,
        AddMethodOperation, AddActionOperation
    )

    # Get the single file
    target_file_rel = list(plan.operations_by_file.keys())[0]
    operations = plan.operations_by_file[target_file_rel]

    operations_text = ""
    for i, op in enumerate(operations, 1):
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
        elif isinstance(op, AddImportOperation):
            operations_text += f"{i}. ADD_IMPORT:\n   {op.import_statement}\n"
        elif isinstance(op, AddTemplateElementOperation):
            operations_text += (
                f"{i}. ADD_TEMPLATE_ELEMENT (next to '{op.insert_adjacent_to}'):\n"
                f"   {op.element_html}\n"
            )
        elif isinstance(op, AddSetupBindingOperation):
            operations_text += f"{i}. ADD_SETUP_BINDING:\n   {op.binding_statement}\n"
        elif isinstance(op, AddMethodOperation):
            operations_text += (
                f"{i}. ADD_METHOD '{op.method_name}':\n"
                f"   {op.method_body}\n"
            )
        elif isinstance(op, AddActionOperation):
            operations_text += (
                f"{i}. ADD_ACTION '{op.action_name}':\n"
                f"   {op.action_body}\n"
            )

    # Strip noise from file content (use Vue stripper for .vue files)
    is_vue = target_file_rel.endswith('.vue')
    content = plan.file_contents[target_file_rel]
    clean_content = _extract_vue_skeleton(content) if is_vue else _extract_store_skeleton(content)

    # Use appropriate instructions based on file type
    file_type = "Vue component" if is_vue else "TypeScript/Pinia store"
    instructions = (
        f"Render the complete updated {file_type}.\n"
        f"Do not add imports that are not already present (except ADD_IMPORT operations).\n"
        f"Do not add fields or actions beyond those specified above."
        if not is_vue
        else f"Render the complete updated {file_type}.\n"
             f"Preserve all existing template blocks, script sections, and style blocks.\n"
             f"Only add the elements and bindings specified in the operations above."
    )

    spec = f"""SYMBOLIC SPECIFICATION — DO NOT DEVIATE

File: {target_file_rel}
Current content provided below.

Operations to apply:
{operations_text}
{instructions}

=== CURRENT FILE CONTENT ===
{clean_content}"""

    return spec


def _build_multifile_spec(plan: MutationPlan) -> str:
    """Build symbolic spec for multi-file tasks (Medium, Hard)."""
    from .intent_parser import (
        AddFieldOperation, MutateActionOperation, AddImportOperation,
        AddTemplateElementOperation, AddSetupBindingOperation,
        AddMethodOperation, AddActionOperation
    )

    # Build a section for each file
    file_sections = []

    for target_file_rel in sorted(plan.operations_by_file.keys()):
        operations = plan.operations_by_file[target_file_rel]

        # Build operations text for this file
        ops_text = ""
        for i, op in enumerate(operations, 1):
            if isinstance(op, AddImportOperation):
                ops_text += f"{i}. ADD_IMPORT:\n   {op.import_statement}\n"
            elif isinstance(op, AddTemplateElementOperation):
                ops_text += (
                    f"{i}. ADD_TEMPLATE_ELEMENT (next to '{op.insert_adjacent_to}'):\n"
                    f"   {op.element_html}\n"
                )
            elif isinstance(op, AddSetupBindingOperation):
                ops_text += f"{i}. ADD_SETUP_BINDING:\n   {op.binding_statement}\n"
            elif isinstance(op, AddMethodOperation):
                ops_text += (
                    f"{i}. ADD_METHOD '{op.method_name}':\n"
                    f"   {op.method_body}\n"
                )
            elif isinstance(op, AddActionOperation):
                ops_text += (
                    f"{i}. ADD_ACTION '{op.action_name}':\n"
                    f"   {op.action_body}\n"
                )

        # Get file extension to determine content stripper
        is_vue = target_file_rel.endswith('.vue')
        content = plan.file_contents[target_file_rel]
        clean_content = _extract_vue_skeleton(content) if is_vue else _extract_store_skeleton(content)

        # Build file section
        file_sections.append(
            f"─── FILE: {target_file_rel} ───\n"
            f"Operations to apply:\n{ops_text}\n"
            f"Current content:\n{clean_content}"
        )

    # Build constraints section if any
    constraints_text = ""
    if plan.constraints:
        constraints_text = "\nCONSTRAINTS (must enforce):\n"
        for constraint in plan.constraints:
            constraints_text += f"- {constraint.target_file}: must NOT import '{constraint.forbidden_module}'\n"

    spec = f"""SYMBOLIC SPECIFICATION — DO NOT DEVIATE

This task spans multiple files. Render each file completely and separately.

{chr(10).join(file_sections)}{constraints_text}

Instructions:
1. Render each file completely — including all existing code
2. Apply ONLY the operations specified above
3. Do not add imports that are not already present (except ADD_IMPORT operations)
4. Enforce all constraints — do not violate forbidden import rules
5. For Vue files: preserve template, script, and style blocks exactly as they are, only adding to them
6. For TypeScript files: preserve all existing exports and functionality

Output format: one code block per file, clearly labeled with the filename."""

    return spec


def _extract_vue_skeleton(content: str) -> str:
    """
    Extract meaningful parts of a Vue .vue file.

    Preserves:
    - All <template> block content (up to 50 lines if very long)
    - <script setup> or <script> block (imports + bindings, not styles)
    - Omits: <style> blocks, DUMMY_ASSETS, boilerplate comments

    Returns: cleaned Vue file skeleton
    """
    lines = content.split('\n')
    skeleton = []
    in_style = False
    in_dummy = False
    in_template = False
    in_script = False
    template_lines = 0

    for line in lines:
        stripped = line.strip()

        # Skip DUMMY_ASSETS block
        if 'DUMMY_ASSETS' in line and '=' in line:
            in_dummy = True
        if in_dummy:
            if stripped.endswith('};'):
                in_dummy = False
            continue

        # Skip <style> blocks
        if '<style' in line.lower():
            in_style = True
        if in_style:
            skeleton.append(line)
            if '</style>' in line.lower():
                in_style = False
            continue

        # Collect <template> block
        if '<template' in line.lower():
            in_template = True
        if in_template:
            skeleton.append(line)
            template_lines += 1
            if template_lines > 50 and '</template>' in line.lower():
                in_template = False
            if '</template>' in line.lower():
                in_template = False
            continue

        # Collect <script> block
        if '<script' in line.lower():
            in_script = True
        if in_script:
            skeleton.append(line)
            if '</script>' in line.lower():
                in_script = False
            continue

        # Skip random boilerplate comments
        if '/**' in line or (stripped.startswith('*') and len(stripped) > 80):
            continue

        # Collect other meaningful lines (imports, exports, etc)
        if stripped.startswith('import ') or stripped.startswith('export '):
            skeleton.append(line)

    return '\n'.join(skeleton).strip()


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
