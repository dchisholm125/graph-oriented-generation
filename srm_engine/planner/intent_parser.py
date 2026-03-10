"""
intent_parser.py - Rule-based Natural Language Intent Parser

Module contract: Takes a natural language prompt string, returns List[OperationSpec]
— an ordered list of AddFieldOperation and/or MutateActionOperation dataclasses.

This module is a no-LLM zone. All parsing is deterministic pattern matching.
If the prompt matches known patterns, it produces structured OperationSpec.
If not, it raises IntentParseError. The caller is responsible for handling failure.
"""

import re
from dataclasses import dataclass
from typing import Literal, Union, List


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures (Operation Specifications)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AddFieldOperation:
    """Specification: Add a new field to a store's state object."""
    operation: Literal["ADD_FIELD"]
    target_file: str    # e.g. "src/stores/authStore.ts"
    target_node: str    # e.g. "state"
    field_name: str     # e.g. "lastLogin"
    field_type: str     # e.g. "string"
    default_value: str  # e.g. "''"


@dataclass
class MutateActionOperation:
    """Specification: Add a statement to a store action."""
    operation: Literal["MUTATE_ACTION"]
    target_file: str        # e.g. "src/stores/authStore.ts"
    target_action: str      # e.g. "login"
    add_statement: str      # e.g. "this.lastLogin = '2026-03-08'"


OperationSpec = Union[AddFieldOperation, MutateActionOperation]


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class IntentParseError(Exception):
    """Raised when natural language prompt doesn't match known patterns."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Regex Patterns for Easy Task (ADD_FIELD + MUTATE_ACTION)
# ─────────────────────────────────────────────────────────────────────────────

# Matches file paths like `src/stores/authStore.ts` or `src/stores/authStore.vue`
FILE_PATTERN = re.compile(r'`(src/[^`]+\.(?:ts|vue))`')

# Matches patterns like "add a `lastLogin` string" or "add `lastLogin` string"
ADD_FIELD_RE = re.compile(r'add\s+a?\s*`(\w+)`\s+(\w+)', re.IGNORECASE)

# Checks if context is "to the default state" or similar
STATE_RE = re.compile(r'to\s+the\s+(?:default\s+)?state', re.IGNORECASE)

# Matches patterns like "update the `login` action"
ACTION_RE = re.compile(r'update\s+the\s+`(\w+)`\s+action', re.IGNORECASE)

# Matches patterns like "set it to '2026-03-08'" or 'set it to "value"'
SET_VALUE_RE = re.compile(r"set\s+it\s+to\s+['\"]([^'\"]+)['\"]", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent(prompt: str) -> List[OperationSpec]:
    """
    Rule-based parser: converts natural language prompt to structured OperationSpec.

    For the Easy task (ADD_FIELD + MUTATE_ACTION), extracts:
    1. Target file path (via FILE_PATTERN)
    2. ADD_FIELD: field name and type (via ADD_FIELD_RE + STATE_RE)
    3. MUTATE_ACTION: action name and statement to add (via ACTION_RE + SET_VALUE_RE)

    Args:
        prompt: Natural language instruction string.

    Returns:
        List[OperationSpec]: Ordered list of operations (ADD_FIELD before MUTATE_ACTION).

    Raises:
        IntentParseError: If mandatory patterns don't match.
    """
    operations: List[OperationSpec] = []

    # ── Extract target file ──────────────────────────────────────────────────
    file_match = FILE_PATTERN.search(prompt)
    if not file_match:
        raise IntentParseError(
            f"Could not extract target file. Expected format: `src/stores/...`"
        )
    target_file = file_match.group(1)

    # ── Try ADD_FIELD pattern ────────────────────────────────────────────────
    add_field_match = ADD_FIELD_RE.search(prompt)
    if add_field_match:
        field_name = add_field_match.group(1)
        field_type = add_field_match.group(2)

        # Verify it's "to the default state" context
        if not STATE_RE.search(prompt):
            raise IntentParseError(
                f"ADD_FIELD pattern matched but not in state context. "
                f"Expected: 'to the default state'"
            )

        # Default value is empty string for string types
        default_value = "''"

        add_field_op = AddFieldOperation(
            operation="ADD_FIELD",
            target_file=target_file,
            target_node="state",
            field_name=field_name,
            field_type=field_type,
            default_value=default_value,
        )
        operations.append(add_field_op)

    # ── Try MUTATE_ACTION pattern ────────────────────────────────────────────
    action_match = ACTION_RE.search(prompt)
    if action_match:
        target_action = action_match.group(1)

        # Extract the value to set
        value_match = SET_VALUE_RE.search(prompt)
        if not value_match:
            raise IntentParseError(
                f"MUTATE_ACTION pattern matched (action: '{target_action}') "
                f"but could not extract value. Expected: 'set it to \"value\"'"
            )
        value = value_match.group(1)

        # Build the statement. For the Easy task, assume we're setting a field
        # that was just added (same name as the ADD_FIELD field_name).
        if operations and isinstance(operations[0], AddFieldOperation):
            field_name = operations[0].field_name
        else:
            # If no ADD_FIELD preceded, try to infer field name from context or error
            raise IntentParseError(
                f"MUTATE_ACTION requires preceding ADD_FIELD to infer field name."
            )

        add_statement = f"this.{field_name} = '{value}'"

        mutate_op = MutateActionOperation(
            operation="MUTATE_ACTION",
            target_file=target_file,
            target_action=target_action,
            add_statement=add_statement,
        )
        operations.append(mutate_op)

    if not operations:
        raise IntentParseError(
            f"No recognized patterns matched. "
            f"Supported: ADD_FIELD + MUTATE_ACTION (Easy task)."
        )

    return operations


if __name__ == "__main__":
    # Test the parser
    test_prompt = (
        "Write the code to add a `lastLogin` string timestamp to the default state "
        "in `src/stores/authStore.ts` and update the `login` action to set it to '2026-03-08'."
    )

    try:
        ops = parse_intent(test_prompt)
        print(f"✓ Parsed {len(ops)} operations:")
        for i, op in enumerate(ops, 1):
            print(f"  {i}. {op}")
    except IntentParseError as e:
        print(f"✗ Parse error: {e}")
