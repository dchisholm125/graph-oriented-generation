"""
srm_engine/salience_evaluator.py
─────────────────────────────────
The Neuro-Symbolic "Salience Membrane".

Background
----------
Graph-Oriented Generation (GOG) isolates a precise topological sub-graph of
files relevant to a given prompt.  This is strictly better than Vector RAG for
architectural tasks because the DAG is built from real import statements — it
cannot hallucinate relationships that do not exist in source code.

However, GOG on its own is a *retrieval* guarantee, not a *generation* guarantee.
Even with a perfectly curated context, an LLM can still emit code that imports
beyond the provided boundary — a condition we call an "architectural
hallucination".

This module implements the Salience Membrane: the deterministic filter that
separates the LLM's stochastic output space from the downstream build system.

Neuro-Symbolic Architecture
---------------------------
The term "Neuro-Symbolic" describes a hybrid AI system combining:

  • Neuro (Neural LLM)  —  creative, stochastic.  Writes code, generalises
                            patterns.  Can hallucinate imports.
  • Symbolic (SRM Graph) —  deterministic, ground-truth.  Every edge is a
                            real `import` statement from source code.  Cannot
                            hallucinate.

The Membrane sits at the boundary.  It:
  1. Accepts  LLM output whose proposed imports all resolve to `allowed_nodes`.
  2. Rejects  LLM output that attempts to reach outside the graph boundary,
              returning a human-readable reason so the calling loop can build a
              corrective prompt and retry (Rejection Sampling).

Rejection Sampling
------------------
"Rejection Sampling" is a Monte Carlo technique: draw a sample, test a
constraint, discard failures, draw again.  Here::

    sample      = one LLM response
    distribution = the LLM's generative output space (temperature-driven)
    constraint  = the GOG symbolic dependency graph

The LLM is trapped in a corrective feedback loop until it produces a
topologically valid response or exhausts `max_attempts`.  Each rejection
injecting the specific illegal import names steers the model toward compliance
on the next draw.

AST / Regex Dual-Layer Extraction
----------------------------------
Import extraction uses two layers:

  1. Primary: tree-sitter AST via `TypeScriptParser.extract_imports()`.
     Structurally precise but has a known quirk: it silently returns `[]` for
     files missing a trailing newline (tree-sitter does not parse the last
     AST node if the file is not newline-terminated).  We always append `\\n`
     before writing the temp file to work around this.

  2. Fallback: module-level `_IMPORT_RE` regex.  If the AST yields zero
     results but the code string contains the word `import`, we fall back to
     regex extraction — the same defensive pattern already used in
     `ast_parser.py`.  This prevents the most dangerous failure mode:
     silently accepting a hallucinated import because the parser had nothing
     to report.

Contributing
------------
See CONTRIBUTING.md for code style and testing guidance.
All new language parsers should expose the same `extract_imports(file_path)`
interface as `TypeScriptParser` so this evaluator can accept them as a
drop-in `self.parser`.
"""
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from srm_engine.ts_parser import TypeScriptParser

# ─────────────────────────────────────────────────────────────────────────────
# Module-level compiled regex for the AST fallback import extractor.
# This mirrors the exact pattern used in `srm_engine/ast_parser.py` so that
# both layers of the toolchain agree on what constitutes an import path.
# `re.MULTILINE` is required to correctly anchor `import` at line boundaries
# across multi-line source files.
_IMPORT_RE = re.compile(r"import\s+.*?from\s+['\"](.+?)['\"]", re.MULTILINE)


@dataclass
class EvaluationResult:
    """
    The verdict returned by SalienceEvaluator.evaluate().

    Attributes
    ----------
    is_valid : bool
        True when the LLM's proposed imports all resolve to files within the
        permitted topological boundary.  False when any local import is
        outside the boundary or when the code cannot be parsed.
    reason : str | None
        Human-readable explanation of the rejection, designed to be injected
        directly back into the LLM prompt as corrective SYSTEM FEEDBACK.
        None on acceptance.
    extracted_code : str | None
        The raw TypeScript / Vue code extracted from the LLM's markdown
        response.  Populated on both accept and reject so the calling loop
        can inspect or log the last generated code.
    """
    is_valid: bool
    reason: Optional[str] = None
    extracted_code: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Import classifier helpers.
# ─────────────────────────────────────────────────────────────────────────────

# Third-party / node_module import specifiers that should NEVER be resolved
# against the local file-system.  Anything that is a relative path (starts
# with '.' or '..') falls outside this set and MUST be validated against the
# DAG's allowed_nodes.  This list is intentionally minimal — extend it if
# your project uses other non-local package namespaces.
_NODE_MODULE_PREFIXES = (
    "vue",
    "pinia",
    "vite",
    "axios",
    "lodash",
    "@vueuse",
    "@vue",
    "vue-router",
)


def _is_local_import(specifier: str) -> bool:
    """Return True when *specifier* refers to a local file (relative path).

    We treat any specifier starting with '.' (relative: ``./foo``, ``../bar``)
    or '/' (absolute path — rare but possible in monorepos) as a local import
    that must be validated.  Everything else is assumed to be a node_module
    and is allowed unconditionally.
    """
    return specifier.startswith(".") or specifier.startswith("/")


class SalienceEvaluator:
    """
    The Neuro-Symbolic Membrane.

    Acts as a strict deterministic filter between the LLM's stochastic output
    and the downstream build system.  Given a topological boundary defined by
    `allowed_nodes` (the set of absolute paths the SRM isolated), it rejects
    any LLM response that tries to import a file that lives outside those
    boundaries.
    """

    def __init__(self, allowed_nodes: List[str]):
        """
        Parameters
        ----------
        allowed_nodes:
            List of absolute paths representing the permitted topological
            boundaries.  These are exactly the files returned by
            ``graph_search.isolate_context()``.
        """
        self.allowed_nodes: List[str] = [os.path.abspath(p) for p in allowed_nodes]
        # Pre-compute a set of *basenames* so that relative imports that resolve
        # to a permitted file are accepted even when the exact relative path
        # differs from the absolute path stored in the graph.
        self._allowed_basenames: set = {
            os.path.basename(p) for p in self.allowed_nodes
        }
        self.parser = TypeScriptParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_code_blocks(self, markdown_text: str) -> str:
        """
        Extracts and joins every ```ts, ```typescript, or ```vue fenced code
        block found in *markdown_text*.

        Returns an empty string when no matching block is found.
        """
        pattern = re.compile(
            r"```(?:typescript|ts|vue)\s*\n(.*?)```",
            re.DOTALL | re.IGNORECASE,
        )
        blocks = pattern.findall(markdown_text)
        if not blocks:
            return ""
        return "\n\n".join(block.strip() for block in blocks)

    def evaluate(self, llm_response_text: str) -> EvaluationResult:
        """
        Parses the LLM's output and validates its topological constraints.

        Algorithm
        ---------
        1. Extract code blocks from the raw markdown.
        2. Write the code to a temporary file so the AST parser can read it.
        3. Call ``TypeScriptParser.extract_imports()`` on the temp file.
        4. For each local (relative) import, verify that it resolves to a
           file that is within ``allowed_nodes``.  Package imports from
           node_modules are ignored.
        5. Return an ``EvaluationResult`` with ``is_valid=True`` on success or
           ``is_valid=False`` with a human-readable ``reason`` on failure.
        """
        # ── Step 1: Extract code ────────────────────────────────────────
        code = self.extract_code_blocks(llm_response_text)

        if not code:
            return EvaluationResult(
                is_valid=False,
                reason=(
                    "No TypeScript or Vue code block found in the LLM response. "
                    "The response must contain at least one ```ts, ```typescript, "
                    "or ```vue fenced code block."
                ),
                extracted_code=None,
            )

        # ── Step 2: Write to a temporary TS file ────────────────────────
        # We always use the .ts extension so the parser's AST query works
        # uniformly regardless of whether the original block was Vue or TS.
        # A trailing newline is required for tree-sitter to parse the final
        # statement correctly — without it single-line files return no nodes.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ts", prefix="salience_eval_")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
                tmp_file.write(code if code.endswith("\n") else code + "\n")

            # ── Step 3: AST-parse the imports ───────────────────────────
            try:
                proposed_imports: List[str] = self.parser.extract_imports(tmp_path)
            except Exception as parse_exc:
                return EvaluationResult(
                    is_valid=False,
                    reason=(
                        f"Syntax error — the AST parser could not parse the generated "
                        f"code: {parse_exc}"
                    ),
                    extracted_code=code,
                )

        finally:
            # Always clean up the temp file, even on error.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # ── Step 3b: Regex fallback ──────────────────────────────────────
        # tree-sitter can return an empty list for valid-but-complex code.
        # If the code visually contains import statements, use the regex
        # pattern from ast_parser.py as a safety net so we never silently
        # let hallucinated imports through.
        if not proposed_imports and "import" in code:
            proposed_imports = _IMPORT_RE.findall(code)

        # ── Step 4: Validate each local import ──────────────────────────
        violations: List[str] = []

        for specifier in proposed_imports:
            if not _is_local_import(specifier):
                # node_modules / package import — skip
                continue

            # Resolve the basename the import is pointing at.
            # We strip trailing extensions that might be omitted by the LLM.
            basename = os.path.basename(specifier)

            # Add common extensions to probe if the basename has none
            candidates = [basename]
            if not os.path.splitext(basename)[1]:
                candidates = [
                    f"{basename}.ts",
                    f"{basename}.vue",
                    f"{basename}/index.ts",
                ]

            matched = False
            for candidate in candidates:
                if candidate in self._allowed_basenames:
                    matched = True
                    break

            if not matched:
                violations.append(specifier)

        # ── Step 5: Return result ────────────────────────────────────────
        if violations:
            violation_list = ", ".join(f"'{v}'" for v in violations)
            allowed_list = ", ".join(
                os.path.basename(p) for p in self.allowed_nodes
            )
            return EvaluationResult(
                is_valid=False,
                reason=(
                    f"Architectural hallucination detected. The following local imports "
                    f"are outside the permitted topological boundary: [{violation_list}]. "
                    f"You may ONLY import from these files: [{allowed_list}]."
                ),
                extracted_code=code,
            )

        return EvaluationResult(
            is_valid=True,
            reason=None,
            extracted_code=code,
        )
