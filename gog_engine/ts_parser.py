import os
from tree_sitter import Language, Parser
import tree_sitter_typescript as tstypescript

# Try multiple language loaders for different tree-sitter-typescript versions
try:
    TS_LANGUAGE = Language(tstypescript.language_typescript())
except AttributeError:
    try:
        TS_LANGUAGE = Language(tstypescript.language())
    except AttributeError:
        # Last resort fallback (some versions use the language object directly)
        TS_LANGUAGE = Language(tstypescript.typescript)  # type: ignore
except Exception as e:
    print(f"Error loading tree-sitter-typescript: {e}")
    TS_LANGUAGE = None


def _walk_import_paths(node, results, content):
    """Recursive tree walker to find import_statement → string → import paths."""
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "string":
                raw = content[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
                decoded = raw.strip("'\"")
                if decoded:
                    results.append(decoded)
        return
    for child in getattr(node, "children", []):
        _walk_import_paths(child, results, content)


class TypeScriptParser:
    """Uses tree-sitter to perform precise AST analysis on TS/Vue files."""

    def __init__(self):
        self.parser = Parser(TS_LANGUAGE) if TS_LANGUAGE else None

    def extract_imports(self, file_path):
        """Extracts import specifiers from a TypeScript / Vue file using AST parsing."""
        if not self.parser or not TS_LANGUAGE:
            return []

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # If it's a Vue file, we only want to parse the <script> block
            if file_path.endswith(".vue"):
                script_match = self._extract_vue_script(content)
                if not script_match:
                    return []
                content = script_match

            tree = self.parser.parse(content)
            if not tree:
                return []

            # Use manual tree walk because Query API is version-dependent
            results = []
            _walk_import_paths(tree.root_node, results, content)
            return results

        except Exception:
            return []

    def _extract_vue_script(self, content):
        """Poor man's Vue script extractor for the benchmark."""
        content_str = content.decode("utf8", errors="ignore")
        import re
        match = re.search(r"<script.*?>(.*?)</script>", content_str, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).encode("utf8")
        return None

if __name__ == "__main__":
    # Test
    parser = TypeScriptParser()
    test_file = "target_repo/src/components/HeaderWidget.vue"
    if os.path.exists(test_file):
        print(f"Imports in {test_file}:")
        for imp in parser.extract_imports(test_file):
            print(f"  - {imp}")
