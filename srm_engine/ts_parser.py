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
        TS_LANGUAGE = Language(tstypescript.typescript) # type: ignore
except Exception as e:
    print(f"Error loading tree-sitter-typescript: {e}")
    TS_LANGUAGE = None

class TypeScriptParser:
    """Uses tree-sitter to perform precise AST analysis on TS/Vue files."""
    
    def __init__(self):
        self.parser = Parser(TS_LANGUAGE) if TS_LANGUAGE else None

    def extract_imports(self, file_path):
        """Extracts import paths from a TypeScript file using AST parsing."""
        if not self.parser or not TS_LANGUAGE:
            return []

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            
            # If it's a Vue file, we only want to parse the <script> block
            if file_path.endswith('.vue'):
                script_match = self._extract_vue_script(content)
                if not script_match:
                    return []
                content = script_match

            tree = self.parser.parse(content)
            if not tree:
                return []
                
            query = TS_LANGUAGE.query("""
                (import_statement
                    source: (string (string_fragment) @import_path))
            """)
            
            # Version-resilient capture execution
            captures = []
            if hasattr(query, 'captures'):
                captures = query.captures(tree.root_node)
            elif hasattr(query, 'matches'):
                # Some versions use matches to return captures too
                captures = query.matches(tree.root_node)
            elif callable(query):
                captures = query(tree.root_node)

            imports = []
            for item in captures:
                # captures can return (node, tag) or a match dict depending on version
                if isinstance(item, tuple) and len(item) == 2:
                    node, tag = item
                    if tag == "import_path":
                        imp_content = content[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                        imports.append(imp_content.strip("'\""))
                elif isinstance(item, dict):
                    # Handle matches format if captures is missing
                    for tag, nodes in item.items():
                        if tag == "import_path":
                            for node in nodes:
                                imp_content = content[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
                                imports.append(imp_content.strip("'\""))
                    
            return imports
        except Exception as e:
            # Fallback handled in ast_parser.py
            raise e

    def _extract_vue_script(self, content):
        """Poor man's Vue script extractor for the benchmark."""
        content_str = content.decode('utf8', errors='ignore')
        import re
        match = re.search(r'<script.*?>\s*(.*?)\s*</script>', content_str, re.DOTALL)
        if match:
            return match.group(1).encode('utf8')
        return None

if __name__ == "__main__":
    # Test
    parser = TypeScriptParser()
    test_file = "target_repo/src/components/HeaderWidget.vue"
    if os.path.exists(test_file):
        print(f"Imports in {test_file}:")
        for imp in parser.extract_imports(test_file):
            print(f"  - {imp}")
