import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_typescript as tstypescript

# Setup language
try:
    TS_LANGUAGE = Language(tstypescript.language_typescript())
except AttributeError:
    TS_LANGUAGE = Language(tstypescript.language())

query_text = """
(import_statement
    source: (string (string_fragment) @import_path))
"""
query = TS_LANGUAGE.query(query_text)

print(f"Query object: {query}")
print(f"Attributes of Query: {dir(query)}")

parser = Parser(TS_LANGUAGE)
tree = parser.parse(b"import { x } from 'y';")
print(f"Capturing from root node: {tree.root_node}")

try:
    captures = query.captures(tree.root_node)
    print("query.captures(node) worked!")
except AttributeError as e:
    print(f"query.captures(node) failed: {e}")

# Check if it's on the Parser or something else?
# In 0.22+ it might be query.captures(node)
# But the error says it has NO attribute 'captures'.
