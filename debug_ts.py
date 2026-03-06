import tree_sitter_typescript as tstypescript
print("Attributes in tree_sitter_typescript:")
print(dir(tstypescript))
try:
    print(f"language_typescript: {tstypescript.language_typescript()}")
except AttributeError:
    print("No language_typescript()")
try:
    print(f"language: {tstypescript.language()}")
except AttributeError:
    print("No language()")

import tree_sitter
print(f"Tree-sitter version: {tree_sitter.__version__ if hasattr(tree_sitter, '__version__') else 'unknown'}")
print(dir(tree_sitter.Language))
