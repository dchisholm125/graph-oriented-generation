"""GOG-Lite: Public reference implementation for Graph-Oriented Generation.

This module provides a simplified graph-native context layer that
demonstrates the GOG thesis without production-grade heuristics.

The goal is reproducibility of the core claim: graph-routed context
can compete with or exceed flat retrieved context for coding tasks.
"""

from .import_graph import build_import_graph
from .graph_search import isolate_context, distance_from_seeds

__all__ = [
    "build_import_graph",
    "isolate_context",
    "distance_from_seeds",
]