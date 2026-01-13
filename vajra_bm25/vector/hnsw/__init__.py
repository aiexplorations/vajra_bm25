"""
Native HNSW (Hierarchical Navigable Small World) Implementation

A coalgebraic approach to approximate nearest neighbor search.
HNSW navigation is modeled as an unfolding process where the
structure map produces candidate successors at each step.

Key insight: HNSW search is a coalgebra X → F(X) where:
- X is the search state (current position, candidates, visited set)
- F is List (branching/nondeterminism)
- The structure map unfolds to successor states

Components:
- HNSWGraph: Multi-layer graph data structure
- HNSWSearchState: Immutable state for coalgebra
- HNSWNavigationCoalgebra: The coalgebra that models search
- NativeHNSWIndex: Complete index implementation

This is a pure Python+Numba implementation with no external dependencies.
"""

from vajra_bm25.vector.hnsw.graph import HNSWGraph
from vajra_bm25.vector.hnsw.state import HNSWSearchState
from vajra_bm25.vector.hnsw.coalgebra import HNSWNavigationCoalgebra
from vajra_bm25.vector.hnsw.index import NativeHNSWIndex

__all__ = [
    "HNSWGraph",
    "HNSWSearchState",
    "HNSWNavigationCoalgebra",
    "NativeHNSWIndex",
]
