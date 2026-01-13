"""
HNSW Search State

Immutable state for the HNSW navigation coalgebra.
The coalgebra structure map transforms this state to produce successors.

State components:
- query: The query vector we're searching for
- current_node: Current position in the graph
- current_level: Current layer in the hierarchy
- candidates: Priority queue of candidates to explore
- results: Best results found so far
- visited: Set of already-visited nodes (at current level)
- ef: Expansion factor (beam width)
"""

from dataclasses import dataclass
from typing import Tuple, FrozenSet, List
import heapq


@dataclass(frozen=True)
class HNSWSearchState:
    """
    Immutable state for HNSW navigation coalgebra.

    The coalgebra structure map: State → F(State) where F = List
    unfolds the search by producing next candidate states.

    Immutability is important for:
    - Referential transparency
    - Easy backtracking/branching
    - Functional composition
    """

    query: Tuple[float, ...]  # Query vector (immutable tuple)
    current_node: int  # Current position in graph
    current_level: int  # Current layer
    ef: int  # Expansion factor

    # Candidate queue: tuple of (distance, node_idx) pairs
    # Using tuple for immutability; represents a min-heap
    candidates: Tuple[Tuple[float, int], ...] = ()

    # Result queue: tuple of (-distance, node_idx) pairs (max-heap via negation)
    results: Tuple[Tuple[float, int], ...] = ()

    # Visited nodes at current level
    visited: FrozenSet[int] = frozenset()

    def __hash__(self):
        return hash((self.current_node, self.current_level, len(self.visited)))

    def with_candidates(
        self, candidates: List[Tuple[float, int]]
    ) -> "HNSWSearchState":
        """Return new state with updated candidates"""
        return HNSWSearchState(
            query=self.query,
            current_node=self.current_node,
            current_level=self.current_level,
            ef=self.ef,
            candidates=tuple(candidates),
            results=self.results,
            visited=self.visited,
        )

    def with_results(self, results: List[Tuple[float, int]]) -> "HNSWSearchState":
        """Return new state with updated results"""
        return HNSWSearchState(
            query=self.query,
            current_node=self.current_node,
            current_level=self.current_level,
            ef=self.ef,
            candidates=self.candidates,
            results=tuple(results),
            visited=self.visited,
        )

    def with_visited(self, visited: FrozenSet[int]) -> "HNSWSearchState":
        """Return new state with updated visited set"""
        return HNSWSearchState(
            query=self.query,
            current_node=self.current_node,
            current_level=self.current_level,
            ef=self.ef,
            candidates=self.candidates,
            results=self.results,
            visited=visited,
        )

    def descend_to_level(self, level: int) -> "HNSWSearchState":
        """Return new state at a lower level with reset visited set"""
        return HNSWSearchState(
            query=self.query,
            current_node=self.current_node,
            current_level=level,
            ef=self.ef,
            candidates=(),  # Reset candidates for new level
            results=self.results,
            visited=frozenset(),  # Reset visited for new level
        )

    def move_to(self, node: int) -> "HNSWSearchState":
        """Return new state with updated current node"""
        return HNSWSearchState(
            query=self.query,
            current_node=node,
            current_level=self.current_level,
            ef=self.ef,
            candidates=self.candidates,
            results=self.results,
            visited=self.visited | {self.current_node},
        )

    @property
    def is_at_layer_zero(self) -> bool:
        return self.current_level == 0

    def worst_result_distance(self) -> float:
        """Get distance of worst result (for pruning)"""
        if not self.results:
            return float("inf")
        # Results are stored as (-distance, idx), so negate to get distance
        return -self.results[0][0]

    def __repr__(self) -> str:
        return (
            f"HNSWSearchState(level={self.current_level}, "
            f"node={self.current_node}, "
            f"candidates={len(self.candidates)}, "
            f"results={len(self.results)}, "
            f"visited={len(self.visited)})"
        )
