"""
HNSW Navigation Coalgebra

The key categorical insight: HNSW search is a coalgebra.

A coalgebra for functor F is a pair (X, α) where α: X → F(X).
For HNSW navigation:
- X is HNSWSearchState (position, candidates, visited, etc.)
- F is List (nondeterministic branching)
- α (structure_map) produces successor states

The search process is an unfolding (anamorphism):
- Start from initial state at entry point
- Repeatedly apply structure map
- Terminate when at layer 0 with no improvement

This coalgebraic view provides:
- Clean separation of state and transition logic
- Composability with other coalgebras
- Natural correspondence to corecursion
"""

from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import heapq

from vajra_bm25.categorical import Coalgebra
from vajra_bm25.vector.hnsw.graph import HNSWGraph
from vajra_bm25.vector.hnsw.state import HNSWSearchState


class HNSWNavigationCoalgebra(Coalgebra[HNSWSearchState, List[HNSWSearchState]]):
    """
    Coalgebra for HNSW graph navigation.

    Structure map: SearchState → List[SearchState]

    This models greedy search as coalgebraic unfolding:
    - Given current state (position, candidates, visited)
    - Produce next states by exploring neighbors
    - Continue until local minimum found

    The coalgebra naturally captures:
    - Hierarchical descent through layers
    - Beam search within each layer
    - Termination conditions
    """

    def __init__(
        self,
        graph: HNSWGraph,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        """
        Initialize navigation coalgebra.

        Args:
            graph: The HNSW graph to navigate
            distance_fn: Function (query, vectors) -> distances
        """
        self.graph = graph
        self.distance_fn = distance_fn

    def structure_map(self, state: HNSWSearchState) -> List[HNSWSearchState]:
        """
        One step of the coalgebraic unfolding.

        Returns:
        - Empty list: Terminal state (search complete)
        - Single state: Continue at same/lower level
        """
        if self.graph.vectors is None:
            return []

        query = np.array(state.query, dtype=np.float32)

        # At layers > 0: greedy descent to find entry point for lower layer
        if state.current_level > 0:
            return self._greedy_step(state, query)

        # At layer 0: beam search with ef candidates
        return self._beam_step(state, query)

    def _greedy_step(
        self, state: HNSWSearchState, query: np.ndarray
    ) -> List[HNSWSearchState]:
        """
        Greedy navigation at upper layers.

        Move to the closest unvisited neighbor, or descend if no improvement.
        """
        neighbors = self.graph.get_neighbors(state.current_node, state.current_level)
        unvisited = [n for n in neighbors if n not in state.visited]

        if not unvisited:
            # No unvisited neighbors - descend to next layer
            return [state.descend_to_level(state.current_level - 1)]

        # Compute distances to unvisited neighbors
        neighbor_vectors = self.graph.vectors[unvisited]
        distances = self.distance_fn(query, neighbor_vectors)

        # Find closest neighbor
        best_idx = np.argmin(distances)
        best_neighbor = unvisited[best_idx]
        best_distance = distances[best_idx]

        # Current distance
        current_distance = self.distance_fn(
            query, self.graph.vectors[state.current_node : state.current_node + 1]
        )[0]

        if best_distance >= current_distance:
            # No improvement - descend to next layer
            return [state.descend_to_level(state.current_level - 1)]

        # Move to better neighbor
        return [state.move_to(best_neighbor)]

    def _beam_step(
        self, state: HNSWSearchState, query: np.ndarray
    ) -> List[HNSWSearchState]:
        """
        Beam search at layer 0.

        Explore ef candidates, maintaining best results.
        Returns empty list when search is complete.
        """
        # Initialize candidates and results if empty
        if not state.candidates and not state.results:
            # Initialize with current node
            current_dist = self.distance_fn(
                query, self.graph.vectors[state.current_node : state.current_node + 1]
            )[0]

            candidates = [(current_dist, state.current_node)]
            results = [(-current_dist, state.current_node)]
            visited = frozenset([state.current_node])

            return [
                HNSWSearchState(
                    query=state.query,
                    current_node=state.current_node,
                    current_level=0,
                    ef=state.ef,
                    candidates=tuple(candidates),
                    results=tuple(results),
                    visited=visited,
                )
            ]

        # Convert to mutable heaps
        candidates = list(state.candidates)
        heapq.heapify(candidates)
        results = list(state.results)
        heapq.heapify(results)
        visited = set(state.visited)

        # Process one candidate
        if not candidates:
            return []  # Terminal - no more candidates

        dist, current = heapq.heappop(candidates)

        # Check termination: current is worse than worst result
        if len(results) >= state.ef and dist > -results[0][0]:
            return []  # Terminal - candidates worse than results

        # Explore neighbors
        neighbors = self.graph.get_neighbors(current, 0)
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if unvisited_neighbors:
            # Compute distances
            neighbor_vectors = self.graph.vectors[unvisited_neighbors]
            distances = self.distance_fn(query, neighbor_vectors)

            for neighbor, neighbor_dist in zip(unvisited_neighbors, distances):
                visited.add(neighbor)
                heapq.heappush(candidates, (neighbor_dist, neighbor))

                if len(results) < state.ef:
                    heapq.heappush(results, (-neighbor_dist, neighbor))
                elif neighbor_dist < -results[0][0]:
                    heapq.heapreplace(results, (-neighbor_dist, neighbor))

        # Return new state for next iteration
        return [
            HNSWSearchState(
                query=state.query,
                current_node=current,
                current_level=0,
                ef=state.ef,
                candidates=tuple(candidates),
                results=tuple(results),
                visited=frozenset(visited),
            )
        ]

    def unfold(self, initial_state: HNSWSearchState) -> List[Tuple[int, float]]:
        """
        Unfold the coalgebra to completion.

        Repeatedly applies structure_map until terminal state.

        Returns:
            List of (node_idx, distance) pairs, sorted by distance
        """
        state = initial_state

        # Navigate through layers (corecursive unfolding)
        while True:
            next_states = self.structure_map(state)
            if not next_states:
                break  # Terminal state reached
            state = next_states[0]

        # Extract results from final state
        if not state.results:
            return []

        # Results are stored as (-distance, idx), convert to (idx, distance)
        final = [(idx, -dist) for dist, idx in state.results]
        final.sort(key=lambda x: x[1])  # Sort by distance
        return final

    def search(
        self, query: np.ndarray, k: int, ef: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Convenience method to search from entry point.

        Args:
            query: Query vector
            k: Number of results
            ef: Expansion factor (defaults to max(k, 50))

        Returns:
            List of (node_idx, distance) pairs
        """
        if self.graph.entry_point < 0:
            return []

        if ef is None:
            ef = max(k, 50)

        initial_state = HNSWSearchState(
            query=tuple(query.astype(np.float32).tolist()),
            current_node=self.graph.entry_point,
            current_level=self.graph.max_level,
            ef=ef,
        )

        results = self.unfold(initial_state)
        return results[:k]
