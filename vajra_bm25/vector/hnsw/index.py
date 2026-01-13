"""
Native HNSW Vector Index

Complete HNSW implementation combining:
- HNSWGraph for storage
- Insertion algorithm for index building
- HNSWNavigationCoalgebra for search

This is a pure Python+Numba implementation with no external dependencies.
Trade-off vs hnswlib: Slower but more transparent and extensible.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pickle
import heapq

from vajra_bm25.vector.index import VectorIndex, VectorSearchResult
from vajra_bm25.vector.hnsw.graph import HNSWGraph
from vajra_bm25.vector.hnsw.state import HNSWSearchState
from vajra_bm25.vector.hnsw.coalgebra import HNSWNavigationCoalgebra
from vajra_bm25.vector.scorer import get_distance_function


class NativeHNSWIndex(VectorIndex):
    """
    Native Python HNSW index with coalgebraic search.

    Features:
    - Pure Python + optional Numba acceleration
    - Coalgebraic search (HNSWNavigationCoalgebra)
    - Full categorical abstraction integration
    - No external dependencies

    Usage:
        index = NativeHNSWIndex(dimension=384, metric="cosine")
        index.add(["doc1", "doc2"], vectors)
        results = index.search(query, k=10)

    Parameters:
        dimension: Vector dimensionality
        metric: Distance metric ("cosine", "l2", "ip")
        M: Max connections per node at layers > 0
        M0: Max connections at layer 0
        ef_construction: Beam width during construction
        ef_search: Default beam width during search
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        M: int = 16,
        M0: Optional[int] = None,
        ef_construction: int = 200,
        ef_search: int = 50,
    ):
        self._dimension = dimension
        self.metric = metric
        self.ef_search = ef_search

        # Get distance function
        self._distance_single, self._distance_batch = get_distance_function(
            metric, use_numba=True
        )

        # Initialize graph
        self.graph = HNSWGraph(
            dimension=dimension,
            M=M,
            M0=M0 if M0 is not None else 2 * M,
            ef_construction=ef_construction,
        )

        # Coalgebra will be initialized after first add
        self.coalgebra: Optional[HNSWNavigationCoalgebra] = None

        # Metadata storage
        self._metadata: Dict[str, Dict[str, Any]] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return self.graph.num_nodes()

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add vectors to the index"""
        vectors = vectors.astype(np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vectors have dimension {vectors.shape[1]}, "
                f"expected {self._dimension}"
            )

        # Normalize for cosine
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vectors = vectors / norms

        # Add each vector
        for i, (id_, vector) in enumerate(zip(ids, vectors)):
            if id_ in self.graph.id_to_idx:
                raise ValueError(f"Duplicate ID: {id_}")

            self._insert_vector(id_, vector)

            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]

        # Initialize/update coalgebra
        self.coalgebra = HNSWNavigationCoalgebra(self.graph, self._distance_batch)

    def _insert_vector(self, vector_id: str, vector: np.ndarray) -> None:
        """
        Insert a single vector into the HNSW graph.

        Algorithm:
        1. Sample random level for new node
        2. Navigate from entry point to insertion level
        3. At each layer from insertion_level to 0:
           - Find ef_construction nearest neighbors
           - Add bidirectional edges to M best neighbors
           - Prune if needed
        """
        # Add vector to storage
        idx = len(self.graph.ids)
        if self.graph.vectors is None:
            self.graph.vectors = vector.reshape(1, -1)
        else:
            self.graph.vectors = np.vstack([self.graph.vectors, vector])

        self.graph.ids.append(vector_id)
        self.graph.id_to_idx[vector_id] = idx

        # Sample level for new node
        level = self.graph.get_random_level()

        # Handle first node
        if self.graph.entry_point < 0:
            self.graph.entry_point = idx
            self.graph.max_level = level
            # Initialize layers for this node
            for l in range(level + 1):
                self.graph.set_neighbors(idx, l, [])
            return

        # Navigate from entry point to insertion level
        current = self.graph.entry_point

        # Descend through upper layers (greedy search)
        for l in range(self.graph.max_level, level, -1):
            current = self._greedy_search_layer(vector, current, l)

        # Insert at each layer from level down to 0
        for l in range(min(level, self.graph.max_level), -1, -1):
            # Find neighbors using beam search
            neighbors = self._search_layer(
                vector, current, l, self.graph.ef_construction
            )

            # Select M best neighbors
            M = self.graph.get_max_neighbors(l)
            selected = self._select_neighbors(vector, neighbors, M)

            # Initialize neighbor list for new node
            self.graph.set_neighbors(idx, l, [])

            # Add edges
            for neighbor_idx, _ in selected:
                self.graph.add_edge(idx, neighbor_idx, l)

            # Prune neighbor connections if needed
            for neighbor_idx, _ in selected:
                self._prune_connections(neighbor_idx, l, M)

            # Update entry point for next layer
            if neighbors:
                current = neighbors[0][0]

        # Update entry point if new node has higher level
        if level > self.graph.max_level:
            self.graph.entry_point = idx
            self.graph.max_level = level

    def _greedy_search_layer(
        self, query: np.ndarray, entry: int, level: int
    ) -> int:
        """Greedy search to find closest node at layer"""
        current = entry
        current_dist = self._distance_batch(
            query, self.graph.vectors[current : current + 1]
        )[0]

        while True:
            neighbors = self.graph.get_neighbors(current, level)
            if not neighbors:
                break

            neighbor_vectors = self.graph.vectors[neighbors]
            distances = self._distance_batch(query, neighbor_vectors)

            best_idx = np.argmin(distances)
            if distances[best_idx] >= current_dist:
                break

            current = neighbors[best_idx]
            current_dist = distances[best_idx]

        return current

    def _search_layer(
        self, query: np.ndarray, entry: int, level: int, ef: int
    ) -> List[Tuple[int, float]]:
        """Beam search at layer, returning ef nearest neighbors"""
        candidates = []  # Min-heap: (distance, node)
        results = []  # Max-heap: (-distance, node)
        visited = {entry}

        entry_dist = self._distance_batch(
            query, self.graph.vectors[entry : entry + 1]
        )[0]
        heapq.heappush(candidates, (entry_dist, entry))
        heapq.heappush(results, (-entry_dist, entry))

        while candidates:
            dist, current = heapq.heappop(candidates)

            if len(results) >= ef and dist > -results[0][0]:
                break

            for neighbor in self.graph.get_neighbors(current, level):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                neighbor_dist = self._distance_batch(
                    query, self.graph.vectors[neighbor : neighbor + 1]
                )[0]

                heapq.heappush(candidates, (neighbor_dist, neighbor))

                if len(results) < ef:
                    heapq.heappush(results, (-neighbor_dist, neighbor))
                elif neighbor_dist < -results[0][0]:
                    heapq.heapreplace(results, (-neighbor_dist, neighbor))

        # Return sorted by distance (ascending)
        return [(idx, -dist) for dist, idx in sorted(results, reverse=True)]

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[Tuple[int, float]], M: int
    ) -> List[Tuple[int, float]]:
        """Select M neighbors (simple: take M closest)"""
        return candidates[:M]

    def _prune_connections(self, node_idx: int, level: int, M: int) -> None:
        """Prune connections if node has more than M neighbors"""
        neighbors = self.graph.get_neighbors(node_idx, level)
        if len(neighbors) <= M:
            return

        # Keep M closest neighbors
        node_vector = self.graph.vectors[node_idx]
        neighbor_vectors = self.graph.vectors[neighbors]
        distances = self._distance_batch(node_vector, neighbor_vectors)

        sorted_idx = np.argsort(distances)[:M]
        self.graph.set_neighbors(node_idx, level, [neighbors[i] for i in sorted_idx])

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """
        Search using coalgebraic unfolding.

        Args:
            query: Query vector
            k: Number of results

        Returns:
            List of VectorSearchResult
        """
        if self.graph.entry_point < 0 or self.coalgebra is None:
            return []

        query = query.astype(np.float32)

        # Normalize for cosine
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm

        # Search using coalgebra
        ef = max(k, self.ef_search)
        results = self.coalgebra.search(query, k, ef=ef)

        # Convert to VectorSearchResult
        return [
            VectorSearchResult(
                id=self.graph.ids[idx],
                score=self._distance_to_score(dist),
                vector=self.graph.vectors[idx].copy(),
                metadata=self._metadata.get(self.graph.ids[idx], {}),
            )
            for idx, dist in results
        ]

    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score"""
        if self.metric == "cosine":
            return 1.0 - distance  # Cosine distance to similarity
        elif self.metric == "ip":
            return -distance  # Negative inner product back to ip
        else:
            return -distance  # L2: negate so higher is better

    def search_batch(
        self, queries: np.ndarray, k: int
    ) -> List[List[VectorSearchResult]]:
        """Batch search"""
        return [self.search(q, k) for q in queries]

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get vector by ID"""
        idx = self.graph.id_to_idx.get(id)
        if idx is not None and self.graph.vectors is not None:
            return self.graph.vectors[idx].copy()
        return None

    def save(self, path: str) -> None:
        """Persist index to disk"""
        data = {
            "dimension": self._dimension,
            "metric": self.metric,
            "ef_search": self.ef_search,
            "graph": {
                "dimension": self.graph.dimension,
                "M": self.graph.M,
                "M0": self.graph.M0,
                "ef_construction": self.graph.ef_construction,
                "ml": self.graph.ml,
                "vectors": self.graph.vectors,
                "ids": self.graph.ids,
                "layers": self.graph.layers,
                "entry_point": self.graph.entry_point,
                "max_level": self.graph.max_level,
            },
            "metadata": self._metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "NativeHNSWIndex":
        """Load index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        graph_data = data["graph"]

        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            M=graph_data["M"],
            M0=graph_data["M0"],
            ef_construction=graph_data["ef_construction"],
            ef_search=data["ef_search"],
        )

        # Restore graph
        index.graph.ml = graph_data["ml"]
        index.graph.vectors = graph_data["vectors"]
        index.graph.ids = graph_data["ids"]
        index.graph.id_to_idx = {id_: i for i, id_ in enumerate(graph_data["ids"])}
        index.graph.layers = graph_data["layers"]
        index.graph.entry_point = graph_data["entry_point"]
        index.graph.max_level = graph_data["max_level"]

        # Restore metadata
        index._metadata = data.get("metadata", {})

        # Initialize coalgebra
        if index.graph.entry_point >= 0:
            index.coalgebra = HNSWNavigationCoalgebra(
                index.graph, index._distance_batch
            )

        return index

    def stats(self) -> Dict:
        """Get index statistics"""
        return {
            "type": "NativeHNSWIndex",
            "dimension": self._dimension,
            "metric": self.metric,
            "ef_search": self.ef_search,
            **self.graph.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"NativeHNSWIndex(dimension={self._dimension}, "
            f"metric={self.metric!r}, size={self.size})"
        )
