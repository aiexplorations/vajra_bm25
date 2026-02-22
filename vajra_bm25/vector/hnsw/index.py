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
from vajra_bm25.vector.scorer import get_distance_function, CosineSimilarity


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

        # Get distance function.
        # For cosine, both stored vectors and query are pre-normalised (add()
        # normalises at insertion; search() normalises the query).  Using
        # score_batch_normalized (pure dot-product) avoids recomputing norms
        # on every call — a 7–17× per-call speedup over score_batch.
        if metric == "cosine":
            _scorer = CosineSimilarity()
            self._distance_single = lambda a, b: float(1.0 - float(np.dot(a, b)))
            self._distance_batch = lambda q, vs: 1.0 - _scorer.score_batch_normalized(q, vs)
        else:
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

        # Normalize for cosine in one vectorised pass before any insertion
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            vectors = vectors / norms

        n_new = len(ids)
        existing = len(self.graph.ids)

        # Pre-allocate the full vector block for all incoming vectors.
        # This avoids the O(N²) copy cost of growing the array one row at a
        # time inside _insert_node.  A single allocation + one bulk fill
        # replaces N incremental np.vstack calls.
        if self.graph.vectors is None:
            self.graph.vectors = np.empty((n_new, self._dimension), dtype=np.float32)
        else:
            self.graph.vectors = np.concatenate(
                [self.graph.vectors, np.empty((n_new, self._dimension), dtype=np.float32)]
            )
        self.graph.vectors[existing : existing + n_new] = vectors

        # Run HNSW insertion for each node; storage is already in place
        for i, id_ in enumerate(ids):
            if id_ in self.graph.id_to_idx:
                raise ValueError(f"Duplicate ID: {id_}")

            self._insert_node(existing + i, id_)

            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]

        # Initialize/update coalgebra
        self.coalgebra = HNSWNavigationCoalgebra(self.graph, self._distance_batch)

    def _insert_node(self, idx: int, vector_id: str) -> None:
        """
        Insert a node into the HNSW graph.

        The vector at self.graph.vectors[idx] must already be written by
        add() before this method is called.

        Algorithm:
        1. Sample random level for new node
        2. Navigate from entry point to insertion level
        3. At each layer from insertion_level to 0:
           - Find ef_construction nearest neighbors
           - Add bidirectional edges to M best neighbors
           - Prune if needed
        """
        # Register identity — storage already written by add()
        self.graph.ids.append(vector_id)
        self.graph.id_to_idx[vector_id] = idx

        # Read vector from pre-allocated storage (no copy needed)
        vector = self.graph.vectors[idx]

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
        vectors = self.graph.vectors  # cache: constant throughout this call

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

            # Prune overfull neighbor lists inline.
            # Inlining eliminates ~329k Python function-call overheads and
            # avoids a second get_neighbors call inside _prune_connections.
            layer = self.graph.layers[l]
            for neighbor_idx, _ in selected:
                nbrs = layer[neighbor_idx]
                if len(nbrs) > M:
                    dists = self._distance_batch(vectors[neighbor_idx], vectors[nbrs])
                    keep = np.argsort(dists)[:M]
                    layer[neighbor_idx] = [nbrs[i] for i in keep]

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
        # Cache as locals — attribute lookup in a tight loop is measurably slower.
        # Direct dict access eliminates the get_neighbors() call overhead
        # (~2M calls at N=10k, each adding a Python frame + bounds check).
        vectors = self.graph.vectors
        dist_fn = self._distance_batch
        layer_dict = (
            self.graph.layers[level] if level < len(self.graph.layers) else {}
        )

        entry_dist = float(dist_fn(query, vectors[entry : entry + 1])[0])
        candidates = [(entry_dist, entry)]   # min-heap by distance
        results = [(-entry_dist, entry)]     # max-heap (negated), capped at ef
        visited = {entry}

        # Track result count as an integer to avoid repeated len() calls in the
        # hot inner loop.  Once the heap fills to ef, results_full stays True
        # for the rest of the search — a branch-prediction-friendly fast path.
        n_results = 1
        results_full = (n_results >= ef)
        worst = entry_dist   # tracks -results[0][0] without a heap read each iter

        while candidates:
            dist, current = heapq.heappop(candidates)

            if results_full and dist > worst:
                break

            unvisited = [n for n in layer_dict.get(current, []) if n not in visited]
            if not unvisited:
                continue

            visited.update(unvisited)
            neighbor_dists = dist_fn(query, vectors[unvisited])

            for neighbor, neighbor_dist in zip(unvisited, neighbor_dists):
                nd = float(neighbor_dist)
                heapq.heappush(candidates, (nd, neighbor))
                if results_full:
                    if nd < worst:
                        heapq.heapreplace(results, (-nd, neighbor))
                        worst = -results[0][0]
                else:
                    heapq.heappush(results, (-nd, neighbor))
                    n_results += 1
                    worst = -results[0][0]
                    if n_results >= ef:
                        results_full = True

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

    def _beam_search_fast(
        self, query: np.ndarray, entry: int, ef: int
    ) -> List[Tuple[int, float]]:
        """
        Imperative beam search at layer 0.

        Uses mutable Python state — no frozenset copies, no HNSWSearchState
        allocations per step.  Same algorithm as the coalgebra unfold but
        avoids the O(visited) frozenset copy that otherwise dominates query
        latency at ef=50 (800 steps × O(step) = O(step²) total copies).

        Returns list of (idx, distance) sorted ascending by distance.
        """
        vectors = self.graph.vectors
        dist_fn = self._distance_batch
        # Direct layer-0 dict access — same optimisation as _search_layer.
        layer_0 = self.graph.layers[0] if self.graph.layers else {}

        d0 = float(dist_fn(query, vectors[entry : entry + 1])[0])
        candidates = [(d0, entry)]    # min-heap by distance
        results = [(-d0, entry)]      # max-heap (negated), capped at ef
        visited = {entry}

        n_results = 1
        results_full = (n_results >= ef)
        worst = d0

        while candidates:
            dist, current = heapq.heappop(candidates)
            if results_full and dist > worst:
                break

            unvisited = [n for n in layer_0.get(current, []) if n not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)
            dists = dist_fn(query, vectors[unvisited])

            for n, nd_raw in zip(unvisited, dists):
                nd = float(nd_raw)
                heapq.heappush(candidates, (nd, n))
                if results_full:
                    if nd < worst:
                        heapq.heapreplace(results, (-nd, n))
                        worst = -results[0][0]
                else:
                    heapq.heappush(results, (-nd, n))
                    n_results += 1
                    worst = -results[0][0]
                    if n_results >= ef:
                        results_full = True

        return [(idx, -d) for d, idx in sorted(results, reverse=True)]

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """
        Search the index.

        Descends upper layers with greedy 1-best search, then runs a full beam
        search at layer 0 via the imperative fast path.  The
        HNSWNavigationCoalgebra is preserved as the documented public abstraction
        (accessible via self.coalgebra) but is not used in this hot path.
        """
        if self.graph.entry_point < 0:
            return []

        query = query.astype(np.float32)

        # Normalize for cosine
        if self.metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm

        ef = max(k, self.ef_search)

        # Greedy descent through upper layers to reach the layer-0 entry point
        current = self.graph.entry_point
        for l in range(self.graph.max_level, 0, -1):
            current = self._greedy_search_layer(query, current, l)

        # Full beam search at layer 0
        raw = self._beam_search_fast(query, current, ef)

        ids = self.graph.ids
        vectors = self.graph.vectors
        return [
            VectorSearchResult(
                id=ids[idx],
                score=self._distance_to_score(dist),
                vector=vectors[idx].copy(),
                metadata=self._metadata.get(ids[idx], {}),
            )
            for idx, dist in raw[:k]
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
