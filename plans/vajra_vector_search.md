# Vajra Vector Search Implementation Plan

## Objective

Build a high-performance vector search engine within Vajra that:
- Scales to 100K-500K+ vectors
- Maintains categorical abstractions (Morphism, Coalgebra, Functor)
- **Native HNSW implementation** in Python + Numba (no external hnswlib dependency)
- Achieves competitive performance with production systems
- Models graph navigation as coalgebraic unfolding

## Architecture Overview

```
vajra_bm25/
├── categorical/              # Existing - reuse and extend
│   ├── category.py          # Morphism composition
│   ├── functor.py           # ListFunctor, MaybeFunctor
│   └── coalgebra.py         # SearchCoalgebra, HNSWCoalgebra (NEW)
│
├── vector/                   # NEW - vector search module
│   ├── __init__.py
│   ├── embeddings.py        # Embedding morphisms
│   ├── index.py             # Vector index abstractions
│   ├── hnsw/                 # Native HNSW implementation
│   │   ├── __init__.py
│   │   ├── graph.py         # HNSWGraph data structure
│   │   ├── coalgebra.py     # HNSWNavigationCoalgebra
│   │   ├── insert.py        # Insertion algorithm
│   │   ├── search.py        # Search algorithm
│   │   └── numba_kernels.py # JIT-compiled distance functions
│   ├── index_flat.py        # Brute-force (small scale, exact)
│   ├── scorer.py            # Similarity morphisms (cosine, L2, dot)
│   ├── search.py            # VajraVectorSearch (coalgebra-based)
│   └── optimized.py         # Batching, caching, quantization
│
├── hybrid/                   # NEW - BM25 + Vector fusion
│   ├── __init__.py
│   └── fusion.py            # Score fusion strategies
│
└── [existing BM25 modules]
```

## Implementation Phases

### Phase 1: Core Abstractions (Foundation)

**1.1 Embedding Morphism**

```python
# vector/embeddings.py

class EmbeddingMorphism(Morphism[T, np.ndarray]):
    """Morphism: T → ℝ^d (embedding space)"""

    @abstractmethod
    def embed(self, item: T) -> np.ndarray:
        """Single item embedding"""
        pass

    @abstractmethod
    def embed_batch(self, items: List[T]) -> np.ndarray:
        """Batch embedding for efficiency"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality"""
        pass


class TextEmbeddingMorphism(EmbeddingMorphism[str, np.ndarray]):
    """Text → Embedding using sentence-transformers or similar"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)


class PrecomputedEmbeddingMorphism(EmbeddingMorphism[str, np.ndarray]):
    """Lookup morphism for pre-computed embeddings"""

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        self._embeddings = embeddings
        self._dimension = next(iter(embeddings.values())).shape[0]

    def embed(self, item_id: str) -> np.ndarray:
        return self._embeddings[item_id]
```

**1.2 Similarity Morphism**

```python
# vector/scorer.py

class SimilarityMorphism(Morphism[Tuple[np.ndarray, np.ndarray], float]):
    """Morphism: (Embedding, Embedding) → ℝ (similarity score)"""
    pass


class CosineSimilarity(SimilarityMorphism):
    """Cosine similarity: dot(a, b) / (||a|| * ||b||)"""

    def apply(self, pair: Tuple[np.ndarray, np.ndarray]) -> float:
        a, b = pair
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def batch_scores(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch scoring (assumes normalized vectors)"""
        return vectors @ query
```

**1.3 Vector Index Interface**

```python
# vector/index.py

@dataclass
class VectorSearchResult:
    """Result from vector index search"""
    id: str
    score: float
    vector: Optional[np.ndarray] = None


class VectorIndex(ABC):
    """Abstract vector index - morphism from query to candidates"""

    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add vectors to index"""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """Find k nearest neighbors"""
        pass

    @abstractmethod
    def search_batch(self, queries: np.ndarray, k: int) -> List[List[VectorSearchResult]]:
        """Batch search for efficiency"""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors in index"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist index to disk"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load index from disk"""
        pass
```

### Phase 2: Native HNSW Implementation (Coalgebraic)

This is the core innovation: implementing HNSW from scratch with categorical abstractions.

**2.1 HNSW Graph Data Structure**

```python
# vector/hnsw/graph.py

@dataclass
class HNSWGraph:
    """
    Hierarchical Navigable Small World graph.

    Multi-layer structure where:
    - Layer 0: All nodes, dense local connections
    - Layer L: Sparse subset, long-range connections
    - Entry point at highest layer
    """
    dimension: int
    M: int = 16                    # Max connections per node per layer
    M0: int = 32                   # Max connections at layer 0
    ef_construction: int = 200     # Neighbors explored during construction
    ml: float = 1.0 / np.log(16)   # Level multiplier

    # Node data
    vectors: np.ndarray = None     # (n, dim) array of all vectors
    ids: List[str] = field(default_factory=list)
    id_to_idx: Dict[str, int] = field(default_factory=dict)

    # Graph structure: layers[level][node_idx] = List[neighbor_idx]
    layers: List[Dict[int, List[int]]] = field(default_factory=list)

    # Entry point (node index at highest layer)
    entry_point: int = -1
    max_level: int = -1

    def get_random_level(self) -> int:
        """Sample level with exponential decay: P(level=l) ∝ exp(-l/ml)"""
        return int(-np.log(np.random.random()) * self.ml)

    def get_neighbors(self, node_idx: int, level: int) -> List[int]:
        """Get neighbors of node at given level"""
        if level >= len(self.layers):
            return []
        return self.layers[level].get(node_idx, [])

    def add_edge(self, node_a: int, node_b: int, level: int) -> None:
        """Add bidirectional edge between nodes at level"""
        while len(self.layers) <= level:
            self.layers.append({})

        if node_a not in self.layers[level]:
            self.layers[level][node_a] = []
        if node_b not in self.layers[level]:
            self.layers[level][node_b] = []

        # Add edges (will be pruned later if needed)
        if node_b not in self.layers[level][node_a]:
            self.layers[level][node_a].append(node_b)
        if node_a not in self.layers[level][node_b]:
            self.layers[level][node_b].append(node_a)
```

**2.2 HNSW Navigation Coalgebra**

The key insight: HNSW search is a coalgebraic unfolding where each state produces candidate successors.

```python
# vector/hnsw/coalgebra.py

@dataclass(frozen=True)
class HNSWSearchState:
    """
    Immutable state for HNSW navigation coalgebra.

    The coalgebra structure map: State → F(State) where F = List
    unfolds the search by producing next candidate states.
    """
    query: Tuple[float, ...]       # Query vector (immutable)
    current_node: int               # Current position in graph
    current_level: int              # Current layer
    candidates: Tuple[int, ...]     # Priority queue of candidates (node_idx)
    visited: frozenset              # Already visited nodes
    ef: int                         # Expansion factor

    def __hash__(self):
        return hash((self.current_node, self.current_level, len(self.visited)))


class HNSWNavigationCoalgebra(Coalgebra[HNSWSearchState, List[HNSWSearchState]]):
    """
    Coalgebra for HNSW graph navigation.

    Structure map: SearchState → List[SearchState]

    This models the greedy search as an unfolding:
    - Given current state (position, candidates, visited)
    - Produce next states by exploring neighbors
    - Continue until local minimum found
    """

    def __init__(self, graph: HNSWGraph, distance_fn: Callable):
        self.graph = graph
        self.distance_fn = distance_fn  # (query, vector) → distance

    def structure_map(self, state: HNSWSearchState) -> List[HNSWSearchState]:
        """
        One step of greedy navigation.

        Returns:
        - Empty list: local minimum reached (terminal state)
        - Single state: continue at same level
        - State with lower level: descend to next layer
        """
        query = np.array(state.query, dtype=np.float32)

        # Get unvisited neighbors of current node
        neighbors = self.graph.get_neighbors(state.current_node, state.current_level)
        unvisited = [n for n in neighbors if n not in state.visited]

        if not unvisited:
            # No unvisited neighbors - check if we should descend
            if state.current_level > 0:
                # Descend to next layer
                return [HNSWSearchState(
                    query=state.query,
                    current_node=state.current_node,
                    current_level=state.current_level - 1,
                    candidates=state.candidates,
                    visited=frozenset(),  # Reset visited for new layer
                    ef=state.ef
                )]
            else:
                # At layer 0 with no unvisited neighbors - terminal
                return []

        # Compute distances to unvisited neighbors
        neighbor_vectors = self.graph.vectors[unvisited]
        distances = self.distance_fn(query, neighbor_vectors)

        # Find closest unvisited neighbor
        best_idx = np.argmin(distances)
        best_neighbor = unvisited[best_idx]
        best_distance = distances[best_idx]

        # Check if we've improved
        current_distance = self.distance_fn(
            query,
            self.graph.vectors[state.current_node:state.current_node+1]
        )[0]

        if best_distance >= current_distance:
            # No improvement - local minimum at this layer
            if state.current_level > 0:
                return [HNSWSearchState(
                    query=state.query,
                    current_node=state.current_node,
                    current_level=state.current_level - 1,
                    candidates=state.candidates,
                    visited=frozenset(),
                    ef=state.ef
                )]
            else:
                return []  # Terminal

        # Move to better neighbor
        new_visited = state.visited | {state.current_node}

        return [HNSWSearchState(
            query=state.query,
            current_node=best_neighbor,
            current_level=state.current_level,
            candidates=state.candidates,
            visited=new_visited,
            ef=state.ef
        )]

    def unfold(self, initial_state: HNSWSearchState) -> List[int]:
        """
        Unfold the coalgebra to completion.
        Returns: List of k nearest neighbor indices
        """
        state = initial_state

        # Navigate through layers (corecursive unfolding)
        while True:
            next_states = self.structure_map(state)
            if not next_states:
                break  # Terminal state reached
            state = next_states[0]

        # At layer 0, do ef-bounded search for final candidates
        return self._layer0_search(state)

    def _layer0_search(self, state: HNSWSearchState) -> List[Tuple[int, float]]:
        """Beam search at layer 0 with ef candidates"""
        import heapq

        query = np.array(state.query, dtype=np.float32)

        # Priority queues: (distance, node_idx)
        candidates = []  # Min-heap of candidates to explore
        results = []     # Max-heap of best results (negated for max behavior)
        visited = set()

        # Start from current node
        start_dist = self.distance_fn(
            query, self.graph.vectors[state.current_node:state.current_node+1]
        )[0]
        heapq.heappush(candidates, (start_dist, state.current_node))
        heapq.heappush(results, (-start_dist, state.current_node))
        visited.add(state.current_node)

        while candidates:
            dist, current = heapq.heappop(candidates)

            # Stop if current is worse than worst result and we have enough
            if len(results) >= state.ef and dist > -results[0][0]:
                break

            # Explore neighbors
            for neighbor in self.graph.get_neighbors(current, 0):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                neighbor_dist = self.distance_fn(
                    query, self.graph.vectors[neighbor:neighbor+1]
                )[0]

                heapq.heappush(candidates, (neighbor_dist, neighbor))

                if len(results) < state.ef:
                    heapq.heappush(results, (-neighbor_dist, neighbor))
                elif neighbor_dist < -results[0][0]:
                    heapq.heapreplace(results, (-neighbor_dist, neighbor))

        # Return sorted results
        final = [(-d, idx) for d, idx in results]
        final.sort()
        return [(idx, d) for d, idx in final]
```

**2.3 HNSW Insert Algorithm**

```python
# vector/hnsw/insert.py

class HNSWInserter:
    """Insert vectors into HNSW graph"""

    def __init__(self, graph: HNSWGraph, distance_fn: Callable):
        self.graph = graph
        self.distance_fn = distance_fn
        self.coalgebra = HNSWNavigationCoalgebra(graph, distance_fn)

    def insert(self, vector_id: str, vector: np.ndarray) -> None:
        """
        Insert a single vector into the graph.

        Algorithm (from paper):
        1. Sample random level for new node
        2. Navigate from entry point to insertion layer
        3. At each layer from insertion_level to 0:
           - Find ef_construction nearest neighbors
           - Add bidirectional edges to M best neighbors
           - Prune if needed to maintain M limit
        """
        # Add vector to storage
        idx = len(self.graph.ids)
        if self.graph.vectors is None:
            self.graph.vectors = vector.reshape(1, -1).astype(np.float32)
        else:
            self.graph.vectors = np.vstack([
                self.graph.vectors,
                vector.astype(np.float32)
            ])
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
                while len(self.graph.layers) <= l:
                    self.graph.layers.append({})
                self.graph.layers[l][idx] = []
            return

        # Navigate from entry point to insertion level
        current = self.graph.entry_point

        # Descend through upper layers (greedy search)
        for l in range(self.graph.max_level, level, -1):
            current = self._greedy_search_layer(vector, current, l)

        # Insert at each layer from level down to 0
        for l in range(min(level, self.graph.max_level), -1, -1):
            neighbors = self._search_layer(vector, current, l, self.graph.ef_construction)

            # Select M best neighbors
            M = self.graph.M0 if l == 0 else self.graph.M
            selected = self._select_neighbors(vector, neighbors, M)

            # Add edges
            while len(self.graph.layers) <= l:
                self.graph.layers.append({})
            self.graph.layers[l][idx] = []

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

    def _greedy_search_layer(self, query: np.ndarray, entry: int, level: int) -> int:
        """Greedy search to find closest node at layer"""
        current = entry
        current_dist = self.distance_fn(
            query, self.graph.vectors[current:current+1]
        )[0]

        while True:
            neighbors = self.graph.get_neighbors(current, level)
            if not neighbors:
                break

            neighbor_vectors = self.graph.vectors[neighbors]
            distances = self.distance_fn(query.reshape(1, -1), neighbor_vectors)[0]

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
        import heapq

        candidates = []
        results = []
        visited = {entry}

        entry_dist = self.distance_fn(query, self.graph.vectors[entry:entry+1])[0]
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

                neighbor_dist = self.distance_fn(
                    query, self.graph.vectors[neighbor:neighbor+1]
                )[0]

                heapq.heappush(candidates, (neighbor_dist, neighbor))

                if len(results) < ef:
                    heapq.heappush(results, (-neighbor_dist, neighbor))
                elif neighbor_dist < -results[0][0]:
                    heapq.heapreplace(results, (-neighbor_dist, neighbor))

        return [(idx, -d) for d, idx in sorted(results, reverse=True)]

    def _select_neighbors(
        self, query: np.ndarray, candidates: List[Tuple[int, float]], M: int
    ) -> List[Tuple[int, float]]:
        """Select M neighbors using simple heuristic"""
        # Simple: take M closest
        return candidates[:M]

    def _prune_connections(self, node_idx: int, level: int, M: int) -> None:
        """Prune connections if node has more than M neighbors"""
        neighbors = self.graph.layers[level].get(node_idx, [])
        if len(neighbors) <= M:
            return

        # Keep M closest neighbors
        node_vector = self.graph.vectors[node_idx]
        neighbor_vectors = self.graph.vectors[neighbors]
        distances = self.distance_fn(node_vector.reshape(1, -1), neighbor_vectors)[0]

        sorted_idx = np.argsort(distances)[:M]
        self.graph.layers[level][node_idx] = [neighbors[i] for i in sorted_idx]
```

**2.4 Numba-Accelerated Distance Functions**

```python
# vector/hnsw/numba_kernels.py

import numba
import numpy as np


@numba.jit(nopython=True, fastmath=True)
def l2_distance_single(query: np.ndarray, vector: np.ndarray) -> float:
    """L2 distance between two vectors"""
    diff = query - vector
    return np.sqrt(np.dot(diff, diff))


@numba.jit(nopython=True, fastmath=True, parallel=True)
def l2_distance_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """L2 distances from query to multiple vectors"""
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float32)

    for i in numba.prange(n):
        diff = query - vectors[i]
        distances[i] = np.sqrt(np.dot(diff, diff))

    return distances


@numba.jit(nopython=True, fastmath=True)
def cosine_distance_single(query: np.ndarray, vector: np.ndarray) -> float:
    """Cosine distance (1 - cosine_similarity)"""
    dot = np.dot(query, vector)
    norm_q = np.sqrt(np.dot(query, query))
    norm_v = np.sqrt(np.dot(vector, vector))
    return 1.0 - dot / (norm_q * norm_v + 1e-8)


@numba.jit(nopython=True, fastmath=True, parallel=True)
def cosine_distance_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Cosine distances from query to multiple vectors"""
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float32)

    norm_q = np.sqrt(np.dot(query, query))

    for i in numba.prange(n):
        dot = np.dot(query, vectors[i])
        norm_v = np.sqrt(np.dot(vectors[i], vectors[i]))
        distances[i] = 1.0 - dot / (norm_q * norm_v + 1e-8)

    return distances


@numba.jit(nopython=True, fastmath=True)
def inner_product_distance_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Negative inner product (for max inner product search)"""
    return -vectors @ query
```

**2.5 HNSW Vector Index (Complete Implementation)**

```python
# vector/hnsw/__init__.py

class NativeHNSWIndex(VectorIndex):
    """
    Native Python+Numba HNSW implementation.

    Features:
    - Coalgebraic search (HNSWNavigationCoalgebra)
    - Numba-accelerated distance computations
    - No external dependencies (pure Python)
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50
    ):
        self.dimension = dimension
        self.metric = metric
        self.ef_search = ef_search

        # Select distance function
        if metric == "cosine":
            self._distance_fn = cosine_distance_batch
        elif metric == "l2":
            self._distance_fn = l2_distance_batch
        else:
            self._distance_fn = inner_product_distance_batch

        # Initialize graph
        self.graph = HNSWGraph(
            dimension=dimension,
            M=M,
            ef_construction=ef_construction
        )

        # Initialize algorithms
        self.inserter = HNSWInserter(self.graph, self._distance_fn)
        self.coalgebra = HNSWNavigationCoalgebra(self.graph, self._distance_fn)

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add vectors to index"""
        vectors = vectors.astype(np.float32)
        for id_, vector in zip(ids, vectors):
            self.inserter.insert(id_, vector)

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """Search using coalgebraic unfolding"""
        if self.graph.entry_point < 0:
            return []

        query = query.astype(np.float32)

        # Create initial state
        initial_state = HNSWSearchState(
            query=tuple(query.tolist()),
            current_node=self.graph.entry_point,
            current_level=self.graph.max_level,
            candidates=(),
            visited=frozenset(),
            ef=max(k, self.ef_search)
        )

        # Unfold coalgebra
        results = self.coalgebra.unfold(initial_state)

        # Convert to VectorSearchResult
        return [
            VectorSearchResult(
                id=self.graph.ids[idx],
                score=1.0 - dist if self.metric == "cosine" else -dist,
                vector=self.graph.vectors[idx]
            )
            for idx, dist in results[:k]
        ]

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[VectorSearchResult]]:
        """Batch search"""
        return [self.search(q, k) for q in queries]

    @property
    def size(self) -> int:
        return len(self.graph.ids)

    def save(self, path: str) -> None:
        """Persist index to disk"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'dimension': self.dimension,
                'metric': self.metric,
                'ef_search': self.ef_search
            }, f)

    @classmethod
    def load(cls, path: str) -> "NativeHNSWIndex":
        """Load index from disk"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            dimension=data['dimension'],
            metric=data['metric'],
            ef_search=data['ef_search']
        )
        instance.graph = data['graph']
        instance.inserter = HNSWInserter(instance.graph, instance._distance_fn)
        instance.coalgebra = HNSWNavigationCoalgebra(instance.graph, instance._distance_fn)
        return instance
```

### Phase 3: Flat Index (Exact Search Baseline)

```python
# vector/index_flat.py

class FlatVectorIndex(VectorIndex):
    """Brute-force exact search - baseline and small-scale use"""

    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / norms

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        for id_ in ids:
            self._id_to_idx[id_] = len(self._ids)
            self._ids.append(id_)

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        if self._vectors is None:
            return []

        query = query.astype(np.float32)
        if self.metric == "cosine":
            query = query / np.linalg.norm(query)
            scores = self._vectors @ query
        else:
            scores = -np.linalg.norm(self._vectors - query, axis=1)

        if k < len(scores):
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(scores)[::-1]

        return [
            VectorSearchResult(
                id=self._ids[idx],
                score=float(scores[idx]),
                vector=self._vectors[idx]
            )
            for idx in top_k_idx[:k]
        ]
```

### Phase 4: Vector Search Engine (Coalgebra-Based)

```python
# vector/search.py

@dataclass(frozen=True)
class VectorQueryState:
    """Immutable query state for coalgebra"""
    query_embedding: Tuple[float, ...]
    top_k: int
    seen_ids: frozenset = frozenset()


class VectorSearchCoalgebra(Coalgebra[VectorQueryState, List[VectorSearchResult]]):
    """Coalgebra: QueryState → List[Result] via vector index"""

    def __init__(self, index: VectorIndex):
        self.index = index

    def structure_map(self, state: VectorQueryState) -> List[VectorSearchResult]:
        """One-step unfolding: query → results"""
        query = np.array(state.query_embedding, dtype=np.float32)
        fetch_k = state.top_k + len(state.seen_ids)
        results = self.index.search(query, k=fetch_k)
        filtered = [r for r in results if r.id not in state.seen_ids]
        return filtered[:state.top_k]


class VajraVectorSearch:
    """Main vector search engine - categorical design"""

    def __init__(
        self,
        embedder: EmbeddingMorphism,
        index: VectorIndex,
        cache_size: int = 1000
    ):
        self.embedder = embedder
        self.index = index
        self.coalgebra = VectorSearchCoalgebra(index)
        self._cache = LRUCache(maxsize=cache_size)

    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 256,
        show_progress: bool = True
    ) -> None:
        """Build index from documents"""
        ids = []
        embeddings = []

        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        iterator = tqdm(batches, desc="Indexing") if show_progress else batches

        for batch in iterator:
            texts = [doc.content for doc in batch]
            batch_embeddings = self.embedder.embed_batch(texts)
            ids.extend([doc.id for doc in batch])
            embeddings.append(batch_embeddings)

        all_embeddings = np.vstack(embeddings)
        self.index.add(ids, all_embeddings)

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search with caching"""
        cache_key = (query, top_k)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        query_embedding = self.embedder.embed(query)
        state = VectorQueryState(
            query_embedding=tuple(query_embedding.tolist()),
            top_k=top_k
        )
        vector_results = self.coalgebra.structure_map(state)

        results = [
            SearchResult(
                document=self._get_document(r.id),
                score=r.score,
                rank=i + 1
            )
            for i, r in enumerate(vector_results)
        ]

        self._cache.put(cache_key, results)
        return results
```

### Phase 5: Hybrid Search (BM25 + Vector)

```python
# hybrid/fusion.py

class HybridSearchEngine:
    """Combine BM25 and vector search using Reciprocal Rank Fusion"""

    def __init__(
        self,
        bm25_engine: VajraSearchOptimized,
        vector_engine: VajraVectorSearch,
        alpha: float = 0.5
    ):
        self.bm25 = bm25_engine
        self.vector = vector_engine
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Reciprocal Rank Fusion of BM25 and vector results"""
        k_rrf = 60

        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}

        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha / (k_rrf + rank)

        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) / (k_rrf + rank)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        doc_map = {r.document.id: r.document for r in bm25_results + vector_results}

        return [
            SearchResult(
                document=doc_map[doc_id],
                score=rrf_scores[doc_id],
                rank=i + 1
            )
            for i, doc_id in enumerate(sorted_ids[:top_k])
        ]
```

### Phase 6: Optimizations

**6.1 Embedding Cache (Comonadic)**

```python
# vector/optimized.py

class EmbeddingCache:
    """Comonadic cache for expensive embeddings"""

    def __init__(self, embedder: EmbeddingMorphism, maxsize: int = 10000):
        self.embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> np.ndarray:
        if text in self._cache:
            self.hits += 1
            return self._cache[text]

        self.misses += 1
        embedding = self.embedder.embed(text)
        self._store(text, embedding)
        return embedding

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

**6.2 Product Quantization (Memory Optimization)**

```python
class QuantizedVectorIndex:
    """Product quantization for memory-efficient storage"""

    def __init__(
        self,
        dimension: int,
        n_subvectors: int = 8,
        n_centroids: int = 256
    ):
        self.dimension = dimension
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.subvector_dim = dimension // n_subvectors
        self.codebooks: Optional[np.ndarray] = None
        self.codes: Optional[np.ndarray] = None

    # Memory savings: 768-dim float32 = 3KB per vector
    # With PQ (8 subvectors, 256 centroids): 8 bytes per vector
    # 375x memory reduction!
```

## Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
vector = [
    "sentence-transformers>=2.2.0",
]
vector-full = [
    "sentence-transformers>=2.2.0",
    "hnswlib>=0.7.0",  # Optional: for comparison/fallback
]
all = [
    "vajra-bm25[optimized,persistence,vector]",
]
```

## Benchmarking Plan

| Metric | Target | Baseline (hnswlib) |
|--------|--------|-------------------|
| Index build (100K docs) | < 5 min | ~2 min |
| Query latency (p50) | < 10ms | ~1ms |
| Query latency (p99) | < 50ms | ~5ms |
| QPS (single thread) | > 500 | ~2000 |
| Recall@10 | > 0.95 | 0.98 |
| Memory (100K, 768d) | < 1GB | 300MB |

Note: Native Python+Numba will be slower than C++ hnswlib, but provides:
- Zero external dependencies
- Full categorical abstraction integration
- Easier debugging and modification
- Educational value

## Testing Strategy

```python
# tests/vector/test_hnsw_coalgebra.py

def test_hnsw_navigation_coalgebra():
    """Verify coalgebra structure map produces valid successors"""
    graph = HNSWGraph(dimension=128)
    # ... add test vectors ...

    coalgebra = HNSWNavigationCoalgebra(graph, l2_distance_batch)

    state = HNSWSearchState(
        query=tuple(np.random.randn(128).tolist()),
        current_node=graph.entry_point,
        current_level=graph.max_level,
        candidates=(),
        visited=frozenset(),
        ef=50
    )

    next_states = coalgebra.structure_map(state)

    # Verify: either descends a layer or stays at same layer
    for ns in next_states:
        assert ns.current_level <= state.current_level


def test_native_hnsw_recall():
    """Verify native HNSW achieves target recall vs flat index"""
    flat = FlatVectorIndex(dimension=128)
    hnsw = NativeHNSWIndex(dimension=128, M=16, ef_construction=200)

    vectors = np.random.randn(10000, 128).astype(np.float32)
    ids = [f"doc_{i}" for i in range(10000)]

    flat.add(ids, vectors)
    hnsw.add(ids, vectors)

    query = np.random.randn(128).astype(np.float32)
    exact_results = flat.search(query, k=10)
    approx_results = hnsw.search(query, k=10)

    exact_ids = {r.id for r in exact_results}
    approx_ids = {r.id for r in approx_results}

    recall = len(exact_ids & approx_ids) / len(exact_ids)
    assert recall >= 0.9
```

## Implementation Timeline

| Phase | Description | Effort |
|-------|-------------|--------|
| 1 | Core abstractions (Morphisms, interfaces) | 2 days |
| 2 | Native HNSW graph + coalgebra | 5 days |
| 3 | Flat index baseline | 1 day |
| 4 | Vector search engine | 2 days |
| 5 | Hybrid BM25+Vector | 2 days |
| 6 | Optimizations (cache, PQ) | 3 days |
| 7 | Testing & benchmarking | 3 days |
| **Total** | | **~18 days** |

## Future Extensions

1. **Filtered Search**: Add metadata filtering during ANN search
2. **Incremental Updates**: Delete/update vectors without full rebuild
3. **Multi-Modal**: Extend to images, audio via CLIP-like embeddings
4. **Distributed**: Shard index across machines for billion-scale
5. **GPU Acceleration**: CUDA kernels for batch distance computation
