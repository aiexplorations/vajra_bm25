# Vajra Vector Search Implementation Plan

## Objective

Build a high-performance vector search engine within Vajra that:
- Scales to 100K-500K+ vectors
- Maintains categorical abstractions (Morphism, Coalgebra, Functor)
- Uses Python + Numba (no Rust rewrite)
- Achieves competitive performance with production systems

## Architecture Overview

```
vajra_bm25/
├── categorical/              # Existing - reuse unchanged
│   ├── category.py          # Morphism composition
│   ├── functor.py           # ListFunctor, MaybeFunctor
│   └── coalgebra.py         # SearchCoalgebra (extend)
│
├── vector/                   # NEW - vector search module
│   ├── __init__.py
│   ├── embeddings.py        # Embedding morphisms
│   ├── index.py             # Vector index abstractions
│   ├── index_hnsw.py        # HNSW implementation (hnswlib backend)
│   ├── index_flat.py        # Brute-force (small scale, exact)
│   ├── scorer.py            # Similarity morphisms (cosine, L2, dot)
│   ├── search.py            # VajraVectorSearch (coalgebra-based)
│   └── optimized.py         # Batching, caching, Numba kernels
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
        """Numba-accelerated batch scoring"""
        # Assumes normalized vectors
        return vectors @ query


class L2Distance(SimilarityMorphism):
    """Euclidean distance (lower is more similar)"""

    def apply(self, pair: Tuple[np.ndarray, np.ndarray]) -> float:
        a, b = pair
        return -np.linalg.norm(a - b)  # Negative for "higher is better"

    @staticmethod
    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def batch_distances(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch L2 distances"""
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float32)
        for i in numba.prange(n):
            diff = vectors[i] - query
            distances[i] = np.sqrt(np.dot(diff, diff))
        return distances
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
    """Abstract vector index - morphism from query bounds to candidates"""

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

### Phase 2: Index Implementations

**2.1 Flat Index (Exact Search)**

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
            # Pre-normalize for fast dot product
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
            scores = self._vectors @ query  # Dot product on normalized vectors
        else:
            scores = -np.linalg.norm(self._vectors - query, axis=1)

        # Partial sort for top-k
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

**2.2 HNSW Index (Approximate Search)**

```python
# vector/index_hnsw.py

class HNSWVectorIndex(VectorIndex):
    """HNSW index using hnswlib backend"""

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 100_000
    ):
        self.dimension = dimension
        self.metric = metric
        self.ef_construction = ef_construction
        self.M = M

        # Map metric names
        space = {"cosine": "cosine", "l2": "l2", "dot": "ip"}[metric]

        self._index = hnswlib.Index(space=space, dim=dimension)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        self._index.set_ef(50)  # Query-time ef

        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32)
        start_idx = len(self._ids)
        indices = np.arange(start_idx, start_idx + len(ids))

        self._index.add_items(vectors, indices)

        for i, id_ in enumerate(ids):
            self._id_to_idx[id_] = start_idx + i
            self._ids.append(id_)

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        query = query.astype(np.float32).reshape(1, -1)
        indices, distances = self._index.knn_query(query, k=k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self._ids):
                # Convert distance to similarity score
                if self.metric == "cosine":
                    score = 1.0 - dist  # hnswlib returns 1 - cosine
                elif self.metric == "l2":
                    score = -dist  # Lower distance = higher score
                else:
                    score = dist  # Inner product

                results.append(VectorSearchResult(
                    id=self._ids[idx],
                    score=float(score)
                ))

        return results

    def search_batch(self, queries: np.ndarray, k: int) -> List[List[VectorSearchResult]]:
        queries = queries.astype(np.float32)
        indices, distances = self._index.knn_query(queries, k=k)

        all_results = []
        for query_indices, query_distances in zip(indices, distances):
            results = []
            for idx, dist in zip(query_indices, query_distances):
                if idx < len(self._ids):
                    score = 1.0 - dist if self.metric == "cosine" else -dist
                    results.append(VectorSearchResult(
                        id=self._ids[idx],
                        score=float(score)
                    ))
            all_results.append(results)

        return all_results

    def set_ef(self, ef: int) -> None:
        """Set query-time ef parameter (accuracy vs speed tradeoff)"""
        self._index.set_ef(ef)

    def save(self, path: str) -> None:
        self._index.save_index(f"{path}.hnsw")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "ids": self._ids,
                "id_to_idx": self._id_to_idx,
                "dimension": self.dimension,
                "metric": self.metric
            }, f)

    @classmethod
    def load(cls, path: str) -> "HNSWVectorIndex":
        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)

        instance = cls(dimension=meta["dimension"], metric=meta["metric"])
        instance._index.load_index(f"{path}.hnsw")
        instance._ids = meta["ids"]
        instance._id_to_idx = meta["id_to_idx"]
        return instance
```

### Phase 3: Search Engine (Coalgebra-Based)

**3.1 Vector Search Coalgebra**

```python
# vector/search.py

@dataclass(frozen=True)
class VectorQueryState:
    """Immutable query state for coalgebra"""
    query_embedding: Tuple[float, ...]  # Immutable
    top_k: int
    seen_ids: frozenset = frozenset()

    def __hash__(self):
        return hash((self.query_embedding[:8], self.top_k, self.seen_ids))


class VectorSearchCoalgebra(Coalgebra[VectorQueryState, List[VectorSearchResult]]):
    """Coalgebra: QueryState → List[Result] via vector index"""

    def __init__(self, index: VectorIndex):
        self.index = index

    def structure_map(self, state: VectorQueryState) -> List[VectorSearchResult]:
        """One-step unfolding: query → results"""
        query = np.array(state.query_embedding, dtype=np.float32)

        # Fetch more than needed to filter seen
        fetch_k = state.top_k + len(state.seen_ids)
        results = self.index.search(query, k=fetch_k)

        # Filter already-seen documents
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

        # Embed query (morphism application)
        query_embedding = self.embedder.embed(query)

        # Create query state
        state = VectorQueryState(
            query_embedding=tuple(query_embedding.tolist()),
            top_k=top_k
        )

        # Apply coalgebra structure map
        vector_results = self.coalgebra.structure_map(state)

        # Convert to SearchResult
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

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """Batch search - vectorized embedding + parallel index search"""
        # Batch embed all queries
        query_embeddings = self.embedder.embed_batch(queries)

        # Batch search
        all_vector_results = self.index.search_batch(query_embeddings, k=top_k)

        # Convert results
        all_results = []
        for vector_results in all_vector_results:
            results = [
                SearchResult(
                    document=self._get_document(r.id),
                    score=r.score,
                    rank=i + 1
                )
                for i, r in enumerate(vector_results)
            ]
            all_results.append(results)

        return all_results
```

### Phase 4: Optimizations

**4.1 Numba-Accelerated Kernels**

```python
# vector/optimized.py

@numba.jit(nopython=True, fastmath=True, parallel=True)
def batch_cosine_similarity(
    queries: np.ndarray,  # (n_queries, dim)
    vectors: np.ndarray   # (n_vectors, dim)
) -> np.ndarray:
    """Compute all pairwise cosine similarities"""
    n_queries = queries.shape[0]
    n_vectors = vectors.shape[0]
    scores = np.empty((n_queries, n_vectors), dtype=np.float32)

    for i in numba.prange(n_queries):
        for j in range(n_vectors):
            dot = 0.0
            norm_q = 0.0
            norm_v = 0.0
            for k in range(queries.shape[1]):
                dot += queries[i, k] * vectors[j, k]
                norm_q += queries[i, k] * queries[i, k]
                norm_v += vectors[j, k] * vectors[j, k]
            scores[i, j] = dot / (np.sqrt(norm_q) * np.sqrt(norm_v) + 1e-8)

    return scores


@numba.jit(nopython=True, fastmath=True)
def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Get top-k indices using partial sort"""
    n = len(scores)
    if k >= n:
        return np.argsort(scores)[::-1]

    # Argpartition is O(n) average
    indices = np.argpartition(scores, -k)[-k:]
    # Sort only the top k
    sorted_indices = indices[np.argsort(scores[indices])[::-1]]
    return sorted_indices
```

**4.2 Embedding Cache (Comonadic)**

```python
# vector/optimized.py

class EmbeddingCache:
    """Comonadic cache for expensive embeddings"""

    def __init__(self, embedder: EmbeddingMorphism, maxsize: int = 10000):
        self.embedder = embedder
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def embed(self, text: str) -> np.ndarray:
        """Cached single embedding"""
        if text in self._cache:
            self.hits += 1
            return self._cache[text]

        self.misses += 1
        embedding = self.embedder.embed(text)
        self._store(text, embedding)
        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding with cache awareness"""
        # Separate cached and uncached
        cached_indices = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                cached_indices.append(i)
                self.hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self.misses += 1

        # Only compute uncached
        if uncached_texts:
            new_embeddings = self.embedder.embed_batch(uncached_texts)
            for text, emb in zip(uncached_texts, new_embeddings):
                self._store(text, emb)

        # Assemble result
        dim = self.embedder.dimension
        result = np.empty((len(texts), dim), dtype=np.float32)

        for i in cached_indices:
            result[i] = self._cache[texts[i]]
        for i, idx in enumerate(uncached_indices):
            result[idx] = self._cache[uncached_texts[i]]

        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

**4.3 Quantization (Memory Optimization)**

```python
# vector/optimized.py

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

        # Codebooks: (n_subvectors, n_centroids, subvector_dim)
        self.codebooks: Optional[np.ndarray] = None
        # Codes: (n_vectors, n_subvectors) as uint8
        self.codes: Optional[np.ndarray] = None

    def train(self, vectors: np.ndarray) -> None:
        """Train codebooks using k-means"""
        from sklearn.cluster import KMeans

        self.codebooks = np.empty(
            (self.n_subvectors, self.n_centroids, self.subvector_dim),
            dtype=np.float32
        )

        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            kmeans = KMeans(n_clusters=self.n_centroids, n_init=1)
            kmeans.fit(subvectors)
            self.codebooks[i] = kmeans.cluster_centers_

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to codes"""
        n = vectors.shape[0]
        codes = np.empty((n, self.n_subvectors), dtype=np.uint8)

        for i in range(self.n_subvectors):
            start = i * self.subvector_dim
            end = start + self.subvector_dim
            subvectors = vectors[:, start:end]

            # Find nearest centroid for each subvector
            distances = np.sum(
                (subvectors[:, np.newaxis] - self.codebooks[i]) ** 2,
                axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    # Memory savings: 768-dim float32 = 3KB per vector
    # With PQ (8 subvectors, 256 centroids): 8 bytes per vector
    # 375x memory reduction!
```

### Phase 5: Hybrid Search (BM25 + Vector)

```python
# hybrid/fusion.py

class HybridSearchEngine:
    """Combine BM25 and vector search"""

    def __init__(
        self,
        bm25_engine: VajraSearchOptimized,
        vector_engine: VajraVectorSearch,
        alpha: float = 0.5  # Weight for BM25 vs vector
    ):
        self.bm25 = bm25_engine
        self.vector = vector_engine
        self.alpha = alpha

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Reciprocal Rank Fusion of BM25 and vector results"""
        k_rrf = 60  # RRF constant

        # Get results from both
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)

        # Compute RRF scores
        rrf_scores: Dict[str, float] = {}

        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + self.alpha / (k_rrf + rank)

        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - self.alpha) / (k_rrf + rank)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build final results
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

## Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
vector = [
    "hnswlib>=0.7.0",
    "sentence-transformers>=2.2.0",
]
vector-gpu = [
    "faiss-gpu>=1.7.0",
    "sentence-transformers>=2.2.0",
]
all = [
    "vajra-bm25[optimized,persistence,vector]",
]
```

## Benchmarking Plan

| Metric | Target | Baseline |
|--------|--------|----------|
| Index build (100K docs) | < 5 min | - |
| Query latency (p50) | < 10ms | hnswlib raw: ~1ms |
| Query latency (p99) | < 50ms | - |
| QPS (single thread) | > 1000 | - |
| Recall@10 | > 0.95 | Flat index: 1.0 |
| Memory (100K, 768d) | < 1GB | Raw: 300MB |

## Testing Strategy

```python
# tests/vector/test_embeddings.py
def test_embedding_morphism_composition():
    """Verify morphism laws hold"""
    embedder = TextEmbeddingMorphism()
    text = "hello world"

    # Identity: embed(text) should be deterministic
    e1 = embedder.embed(text)
    e2 = embedder.embed(text)
    assert np.allclose(e1, e2)

    # Batch consistency
    batch_result = embedder.embed_batch([text])
    assert np.allclose(e1, batch_result[0])


# tests/vector/test_index.py
def test_index_recall():
    """Verify HNSW achieves target recall"""
    flat = FlatVectorIndex(dimension=128)
    hnsw = HNSWVectorIndex(dimension=128, ef_construction=200)

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
    assert recall >= 0.9  # 90% recall minimum
```

## Future Extensions

1. **Point Cloud Support**: Add `PointCloudEmbeddingMorphism` using PointNet/DGCNN
2. **Multi-Modal**: Extend to images, audio via CLIP-like embeddings
3. **Filtered Search**: Add metadata filtering during ANN search
4. **Distributed**: Shard index across machines for billion-scale
