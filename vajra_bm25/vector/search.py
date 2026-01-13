"""
Vajra Vector Search Engine

High-level vector search engine combining:
- Embedding morphisms for text-to-vector transformation
- Vector indices for similarity search
- Coalgebraic search for clean abstraction

This mirrors the VajraSearch/VajraSearchOptimized pattern from BM25,
providing a consistent API for vector-based retrieval.

Usage:
    from vajra_bm25.vector import VajraVectorSearch, NativeHNSWIndex
    from vajra_bm25.vector import TextEmbeddingMorphism

    embedder = TextEmbeddingMorphism("all-MiniLM-L6-v2")
    index = NativeHNSWIndex(dimension=384, metric="cosine")

    engine = VajraVectorSearch(embedder, index)
    engine.index_documents(documents)

    results = engine.search("semantic query", top_k=10)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from vajra_bm25.categorical import Coalgebra
from vajra_bm25.documents import Document
from vajra_bm25.search import SearchResult
from vajra_bm25.vector.embeddings import EmbeddingMorphism
from vajra_bm25.vector.index import VectorIndex, VectorSearchResult


@dataclass(frozen=True)
class VectorQueryState:
    """
    Immutable query state for vector search coalgebra.

    The state captures:
    - Query embedding
    - Number of results to return
    - Already-seen IDs (for pagination/exclusion)
    """

    query_embedding: Tuple[float, ...]
    top_k: int
    seen_ids: frozenset = frozenset()

    def exclude(self, ids: List[str]) -> "VectorQueryState":
        """Return new state with additional excluded IDs"""
        return VectorQueryState(
            query_embedding=self.query_embedding,
            top_k=self.top_k,
            seen_ids=self.seen_ids | set(ids),
        )


class VectorSearchCoalgebra(Coalgebra[VectorQueryState, List[VectorSearchResult]]):
    """
    Coalgebra: QueryState → List[VectorSearchResult]

    Structure map transforms a query state into search results
    via the underlying vector index.
    """

    def __init__(self, index: VectorIndex):
        self.index = index

    def structure_map(self, state: VectorQueryState) -> List[VectorSearchResult]:
        """One-step unfolding: query state → results"""
        query = np.array(state.query_embedding, dtype=np.float32)

        # Fetch extra results to account for exclusions
        fetch_k = state.top_k + len(state.seen_ids)
        results = self.index.search(query, k=fetch_k)

        # Filter excluded IDs
        filtered = [r for r in results if r.id not in state.seen_ids]
        return filtered[: state.top_k]


class VajraVectorSearch:
    """
    Main vector search engine with categorical design.

    Combines embedding morphisms and vector indices for semantic search.
    Provides a high-level API similar to VajraSearch for BM25.

    Features:
    - Text-to-vector embedding via morphisms
    - Approximate nearest neighbor search
    - LRU caching for query embeddings
    - Document-based interface

    Usage:
        engine = VajraVectorSearch(embedder, index)
        engine.index_documents(documents)
        results = engine.search("query", top_k=10)
    """

    def __init__(
        self,
        embedder: EmbeddingMorphism,
        index: VectorIndex,
        cache_size: int = 1000,
    ):
        """
        Initialize vector search engine.

        Args:
            embedder: Morphism to convert text to vectors
            index: Vector index for similarity search
            cache_size: Size of embedding cache
        """
        self.embedder = embedder
        self.index = index
        self.coalgebra = VectorSearchCoalgebra(index)

        # Document storage (for returning full documents)
        self._documents: Dict[str, Document] = {}

        # Simple LRU cache for query embeddings
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_size = cache_size

    def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> int:
        """
        Build index from documents.

        Args:
            documents: List of Document objects
            batch_size: Batch size for embedding
            show_progress: Show progress bar

        Returns:
            Number of documents indexed
        """
        try:
            from tqdm import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        ids = []
        all_embeddings = []

        # Process in batches
        batches = [
            documents[i : i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        if show_progress and has_tqdm and len(batches) > 1:
            iterator = tqdm(batches, desc="Indexing documents")
        else:
            iterator = batches

        for batch in iterator:
            texts = [doc.content for doc in batch]
            batch_embeddings = self.embedder.embed_batch(texts)

            ids.extend([doc.id for doc in batch])
            all_embeddings.append(batch_embeddings)

            # Store documents for retrieval
            for doc in batch:
                self._documents[doc.id] = doc

        # Add all embeddings to index
        embeddings = np.vstack(all_embeddings)
        self.index.add(ids, embeddings)

        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to query.

        Args:
            query: Search query text
            top_k: Number of results
            exclude_ids: IDs to exclude from results

        Returns:
            List of SearchResult objects
        """
        # Get query embedding (with caching)
        query_embedding = self._get_cached_embedding(query)

        # Create query state
        seen_ids = frozenset(exclude_ids) if exclude_ids else frozenset()
        state = VectorQueryState(
            query_embedding=tuple(query_embedding.tolist()),
            top_k=top_k,
            seen_ids=seen_ids,
        )

        # Execute search via coalgebra
        vector_results = self.coalgebra.structure_map(state)

        # Convert to SearchResult
        results = []
        for i, vr in enumerate(vector_results):
            doc = self._documents.get(vr.id)
            if doc:
                results.append(
                    SearchResult(
                        document=doc,
                        score=vr.score,
                        rank=i + 1,
                    )
                )

        return results

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> List[VectorSearchResult]:
        """
        Search by pre-computed vector.

        Args:
            query_vector: Query embedding
            top_k: Number of results

        Returns:
            List of VectorSearchResult
        """
        return self.index.search(query_vector, k=top_k)

    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """Get embedding with LRU caching"""
        if text in self._cache:
            # Move to end of cache order (most recently used)
            self._cache_order.remove(text)
            self._cache_order.append(text)
            return self._cache[text]

        # Compute embedding
        embedding = self.embedder.embed(text)

        # Add to cache
        self._cache[text] = embedding
        self._cache_order.append(text)

        # Evict oldest if cache is full
        while len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return embedding

    def add_document(self, document: Document) -> None:
        """Add a single document to the index"""
        embedding = self.embedder.embed(document.content)
        self.index.add([document.id], embedding.reshape(1, -1))
        self._documents[document.id] = document

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID"""
        return self._documents.get(doc_id)

    @property
    def num_documents(self) -> int:
        """Number of indexed documents"""
        return self.index.size

    def save(self, path: str) -> None:
        """
        Save the search engine to disk.

        Note: Only saves the index. Documents must be re-added
        or stored separately.
        """
        self.index.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        embedder: EmbeddingMorphism,
        index_class: type,
    ) -> "VajraVectorSearch":
        """
        Load search engine from disk.

        Args:
            path: Path to saved index
            embedder: Embedder to use
            index_class: Class of index to load

        Returns:
            Loaded VajraVectorSearch instance
        """
        index = index_class.load(path)
        return cls(embedder, index)

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self._cache.clear()
        self._cache_order.clear()

    def __repr__(self) -> str:
        return (
            f"VajraVectorSearch(embedder={self.embedder.__class__.__name__}, "
            f"index={self.index.__class__.__name__}, "
            f"documents={self.num_documents})"
        )
