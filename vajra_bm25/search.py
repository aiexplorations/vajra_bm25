"""
BM25 Search as Coalgebra

Query unfolding: Query -> List[(Document, Score)]

This is the coalgebraic structure for information retrieval:
- Carrier: Query state
- Structure map: alpha: Query -> List[(Document, Score)]
- The list functor captures multiple results with scores

Different from graph search:
- State space is query space (can be refined/expanded)
- Unfolding produces ranked documents, not next states
- Terminal condition: user satisfied or top-k retrieved

Supports three search modes:
- bm25: Traditional keyword search (default)
- vector: Semantic vector search (lazy-loaded)
- hybrid: Combined BM25 + vector with score fusion
"""

from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass

from vajra_bm25.categorical import Coalgebra
from vajra_bm25.documents import Document, DocumentCorpus
from vajra_bm25.inverted_index import InvertedIndex
from vajra_bm25.scorer import BM25Scorer, BM25Parameters
from vajra_bm25.text_processing import preprocess_text
from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("search")

# Type alias for search modes
SearchMode = Literal["bm25", "vector", "hybrid"]


@dataclass(frozen=True)
class QueryState:
    """
    A query state in the search space.

    Immutable because states are objects in a category.

    Can include:
    - Original query
    - Refined/expanded query terms
    - Already seen documents (for iterative search)
    """
    query: str
    query_terms: Tuple[str, ...]  # Immutable tuple
    expansion_depth: int = 0  # How many refinements applied
    seen_docs: frozenset = frozenset()  # Documents already shown

    def __hash__(self):
        return hash((self.query, self.query_terms, self.expansion_depth, self.seen_docs))


@dataclass
class SearchResult:
    """A search result: document with relevance score"""
    document: Document
    score: float
    rank: int

    def __repr__(self):
        return f"Result(rank={self.rank}, score={self.score:.3f}, doc={self.document.id})"


class BM25SearchCoalgebra(Coalgebra[QueryState, List[SearchResult]]):
    """
    BM25 search as a coalgebra.

    Structure map: QueryState -> List[SearchResult]

    The query "unfolds" into a ranked list of documents.
    This is one-step unfolding (retrieve and rank).

    Categorical interpretation:
    - alpha: Query -> List[(Document, Score)]
    - List functor captures multiple results
    - Score is a morphism: (Query, Document) -> R
    """

    def __init__(
        self,
        corpus: DocumentCorpus,
        index: InvertedIndex,
        scorer: BM25Scorer,
        top_k: int = 10
    ):
        self.corpus = corpus
        self.index = index
        self.scorer = scorer
        self.top_k = top_k

    def structure_map(self, state: QueryState) -> List[SearchResult]:
        """
        alpha: QueryState -> List[SearchResult]

        Unfold query into ranked documents.

        Process:
        1. Get candidate documents (inverted index lookup)
        2. Score candidates with BM25
        3. Rank and return top-k

        This is coalgebraic unfolding!
        """
        # Get candidates from index
        candidates = self.index.get_candidate_documents(list(state.query_terms))

        # Filter out already seen documents (for iterative search)
        candidates = candidates - state.seen_docs

        if not candidates:
            return []

        # Rank candidates
        ranked = self.scorer.rank_documents(list(state.query_terms), list(candidates))

        # Take top-k and create SearchResult objects
        results = []
        for rank, (doc_id, score) in enumerate(ranked[:self.top_k], 1):
            doc = self.corpus.get(doc_id)
            if doc:
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    rank=rank
                ))

        return results


class VajraSearch:
    """
    Vajra BM25 search system using categorical framework.

    Vajra (Sanskrit: vajra, "thunderbolt/diamond") implements BM25 search
    using pure category theory.

    Supports three search modes:
    - bm25: Traditional keyword search (default, always available)
    - vector: Semantic vector search (lazy-loaded on first use)
    - hybrid: Combined BM25 + vector with score fusion

    Usage:
        engine = VajraSearch(corpus)

        # BM25 search (default)
        results = engine.search("query")

        # Vector search (lazy-loads embedding model)
        results = engine.search("semantic query", mode="vector")

        # Hybrid search (combines both)
        results = engine.search("query", mode="hybrid", alpha=0.5)
    """

    # Default embedding model for vector search
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        corpus: DocumentCorpus,
        params: BM25Parameters = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize search engine.

        Args:
            corpus: Document corpus to search
            params: BM25 parameters (k1, b)
            embedding_model: Optional embedding model name for vector search.
                           If None, uses default when vector search is first called.
        """
        self.corpus = corpus
        self._embedding_model_name = embedding_model or self.DEFAULT_EMBEDDING_MODEL

        # Build BM25 index
        logger.info("Building inverted index...")
        self.index = InvertedIndex()
        self.index.build(corpus)

        # Create BM25 scorer
        self.scorer = BM25Scorer(self.index, params or BM25Parameters())

        logger.info(f"Index built: {self.index}")

        # Lazy-loaded vector search components
        self._vector_engine = None
        self._hybrid_engine = None
        self._vector_initialized = False

    def _init_vector_search(self) -> bool:
        """
        Lazy-initialize vector search components.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._vector_initialized:
            return self._vector_engine is not None

        self._vector_initialized = True

        try:
            from vajra_bm25.vector import (
                VajraVectorSearch,
                FlatVectorIndex,
                NativeHNSWIndex,
                TextEmbeddingMorphism,
                HybridSearchEngine,
            )

            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            embedder = TextEmbeddingMorphism(
                model_name=self._embedding_model_name,
                normalize=True
            )
            dimension = embedder.dimension

            # Use HNSW for larger corpora, Flat for small ones
            corpus_size = len(self.corpus)
            if corpus_size > 1000:
                logger.info(f"Building HNSW index ({corpus_size:,} documents)")
                index = NativeHNSWIndex(
                    dimension=dimension,
                    metric="cosine",
                    M=16,
                    ef_construction=200,
                    ef_search=50
                )
            else:
                logger.info(f"Building flat index ({corpus_size:,} documents)")
                index = FlatVectorIndex(dimension=dimension, metric="cosine")

            self._vector_engine = VajraVectorSearch(embedder, index)

            # Index documents
            logger.info("Indexing documents for vector search...")
            docs = list(self.corpus.documents)
            self._vector_engine.index_documents(docs, batch_size=64, show_progress=False)

            # Create hybrid engine
            self._hybrid_engine = HybridSearchEngine(
                bm25_engine=self,
                vector_engine=self._vector_engine,
                method="rrf"
            )

            logger.info("Vector search initialized successfully")
            return True

        except ImportError as e:
            logger.warning(
                f"Vector search not available: {e}. "
                "Install with: pip install vajra-bm25[vector]"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = "bm25",
        alpha: float = 0.5,
    ) -> List[SearchResult]:
        """
        Execute search in specified mode.

        Args:
            query: Search query string
            top_k: Number of results to return
            mode: Search mode - "bm25", "vector", or "hybrid"
            alpha: BM25 weight for hybrid search (0-1). Higher = more BM25.

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If mode is invalid or vector search unavailable
        """
        if mode == "bm25":
            return self._search_bm25(query, top_k)
        elif mode == "vector":
            return self._search_vector(query, top_k)
        elif mode == "hybrid":
            return self._search_hybrid(query, top_k, alpha)
        else:
            raise ValueError(f"Unknown search mode: {mode}. Use 'bm25', 'vector', or 'hybrid'.")

    def _search_bm25(self, query: str, top_k: int) -> List[SearchResult]:
        """Execute BM25 search."""
        query_terms = preprocess_text(query)

        if not query_terms:
            return []

        # Create query state
        state = QueryState(
            query=query,
            query_terms=tuple(query_terms)
        )

        # Create coalgebra
        coalgebra = BM25SearchCoalgebra(
            corpus=self.corpus,
            index=self.index,
            scorer=self.scorer,
            top_k=top_k
        )

        # Unfold: apply structure map
        return coalgebra.structure_map(state)

    def _search_vector(self, query: str, top_k: int) -> List[SearchResult]:
        """Execute vector search (lazy-loads on first use)."""
        if not self._init_vector_search():
            raise ValueError(
                "Vector search not available. "
                "Install with: pip install vajra-bm25[vector]"
            )

        return self._vector_engine.search(query, top_k=top_k)

    def _search_hybrid(self, query: str, top_k: int, alpha: float) -> List[SearchResult]:
        """Execute hybrid search (lazy-loads on first use)."""
        if not self._init_vector_search():
            raise ValueError(
                "Hybrid search requires vector search. "
                "Install with: pip install vajra-bm25[vector]"
            )

        return self._hybrid_engine.search(query, top_k=top_k, bm25_weight=alpha)

    @property
    def vector_available(self) -> bool:
        """Check if vector search is available (without initializing)."""
        try:
            from vajra_bm25.vector import VajraVectorSearch
            return True
        except ImportError:
            return False

    @property
    def vector_initialized(self) -> bool:
        """Check if vector search has been initialized."""
        return self._vector_initialized and self._vector_engine is not None

    def explain_result(self, query: str, doc_id: str) -> dict:
        """
        Explain why a document was ranked for a query.

        Shows the BM25 score breakdown.
        """
        query_terms = preprocess_text(query)
        score = self.scorer.score(query_terms, doc_id)
        breakdown = self.scorer.explain_score(query_terms, doc_id)

        return {
            "total_score": score,
            "term_scores": breakdown,
            "query_terms": query_terms
        }


if __name__ == "__main__":
    from vajra_bm25.documents import create_sample_corpus

    # Create corpus
    corpus = create_sample_corpus()
    logger.info(f"Created corpus with {len(corpus)} documents")

    # Create search system
    search_engine = VajraSearch(corpus)

    # Test queries
    test_queries = [
        "category theory functors",
        "search algorithms breadth first",
        "functional programming monads",
        "lambda calculus computation"
    ]

    for query in test_queries:
        logger.info("=" * 70)
        logger.info(f"Query: '{query}'")
        logger.info("=" * 70)

        results = search_engine.search(query, top_k=5)

        if not results:
            logger.info("No results found.")
            continue

        logger.info(f"Found {len(results)} results:")

        for result in results:
            logger.info(f"{result.rank}. [{result.document.id}] {result.document.title}")
            logger.info(f"   Score: {result.score:.3f}")
            logger.info(f"   {result.document.content[:120]}...")

        # Explain top result
        if results:
            top_result = results[0]
            logger.info(f"Score breakdown for top result:")
            explanation = search_engine.explain_result(query, top_result.document.id)
            for term, score in sorted(explanation['term_scores'].items(),
                                     key=lambda x: x[1], reverse=True):
                if score > 0:
                    logger.info(f"  '{term}': {score:.3f}")
