"""
Vajra Vector Search: Categorical Vector Similarity Search

This module provides vector search capabilities with the same categorical
abstractions used in BM25 search:

- Embeddings as morphisms: Text → ℝ^d
- Similarity as morphisms: (ℝ^d, ℝ^d) → ℝ
- HNSW navigation as coalgebraic unfolding
- Hybrid BM25 + Vector fusion

Basic Usage:
    from vajra_bm25.vector import VajraVectorSearch, NativeHNSWIndex
    from vajra_bm25.vector import TextEmbeddingMorphism

    # Create embedder and index
    embedder = TextEmbeddingMorphism(model_name="all-MiniLM-L6-v2")
    index = NativeHNSWIndex(dimension=384, metric="cosine")

    # Build search engine
    engine = VajraVectorSearch(embedder, index)
    engine.index_documents(documents)

    # Search
    results = engine.search("semantic query", top_k=10)

Hybrid Search:
    from vajra_bm25.vector import HybridSearchEngine
    from vajra_bm25 import VajraSearchOptimized

    hybrid = HybridSearchEngine(
        bm25_engine=bm25_engine,
        vector_engine=vector_engine,
        alpha=0.5  # BM25 weight
    )
    results = hybrid.search("query", top_k=10)
"""

# Core abstractions
from vajra_bm25.vector.embeddings import (
    EmbeddingMorphism,
    TextEmbeddingMorphism,
    PrecomputedEmbeddingMorphism,
)

from vajra_bm25.vector.scorer import (
    SimilarityMorphism,
    CosineSimilarity,
    L2Distance,
    InnerProduct,
)

from vajra_bm25.vector.index import (
    VectorSearchResult,
    VectorIndex,
)

from vajra_bm25.vector.index_flat import FlatVectorIndex

# HNSW implementation
from vajra_bm25.vector.hnsw import (
    HNSWGraph,
    HNSWSearchState,
    HNSWNavigationCoalgebra,
    NativeHNSWIndex,
)

# Search engine
from vajra_bm25.vector.search import (
    VectorQueryState,
    VectorSearchCoalgebra,
    VajraVectorSearch,
)

# Hybrid search
from vajra_bm25.vector.hybrid import HybridSearchEngine

__all__ = [
    # Embeddings
    "EmbeddingMorphism",
    "TextEmbeddingMorphism",
    "PrecomputedEmbeddingMorphism",
    # Similarity
    "SimilarityMorphism",
    "CosineSimilarity",
    "L2Distance",
    "InnerProduct",
    # Index
    "VectorSearchResult",
    "VectorIndex",
    "FlatVectorIndex",
    # HNSW
    "HNSWGraph",
    "HNSWSearchState",
    "HNSWNavigationCoalgebra",
    "NativeHNSWIndex",
    # Search
    "VectorQueryState",
    "VectorSearchCoalgebra",
    "VajraVectorSearch",
    # Hybrid
    "HybridSearchEngine",
]
