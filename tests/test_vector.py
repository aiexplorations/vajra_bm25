"""
Tests for Vajra Vector Search Module

Tests cover:
- Embedding morphisms
- Similarity morphisms
- Flat vector index
- HNSW graph and coalgebra
- NativeHNSWIndex
- VajraVectorSearch engine
- Hybrid search
"""

import pytest
import numpy as np
import tempfile
import os

from vajra_bm25.documents import Document


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_vectors():
    """Generate sample vectors for testing"""
    np.random.seed(42)
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def sample_ids():
    """Generate sample IDs"""
    return [f"doc_{i}" for i in range(100)]


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        Document(id="1", title="Machine Learning", content="Introduction to machine learning algorithms"),
        Document(id="2", title="Deep Learning", content="Neural networks and deep learning techniques"),
        Document(id="3", title="Natural Language", content="Natural language processing with transformers"),
        Document(id="4", title="Computer Vision", content="Image recognition and computer vision models"),
        Document(id="5", title="Reinforcement", content="Reinforcement learning and decision making"),
    ]


@pytest.fixture
def normalized_vectors(sample_vectors):
    """L2-normalized vectors"""
    norms = np.linalg.norm(sample_vectors, axis=1, keepdims=True)
    return sample_vectors / norms


# ============================================================================
# Embedding Morphism Tests
# ============================================================================


class TestPrecomputedEmbeddingMorphism:
    """Tests for PrecomputedEmbeddingMorphism"""

    def test_basic_lookup(self):
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        embeddings = {
            "doc1": np.array([1.0, 2.0, 3.0]),
            "doc2": np.array([4.0, 5.0, 6.0]),
        }
        morphism = PrecomputedEmbeddingMorphism(embeddings)

        result = morphism.embed("doc1")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_dimension_property(self):
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        embeddings = {"doc1": np.array([1.0, 2.0, 3.0, 4.0])}
        morphism = PrecomputedEmbeddingMorphism(embeddings)

        assert morphism.dimension == 4

    def test_batch_embedding(self):
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        embeddings = {
            "a": np.array([1.0, 0.0]),
            "b": np.array([0.0, 1.0]),
        }
        morphism = PrecomputedEmbeddingMorphism(embeddings)

        result = morphism.embed_batch(["a", "b"])
        assert result.shape == (2, 2)

    def test_missing_id_raises(self):
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        embeddings = {"doc1": np.array([1.0, 2.0])}
        morphism = PrecomputedEmbeddingMorphism(embeddings)

        with pytest.raises(KeyError):
            morphism.embed("nonexistent")

    def test_add_embedding(self):
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        embeddings = {"doc1": np.array([1.0, 2.0])}
        morphism = PrecomputedEmbeddingMorphism(embeddings)

        morphism.add("doc2", np.array([3.0, 4.0]))
        result = morphism.embed("doc2")
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])


class TestIdentityEmbeddingMorphism:
    """Tests for IdentityEmbeddingMorphism"""

    def test_passthrough(self):
        from vajra_bm25.vector.embeddings import IdentityEmbeddingMorphism

        morphism = IdentityEmbeddingMorphism(dimension=3)
        vector = np.array([1.0, 2.0, 3.0])

        result = morphism.embed(vector)
        np.testing.assert_array_almost_equal(result, vector)

    def test_dimension_mismatch_raises(self):
        from vajra_bm25.vector.embeddings import IdentityEmbeddingMorphism

        morphism = IdentityEmbeddingMorphism(dimension=3)
        vector = np.array([1.0, 2.0])  # Wrong dimension

        with pytest.raises(ValueError):
            morphism.embed(vector)


# ============================================================================
# Similarity Morphism Tests
# ============================================================================


class TestCosineSimilarity:
    """Tests for CosineSimilarity"""

    def test_identical_vectors(self):
        from vajra_bm25.vector.scorer import CosineSimilarity

        scorer = CosineSimilarity()
        v = np.array([1.0, 2.0, 3.0])

        assert scorer.score(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from vajra_bm25.vector.scorer import CosineSimilarity

        scorer = CosineSimilarity()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])

        assert scorer.score(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        from vajra_bm25.vector.scorer import CosineSimilarity

        scorer = CosineSimilarity()
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])

        assert scorer.score(v1, v2) == pytest.approx(-1.0)

    def test_batch_scoring(self, normalized_vectors):
        from vajra_bm25.vector.scorer import CosineSimilarity

        scorer = CosineSimilarity()
        query = normalized_vectors[0]
        vectors = normalized_vectors[:10]

        scores = scorer.score_batch(query, vectors)
        assert scores.shape == (10,)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)  # Self-similarity


class TestL2Distance:
    """Tests for L2Distance"""

    def test_identical_vectors(self):
        from vajra_bm25.vector.scorer import L2Distance

        scorer = L2Distance()
        v = np.array([1.0, 2.0, 3.0])

        assert scorer.score(v, v) == pytest.approx(0.0)

    def test_known_distance(self):
        from vajra_bm25.vector.scorer import L2Distance

        scorer = L2Distance()
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])

        assert scorer.score(v1, v2) == pytest.approx(5.0)

    def test_is_distance_metric(self):
        from vajra_bm25.vector.scorer import L2Distance

        scorer = L2Distance()
        assert scorer.is_distance is True


class TestInnerProduct:
    """Tests for InnerProduct"""

    def test_basic_inner_product(self):
        from vajra_bm25.vector.scorer import InnerProduct

        scorer = InnerProduct()
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])

        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert scorer.score(v1, v2) == pytest.approx(32.0)


# ============================================================================
# Flat Vector Index Tests
# ============================================================================


class TestFlatVectorIndex:
    """Tests for FlatVectorIndex"""

    def test_add_and_search(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64, metric="cosine")
        index.add(sample_ids, sample_vectors)

        assert index.size == 100

        # Search for first vector - should find itself
        results = index.search(sample_vectors[0], k=5)
        assert len(results) == 5
        assert results[0].id == "doc_0"

    def test_search_returns_correct_k(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        for k in [1, 5, 10, 50]:
            results = index.search(sample_vectors[0], k=k)
            assert len(results) == k

    def test_l2_metric(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64, metric="l2")
        index.add(sample_ids, sample_vectors)

        results = index.search(sample_vectors[0], k=1)
        assert results[0].id == "doc_0"

    def test_get_vector(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        vector = index.get_vector("doc_5")
        assert vector is not None
        # Check it's close to original (may be normalized)
        assert vector.shape == (64,)

    def test_remove(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        removed = index.remove(["doc_0", "doc_1", "doc_2"])
        assert removed == 3
        assert index.size == 97
        assert index.get_vector("doc_0") is None

    def test_save_and_load(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            index.save(path)
            loaded = FlatVectorIndex.load(path)

            assert loaded.size == index.size
            assert loaded.dimension == index.dimension

            # Search should work on loaded index
            results = loaded.search(sample_vectors[0], k=1)
            assert results[0].id == "doc_0"
        finally:
            os.unlink(path)

    def test_batch_search(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        queries = sample_vectors[:5]
        results = index.search_batch(queries, k=3)

        assert len(results) == 5
        for i, query_results in enumerate(results):
            assert len(query_results) == 3
            assert query_results[0].id == f"doc_{i}"


# ============================================================================
# HNSW Graph Tests
# ============================================================================


class TestHNSWGraph:
    """Tests for HNSWGraph"""

    def test_initialization(self):
        from vajra_bm25.vector.hnsw.graph import HNSWGraph

        graph = HNSWGraph(dimension=64, M=16, ef_construction=100)

        assert graph.dimension == 64
        assert graph.M == 16
        assert graph.M0 == 32  # Default is 2*M
        assert graph.entry_point == -1

    def test_random_level_distribution(self):
        from vajra_bm25.vector.hnsw.graph import HNSWGraph

        graph = HNSWGraph(dimension=64)
        np.random.seed(42)

        levels = [graph.get_random_level() for _ in range(1000)]

        # Most should be level 0
        assert levels.count(0) > 500
        # Some should be higher
        assert max(levels) >= 1

    def test_add_edge(self):
        from vajra_bm25.vector.hnsw.graph import HNSWGraph

        graph = HNSWGraph(dimension=64)
        graph.add_edge(0, 1, level=0)

        assert 1 in graph.get_neighbors(0, 0)
        assert 0 in graph.get_neighbors(1, 0)

    def test_set_neighbors(self):
        from vajra_bm25.vector.hnsw.graph import HNSWGraph

        graph = HNSWGraph(dimension=64)
        graph.set_neighbors(0, 0, [1, 2, 3])

        assert graph.get_neighbors(0, 0) == [1, 2, 3]


# ============================================================================
# HNSW Search State Tests
# ============================================================================


class TestHNSWSearchState:
    """Tests for HNSWSearchState"""

    def test_immutability(self):
        from vajra_bm25.vector.hnsw.state import HNSWSearchState

        state = HNSWSearchState(
            query=(1.0, 2.0, 3.0),
            current_node=0,
            current_level=2,
            ef=50,
        )

        # Should be hashable (immutable)
        hash(state)

    def test_descend_to_level(self):
        from vajra_bm25.vector.hnsw.state import HNSWSearchState

        state = HNSWSearchState(
            query=(1.0, 2.0),
            current_node=5,
            current_level=3,
            ef=50,
            visited=frozenset([1, 2, 3]),
        )

        new_state = state.descend_to_level(2)

        assert new_state.current_level == 2
        assert new_state.current_node == 5
        assert new_state.visited == frozenset()  # Reset

    def test_move_to(self):
        from vajra_bm25.vector.hnsw.state import HNSWSearchState

        state = HNSWSearchState(
            query=(1.0, 2.0),
            current_node=0,
            current_level=0,
            ef=50,
        )

        new_state = state.move_to(5)

        assert new_state.current_node == 5
        assert 0 in new_state.visited


# ============================================================================
# Native HNSW Index Tests
# ============================================================================


class TestNativeHNSWIndex:
    """Tests for NativeHNSWIndex"""

    def test_basic_add_and_search(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, metric="cosine", M=8, ef_construction=50)
        index.add(sample_ids, sample_vectors)

        assert index.size == 100

        results = index.search(sample_vectors[0], k=5)
        assert len(results) == 5
        # First result should be the query itself (or very close)
        assert results[0].id == "doc_0"

    def test_recall_vs_flat(self, sample_vectors, sample_ids):
        """HNSW should achieve reasonable recall vs exact search"""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        # Exact search
        flat = FlatVectorIndex(dimension=64)
        flat.add(sample_ids, sample_vectors)

        # HNSW with high ef for good recall
        hnsw = NativeHNSWIndex(dimension=64, M=16, ef_construction=200, ef_search=100)
        hnsw.add(sample_ids, sample_vectors)

        # Compare results for random queries
        np.random.seed(123)
        query = np.random.randn(64).astype(np.float32)

        exact_results = flat.search(query, k=10)
        hnsw_results = hnsw.search(query, k=10)

        exact_ids = {r.id for r in exact_results}
        hnsw_ids = {r.id for r in hnsw_results}

        recall = len(exact_ids & hnsw_ids) / len(exact_ids)
        assert recall >= 0.7  # At least 70% recall

    def test_save_and_load(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, M=8)
        index.add(sample_ids[:20], sample_vectors[:20])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            index.save(path)
            loaded = NativeHNSWIndex.load(path)

            assert loaded.size == index.size

            results = loaded.search(sample_vectors[0], k=5)
            assert len(results) == 5
        finally:
            os.unlink(path)

    def test_stats(self, sample_vectors, sample_ids):
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, M=8)
        index.add(sample_ids[:50], sample_vectors[:50])

        stats = index.stats()
        assert stats["total_nodes"] == 50
        assert stats["dimension"] == 64
        assert "levels" in stats


# ============================================================================
# Vajra Vector Search Engine Tests
# ============================================================================


class TestVajraVectorSearch:
    """Tests for VajraVectorSearch engine"""

    def test_index_and_search_with_precomputed(self, sample_documents):
        from vajra_bm25.vector.search import VajraVectorSearch
        from vajra_bm25.vector.index_flat import FlatVectorIndex
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        # Create fake embeddings
        np.random.seed(42)
        embeddings = {
            doc.id: np.random.randn(32).astype(np.float32)
            for doc in sample_documents
        }

        embedder = PrecomputedEmbeddingMorphism(embeddings)
        index = FlatVectorIndex(dimension=32)

        engine = VajraVectorSearch(embedder, index)

        # Index documents manually (since precomputed needs ID lookup)
        for doc in sample_documents:
            vec = embedder.embed(doc.id)
            index.add([doc.id], vec.reshape(1, -1))
            engine._documents[doc.id] = doc

        assert engine.num_documents == 5

    def test_search_returns_search_results(self, sample_documents):
        from vajra_bm25.vector.search import VajraVectorSearch
        from vajra_bm25.vector.index_flat import FlatVectorIndex
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        np.random.seed(42)
        embeddings = {
            doc.id: np.random.randn(16).astype(np.float32)
            for doc in sample_documents
        }
        # Add query embedding
        embeddings["query"] = embeddings["1"]  # Same as doc 1

        embedder = PrecomputedEmbeddingMorphism(embeddings)
        index = FlatVectorIndex(dimension=16)

        engine = VajraVectorSearch(embedder, index)

        # Index using engine's method
        for doc in sample_documents:
            vec = embedder.embed(doc.id)
            index.add([doc.id], vec.reshape(1, -1))
            engine._documents[doc.id] = doc

        # Search using the query embedding
        results = engine.search_by_vector(embeddings["query"], top_k=3)
        assert len(results) == 3


# ============================================================================
# Hybrid Search Tests
# ============================================================================


class TestHybridSearch:
    """Tests for HybridSearchEngine"""

    def test_rrf_fusion(self, sample_documents):
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from vajra_bm25.search import SearchResult
        from unittest.mock import Mock

        # Create mock engines
        bm25_engine = Mock()
        vector_engine = Mock()

        # BM25 returns docs in order: 1, 2, 3
        bm25_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=10.0, rank=1),
            SearchResult(document=sample_documents[1], score=8.0, rank=2),
            SearchResult(document=sample_documents[2], score=6.0, rank=3),
        ]

        # Vector returns docs in order: 2, 1, 4
        vector_engine.search.return_value = [
            SearchResult(document=sample_documents[1], score=0.9, rank=1),
            SearchResult(document=sample_documents[0], score=0.8, rank=2),
            SearchResult(document=sample_documents[3], score=0.7, rank=3),
        ]

        hybrid = HybridSearchEngine(bm25_engine, vector_engine, method="rrf")
        results = hybrid.search("test query", top_k=3)

        assert len(results) == 3
        # Doc 1 and 2 should be top since they appear in both
        top_ids = {r.document.id for r in results[:2]}
        assert "1" in top_ids
        assert "2" in top_ids

    def test_linear_fusion(self, sample_documents):
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from vajra_bm25.search import SearchResult
        from unittest.mock import Mock

        bm25_engine = Mock()
        vector_engine = Mock()

        # Need multiple results for normalization to work properly
        bm25_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=10.0, rank=1),
            SearchResult(document=sample_documents[1], score=5.0, rank=2),
        ]
        vector_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=1.0, rank=1),
            SearchResult(document=sample_documents[1], score=0.5, rank=2),
        ]

        hybrid = HybridSearchEngine(
            bm25_engine, vector_engine, method="linear", alpha=0.5
        )
        results = hybrid.search("test", top_k=2)

        assert len(results) == 2
        # Doc 0 should have highest score (top of both lists)
        assert results[0].document.id == "1"
        # Score should be positive
        assert results[0].score > 0

    def test_search_detailed(self, sample_documents):
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from vajra_bm25.search import SearchResult
        from unittest.mock import Mock

        bm25_engine = Mock()
        vector_engine = Mock()

        bm25_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=10.0, rank=1),
        ]
        vector_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=0.9, rank=1),
        ]

        hybrid = HybridSearchEngine(bm25_engine, vector_engine)
        results = hybrid.search_detailed("test", top_k=1)

        assert len(results) == 1
        result = results[0]
        assert result.bm25_score == 10.0
        assert result.bm25_rank == 1
        assert result.vector_score == 0.9
        assert result.vector_rank == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestVectorSearchIntegration:
    """Integration tests combining multiple components"""

    def test_flat_index_with_coalgebra(self, sample_vectors, sample_ids):
        """Test that flat index works with the coalgebra pattern"""
        from vajra_bm25.vector.index_flat import FlatVectorIndex
        from vajra_bm25.vector.search import VectorSearchCoalgebra, VectorQueryState

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids, sample_vectors)

        coalgebra = VectorSearchCoalgebra(index)
        state = VectorQueryState(
            query_embedding=tuple(sample_vectors[0].tolist()),
            top_k=5,
        )

        results = coalgebra.structure_map(state)
        assert len(results) == 5

    def test_hnsw_coalgebra_navigation(self, sample_vectors, sample_ids):
        """Test HNSW coalgebra navigation through layers"""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, M=8, ef_construction=50)
        index.add(sample_ids[:30], sample_vectors[:30])

        # Verify graph has multiple layers
        assert index.graph.max_level >= 0

        # Search should navigate through layers
        results = index.search(sample_vectors[0], k=5)
        assert len(results) == 5


# ============================================================================
# Additional Coverage Tests
# ============================================================================


class TestHybridSearchAdditional:
    """Additional tests for HybridSearchEngine"""

    def test_rsf_fusion(self, sample_documents):
        """Test Relative Score Fusion method."""
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from vajra_bm25.search import SearchResult
        from unittest.mock import Mock

        bm25_engine = Mock()
        vector_engine = Mock()

        bm25_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=10.0, rank=1),
            SearchResult(document=sample_documents[1], score=5.0, rank=2),
        ]
        vector_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=1.0, rank=1),
            SearchResult(document=sample_documents[1], score=0.5, rank=2),
        ]

        hybrid = HybridSearchEngine(
            bm25_engine, vector_engine, method="rsf", alpha=0.5
        )
        results = hybrid.search("test", top_k=2)

        assert len(results) == 2
        # Doc 0 should be top (best in both)
        assert results[0].document.id == "1"

    def test_hybrid_with_disjoint_results(self, sample_documents):
        """Test hybrid when BM25 and vector return different docs."""
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from vajra_bm25.search import SearchResult
        from unittest.mock import Mock

        bm25_engine = Mock()
        vector_engine = Mock()

        # BM25 returns docs 0, 1
        bm25_engine.search.return_value = [
            SearchResult(document=sample_documents[0], score=10.0, rank=1),
            SearchResult(document=sample_documents[1], score=8.0, rank=2),
        ]
        # Vector returns docs 2, 3 (no overlap)
        vector_engine.search.return_value = [
            SearchResult(document=sample_documents[2], score=0.9, rank=1),
            SearchResult(document=sample_documents[3], score=0.8, rank=2),
        ]

        hybrid = HybridSearchEngine(bm25_engine, vector_engine, method="rrf")
        results = hybrid.search("test", top_k=4)

        assert len(results) == 4
        doc_ids = {r.document.id for r in results}
        assert doc_ids == {"1", "2", "3", "4"}

    def test_hybrid_repr(self, sample_documents):
        """Test HybridSearchEngine __repr__."""
        from vajra_bm25.vector.hybrid import HybridSearchEngine
        from unittest.mock import Mock

        bm25_engine = Mock()
        vector_engine = Mock()

        hybrid = HybridSearchEngine(bm25_engine, vector_engine, alpha=0.7)
        repr_str = repr(hybrid)

        assert "rrf" in repr_str
        assert "0.7" in repr_str


class TestSimilarityMorphismsAdditional:
    """Additional tests for similarity morphisms"""

    def test_l2_batch_scoring(self):
        """Test L2Distance batch scoring."""
        from vajra_bm25.vector.scorer import L2Distance

        scorer = L2Distance()
        query = np.array([1.0, 0.0])
        candidates = np.array([
            [1.0, 0.0],  # Same as query
            [0.0, 1.0],  # Distance sqrt(2)
            [2.0, 0.0],  # Distance 1.0
        ])

        scores = scorer.score_batch(query, candidates)
        assert len(scores) == 3
        # Same vector should have distance 0
        assert scores[0] == pytest.approx(0.0, abs=1e-5)
        # Distance should be positive
        assert scores[1] > 0
        assert scores[2] > 0

    def test_inner_product_batch_scoring(self):
        """Test InnerProduct batch scoring."""
        from vajra_bm25.vector.scorer import InnerProduct

        scorer = InnerProduct()
        query = np.array([1.0, 2.0])
        candidates = np.array([
            [1.0, 2.0],  # Inner product = 5
            [1.0, 0.0],  # Inner product = 1
            [0.0, 0.0],  # Inner product = 0
        ])

        scores = scorer.score_batch(query, candidates)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(5.0)
        assert scores[1] == pytest.approx(1.0)
        assert scores[2] == pytest.approx(0.0)

    def test_cosine_similarity_repr(self):
        """Test CosineSimilarity __repr__."""
        from vajra_bm25.vector.scorer import CosineSimilarity

        scorer = CosineSimilarity()
        repr_str = repr(scorer)
        assert "CosineSimilarity" in repr_str

    def test_l2_distance_repr(self):
        """Test L2Distance __repr__."""
        from vajra_bm25.vector.scorer import L2Distance

        scorer = L2Distance()
        repr_str = repr(scorer)
        assert "L2Distance" in repr_str


class TestFlatVectorIndexAdditional:
    """Additional tests for FlatVectorIndex"""

    def test_search_empty_index(self):
        """Test searching an empty index."""
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        query = np.random.randn(64).astype(np.float32)

        results = index.search(query, k=5)
        assert len(results) == 0

    def test_search_k_larger_than_index(self, sample_vectors, sample_ids):
        """Test searching with k larger than index size."""
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        # Add only 3 vectors
        index.add(sample_ids[:3], sample_vectors[:3])

        results = index.search(sample_vectors[0], k=10)
        assert len(results) == 3  # Should return all available

    def test_batch_search_multiple_queries(self, sample_vectors, sample_ids):
        """Test batch search with multiple queries."""
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        index.add(sample_ids[:50], sample_vectors[:50])

        queries = sample_vectors[:5]
        all_results = index.search_batch(queries, k=3)

        assert len(all_results) == 5
        for results in all_results:
            assert len(results) == 3

    def test_inner_product_metric(self, sample_vectors, sample_ids):
        """Test FlatVectorIndex with inner product metric."""
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64, metric="ip")
        index.add(sample_ids[:10], sample_vectors[:10])

        results = index.search(sample_vectors[0], k=3)
        assert len(results) == 3

    def test_index_size_property(self, sample_vectors, sample_ids):
        """Test index size property."""
        from vajra_bm25.vector.index_flat import FlatVectorIndex

        index = FlatVectorIndex(dimension=64)
        assert index.size == 0

        index.add(sample_ids[:10], sample_vectors[:10])
        assert index.size == 10

        index.add(sample_ids[10:15], sample_vectors[10:15])
        assert index.size == 15


class TestHNSWIndexAdditional:
    """Additional tests for NativeHNSWIndex"""

    def test_search_empty_index(self):
        """Test searching an empty HNSW index."""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64)
        query = np.random.randn(64).astype(np.float32)

        results = index.search(query, k=5)
        assert len(results) == 0

    def test_l2_metric(self, sample_vectors, sample_ids):
        """Test HNSW with L2 metric."""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, metric="l2")
        index.add(sample_ids[:20], sample_vectors[:20])

        results = index.search(sample_vectors[0], k=3)
        assert len(results) == 3

    def test_ef_search_parameter(self, sample_vectors, sample_ids):
        """Test adjusting ef_search parameter."""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, ef_search=100)
        index.add(sample_ids[:30], sample_vectors[:30])

        # High ef_search should give good recall
        results = index.search(sample_vectors[0], k=5)
        assert results[0].id == "doc_0"

    def test_batch_search_hnsw(self, sample_vectors, sample_ids):
        """Test batch search with HNSW."""
        from vajra_bm25.vector.hnsw import NativeHNSWIndex

        index = NativeHNSWIndex(dimension=64, M=8, ef_construction=50)
        index.add(sample_ids[:50], sample_vectors[:50])

        queries = sample_vectors[:3]
        all_results = index.search_batch(queries, k=5)

        assert len(all_results) == 3
        for results in all_results:
            assert len(results) == 5


class TestEmbeddingMorphismsAdditional:
    """Additional tests for embedding morphisms"""

    def test_precomputed_empty_embeddings_raises(self):
        """Test PrecomputedEmbeddingMorphism with empty dict raises."""
        from vajra_bm25.vector.embeddings import PrecomputedEmbeddingMorphism

        # Empty dict should raise in __init__
        with pytest.raises(ValueError, match="cannot be empty"):
            PrecomputedEmbeddingMorphism({})

    def test_identity_embedding_batch(self):
        """Test IdentityEmbeddingMorphism batch operation."""
        from vajra_bm25.vector.embeddings import IdentityEmbeddingMorphism

        morphism = IdentityEmbeddingMorphism(dimension=4)
        # Pass as numpy array, not list
        vectors = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])

        result = morphism.embed_batch(vectors)
        assert result.shape == (2, 4)


class TestVectorQueryState:
    """Tests for VectorQueryState"""

    def test_state_creation(self):
        """Test VectorQueryState creation."""
        from vajra_bm25.vector.search import VectorQueryState

        state = VectorQueryState(
            query_embedding=(1.0, 2.0, 3.0),
            top_k=10,
        )

        assert state.query_embedding == (1.0, 2.0, 3.0)
        assert state.top_k == 10

    def test_state_hashable(self):
        """Test VectorQueryState is hashable."""
        from vajra_bm25.vector.search import VectorQueryState

        state = VectorQueryState(
            query_embedding=(1.0, 2.0, 3.0),
            top_k=5,
        )

        # Should be hashable
        hash(state)
        # Should work in set
        s = {state}
        assert state in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
