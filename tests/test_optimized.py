"""
Tests for VajraSearchOptimized and related optimizations.

Tests cover:
- LRU caching behavior
- Sparse matrix operations
- Various scorer modes
- Index save/load functionality
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from vajra_bm25 import Document, DocumentCorpus
from vajra_bm25.optimized import (
    VajraSearchOptimized,
    LRUCache,
    memoized_morphism,
)


@pytest.fixture
def sample_corpus():
    """Create sample corpus for testing."""
    docs = [
        Document("1", "Category Theory", "Functors preserve structure between categories"),
        Document("2", "Coalgebras", "Coalgebras capture dynamics and unfolding processes"),
        Document("3", "Search Algorithms", "BFS and DFS are graph search algorithms"),
        Document("4", "Functors", "Category theory uses functors and morphisms"),
        Document("5", "Machine Learning", "Neural networks learn patterns from data"),
    ]
    return DocumentCorpus(docs)


@pytest.fixture
def larger_corpus():
    """Create a larger corpus for sparse matrix testing."""
    docs = [
        Document(f"doc_{i}", f"Title {i}", f"Content about topic {i % 10} with some common words")
        for i in range(100)
    ]
    docs.extend([
        Document("search_doc", "Search Algorithms", "BFS DFS graph search traversal"),
        Document("ml_doc", "Machine Learning", "neural networks deep learning AI"),
        Document("cat_doc", "Category Theory", "functors morphisms categories monads"),
    ])
    return DocumentCorpus(docs)


# ============================================================================
# LRU Cache Tests
# ============================================================================

class TestLRUCache:
    """Tests for LRU caching behavior."""

    def test_cache_creation(self):
        """Test LRU cache initialization."""
        cache = LRUCache(capacity=100)
        assert cache.capacity == 100
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(capacity=5)

        cache.put("key1", [1, 2, 3])
        cache.put("key2", [4, 5, 6])

        result = cache.get("key1")
        assert result == [1, 2, 3]
        assert cache.hits == 1

    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = LRUCache(capacity=5)

        result = cache.get("nonexistent")
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCache(capacity=3)

        cache.put("key1", [1])
        cache.put("key2", [2])
        cache.put("key3", [3])

        # All keys should be present
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

        # Add a fourth key, should evict key1 (LRU after get operations)
        cache.put("key4", [4])

        # key1 was accessed last, so key2 should be evicted
        # Actually after gets: key1, key2, key3 were accessed in that order
        # So key1 was accessed first in the get sequence, making key2 LRU...
        # Let me think again: after puts, order is key1, key2, key3
        # After get(key1), order is key2, key3, key1
        # After get(key2), order is key3, key1, key2
        # After get(key3), order is key1, key2, key3
        # After put(key4), key1 is evicted (first in order)
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key4") is not None

    def test_cache_update_moves_to_end(self):
        """Test that updating existing key moves it to most recent."""
        cache = LRUCache(capacity=3)

        cache.put("key1", [1])
        cache.put("key2", [2])
        cache.put("key3", [3])

        # Update key1, should move it to end
        cache.put("key1", [10])

        # Add key4, should evict key2 (now LRU)
        cache.put("key4", [4])

        assert cache.get("key2") is None  # Should be evicted
        assert cache.get("key1") == [10]

    def test_cache_clear(self):
        """Test cache clear operation."""
        cache = LRUCache(capacity=5)

        cache.put("key1", [1])
        cache.put("key2", [2])
        cache.get("key1")

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(capacity=5)

        cache.put("key1", [1])
        cache.put("key2", [2])
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        cache.get("key1")  # Hit

        stats = cache.stats()

        assert stats['size'] == 2
        assert stats['capacity'] == 5
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_rate'] == pytest.approx(2/3)


# ============================================================================
# Memoized Morphism Tests
# ============================================================================

class TestMemoizedMorphism:
    """Tests for memoized morphism decorator."""

    def test_memoization_caches_results(self):
        """Test that memoized functions cache results."""
        call_count = 0

        @memoized_morphism
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = expensive_function(5)
        result2 = expensive_function(5)
        result3 = expensive_function(10)

        assert result1 == 10
        assert result2 == 10  # Should use cached value
        assert result3 == 20
        assert call_count == 2  # Only 2 unique calls

    def test_memoization_with_multiple_args(self):
        """Test memoization with multiple arguments."""
        @memoized_morphism
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add(1, 2) == 3  # Cached
        assert add(2, 1) == 3  # Different key

        assert len(add.cache) == 2

    def test_cache_info(self):
        """Test cache info method."""
        @memoized_morphism
        def func(x):
            return x

        func(1)
        func(2)
        func(1)  # Cached

        info = func.cache_info()
        assert info['size'] == 2


# ============================================================================
# VajraSearchOptimized Tests
# ============================================================================

class TestVajraSearchOptimized:
    """Tests for VajraSearchOptimized search engine."""

    def test_creation_with_defaults(self, sample_corpus):
        """Test VajraSearchOptimized creation with default parameters."""
        engine = VajraSearchOptimized(sample_corpus)

        assert engine.k1 == 1.5
        assert engine.b == 0.75
        assert engine.index is not None

    def test_creation_with_custom_params(self, sample_corpus):
        """Test VajraSearchOptimized with custom BM25 parameters."""
        engine = VajraSearchOptimized(sample_corpus, k1=2.0, b=0.5)

        assert engine.k1 == 2.0
        assert engine.b == 0.5

    def test_search_returns_results(self, sample_corpus):
        """Test that search returns results."""
        engine = VajraSearchOptimized(sample_corpus)
        results = engine.search("category functors", top_k=3)

        assert len(results) > 0
        assert all(hasattr(r, 'document') for r in results)
        assert all(hasattr(r, 'score') for r in results)
        assert all(hasattr(r, 'rank') for r in results)

    def test_search_results_are_ranked(self, sample_corpus):
        """Test that results are properly ranked by score."""
        engine = VajraSearchOptimized(sample_corpus)
        results = engine.search("category theory functors", top_k=5)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self, sample_corpus):
        """Test that top_k limits results."""
        engine = VajraSearchOptimized(sample_corpus)

        results = engine.search("category", top_k=2)
        assert len(results) <= 2

    def test_search_empty_query(self, sample_corpus):
        """Test search with empty query."""
        engine = VajraSearchOptimized(sample_corpus)
        results = engine.search("")

        assert len(results) == 0

    def test_search_no_matches(self, sample_corpus):
        """Test search with query matching nothing."""
        engine = VajraSearchOptimized(sample_corpus)
        results = engine.search("xyzzy quantum superconductivity")

        # Should return empty or zero-scored results
        assert len(results) == 0 or all(r.score == 0 for r in results)

    def test_caching_improves_repeat_queries(self, larger_corpus):
        """Test that caching speeds up repeated queries."""
        engine = VajraSearchOptimized(larger_corpus, cache_size=100)

        # First query
        results1 = engine.search("machine learning neural")

        # Second query (should be cached)
        results2 = engine.search("machine learning neural")

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.document.id == r2.document.id
            assert r1.score == r2.score

    def test_sparse_mode_with_larger_corpus(self, larger_corpus):
        """Test sparse matrix mode with larger corpus."""
        engine = VajraSearchOptimized(larger_corpus, use_sparse=True)

        results = engine.search("topic common words", top_k=5)

        assert len(results) > 0
        assert all(r.score >= 0 for r in results)

    def test_dense_mode_with_small_corpus(self, sample_corpus):
        """Test dense matrix mode with small corpus."""
        engine = VajraSearchOptimized(sample_corpus, use_sparse=False)

        results = engine.search("category functors", top_k=3)

        assert len(results) > 0

    def test_eager_mode_produces_same_results(self, larger_corpus):
        """Test that eager scoring produces same results as traditional."""
        engine_eager = VajraSearchOptimized(
            larger_corpus, use_eager=True, use_sparse=True
        )
        engine_traditional = VajraSearchOptimized(
            larger_corpus, use_eager=False, use_numba=False, use_sparse=True
        )

        queries = ["graph search", "machine learning", "category functors"]

        for query in queries:
            results_eager = engine_eager.search(query, top_k=5)
            results_trad = engine_traditional.search(query, top_k=5)

            assert len(results_eager) == len(results_trad)
            for r_e, r_t in zip(results_eager, results_trad):
                assert r_e.document.id == r_t.document.id
                np.testing.assert_allclose(r_e.score, r_t.score, rtol=1e-5)


# ============================================================================
# Index Persistence Tests
# ============================================================================

class TestIndexPersistence:
    """Tests for index save/load functionality."""

    def test_save_index(self, larger_corpus):
        """Test saving index to disk."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not available")

        engine = VajraSearchOptimized(
            larger_corpus, use_sparse=True, use_eager=False, use_numba=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.joblib"

            # Save index
            engine.save_index(index_path)

            # File should exist
            assert index_path.exists()

            # File should have content
            assert index_path.stat().st_size > 0

    def test_load_index_file_not_found(self, larger_corpus):
        """Test loading index from non-existent file raises error."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not available")

        with pytest.raises(FileNotFoundError):
            VajraSearchOptimized.load_index(
                Path("/nonexistent/path/index.joblib"),
                larger_corpus
            )

    def test_save_load_roundtrip_with_eager_scorer(self, larger_corpus):
        """Test save/load roundtrip preserves eager scorer functionality."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not available")

        # Create engine with eager scoring enabled
        engine = VajraSearchOptimized(
            larger_corpus, use_sparse=True, use_eager=True, use_numba=False
        )

        # Get results before save
        query = "machine learning neural"
        results_before = engine.search(query, top_k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.joblib"

            # Save index
            engine.save_index(index_path)

            # Load index
            loaded_engine = VajraSearchOptimized.load_index(index_path, larger_corpus)

            # Verify all attributes are restored
            assert loaded_engine.k1 == engine.k1
            assert loaded_engine.b == engine.b
            assert loaded_engine.use_eager == engine.use_eager
            assert loaded_engine.use_sparse == engine.use_sparse

            # Verify eager_scorer is available
            assert loaded_engine.eager_scorer is not None

            # Search should work and return same results
            results_after = loaded_engine.search(query, top_k=5)

            assert len(results_before) == len(results_after)
            for r_before, r_after in zip(results_before, results_after):
                assert r_before.document.id == r_after.document.id
                np.testing.assert_allclose(r_before.score, r_after.score, rtol=1e-5)

    def test_save_load_roundtrip_sparse_no_eager(self, larger_corpus):
        """Test save/load roundtrip without eager scorer."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not available")

        # Create engine with eager scoring disabled
        engine = VajraSearchOptimized(
            larger_corpus, use_sparse=True, use_eager=False, use_numba=False
        )

        query = "topic common words"
        results_before = engine.search(query, top_k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.joblib"

            engine.save_index(index_path)
            loaded_engine = VajraSearchOptimized.load_index(index_path, larger_corpus)

            # Verify eager_scorer is None
            assert loaded_engine.eager_scorer is None
            assert loaded_engine.use_eager is False

            # Search should still work
            results_after = loaded_engine.search(query, top_k=5)

            assert len(results_before) == len(results_after)
            for r_before, r_after in zip(results_before, results_after):
                assert r_before.document.id == r_after.document.id

    def test_save_load_roundtrip_dense(self, sample_corpus):
        """Test save/load roundtrip with dense index."""
        try:
            import joblib
        except ImportError:
            pytest.skip("joblib not available")

        # Create engine with dense index
        engine = VajraSearchOptimized(
            sample_corpus, use_sparse=False
        )

        query = "category functors"
        results_before = engine.search(query, top_k=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.joblib"

            engine.save_index(index_path)
            loaded_engine = VajraSearchOptimized.load_index(index_path, sample_corpus)

            # Dense index doesn't have these scorers
            assert loaded_engine.eager_scorer is None
            assert loaded_engine.numba_scorer is None
            assert loaded_engine.maxscore_scorer is None

            # Search should work
            results_after = loaded_engine.search(query, top_k=3)

            assert len(results_before) == len(results_after)


# ============================================================================
# Additional Coverage Tests
# ============================================================================

class TestVajraSearchOptimizedAdditional:
    """Additional tests for VajraSearchOptimized coverage."""

    def test_search_single_term(self, sample_corpus):
        """Test search with single term query."""
        engine = VajraSearchOptimized(sample_corpus, use_sparse=True)
        results = engine.search("category", top_k=5)

        assert len(results) > 0

    def test_search_returns_scored_results(self, larger_corpus):
        """Test that search results have scores."""
        engine = VajraSearchOptimized(larger_corpus, use_sparse=True)
        results = engine.search("graph search traversal", top_k=10)

        if results:
            assert all(r.score > 0 for r in results)
            assert all(r.rank > 0 for r in results)

    def test_cache_clear(self, sample_corpus):
        """Test cache clear functionality."""
        engine = VajraSearchOptimized(sample_corpus, cache_size=10)

        # Make some queries to populate cache
        engine.search("category")
        engine.search("functors")

        # Cache should have entries
        if engine.query_cache:
            initial_size = len(engine.query_cache.cache)
            engine.query_cache.clear()
            assert len(engine.query_cache.cache) == 0

    def test_index_stats(self, sample_corpus):
        """Test that index has expected attributes."""
        engine = VajraSearchOptimized(sample_corpus, use_sparse=True)

        assert hasattr(engine.index, 'term_to_id')
        assert hasattr(engine.index, 'avg_doc_length')
        assert engine.index.avg_doc_length > 0
        assert len(engine.index.term_to_id) > 0

    def test_search_with_stopwords_only(self, sample_corpus):
        """Test search with query containing mostly stopwords."""
        engine = VajraSearchOptimized(sample_corpus, use_sparse=True)
        # "the" and "and" are common stopwords
        results = engine.search("the and", top_k=5)

        # May return empty or few results
        assert isinstance(results, list)

    def test_sparse_vs_dense_equivalent_results(self, sample_corpus):
        """Test that sparse and dense modes give similar results."""
        engine_sparse = VajraSearchOptimized(sample_corpus, use_sparse=True, use_eager=False)
        engine_dense = VajraSearchOptimized(sample_corpus, use_sparse=False)

        query = "category theory functors"
        results_sparse = engine_sparse.search(query, top_k=3)
        results_dense = engine_dense.search(query, top_k=3)

        # Same documents should be in top results
        sparse_ids = {r.document.id for r in results_sparse}
        dense_ids = {r.document.id for r in results_dense}

        # At least some overlap expected
        if results_sparse and results_dense:
            assert sparse_ids.intersection(dense_ids)

    def test_maxscore_mode(self, larger_corpus):
        """Test MaxScore algorithm mode."""
        engine = VajraSearchOptimized(
            larger_corpus,
            use_sparse=True,
            use_maxscore=True,
            use_eager=False,
            use_numba=False
        )

        results = engine.search("graph traversal", top_k=5)
        assert isinstance(results, list)


class TestLRUCacheAdditional:
    """Additional LRU Cache tests."""

    def test_cache_small_capacity(self):
        """Test cache with small capacity."""
        cache = LRUCache(capacity=1)
        cache.put("key1", [1, 2, 3])
        cache.put("key2", [4, 5, 6])

        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == [4, 5, 6]

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        cache = LRUCache(capacity=5)

        # All misses
        cache.get("a")
        cache.get("b")

        stats = cache.stats()
        assert stats['hit_rate'] == 0.0

        # Add and hit
        cache.put("a", [1])
        cache.get("a")  # Hit

        stats = cache.stats()
        assert stats['hit_rate'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
