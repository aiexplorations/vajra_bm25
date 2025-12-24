"""
Tests for VajraSearchParallel - parallel batch query processing.

Tests cover:
- Batch search functionality
- Thread pool behavior
- Result ordering preservation
- Performance characteristics
"""

import pytest
import time

from vajra_bm25 import Document, DocumentCorpus
from vajra_bm25.parallel import VajraSearchParallel


@pytest.fixture
def sample_corpus():
    """Create sample corpus for testing."""
    docs = [
        Document("1", "Category Theory", "Functors preserve structure between categories"),
        Document("2", "Coalgebras", "Coalgebras capture dynamics and unfolding processes"),
        Document("3", "Search Algorithms", "BFS and DFS are graph search algorithms"),
        Document("4", "Functors", "Category theory uses functors and morphisms"),
        Document("5", "Machine Learning", "Neural networks learn patterns from data"),
        Document("6", "Deep Learning", "Deep neural networks with multiple layers"),
        Document("7", "Graph Theory", "Graphs consist of vertices and edges"),
        Document("8", "Data Structures", "Arrays lists trees and hash tables"),
    ]
    return DocumentCorpus(docs)


@pytest.fixture
def larger_corpus():
    """Create a larger corpus for performance testing."""
    docs = [
        Document(f"doc_{i}", f"Title {i}", f"Content about topic {i % 10} with some common words")
        for i in range(200)
    ]
    docs.extend([
        Document("search_doc", "Search Algorithms", "BFS DFS graph search traversal"),
        Document("ml_doc", "Machine Learning", "neural networks deep learning AI"),
        Document("cat_doc", "Category Theory", "functors morphisms categories monads"),
    ])
    return DocumentCorpus(docs)


class TestVajraSearchParallel:
    """Tests for VajraSearchParallel search engine."""

    def test_creation(self, sample_corpus):
        """Test VajraSearchParallel creation."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        assert engine.max_workers == 2
        assert engine.corpus == sample_corpus

    def test_single_query_search(self, sample_corpus):
        """Test that single query search works (inherited from parent)."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        results = engine.search("category functors", top_k=3)

        assert len(results) > 0
        assert all(hasattr(r, 'document') for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_batch_search_returns_correct_count(self, sample_corpus):
        """Test that batch search returns results for all queries."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["category", "machine learning", "graph"]
        batch_results = engine.search_batch(queries, top_k=3)

        assert len(batch_results) == len(queries)

    def test_batch_search_preserves_order(self, sample_corpus):
        """Test that batch results maintain query order."""
        engine = VajraSearchParallel(sample_corpus, max_workers=4)

        queries = [
            "category functors",
            "machine learning neural",
            "graph algorithms search",
            "deep learning",
        ]
        batch_results = engine.search_batch(queries, top_k=3)

        # Each result set should be related to its query
        # First query about category should match doc 1 or 4
        if batch_results[0]:
            doc_ids = [r.document.id for r in batch_results[0]]
            assert "1" in doc_ids or "4" in doc_ids

        # Second query about ML should match doc 5 or 6
        if batch_results[1]:
            doc_ids = [r.document.id for r in batch_results[1]]
            assert "5" in doc_ids or "6" in doc_ids

    def test_batch_search_with_empty_query(self, sample_corpus):
        """Test batch search handles empty queries."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["category", "", "machine learning"]
        batch_results = engine.search_batch(queries, top_k=3)

        assert len(batch_results) == 3
        assert len(batch_results[1]) == 0  # Empty query

    def test_batch_search_all_no_matches(self, sample_corpus):
        """Test batch search with queries that match nothing."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["xyzzy", "superconductivity", "quantum"]
        batch_results = engine.search_batch(queries, top_k=3)

        assert len(batch_results) == 3
        for results in batch_results:
            # Should be empty or zero-scored
            assert len(results) == 0 or all(r.score == 0 for r in results)

    def test_batch_search_with_different_workers(self, sample_corpus):
        """Test batch search with different worker counts."""
        queries = ["category", "machine", "graph", "data"]

        for workers in [1, 2, 4]:
            engine = VajraSearchParallel(sample_corpus, max_workers=workers)
            batch_results = engine.search_batch(queries, top_k=3)

            assert len(batch_results) == len(queries)

    def test_batch_search_results_are_ranked(self, sample_corpus):
        """Test that each result set is properly ranked."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["category theory functors", "machine learning deep"]
        batch_results = engine.search_batch(queries, top_k=5)

        for results in batch_results:
            if len(results) > 1:
                scores = [r.score for r in results]
                assert scores == sorted(scores, reverse=True)

    def test_batch_search_respects_top_k(self, sample_corpus):
        """Test that top_k is respected in batch results."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["category", "machine", "graph"]
        batch_results = engine.search_batch(queries, top_k=2)

        for results in batch_results:
            assert len(results) <= 2

    def test_batch_search_larger_corpus(self, larger_corpus):
        """Test batch search with larger corpus."""
        engine = VajraSearchParallel(larger_corpus, max_workers=4)

        queries = [
            "topic common",
            "machine learning neural",
            "category functors",
            "graph search",
        ]
        batch_results = engine.search_batch(queries, top_k=10)

        assert len(batch_results) == 4
        for results in batch_results:
            assert len(results) <= 10

    def test_batch_search_duplicate_queries(self, sample_corpus):
        """Test batch search with duplicate queries."""
        engine = VajraSearchParallel(sample_corpus, max_workers=2)

        queries = ["category", "category", "category"]
        batch_results = engine.search_batch(queries, top_k=3)

        assert len(batch_results) == 3

        # All results should be identical
        if batch_results[0]:
            ids_0 = [r.document.id for r in batch_results[0]]
            ids_1 = [r.document.id for r in batch_results[1]]
            ids_2 = [r.document.id for r in batch_results[2]]
            assert ids_0 == ids_1 == ids_2

    def test_batch_search_many_queries(self, larger_corpus):
        """Test batch search with many queries."""
        engine = VajraSearchParallel(larger_corpus, max_workers=4)

        # Generate many queries
        queries = [f"topic {i}" for i in range(20)]
        batch_results = engine.search_batch(queries, top_k=5)

        assert len(batch_results) == 20

    def test_custom_bm25_parameters(self, sample_corpus):
        """Test parallel search with custom BM25 parameters."""
        engine = VajraSearchParallel(sample_corpus, k1=2.0, b=0.5, max_workers=2)

        results = engine.search("category functors", top_k=3)
        assert len(results) > 0

        batch_results = engine.search_batch(["category", "machine"], top_k=3)
        assert len(batch_results) == 2


class TestParallelPerformance:
    """Performance-related tests for parallel search."""

    def test_parallel_completes_in_reasonable_time(self, larger_corpus):
        """Test that parallel batch completes in reasonable time."""
        engine = VajraSearchParallel(larger_corpus, max_workers=4)

        queries = ["topic"] * 50

        start = time.time()
        batch_results = engine.search_batch(queries, top_k=5)
        elapsed = time.time() - start

        assert len(batch_results) == 50
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_worker_count_zero_defaults(self, sample_corpus):
        """Test engine creation with default workers."""
        engine = VajraSearchParallel(sample_corpus)

        # Should have some workers set
        assert engine.max_workers >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
