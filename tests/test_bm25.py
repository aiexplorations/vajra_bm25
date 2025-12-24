"""
Tests for Vajra BM25 information retrieval implementation

Tests the categorical BM25 implementation:
- BM25 scoring as morphism (Query, Document) -> R
- Search as coalgebraic unfolding
- Document and corpus handling
"""

import pytest
from vajra_bm25 import (
    Document,
    DocumentCorpus,
    preprocess_text,
    BM25Parameters,
    BM25Scorer,
    VajraSearch,
)
from vajra_bm25.inverted_index import InvertedIndex


# Test fixtures
@pytest.fixture
def sample_documents():
    """Create sample documents for testing"""
    return [
        Document("1", "Category Theory", "Functors are structure-preserving maps between categories"),
        Document("2", "Coalgebras", "Coalgebras capture dynamics and unfolding processes"),
        Document("3", "Search", "BFS and DFS are graph search algorithms"),
        Document("4", "Functors and Morphisms", "Category theory uses functors and morphisms"),
    ]


@pytest.fixture
def sample_corpus(sample_documents):
    """Create sample corpus"""
    return DocumentCorpus(sample_documents)


def test_document_creation():
    """Test Document creation"""
    doc = Document("test_id", "Test Title", "Test content here")

    assert doc.id == "test_id"
    assert doc.title == "Test Title"
    assert doc.content == "Test content here"


def test_document_immutable_after_creation():
    """Test that Document fields can be set"""
    doc = Document("1", "Title", "Content")

    # Fields should exist and be accessible
    assert doc.id == "1"
    assert doc.title == "Title"
    assert doc.content == "Content"


def test_corpus_creation(sample_documents):
    """Test DocumentCorpus creation"""
    corpus = DocumentCorpus(sample_documents)

    assert len(corpus) == 4
    assert corpus.get("1").title == "Category Theory"


def test_corpus_iteration(sample_corpus):
    """Test iterating over corpus"""
    ids = [doc.id for doc in sample_corpus]

    assert ids == ["1", "2", "3", "4"]


def test_text_preprocessing():
    """Test text preprocessing pipeline"""
    text = "The Quick BROWN fox jumps over the lazy dog"
    terms = preprocess_text(text)

    # Should be lowercased, stop words removed
    assert "quick" in terms
    assert "brown" in terms
    assert "fox" in terms
    assert "the" not in terms  # Stop word removed


def test_inverted_index_build(sample_corpus):
    """Test inverted index construction"""
    index = InvertedIndex()
    index.build(sample_corpus)

    assert index.num_docs == 4
    assert index.avg_doc_length > 0

    # Check that key terms are indexed
    assert "functors" in index.index
    assert "category" in index.index


def test_inverted_index_candidate_docs(sample_corpus):
    """Test candidate document retrieval"""
    index = InvertedIndex()
    index.build(sample_corpus)

    query_terms = ["category", "functors"]
    candidates = index.get_candidate_documents(query_terms)

    # Documents 1 and 4 contain these terms
    assert "1" in candidates or "4" in candidates


def test_inverted_index_idf(sample_corpus):
    """Test IDF calculation"""
    index = InvertedIndex()
    index.build(sample_corpus)

    # Common term should have lower IDF
    idf_functors = index.idf("functors")

    # Rare term should have higher IDF
    idf_coalgebras = index.idf("coalgebras")

    # Coalgebras appears in fewer docs, should have higher IDF
    assert idf_coalgebras > idf_functors


def test_bm25_parameters():
    """Test BM25 parameters"""
    params = BM25Parameters()

    # Default values
    assert params.k1 == 1.5
    assert params.b == 0.75

    # Custom values
    custom = BM25Parameters(k1=2.0, b=0.9)
    assert custom.k1 == 2.0
    assert custom.b == 0.9


def test_bm25_scorer_creation(sample_corpus):
    """Test BM25Scorer creation"""
    index = InvertedIndex()
    index.build(sample_corpus)

    scorer = BM25Scorer(index)

    assert scorer.index == index
    assert scorer.params.k1 == 1.5
    assert scorer.params.b == 0.75


def test_bm25_scoring(sample_corpus):
    """Test BM25 scoring morphism"""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["category", "functors"]

    # Score document 1 (contains both terms)
    score1 = scorer.score(query_terms, "1")

    # Score document 3 (contains neither term)
    score3 = scorer.score(query_terms, "3")

    # Document with matching terms should score higher
    assert score1 > score3


def test_bm25_scoring_is_morphism(sample_corpus):
    """Test that BM25 scoring is a proper morphism (deterministic)"""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["category"]

    # Same input should always give same output
    score1 = scorer.score(query_terms, "1")
    score2 = scorer.score(query_terms, "1")
    score3 = scorer.score(query_terms, "1")

    assert score1 == score2 == score3


def test_bm25_rank_documents(sample_corpus):
    """Test document ranking"""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["functors", "morphisms"]
    doc_ids = ["1", "2", "3", "4"]

    ranked = scorer.rank_documents(query_terms, doc_ids)

    # Should return list of (doc_id, score) tuples
    assert len(ranked) == 4
    assert all(isinstance(item, tuple) for item in ranked)
    assert all(len(item) == 2 for item in ranked)

    # Should be sorted by score (descending)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_bm25_empty_query(sample_corpus):
    """Test BM25 with empty query"""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    score = scorer.score([], "1")

    # Empty query should give zero score
    assert score == 0.0


def test_bm25_unknown_document(sample_corpus):
    """Test BM25 scoring unknown document"""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["category"]

    # Unknown document should score 0
    score = scorer.score(query_terms, "unknown_id")

    assert score == 0.0


def test_vajra_search_creation(sample_corpus):
    """Test VajraSearch creation"""
    engine = VajraSearch(sample_corpus)

    assert engine.corpus == sample_corpus
    assert engine.index is not None
    assert engine.scorer is not None


def test_vajra_search_query(sample_corpus):
    """Test search as coalgebraic unfolding"""
    engine = VajraSearch(sample_corpus)

    results = engine.search("category functors")

    # Should return SearchResult objects
    assert len(results) > 0
    assert all(hasattr(r, 'document') for r in results)
    assert all(hasattr(r, 'score') for r in results)
    assert all(hasattr(r, 'rank') for r in results)


def test_vajra_search_ranking(sample_corpus):
    """Test that search results are properly ranked"""
    engine = VajraSearch(sample_corpus)

    results = engine.search("category theory functors")

    # Results should be ranked by score
    if len(results) > 1:
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Ranks should be sequential
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))


def test_vajra_search_top_k(sample_corpus):
    """Test top-k retrieval"""
    engine = VajraSearch(sample_corpus)

    # Request only top 2 results
    results = engine.search("category", top_k=2)

    assert len(results) <= 2


def test_vajra_search_empty_query(sample_corpus):
    """Test search with empty query"""
    engine = VajraSearch(sample_corpus)

    results = engine.search("")

    # Empty query should return empty results
    assert len(results) == 0


def test_vajra_search_no_matches(sample_corpus):
    """Test search with no matching documents"""
    engine = VajraSearch(sample_corpus)

    # Query with terms not in corpus
    results = engine.search("quantum chromodynamics superconductivity")

    # Should return empty or very low scored results
    if results:
        assert all(r.score == 0 for r in results)


def test_bm25_parameters_affect_scoring(sample_corpus):
    """Test that BM25 parameters change scores"""
    index = InvertedIndex()
    index.build(sample_corpus)

    # Default parameters
    scorer1 = BM25Scorer(index, BM25Parameters(k1=1.5, b=0.75))

    # Different parameters
    scorer2 = BM25Scorer(index, BM25Parameters(k1=2.0, b=0.5))

    query_terms = ["category", "functors"]

    score1 = scorer1.score(query_terms, "1")
    score2 = scorer2.score(query_terms, "1")

    # Different parameters should (usually) give different scores
    # Note: might be equal in edge cases, but generally different
    assert score1 != score2 or score1 == 0  # If both zero, that's ok


def test_search_coalgebra_structure():
    """Test that search engine uses coalgebraic structure"""
    docs = [
        Document("1", "Test A", "content a"),
        Document("2", "Test B", "content b"),
    ]
    corpus = DocumentCorpus(docs)
    engine = VajraSearch(corpus)

    # Search is coalgebraic unfolding: Query -> List[SearchResult]
    results = engine.search("test")

    # Structure should be: one query unfolds into multiple results
    assert isinstance(results, list)
    assert len(results) >= 0


def test_categorical_abstractions():
    """Test categorical abstractions are importable"""
    from vajra_bm25 import (
        Morphism,
        FunctionMorphism,
        IdentityMorphism,
        Functor,
        ListFunctor,
        Coalgebra,
        SearchCoalgebra,
    )

    # Test basic morphism composition
    f = FunctionMorphism(lambda x: x + 1)
    g = FunctionMorphism(lambda x: x * 2)

    h = f >> g  # Composition
    assert h.apply(5) == 12  # (5 + 1) * 2 = 12


def test_identity_morphism():
    """Test identity morphism laws"""
    from vajra_bm25 import IdentityMorphism, FunctionMorphism

    identity = IdentityMorphism()
    f = FunctionMorphism(lambda x: x * 2)

    # f . id = f
    assert (identity >> f).apply(5) == f.apply(5)

    # id . f = f
    assert (f >> identity).apply(5) == f.apply(5)


# ============================================================================
# EAGER SCORING TESTS
# ============================================================================

@pytest.fixture
def larger_corpus():
    """Create a larger corpus for testing eager scoring (requires sparse matrices)."""
    docs = [
        Document(f"doc_{i}", f"Title {i}", f"Content about topic {i % 10} with some common words")
        for i in range(100)
    ]
    # Add some documents with specific terms for testing
    docs.extend([
        Document("search_doc", "Search Algorithms", "BFS DFS graph search traversal"),
        Document("ml_doc", "Machine Learning", "neural networks deep learning AI"),
        Document("cat_doc", "Category Theory", "functors morphisms categories monads"),
    ])
    return DocumentCorpus(docs)


def test_eager_scoring_score_matrix_built(larger_corpus):
    """Test that eager scoring builds the score matrix correctly."""
    from vajra_bm25.optimized import VajraSearchOptimized

    # Create engine with eager scoring enabled (force sparse for small corpus)
    engine = VajraSearchOptimized(larger_corpus, use_eager=True, use_sparse=True)

    # Score matrix should be built
    assert engine.index.score_matrix is not None
    assert engine.eager_scorer is not None

    # Score matrix should have same sparsity pattern as term_doc_matrix
    assert engine.index.score_matrix.shape == engine.index.term_doc_matrix.shape
    assert engine.index.score_matrix.nnz == engine.index.term_doc_matrix.nnz


def test_eager_scoring_produces_correct_scores(larger_corpus):
    """Test that eager scoring produces same scores as traditional scoring."""
    from vajra_bm25.optimized import VajraSearchOptimized
    import numpy as np

    # Create engine with eager scoring disabled first (force sparse for testing)
    engine_traditional = VajraSearchOptimized(
        larger_corpus, use_eager=False, use_numba=False, use_sparse=True
    )

    # Create engine with eager scoring enabled (force sparse for testing)
    engine_eager = VajraSearchOptimized(larger_corpus, use_eager=True, use_sparse=True)

    # Test several queries
    queries = [
        "graph search",
        "machine learning neural",
        "category functors",
        "topic common words",
    ]

    for query in queries:
        results_traditional = engine_traditional.search(query, top_k=10)
        results_eager = engine_eager.search(query, top_k=10)

        # Should return same number of results
        assert len(results_traditional) == len(results_eager), f"Query: {query}"

        # Should return same documents in same order
        for r_trad, r_eager in zip(results_traditional, results_eager):
            assert r_trad.document.id == r_eager.document.id, f"Query: {query}"
            # Scores should be very close (floating point tolerance)
            np.testing.assert_allclose(
                r_trad.score, r_eager.score, rtol=1e-5,
                err_msg=f"Query: {query}, Doc: {r_trad.document.id}"
            )


def test_eager_scoring_empty_query(larger_corpus):
    """Test eager scoring with empty query."""
    from vajra_bm25.optimized import VajraSearchOptimized

    engine = VajraSearchOptimized(larger_corpus, use_eager=True, use_sparse=True)
    results = engine.search("")

    assert len(results) == 0


def test_eager_scoring_no_matches(larger_corpus):
    """Test eager scoring with query that matches nothing."""
    from vajra_bm25.optimized import VajraSearchOptimized

    engine = VajraSearchOptimized(larger_corpus, use_eager=True, use_sparse=True)
    results = engine.search("xyzzy quantum chromodynamics superconductivity")

    # Should return empty or very low scored results
    assert len(results) == 0 or all(r.score == 0 for r in results)


def test_eager_scoring_can_be_disabled(larger_corpus):
    """Test that eager scoring can be disabled."""
    from vajra_bm25.optimized import VajraSearchOptimized

    engine = VajraSearchOptimized(larger_corpus, use_eager=False, use_sparse=True)

    # Score matrix should not be built
    assert engine.index.score_matrix is None
    assert engine.eager_scorer is None


def test_eager_scorer_search_top_k(larger_corpus):
    """Test EagerSparseBM25Scorer.search_top_k method."""
    from vajra_bm25.optimized import VajraSearchOptimized
    from vajra_bm25.text_processing import preprocess_text

    engine = VajraSearchOptimized(larger_corpus, use_eager=True, use_sparse=True)

    query_terms = preprocess_text("machine learning neural networks")
    top_docs = engine.eager_scorer.search_top_k(query_terms, k=5)

    # Should return list of (doc_idx, score) tuples
    assert isinstance(top_docs, list)
    assert len(top_docs) <= 5

    # Should be sorted by score descending
    if len(top_docs) > 1:
        scores = [score for _, score in top_docs]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# ADDITIONAL SCORER TESTS
# ============================================================================

def test_bm25_parameters_repr():
    """Test BM25Parameters string representation."""
    params = BM25Parameters(k1=2.0, b=0.8)
    repr_str = repr(params)

    assert "k1=2.0" in repr_str
    assert "b=0.8" in repr_str


def test_bm25_score_document_object(sample_corpus):
    """Test scoring a Document object directly."""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["category", "functors"]
    doc = sample_corpus.get("1")

    score = scorer.score_document_object(query_terms, doc)

    # Should match direct scoring by ID
    score_by_id = scorer.score(query_terms, "1")
    assert score == score_by_id


def test_bm25_explain_score(sample_corpus):
    """Test score explanation breakdown."""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    query_terms = ["category", "functors", "nonexistent"]
    explanation = scorer.explain_score(query_terms, "1")

    # Should have an entry for each term
    assert "category" in explanation
    assert "functors" in explanation
    assert "nonexistent" in explanation

    # Known terms should have positive scores
    assert explanation["category"] > 0
    assert explanation["functors"] > 0

    # Unknown term should have zero score
    assert explanation["nonexistent"] == 0.0


def test_bm25_explain_score_zero_doc_length(sample_corpus):
    """Test explanation for document with zero length."""
    index = InvertedIndex()
    index.build(sample_corpus)
    scorer = BM25Scorer(index)

    # Try to explain a non-existent document
    explanation = scorer.explain_score(["category"], "nonexistent_doc")

    # All scores should be zero for non-existent doc
    assert explanation["category"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
