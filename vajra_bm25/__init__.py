"""
Vajra BM25: Categorical BM25 Search Engine

Vajra (Sanskrit: vajra, "thunderbolt/diamond") implements BM25 information
retrieval using pure category theory.

Core Concepts:
- BM25 scoring as morphisms: (Query, Document) -> R
- Search as coalgebraic unfolding: Query -> List[SearchResult]
- Functors capture multiple-results semantics (List functor)

Basic Usage:
    from vajra_bm25 import VajraSearch, Document, DocumentCorpus

    # Create documents
    docs = [
        Document(id="1", title="Hello", content="Hello world"),
        Document(id="2", title="Goodbye", content="Goodbye world"),
    ]
    corpus = DocumentCorpus(docs)

    # Create search engine
    engine = VajraSearch(corpus)

    # Search
    results = engine.search("hello")
    for r in results:
        print(f"{r.rank}. {r.document.title} (score: {r.score:.3f})")

Optimized Usage (for large corpora):
    from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

    corpus = DocumentCorpus.load_jsonl("corpus.jsonl")
    engine = VajraSearchOptimized(corpus)  # Auto-uses sparse matrices for >10K docs
    results = engine.search("query", top_k=10)

Parallel Batch Processing:
    from vajra_bm25 import VajraSearchParallel

    engine = VajraSearchParallel(corpus, max_workers=4)
    batch_results = engine.search_batch(["query1", "query2", "query3"])
"""

__version__ = "0.2.0"

# Core document types
from vajra_bm25.documents import (
    Document,
    DocumentCorpus,
    create_sample_corpus,
)

# Text processing
from vajra_bm25.text_processing import (
    preprocess_text,
    tokenize,
    Token,
    STOP_WORDS,
)

# Scoring
from vajra_bm25.scorer import (
    BM25Scorer,
    BM25Parameters,
)

# Search engines
from vajra_bm25.search import (
    VajraSearch,
    QueryState,
    SearchResult,
    BM25SearchCoalgebra,
)

from vajra_bm25.optimized import (
    VajraSearchOptimized,
)

from vajra_bm25.parallel import (
    VajraSearchParallel,
)

# Categorical abstractions (for advanced users)
from vajra_bm25.categorical import (
    # Category primitives
    Morphism,
    FunctionMorphism,
    IdentityMorphism,
    ComposedMorphism,
    # Functors
    Functor,
    ListFunctor,
    MaybeFunctor,
    TreeFunctor,
    Tree,
    # Coalgebras
    Coalgebra,
    SearchCoalgebra,
    TreeSearchCoalgebra,
    ConditionalCoalgebra,
)

__all__ = [
    # Version
    "__version__",
    # Documents
    "Document",
    "DocumentCorpus",
    "create_sample_corpus",
    # Text processing
    "preprocess_text",
    "tokenize",
    "Token",
    "STOP_WORDS",
    # Scoring
    "BM25Scorer",
    "BM25Parameters",
    # Search
    "VajraSearch",
    "VajraSearchOptimized",
    "VajraSearchParallel",
    "QueryState",
    "SearchResult",
    "BM25SearchCoalgebra",
    # Categorical (advanced)
    "Morphism",
    "FunctionMorphism",
    "IdentityMorphism",
    "ComposedMorphism",
    "Functor",
    "ListFunctor",
    "MaybeFunctor",
    "TreeFunctor",
    "Tree",
    "Coalgebra",
    "SearchCoalgebra",
    "TreeSearchCoalgebra",
    "ConditionalCoalgebra",
]
