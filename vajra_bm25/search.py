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
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from vajra_bm25.categorical import Coalgebra
from vajra_bm25.documents import Document, DocumentCorpus
from vajra_bm25.inverted_index import InvertedIndex
from vajra_bm25.scorer import BM25Scorer, BM25Parameters
from vajra_bm25.text_processing import preprocess_text


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

    Combines:
    - Document corpus
    - Inverted index construction
    - BM25 scoring (morphism)
    - Search coalgebra (unfolding)
    """

    def __init__(self, corpus: DocumentCorpus, params: BM25Parameters = None):
        self.corpus = corpus

        # Build index
        print("Building inverted index...")
        self.index = InvertedIndex()
        self.index.build(corpus)

        # Create scorer
        self.scorer = BM25Scorer(self.index, params or BM25Parameters())

        print(f"Index built: {self.index}")

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Execute search.

        This creates a coalgebra and applies one unfolding.

        Query -> List[SearchResult]
        """
        # Preprocess query
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
        results = coalgebra.structure_map(state)

        return results

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
    print(f"Created corpus with {len(corpus)} documents\n")

    # Create search system
    search_engine = VajraSearch(corpus)
    print()

    # Test queries
    test_queries = [
        "category theory functors",
        "search algorithms breadth first",
        "functional programming monads",
        "lambda calculus computation"
    ]

    for query in test_queries:
        print("=" * 70)
        print(f"Query: '{query}'")
        print("=" * 70)

        results = search_engine.search(query, top_k=5)

        if not results:
            print("No results found.\n")
            continue

        print(f"\nFound {len(results)} results:\n")

        for result in results:
            print(f"{result.rank}. [{result.document.id}] {result.document.title}")
            print(f"   Score: {result.score:.3f}")
            print(f"   {result.document.content[:120]}...")
            print()

        # Explain top result
        if results:
            top_result = results[0]
            print(f"Score breakdown for top result:")
            explanation = search_engine.explain_result(query, top_result.document.id)
            for term, score in sorted(explanation['term_scores'].items(),
                                     key=lambda x: x[1], reverse=True):
                if score > 0:
                    print(f"  '{term}': {score:.3f}")
            print()
