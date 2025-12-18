"""
BM25 Ranking Algorithm

BM25 is a morphism: (Query, Document) -> R

It's a probabilistic relevance function that scores documents
given a query. The score is a composition of term-level scores.

Categorical structure:
- BM25: (Q, D) -> R is a morphism to the reals
- For fixed Q: D -> R is a morphism (curried form)
- This morphism induces a ranking (total order on documents)
"""

import math
from typing import List, Dict
from dataclasses import dataclass

from vajra_bm25.documents import Document
from vajra_bm25.inverted_index import InvertedIndex
from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("scorer")
from vajra_bm25.text_processing import preprocess_text


@dataclass
class BM25Parameters:
    """
    BM25 hyperparameters.

    These control the functional form of the scoring morphism.
    """
    k1: float = 1.5  # Term frequency saturation parameter
    b: float = 0.75  # Length normalization parameter

    def __repr__(self):
        return f"BM25(k1={self.k1}, b={self.b})"


class BM25Scorer:
    """
    BM25 scoring function.

    This is a categorical morphism:
    BM25: Query x Document -> R

    For a fixed query, this becomes:
    score: Document -> R

    This morphism defines a total order on documents (ranking).
    """

    def __init__(self, index: InvertedIndex, params: BM25Parameters = None):
        self.index = index
        self.params = params or BM25Parameters()

    def score(self, query_terms: List[str], doc_id: str) -> float:
        """
        Compute BM25 score for a document given query terms.

        BM25(q, d) = Sum IDF(qi) x (f(qi, d) x (k1 + 1)) / (f(qi, d) + k1 x (1 - b + b x |d|/avgdl))

        where:
        - IDF(qi) = inverse document frequency of term qi
        - f(qi, d) = frequency of qi in document d
        - |d| = length of document d
        - avgdl = average document length in corpus
        - k1, b = tuning parameters

        This is a morphism: (Query, Document) -> R
        """
        score = 0.0

        doc_length = self.index.doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            return 0.0

        avg_doc_length = self.index.avg_doc_length

        for term in query_terms:
            # Get IDF for term
            idf = self.index.idf(term)

            # Get term frequency in document
            tf = self.index.get_term_frequency(term, doc_id)

            if tf == 0:
                continue

            # Length normalization
            norm_factor = 1 - self.params.b + self.params.b * (doc_length / avg_doc_length)

            # BM25 formula
            term_score = idf * (tf * (self.params.k1 + 1)) / (tf + self.params.k1 * norm_factor)

            score += term_score

        return score

    def score_document_object(self, query_terms: List[str], doc: Document) -> float:
        """
        Score a Document object directly.

        Alternative morphism signature: (Query, Document) -> R
        """
        return self.score(query_terms, doc.id)

    def rank_documents(self, query_terms: List[str], doc_ids: List[str]) -> List[tuple[str, float]]:
        """
        Rank a list of documents by BM25 score.

        Returns: List[(doc_id, score)] sorted by score descending

        This is applying the morphism D -> R to induce a ranking.
        """
        scored = [(doc_id, self.score(query_terms, doc_id)) for doc_id in doc_ids]

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def explain_score(self, query_terms: List[str], doc_id: str) -> Dict[str, float]:
        """
        Break down the BM25 score by term.

        Useful for understanding which terms contribute most.
        """
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.index.avg_doc_length

        term_scores = {}

        for term in query_terms:
            idf = self.index.idf(term)
            tf = self.index.get_term_frequency(term, doc_id)

            if tf > 0:
                norm_factor = 1 - self.params.b + self.params.b * (doc_length / avg_doc_length)
                term_score = idf * (tf * (self.params.k1 + 1)) / (tf + self.params.k1 * norm_factor)
                term_scores[term] = term_score
            else:
                term_scores[term] = 0.0

        return term_scores


if __name__ == "__main__":
    from vajra_bm25.documents import create_sample_corpus
    from vajra_bm25.inverted_index import InvertedIndex

    # Create corpus and index
    corpus = create_sample_corpus()
    index = InvertedIndex()
    index.build(corpus)

    # Create BM25 scorer
    scorer = BM25Scorer(index)
    logger.info(f"BM25 Scorer initialized with {scorer.params}")

    # Test query
    query = "category theory functors"
    query_terms = preprocess_text(query)

    logger.info(f"Query: '{query}'")
    logger.info(f"Query terms: {query_terms}")

    # Get candidates and rank
    candidates = index.get_candidate_documents(query_terms)
    ranked = scorer.rank_documents(query_terms, list(candidates))

    logger.info(f"Top 5 results:")
    logger.info(f"{'Rank':<6} {'Doc ID':<10} {'Score':<10} {'Title':<40}")
    logger.info("-" * 70)

    for i, (doc_id, score) in enumerate(ranked[:5], 1):
        doc = corpus.get(doc_id)
        title = doc.title if doc else "Unknown"
        logger.info(f"{i:<6} {doc_id:<10} {score:<10.3f} {title:<40}")

    # Explain top result
    if ranked:
        top_doc_id = ranked[0][0]
        logger.info(f"Score breakdown for top document ({top_doc_id}):")
        explanation = scorer.explain_score(query_terms, top_doc_id)

        for term, term_score in sorted(explanation.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {term}: {term_score:.3f}")
