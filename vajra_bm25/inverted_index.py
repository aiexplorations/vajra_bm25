"""
Inverted Index

An inverted index is a categorical structure:
- Maps terms to sets of documents
- Functor-like: preserves document relationships through term mappings
- Efficient morphism from query terms to candidate documents
"""

from typing import Dict, Set, List
from collections import defaultdict, Counter
import math
from dataclasses import dataclass

from vajra_bm25.documents import Document, DocumentCorpus
from vajra_bm25.text_processing import preprocess_text
from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("inverted_index")


@dataclass
class PostingList:
    """
    Documents containing a term, with frequency information.

    This is the image of a term under the index morphism.
    """
    doc_ids: Set[str]  # Documents containing the term
    doc_frequencies: Dict[str, int]  # Term frequency in each document

    def __len__(self):
        return len(self.doc_ids)


class InvertedIndex:
    """
    Inverted index: Term -> PostingList

    Categorical interpretation:
    - A structure-preserving map from terms to document sets
    - Enables efficient "unfolding" of queries into candidate documents
    """

    def __init__(self):
        self.index: Dict[str, PostingList] = {}
        self.doc_lengths: Dict[str, int] = {}  # Document length in terms
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def build(self, corpus: DocumentCorpus):
        """
        Build index from corpus.

        Morphism: Corpus -> Index

        This is a one-time computation that enables efficient queries.
        """
        # Temporary structure for building
        term_to_docs: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self.num_docs = len(corpus)
        total_length = 0

        # Process each document
        for doc in corpus:
            # Preprocess document text (title + content)
            full_text = doc.title + " " + doc.content
            terms = preprocess_text(full_text)

            # Store document length
            doc_length = len(terms)
            self.doc_lengths[doc.id] = doc_length
            total_length += doc_length

            # Count term frequencies (Counter is faster than manual loop)
            term_counts = Counter(terms)

            # Add to inverted index
            for term, count in term_counts.items():
                term_to_docs[term][doc.id] = count

        # Calculate average document length
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0

        # Convert to PostingList objects
        for term, doc_freqs in term_to_docs.items():
            self.index[term] = PostingList(
                doc_ids=set(doc_freqs.keys()),
                doc_frequencies=doc_freqs
            )

    def get_posting_list(self, term: str) -> PostingList:
        """
        Retrieve posting list for a term.

        Morphism: Term -> PostingList
        """
        return self.index.get(term, PostingList(set(), {}))

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Get frequency of term in document"""
        posting_list = self.get_posting_list(term)
        return posting_list.doc_frequencies.get(doc_id, 0)

    def get_document_frequency(self, term: str) -> int:
        """
        Number of documents containing term.

        Used for IDF calculation.
        """
        return len(self.get_posting_list(term))

    def get_candidate_documents(self, query_terms: List[str]) -> Set[str]:
        """
        Find all documents containing any query term.

        This is the coalgebraic unfolding:
        Query -> Set[Document]

        Union of posting lists for query terms.
        """
        candidates = set()
        for term in query_terms:
            posting_list = self.get_posting_list(term)
            candidates.update(posting_list.doc_ids)

        return candidates

    def idf(self, term: str) -> float:
        """
        Inverse Document Frequency.

        IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)

        where N = total documents, df = document frequency

        This is a morphism: Term -> R
        """
        df = self.get_document_frequency(term)

        if df == 0:
            return 0.0

        # BM25 IDF formula
        idf_score = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        return idf_score

    def __repr__(self):
        return f"InvertedIndex(terms={len(self.index)}, docs={self.num_docs}, avg_len={self.avg_doc_length:.1f})"


if __name__ == "__main__":
    from vajra_bm25.documents import create_sample_corpus

    # Create corpus
    corpus = create_sample_corpus()
    logger.info(f"Created corpus with {len(corpus)} documents")

    # Build index
    logger.info("Building inverted index...")
    index = InvertedIndex()
    index.build(corpus)
    logger.info(f"{index}")

    # Test queries
    test_terms = ["category", "functor", "search", "programming"]

    logger.info("Term statistics:")
    logger.info(f"{'Term':<15} {'Doc Freq':<10} {'IDF':<10}")
    logger.info("-" * 35)
    for term in test_terms:
        df = index.get_document_frequency(term)
        idf = index.idf(term)
        logger.info(f"{term:<15} {df:<10} {idf:<10.3f}")

    # Test candidate retrieval
    logger.info("="*50)
    query = "category theory functors"
    query_terms = preprocess_text(query)
    logger.info(f"Query: '{query}'")
    logger.info(f"Preprocessed terms: {query_terms}")

    candidates = index.get_candidate_documents(query_terms)
    logger.info(f"Candidate documents: {candidates}")
    logger.info(f"Found {len(candidates)} candidates")
