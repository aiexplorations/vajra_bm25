"""
Optimized Categorical BM25: Performance improvements using category theory

Key optimizations (all categorical):
1. Memoized morphisms (comonadic caching)
2. Vectorized operations (morphisms over vector spaces)
3. Lazy evaluation (monadic delay)
4. Batch processing (functor over query lists)
5. Parallel unfolding (concurrent coalgebra evaluation)
"""

from typing import List, Dict, Set, Tuple, Callable, Optional
from dataclasses import dataclass
import numpy as np
from functools import lru_cache, wraps
from collections import OrderedDict, Counter
import time
from multiprocessing import Pool, cpu_count

try:
    from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from pathlib import Path

from vajra_bm25.documents import Document, DocumentCorpus
from vajra_bm25.text_processing import preprocess_text
from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("optimized")


# ============================================================================
# Helper: Parallel tokenization
# ============================================================================

def _tokenize_document(doc: Document) -> Tuple[str, Dict[str, int], int]:
    """
    Tokenize a single document (for parallel processing).

    Returns:
        (doc_id, term_counts, doc_length)
    """
    full_text = doc.title + " " + doc.content
    terms = preprocess_text(full_text)
    term_counts = Counter(terms)
    return doc.id, term_counts, len(terms)


# ============================================================================
# OPTIMIZATION 1: Memoized Morphisms (Comonadic Caching)
# ============================================================================

def memoized_morphism(func):
    """
    Decorator for memoizing morphisms.

    Categorical interpretation: This is a comonad!
    - extract: get the cached value
    - duplicate: create nested caches

    Caching is structure-preserving: f: A -> B becomes cached_f: A -> B
    with the same mathematical properties.
    """
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create hashable key
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    wrapper.cache_info = lambda: {'size': len(cache), 'hits': getattr(wrapper, '_hits', 0)}
    return wrapper


class LRUCache:
    """
    LRU (Least Recently Used) cache for query results.

    Categorical interpretation:
    - Caching is a comonad
    - extract: retrieve cached value
    - duplicate: nested cache layers

    This is structure-preserving: the cached morphism
    f: Query -> Results has the same type signature.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List]:
        """Get value from cache, moving it to end (most recently used)."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: str, value: List):
        """Put value in cache, evicting LRU item if at capacity."""
        if key in self.cache:
            # Update and move to end
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)

        self.cache[key] = value

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


# ============================================================================
# OPTIMIZATION 2: Vectorized Inverted Index
# ============================================================================

class VectorizedIndex:
    """
    Vectorized inverted index using NumPy.

    Categorical interpretation:
    - Index is still a functor: Term -> PostingList
    - But operations are morphisms over vector spaces
    - Linear algebra preserves categorical structure
    """

    def __init__(self):
        self.term_to_id: Dict[str, int] = {}
        self.id_to_term: Dict[int, str] = {}
        self.doc_to_id: Dict[str, int] = {}
        self.id_to_doc: Dict[int, str] = {}

        # Vectorized structures
        self.term_doc_matrix: Optional[np.ndarray] = None  # Sparse would be better
        self.doc_lengths: Optional[np.ndarray] = None
        self.doc_freqs: Optional[np.ndarray] = None
        self.idf_cache: Optional[np.ndarray] = None

        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0
        self.num_terms: int = 0

    def build(self, corpus: DocumentCorpus):
        """
        Build vectorized index.

        Morphism: Corpus -> VectorizedIndex
        """
        # Build term and doc vocabularies
        term_set = set()
        doc_term_counts = []

        for doc_idx, doc in enumerate(corpus):
            self.doc_to_id[doc.id] = doc_idx
            self.id_to_doc[doc_idx] = doc.id

            full_text = doc.title + " " + doc.content
            terms = preprocess_text(full_text)

            term_counts = {}
            for term in terms:
                term_set.add(term)
                term_counts[term] = term_counts.get(term, 0) + 1

            doc_term_counts.append(term_counts)

        # Assign term IDs
        for term_id, term in enumerate(sorted(term_set)):
            self.term_to_id[term] = term_id
            self.id_to_term[term_id] = term

        self.num_docs = len(corpus)
        self.num_terms = len(term_set)

        # Build term-document matrix (dense for now, sparse would be better)
        self.term_doc_matrix = np.zeros((self.num_terms, self.num_docs), dtype=np.float32)
        self.doc_lengths = np.zeros(self.num_docs, dtype=np.int32)

        for doc_idx, term_counts in enumerate(doc_term_counts):
            for term, count in term_counts.items():
                term_id = self.term_to_id[term]
                self.term_doc_matrix[term_id, doc_idx] = count
                self.doc_lengths[doc_idx] += count

        # Pre-compute document frequencies (DF)
        self.doc_freqs = (self.term_doc_matrix > 0).sum(axis=1)

        # Pre-compute IDF values (vectorized!)
        # IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf_cache = np.log(
            (self.num_docs - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5) + 1.0
        )

        # Average document length
        self.avg_doc_length = self.doc_lengths.mean()

    @memoized_morphism
    def get_term_id(self, term: str) -> Optional[int]:
        """Morphism: Term -> TermID"""
        return self.term_to_id.get(term)

    def get_candidate_docs_vectorized(self, query_terms: List[str]) -> np.ndarray:
        """
        Get candidate documents (vectorized).

        Returns boolean array indicating which docs contain any query term.
        This is a morphism: Query -> DocumentSet (as boolean vector)
        """
        term_ids = [self.term_to_id[t] for t in query_terms if t in self.term_to_id]

        if not term_ids:
            return np.zeros(self.num_docs, dtype=bool)

        # Union of posting lists (vectorized OR)
        candidates = np.any(self.term_doc_matrix[term_ids, :] > 0, axis=0)
        return candidates


# ============================================================================
# OPTIMIZATION 3: Vectorized BM25 Scorer
# ============================================================================

class VectorizedBM25Scorer:
    """
    Vectorized BM25 scoring using NumPy.

    Categorical interpretation:
    - Still a morphism: (Query, Document) -> R
    - But computed via linear algebra (morphisms in vector spaces)
    - Batch scoring: Query x DocumentSet -> R^n (functor application)
    """

    def __init__(self, index: VectorizedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def score_batch(self, query_terms: List[str], doc_mask: np.ndarray) -> np.ndarray:
        """
        Score multiple documents at once (vectorized).

        Morphism: (Query, DocumentSet) -> R^n

        This is compositional: we apply the scoring morphism to all
        candidates simultaneously via vectorization.
        """
        # Get term IDs for query
        term_ids = [self.index.term_to_id[t] for t in query_terms if t in self.index.term_to_id]

        if not term_ids:
            return np.zeros(self.index.num_docs, dtype=np.float32)

        # Get IDF values for query terms (pre-cached!)
        query_idfs = self.index.idf_cache[term_ids]  # Shape: (num_query_terms,)

        # Get term frequencies for candidate docs
        # Shape: (num_query_terms, num_docs)
        tf_matrix = self.index.term_doc_matrix[term_ids, :]

        # BM25 normalization factor (vectorized)
        # norm = 1 - b + b * (doc_length / avg_doc_length)
        norm_factors = 1.0 - self.b + self.b * (self.index.doc_lengths / self.index.avg_doc_length)

        # BM25 formula (fully vectorized!)
        # score = IDF * (TF * (k1 + 1)) / (TF + k1 * norm)
        numerator = tf_matrix * (self.k1 + 1)
        denominator = tf_matrix + self.k1 * norm_factors

        # Broadcast IDF across documents
        # Shape: (num_query_terms, num_docs)
        term_scores = query_idfs[:, np.newaxis] * (numerator / denominator)

        # Sum across query terms
        # Shape: (num_docs,)
        doc_scores = term_scores.sum(axis=0)

        # Apply document mask (only score candidates)
        doc_scores = doc_scores * doc_mask

        return doc_scores

    def get_top_k(self, scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k documents by score.

        Morphism: R^n -> List[(DocID, Score)]

        Uses partial sort for efficiency (O(n + k log k) vs O(n log n))
        """
        # Get indices of top-k scores (argpartition is O(n))
        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort: only sort top-k
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Filter out zero scores
        result = []
        for idx in top_indices:
            if scores[idx] > 0:
                result.append((int(idx), float(scores[idx])))

        return result


# ============================================================================
# OPTIMIZATION 3.5: Sparse Matrix Index (CRITICAL for 100K+ docs)
# ============================================================================

class VectorizedIndexSparse:
    """
    Sparse matrix vectorized index using scipy.sparse.csr_matrix.

    Categorical interpretation:
    - Functor: Term -> PostingList (enriched with max scores)
    - Morphisms over vector spaces with pre-computed bounds
    - Sparse representation = optimization without changing math

    The index is ENRICHED with:
    - max_term_score: Maximum BM25 contribution per term (for coalgebraic guards)
    - This enables early termination in the MaxScore algorithm

    Memory savings: ~100x smaller than dense for typical text corpus
    Speed improvement: 2-5x on sparse operations, plus algorithmic speedup
    """

    def __init__(self):
        self.term_to_id: Dict[str, int] = {}
        self.id_to_term: Dict[int, str] = {}
        self.doc_to_id: Dict[str, int] = {}
        self.id_to_doc: Dict[int, str] = {}

        # Sparse structures
        self.term_doc_matrix: Optional[csr_matrix] = None  # Sparse!
        self.doc_lengths: Optional[np.ndarray] = None
        self.doc_freqs: Optional[np.ndarray] = None
        self.idf_cache: Optional[np.ndarray] = None

        # ENRICHED: Pre-computed bounds for coalgebraic early termination
        self.max_term_score: Optional[np.ndarray] = None  # Max BM25 score per term
        self.norm_factors: Optional[np.ndarray] = None    # Pre-computed doc length norms

        # EAGER SCORING: Pre-computed BM25 scores (BM25S approach)
        self.score_matrix: Optional[csr_matrix] = None  # Pre-computed BM25 scores
        self._eager_k1: Optional[float] = None  # k1 used for eager scoring
        self._eager_b: Optional[float] = None   # b used for eager scoring

        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0
        self.num_terms: int = 0

    def build(self, corpus: DocumentCorpus):
        """
        Build sparse vectorized index with optimizations.

        Optimizations:
        - Parallel tokenization (uses multiprocessing)
        - COO format construction (3-5x faster than LIL)
        - Single-pass vocabulary building

        Morphism: Corpus -> SparseVectorizedIndex
        """
        n_jobs = min(cpu_count(), len(corpus) // 100 + 1)  # At least 100 docs per worker
        logger.info(f"Building sparse index with {n_jobs} workers...")

        # ====================================================================
        # OPTIMIZATION: Parallel tokenization
        # ====================================================================
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_tokenize_document, corpus.documents)

        # Extract results
        doc_ids = []
        doc_term_counts = []
        doc_lengths_list = []

        for doc_id, term_counts, doc_len in results:
            doc_ids.append(doc_id)
            doc_term_counts.append(term_counts)
            doc_lengths_list.append(doc_len)

        # Build vocabularies
        term_set = set()
        for term_counts in doc_term_counts:
            term_set.update(term_counts.keys())

        # Assign term IDs (use insertion order for speed - no sorting)
        self.term_to_id = {term: idx for idx, term in enumerate(term_set)}
        self.id_to_term = {idx: term for term, idx in self.term_to_id.items()}
        self.doc_to_id = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.id_to_doc = {idx: doc_id for doc_id, idx in self.doc_to_id.items()}

        self.num_docs = len(corpus)
        self.num_terms = len(term_set)
        self.doc_lengths = np.array(doc_lengths_list, dtype=np.int32)

        # ====================================================================
        # OPTIMIZATION: COO format construction (batch inserts)
        # ====================================================================
        # Prepare COO format data
        rows = []
        cols = []
        data = []

        for doc_idx, term_counts in enumerate(doc_term_counts):
            for term, count in term_counts.items():
                term_id = self.term_to_id[term]
                rows.append(term_id)
                cols.append(doc_idx)
                data.append(count)

        # Build COO matrix (fast batch construction)
        coo = coo_matrix(
            (data, (rows, cols)),
            shape=(self.num_terms, self.num_docs),
            dtype=np.float32
        )

        # Convert to CSR for efficient row slicing and arithmetic
        self.term_doc_matrix = coo.tocsr()

        logger.info(f"  Built sparse matrix: {self.num_terms:,} terms × {self.num_docs:,} docs")
        logger.info(f"  Non-zero entries: {coo.nnz:,} ({(1.0 - coo.nnz/(self.num_terms*self.num_docs))*100:.2f}% sparse)")

        # Pre-compute document frequencies (DF)
        # Number of non-zero entries per row
        self.doc_freqs = np.asarray((self.term_doc_matrix > 0).sum(axis=1)).flatten()

        # Pre-compute IDF values (vectorized!)
        # IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf_cache = np.log(
            (self.num_docs - self.doc_freqs + 0.5) / (self.doc_freqs + 0.5) + 1.0
        ).astype(np.float32)

        # Average document length
        self.avg_doc_length = float(self.doc_lengths.mean())

        # ====================================================================
        # ENRICHED INDEX: Pre-compute bounds for coalgebraic early termination
        # ====================================================================
        self._compute_term_bounds(k1=1.5, b=0.75)

    def _compute_term_bounds(self, k1: float = 1.5, b: float = 0.75):
        """
        Compute maximum possible BM25 score for each term.

        This is the FUNCTORIAL ENRICHMENT: we lift score bounds from
        query-time to index-time, enabling coalgebraic early termination.

        For term t, max_term_score[t] = IDF[t] * max_d(TF_component[t,d])
        where TF_component = (tf * (k1+1)) / (tf + k1 * norm)
        """
        logger.info("  Computing term bounds for coalgebraic scoring...")

        # Pre-compute document length normalization factors
        # norm[d] = 1 - b + b * (doc_len[d] / avg_doc_len)
        self.norm_factors = (
            1.0 - b + b * (self.doc_lengths / self.avg_doc_length)
        ).astype(np.float32)

        # For each term, find the maximum TF component across all documents
        # TF_component = (tf * (k1+1)) / (tf + k1 * norm)
        self.max_term_score = np.zeros(self.num_terms, dtype=np.float32)

        # Iterate through CSR matrix rows (terms)
        indptr = self.term_doc_matrix.indptr
        indices = self.term_doc_matrix.indices
        data = self.term_doc_matrix.data

        for term_id in range(self.num_terms):
            row_start = indptr[term_id]
            row_end = indptr[term_id + 1]

            if row_start == row_end:
                continue  # Term not in any document

            # Get TF values and doc indices for this term
            doc_indices = indices[row_start:row_end]
            tf_values = data[row_start:row_end]

            # Compute TF component for each doc containing this term
            norms = self.norm_factors[doc_indices]
            tf_components = (tf_values * (k1 + 1)) / (tf_values + k1 * norms)

            # Max TF component × IDF = max possible score for this term
            self.max_term_score[term_id] = self.idf_cache[term_id] * tf_components.max()

        logger.info(f"  Term bounds computed: max={self.max_term_score.max():.3f}, "
                    f"mean={self.max_term_score.mean():.3f}")

    def build_score_matrix(self, k1: float = 1.5, b: float = 0.75):
        """
        Build pre-computed BM25 score matrix (eager scoring).

        This implements the BM25S approach: pre-compute all BM25 scores at
        index time, enabling query-time to be just sparse matrix slicing + sum.

        The score for term t in document d is:
            score[t,d] = IDF[t] * (TF[t,d] * (k1 + 1)) / (TF[t,d] + k1 * norm[d])

        This matrix has the same sparsity pattern as term_doc_matrix (only
        non-zero where term appears in document), but values are BM25 scores
        instead of term frequencies.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
        """
        if self.term_doc_matrix is None:
            raise ValueError("Index not built. Call build() first.")

        logger.info(f"  Building eager score matrix (k1={k1}, b={b})...")
        start_time = time.time()

        # Ensure norm factors are computed
        if self.norm_factors is None:
            self.norm_factors = (
                1.0 - b + b * (self.doc_lengths / self.avg_doc_length)
            ).astype(np.float32)

        # Get CSR components
        indptr = self.term_doc_matrix.indptr
        indices = self.term_doc_matrix.indices
        data = self.term_doc_matrix.data

        # Pre-compute BM25 scores (same sparsity pattern, different values)
        k1_plus_1 = np.float32(k1 + 1.0)
        k1_f32 = np.float32(k1)

        # Vectorized computation over all non-zero entries
        # For each (term, doc) pair with TF > 0:
        #   score = IDF[term] * (TF * (k1+1)) / (TF + k1 * norm[doc])
        tf = data.astype(np.float32)
        doc_norms = self.norm_factors[indices]  # norm for each doc in posting

        # Compute TF component: (TF * (k1+1)) / (TF + k1 * norm)
        tf_component = (tf * k1_plus_1) / (tf + k1_f32 * doc_norms)

        # Multiply by IDF for each term
        # We need to map each entry to its term's IDF
        # Create array of IDFs corresponding to each non-zero entry
        term_ids_for_entries = np.zeros(len(data), dtype=np.int32)
        for term_id in range(self.num_terms):
            row_start = indptr[term_id]
            row_end = indptr[term_id + 1]
            term_ids_for_entries[row_start:row_end] = term_id

        idf_for_entries = self.idf_cache[term_ids_for_entries]
        score_data = (tf_component * idf_for_entries).astype(np.float32)

        # Build score matrix with same structure as term_doc_matrix
        self.score_matrix = csr_matrix(
            (score_data, indices.copy(), indptr.copy()),
            shape=(self.num_terms, self.num_docs),
            dtype=np.float32
        )

        # Store parameters used
        self._eager_k1 = k1
        self._eager_b = b

        build_time = time.time() - start_time
        logger.info(f"  Score matrix built in {build_time:.3f}s "
                    f"({self.score_matrix.nnz:,} entries)")

    @memoized_morphism
    def get_term_id(self, term: str) -> Optional[int]:
        """Morphism: Term -> TermID"""
        return self.term_to_id.get(term)

    def get_candidate_docs_vectorized(self, query_terms: List[str]) -> np.ndarray:
        """
        Get candidate documents (vectorized, sparse-aware).

        Returns boolean array indicating which docs contain any query term.
        This is a morphism: Query -> DocumentSet (as boolean vector)
        """
        term_ids = [self.term_to_id[t] for t in query_terms if t in self.term_to_id]

        if not term_ids:
            return np.zeros(self.num_docs, dtype=bool)

        # Union of posting lists (sparse OR operation)
        # Extract relevant rows and check for any non-zero values per column
        candidates = np.asarray(
            (self.term_doc_matrix[term_ids, :].sum(axis=0) > 0)
        ).flatten()

        return candidates


class SparseBM25Scorer:
    """
    Sparse matrix BM25 scoring.

    Categorical interpretation:
    - Still a morphism: (Query, Document) -> R
    - Sparse operations preserve mathematical structure
    - Linear algebra is still categorical
    """

    def __init__(self, index: VectorizedIndexSparse, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def score_batch(self, query_terms: List[str], doc_mask: np.ndarray) -> np.ndarray:
        """
        Score multiple documents at once (vectorized with sparse matrices).

        OPTIMIZED:
        - Uses pre-computed norm factors from enriched index
        - Consistent float32 dtype throughout
        - Efficient numpy operations

        Morphism: (Query, DocumentSet) -> R^n
        """
        # Get term IDs for query
        term_ids = [self.index.term_to_id[t] for t in query_terms if t in self.index.term_to_id]

        if not term_ids:
            return np.zeros(self.index.num_docs, dtype=np.float32)

        # Get IDF values for query terms (pre-cached, float32)
        query_idfs = self.index.idf_cache[term_ids]

        # Get term frequencies (sparse -> dense, only query term rows)
        tf_dense = self.index.term_doc_matrix[term_ids, :].toarray()

        # Pre-computed norm factors (float32)
        norm_factors = self.index.norm_factors

        # BM25 formula (optimized for float32)
        # score = IDF * (TF * (k1 + 1)) / (TF + k1 * norm)
        k1_plus_1 = np.float32(self.k1 + 1)
        k1 = np.float32(self.k1)

        numerator = tf_dense * k1_plus_1
        denominator = tf_dense + k1 * norm_factors
        np.maximum(denominator, 1e-10, out=denominator)  # In-place, avoid division by zero

        # Compute scores in-place where possible
        term_scores = numerator
        np.divide(numerator, denominator, out=term_scores)
        term_scores *= query_idfs[:, np.newaxis]

        # Sum and apply mask
        doc_scores = term_scores.sum(axis=0, dtype=np.float32)
        doc_scores *= doc_mask

        return doc_scores

    def get_top_k(self, scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """
        Get top-k documents by score.

        OPTIMIZED: Only considers non-zero scores (much smaller set than 200K).

        Morphism: R^n -> List[(DocID, Score)]
        """
        # OPTIMIZATION: Only work with non-zero scores
        nonzero_mask = scores > 0
        nonzero_indices = np.flatnonzero(nonzero_mask)

        if len(nonzero_indices) == 0:
            return []

        nonzero_scores = scores[nonzero_indices]

        # Get top-k from non-zero scores only
        if k >= len(nonzero_scores):
            sorted_order = np.argsort(nonzero_scores)[::-1]
        else:
            # Partial sort on small array
            sorted_order = np.argpartition(nonzero_scores, -k)[-k:]
            sorted_order = sorted_order[np.argsort(nonzero_scores[sorted_order])[::-1]]

        # Map back to original indices
        return [(int(nonzero_indices[i]), float(nonzero_scores[i])) for i in sorted_order]


# ============================================================================
# EAGER SPARSE SCORER: Pre-computed BM25 scores (BM25S approach)
# ============================================================================

class EagerSparseBM25Scorer:
    """
    Eager sparse BM25 scorer using pre-computed scores.

    This implements the BM25S approach: BM25 scores are pre-computed at index
    time and stored in a sparse matrix. Query-time scoring is reduced to:
    1. Select rows for query terms
    2. Sum across terms (axis=0)

    This eliminates all BM25 computation at query time, providing 3-5x speedup
    over computing scores on-the-fly.

    Categorical interpretation:
    - Pre-computation is a functor: Index -> ScoredIndex
    - Query-time is pure linear algebra (slice + sum)
    - Mathematically equivalent to on-the-fly scoring
    """

    def __init__(self, index: VectorizedIndexSparse):
        """
        Initialize eager scorer with pre-computed score matrix.

        Args:
            index: VectorizedIndexSparse with score_matrix already built
        """
        if index.score_matrix is None:
            raise ValueError(
                "Score matrix not built. Call index.build_score_matrix(k1, b) first."
            )
        self.index = index
        self.k1 = index._eager_k1
        self.b = index._eager_b

    def score_batch(self, query_terms: List[str], doc_mask: np.ndarray) -> np.ndarray:
        """
        Score documents using pre-computed scores (just slice + sum).

        This is the key optimization: no BM25 computation happens here.
        We just select the relevant rows and sum them.

        Args:
            query_terms: List of preprocessed query terms
            doc_mask: Boolean mask for candidate documents (unused but kept for API)

        Returns:
            Document scores as numpy array
        """
        # Get term IDs for query
        term_ids = [self.index.term_to_id[t] for t in query_terms
                    if t in self.index.term_to_id]

        if not term_ids:
            return np.zeros(self.index.num_docs, dtype=np.float32)

        # THE KEY OPTIMIZATION: Just slice and sum pre-computed scores
        # No BM25 formula computation needed!
        scores = np.asarray(
            self.index.score_matrix[term_ids, :].sum(axis=0)
        ).flatten().astype(np.float32)

        return scores

    def search_top_k(self, query_terms: List[str], k: int) -> List[Tuple[int, float]]:
        """
        Search for top-k documents using eager scoring.

        Args:
            query_terms: List of preprocessed query terms
            k: Number of top results to return

        Returns:
            List of (doc_id, score) tuples in descending score order
        """
        # Score all documents
        scores = self.score_batch(query_terms, None)

        # Get top-k using efficient partial sort
        nonzero_mask = scores > 0
        nonzero_indices = np.flatnonzero(nonzero_mask)

        if len(nonzero_indices) == 0:
            return []

        nonzero_scores = scores[nonzero_indices]

        # Get top-k from non-zero scores only
        if k >= len(nonzero_scores):
            sorted_order = np.argsort(nonzero_scores)[::-1]
        else:
            # Partial sort on small array
            sorted_order = np.argpartition(nonzero_scores, -k)[-k:]
            sorted_order = sorted_order[np.argsort(nonzero_scores[sorted_order])[::-1]]

        # Map back to original indices
        return [(int(nonzero_indices[i]), float(nonzero_scores[i])) for i in sorted_order]


# ============================================================================
# NUMBA JIT-COMPILED SCORER: Maximum performance through compilation
# ============================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _numba_score_terms_sparse(
        indptr: np.ndarray,      # CSR row pointers (num_terms + 1,)
        indices: np.ndarray,     # CSR column indices (nnz,)
        data: np.ndarray,        # CSR data / TF values (nnz,)
        term_ids: np.ndarray,    # Query term IDs (num_query_terms,)
        idfs: np.ndarray,        # IDF values for query terms (num_query_terms,)
        norm_factors: np.ndarray,  # Pre-computed doc length norms (num_docs,)
        k1: float,
        scores: np.ndarray,      # Output: document scores (num_docs,)
    ) -> None:
        """
        Numba JIT-compiled BM25 scoring using CSR sparse matrix directly.

        This is the hot path - compiled to machine code for maximum speed.

        Term-at-a-time (TAT) approach:
        - Iterate over query terms
        - For each term, iterate over its posting list
        - Accumulate BM25 scores

        Key optimizations:
        - No Python object overhead
        - Direct array access
        - Loop unrolling and SIMD via fastmath
        - No memory allocation in hot loop
        """
        k1_plus_1 = k1 + 1.0

        # Process each query term
        for t_idx in range(len(term_ids)):
            term_id = term_ids[t_idx]
            idf = idfs[t_idx]

            # Get posting list range for this term
            row_start = indptr[term_id]
            row_end = indptr[term_id + 1]

            # Score each document in posting list
            for j in range(row_start, row_end):
                doc_idx = indices[j]
                tf = data[j]
                norm = norm_factors[doc_idx]

                # BM25 formula: IDF * (TF * (k1 + 1)) / (TF + k1 * norm)
                score = idf * (tf * k1_plus_1) / (tf + k1 * norm)
                scores[doc_idx] += score

    @njit(cache=True)
    def _numba_get_top_k(
        scores: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numba JIT-compiled top-k selection.

        Returns (doc_indices, doc_scores) for top-k documents.
        Uses partial sort for O(n + k log k) complexity.
        """
        # Find non-zero scores
        nonzero_count = 0
        for i in range(len(scores)):
            if scores[i] > 0:
                nonzero_count += 1

        if nonzero_count == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        # Collect non-zero indices and scores
        nonzero_indices = np.empty(nonzero_count, dtype=np.int64)
        nonzero_scores = np.empty(nonzero_count, dtype=np.float32)

        j = 0
        for i in range(len(scores)):
            if scores[i] > 0:
                nonzero_indices[j] = i
                nonzero_scores[j] = scores[i]
                j += 1

        # If fewer than k non-zero, return all sorted
        if nonzero_count <= k:
            # Simple insertion sort for small arrays
            for i in range(1, nonzero_count):
                key_score = nonzero_scores[i]
                key_idx = nonzero_indices[i]
                j = i - 1
                while j >= 0 and nonzero_scores[j] < key_score:
                    nonzero_scores[j + 1] = nonzero_scores[j]
                    nonzero_indices[j + 1] = nonzero_indices[j]
                    j -= 1
                nonzero_scores[j + 1] = key_score
                nonzero_indices[j + 1] = key_idx
            return nonzero_indices, nonzero_scores

        # Partial sort: find k-th largest using partition
        # Then sort only top k
        result_indices = np.empty(k, dtype=np.int64)
        result_scores = np.empty(k, dtype=np.float32)

        # Use a simple heap-based approach for top-k
        # Initialize with first k elements
        for i in range(k):
            result_indices[i] = nonzero_indices[i]
            result_scores[i] = nonzero_scores[i]

        # Heapify (min-heap by score)
        for i in range(k // 2 - 1, -1, -1):
            _sift_down(result_scores, result_indices, i, k)

        # Process remaining elements
        for i in range(k, nonzero_count):
            if nonzero_scores[i] > result_scores[0]:
                result_scores[0] = nonzero_scores[i]
                result_indices[0] = nonzero_indices[i]
                _sift_down(result_scores, result_indices, 0, k)

        # Sort the top k (heap sort extract)
        # Min-heap extraction produces descending order (largest first)
        for i in range(k - 1, 0, -1):
            result_scores[0], result_scores[i] = result_scores[i], result_scores[0]
            result_indices[0], result_indices[i] = result_indices[i], result_indices[0]
            _sift_down(result_scores, result_indices, 0, i)

        return result_indices, result_scores

    @njit(cache=True)
    def _sift_down(scores: np.ndarray, indices: np.ndarray, start: int, end: int) -> None:
        """Min-heap sift down operation."""
        root = start
        while True:
            child = 2 * root + 1
            if child >= end:
                break
            if child + 1 < end and scores[child] > scores[child + 1]:
                child += 1
            if scores[root] > scores[child]:
                scores[root], scores[child] = scores[child], scores[root]
                indices[root], indices[child] = indices[child], indices[root]
                root = child
            else:
                break


class NumbaSparseBM25Scorer:
    """
    Numba JIT-compiled sparse BM25 scorer for maximum performance.

    Key differences from SparseBM25Scorer:
    - Works directly with CSR matrix arrays (no scipy overhead)
    - JIT-compiled scoring loop (no Python interpreter overhead)
    - Term-at-a-time approach (cache-friendly access pattern)
    - Pre-allocated score buffer (no allocation in hot path)

    Expected speedup: 5-20x over pure NumPy/SciPy implementation.
    """

    def __init__(self, index: VectorizedIndexSparse, k1: float = 1.5, b: float = 0.75):
        if not NUMBA_AVAILABLE:
            raise ImportError("Numba required for NumbaSparseBM25Scorer. Install with: pip install numba")

        self.index = index
        self.k1 = np.float64(k1)
        self.b = b

        # Extract CSR matrix components for direct access
        self.indptr = index.term_doc_matrix.indptr.astype(np.int64)
        self.indices = index.term_doc_matrix.indices.astype(np.int64)
        self.data = index.term_doc_matrix.data.astype(np.float64)

        # Pre-computed values
        self.idf_cache = index.idf_cache.astype(np.float64)
        self.norm_factors = index.norm_factors.astype(np.float64)

        # Pre-allocate score buffer
        self._scores_buffer = np.zeros(index.num_docs, dtype=np.float64)

        # Warm up JIT compilation
        self._warmup()

    def _warmup(self):
        """Trigger JIT compilation with dummy data."""
        dummy_terms = np.array([0], dtype=np.int64)
        dummy_idfs = np.array([1.0], dtype=np.float64)
        dummy_scores = np.zeros(10, dtype=np.float64)
        dummy_norms = np.ones(10, dtype=np.float64)
        dummy_indptr = np.array([0, 0], dtype=np.int64)
        dummy_indices = np.array([], dtype=np.int64)
        dummy_data = np.array([], dtype=np.float64)

        _numba_score_terms_sparse(
            dummy_indptr, dummy_indices, dummy_data,
            dummy_terms, dummy_idfs, dummy_norms,
            self.k1, dummy_scores
        )
        logger.debug("Numba scorer warmed up")

    def score_query(self, query_terms: List[str]) -> np.ndarray:
        """
        Score all documents for a query using Numba JIT.

        Returns dense score array (most entries will be zero).
        """
        # Get term IDs
        term_ids = []
        idfs = []
        for term in query_terms:
            if term in self.index.term_to_id:
                term_id = self.index.term_to_id[term]
                term_ids.append(term_id)
                idfs.append(self.idf_cache[term_id])

        if not term_ids:
            return np.zeros(self.index.num_docs, dtype=np.float32)

        # Convert to numpy arrays
        term_ids_arr = np.array(term_ids, dtype=np.int64)
        idfs_arr = np.array(idfs, dtype=np.float64)

        # Reset score buffer
        scores = self._scores_buffer
        scores.fill(0.0)

        # JIT-compiled scoring
        _numba_score_terms_sparse(
            self.indptr, self.indices, self.data,
            term_ids_arr, idfs_arr, self.norm_factors,
            self.k1, scores
        )

        return scores.astype(np.float32)

    def search_top_k(self, query_terms: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Score documents and return top-k results.

        Uses Numba JIT for both scoring and top-k selection.
        """
        scores = self.score_query(query_terms)

        # JIT-compiled top-k
        top_indices, top_scores = _numba_get_top_k(scores, top_k)

        return [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]


# ============================================================================
# COALGEBRAIC SCORER: MaxScore with Early Termination
# ============================================================================

class MaxScoreBM25Scorer:
    """
    Coalgebraic BM25 scorer with early termination (MaxScore algorithm).

    Categorical interpretation:
    - This is an ANAMORPHISM (unfold) with GUARDS
    - The guard checks: can this document make it to top-k?
    - If not, we skip it (coalgebraic "stop" branch)

    The key insight is that BM25 is a MONOID HOMOMORPHISM:
    - score(q1 ⊕ q2) = score(q1) + score(q2)
    - This additivity lets us compute upper bounds

    For each document, we track:
    - current_score: sum of scores from processed terms
    - upper_bound: max possible contribution from remaining terms

    If current_score + upper_bound < threshold, skip the document.

    OPTIMIZATION: Uses NumPy arrays instead of dicts for O(1) access.
    """

    def __init__(self, index: VectorizedIndexSparse, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def search_top_k(self, query_terms: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        MaxScore algorithm: coalgebraic search with early termination.

        Returns list of (doc_idx, score) tuples, sorted by score descending.
        """
        import heapq

        # Get term IDs and filter to known terms
        term_data = []
        for term in query_terms:
            if term in self.index.term_to_id:
                term_id = self.index.term_to_id[term]
                max_score = self.index.max_term_score[term_id]
                idf = self.index.idf_cache[term_id]
                term_data.append((term_id, max_score, idf))

        if not term_data:
            return []

        # Sort terms by max_score DESCENDING (essential terms first)
        term_data.sort(key=lambda x: x[1], reverse=True)
        term_ids = np.array([t[0] for t in term_data], dtype=np.int32)
        term_idfs = np.array([t[2] for t in term_data], dtype=np.float32)
        term_max_scores = np.array([t[1] for t in term_data], dtype=np.float32)

        # Compute cumulative upper bounds (remaining potential from each position)
        upper_bounds = np.zeros(len(term_ids) + 1, dtype=np.float32)
        for i in range(len(term_ids) - 1, -1, -1):
            upper_bounds[i] = upper_bounds[i + 1] + term_max_scores[i]

        # OPTIMIZATION: Use numpy array for scores (O(1) access)
        scores = np.zeros(self.index.num_docs, dtype=np.float32)
        touched = np.zeros(self.index.num_docs, dtype=bool)  # Track which docs were touched

        # Use min-heap for efficient top-k tracking (stores negative scores)
        top_k_heap = []  # (neg_score, doc_idx)
        threshold = 0.0

        # CSR matrix data for direct access
        indptr = self.index.term_doc_matrix.indptr
        indices = self.index.term_doc_matrix.indices
        data = self.index.term_doc_matrix.data
        norm_factors = self.index.norm_factors
        k1 = self.k1

        # Process terms in order of decreasing max contribution
        for t_idx in range(len(term_ids)):
            term_id = term_ids[t_idx]
            idf = term_idfs[t_idx]
            remaining_upper = upper_bounds[t_idx + 1]
            current_term_max = term_max_scores[t_idx]

            # Get posting list for this term
            row_start = indptr[term_id]
            row_end = indptr[term_id + 1]

            # Process posting list
            for j in range(row_start, row_end):
                doc_idx = indices[j]

                # ============================================================
                # COALGEBRAIC GUARD: Can this document make it to top-k?
                # ============================================================
                max_possible = scores[doc_idx] + remaining_upper + current_term_max
                if max_possible < threshold:
                    continue  # Skip - can't make top-k

                # Score this term-document pair
                tf = data[j]
                norm = norm_factors[doc_idx]
                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * norm)
                scores[doc_idx] += term_score
                touched[doc_idx] = True

                # Update heap if this doc might be in top-k
                new_score = scores[doc_idx]
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (new_score, doc_idx))
                    if len(top_k_heap) == top_k:
                        threshold = top_k_heap[0][0]  # Smallest in heap
                elif new_score > top_k_heap[0][0]:
                    heapq.heapreplace(top_k_heap, (new_score, doc_idx))
                    threshold = top_k_heap[0][0]

        # Extract top-k results from touched documents
        if not touched.any():
            return []

        # Get final top-k from all touched documents
        touched_indices = np.where(touched)[0]
        touched_scores = scores[touched_indices]

        if len(touched_indices) <= top_k:
            sorted_order = np.argsort(touched_scores)[::-1]
        else:
            top_k_order = np.argpartition(touched_scores, -top_k)[-top_k:]
            sorted_order = top_k_order[np.argsort(touched_scores[top_k_order])[::-1]]

        return [(int(touched_indices[i]), float(touched_scores[i])) for i in sorted_order]


# ============================================================================
# OPTIMIZATION 4: Optimized Search Coalgebra
# ============================================================================

@dataclass(frozen=True)
class QueryState:
    """Query state (unchanged - still an object in our category)"""
    query: str
    query_terms: Tuple[str, ...]


@dataclass
class SearchResult:
    """Search result (unchanged - still an object)"""
    document: Document
    score: float
    rank: int


class OptimizedBM25SearchCoalgebra:
    """
    Optimized coalgebra with vectorized operations.

    Structure map: QueryState -> List[SearchResult]

    Same categorical structure, but:
    - Vectorized candidate retrieval
    - Batch scoring
    - Memoized operations
    - Efficient top-k selection
    """

    def __init__(
        self,
        corpus: DocumentCorpus,
        index: VectorizedIndex,
        scorer: VectorizedBM25Scorer,
        top_k: int = 10
    ):
        self.corpus = corpus
        self.index = index
        self.scorer = scorer
        self.top_k = top_k

    def structure_map(self, state: QueryState) -> List[SearchResult]:
        """
        Optimized coalgebraic unfolding.

        alpha: QueryState -> List[SearchResult]

        Same mathematical structure, optimized implementation:
        1. Vectorized candidate retrieval
        2. Batch scoring (all candidates at once)
        3. Efficient top-k selection
        """
        # Get candidate documents (vectorized)
        candidate_mask = self.index.get_candidate_docs_vectorized(list(state.query_terms))

        if not candidate_mask.any():
            return []

        # Score all candidates at once (vectorized!)
        scores = self.scorer.score_batch(list(state.query_terms), candidate_mask)

        # Get top-k (efficient partial sort)
        top_docs = self.scorer.get_top_k(scores, self.top_k)

        # Convert to SearchResult objects
        results = []
        for rank, (doc_idx, score) in enumerate(top_docs, 1):
            doc_id = self.index.id_to_doc[doc_idx]
            doc = self.corpus.get(doc_id)
            if doc:
                results.append(SearchResult(
                    document=doc,
                    score=score,
                    rank=rank
                ))

        return results


# ============================================================================
# OPTIMIZATION 5: Optimized Search Engine
# ============================================================================

class VajraSearchOptimized:
    """
    High-performance Vajra BM25 search with coalgebraic optimization.

    Vajra (Sanskrit: vajra, "thunderbolt/diamond") optimized implementation.

    Maintains all categorical structure while using:
    - Vectorized operations (morphisms over vector spaces)
    - Memoization (comonadic caching)
    - Efficient data structures
    - Batch processing (functorial composition)
    - Sparse matrices (optional, for 100K+ documents)
    - MaxScore algorithm (coalgebraic early termination)

    The MaxScore algorithm exploits BM25's monoid homomorphism property:
    - score(q1 ⊕ q2) = score(q1) + score(q2)
    - Pre-computed term bounds enable early termination
    - Only documents that CAN make top-k are fully scored
    """

    def __init__(
        self,
        corpus: DocumentCorpus,
        k1: float = 1.5,
        b: float = 0.75,
        use_sparse: bool = False,
        cache_size: int = 1000,
        use_maxscore: bool = False,  # MaxScore disabled by default (Python too slow)
        use_numba: bool = True,  # Use Numba JIT scorer when available (much faster)
        use_eager: bool = True  # Use eager scoring (pre-computed BM25 scores) - fastest!
    ):
        self.corpus = corpus
        self.use_sparse = use_sparse
        self.use_maxscore = use_maxscore
        self.use_numba = use_numba and NUMBA_AVAILABLE
        self.use_eager = use_eager
        self.k1 = k1
        self.b = b

        # Initialize multi-level caching
        self.query_cache = LRUCache(capacity=cache_size) if cache_size > 0 else None

        # Determine whether to use sparse matrices
        # Automatically use sparse for large corpora if scipy available
        if use_sparse or (SCIPY_AVAILABLE and len(corpus) >= 10000):
            if not SCIPY_AVAILABLE:
                logger.warning("scipy not available, falling back to dense matrices. Install with: pip install scipy")
                use_sparse_actual = False
            else:
                use_sparse_actual = True
                logger.info(f"Using sparse matrices for corpus of {len(corpus)} documents")
        else:
            use_sparse_actual = False
            logger.info(f"Using dense matrices for corpus of {len(corpus)} documents")

        self._use_sparse_actual = use_sparse_actual

        # Build index (sparse or dense)
        if use_sparse_actual:
            logger.info("Building optimized SPARSE vectorized index...")
            start = time.time()
            self.index = VectorizedIndexSparse()
            self.index.build(corpus)
            build_time = time.time() - start
            logger.info(f"Built sparse index in {build_time:.3f}s ({self.index.num_terms} terms, {self.index.num_docs} docs)")

            # Create scorers
            self.scorer = SparseBM25Scorer(self.index, k1, b)
            self.maxscore_scorer = MaxScoreBM25Scorer(self.index, k1, b)

            # Create Numba scorer if available (primary scorer for speed)
            if self.use_numba:
                logger.info("Initializing Numba JIT scorer (this may take a moment on first run)...")
                self.numba_scorer = NumbaSparseBM25Scorer(self.index, k1, b)
                logger.info("Numba scorer ready")
            else:
                self.numba_scorer = None
                if use_numba and not NUMBA_AVAILABLE:
                    logger.warning("Numba not available. Install with: pip install numba")

            # Create eager scorer if enabled (fastest for query-time)
            if self.use_eager:
                logger.info("Building eager score matrix for fast query-time scoring...")
                self.index.build_score_matrix(k1, b)
                self.eager_scorer = EagerSparseBM25Scorer(self.index)
                logger.info("Eager scorer ready")
            else:
                self.eager_scorer = None
        else:
            logger.info("Building optimized vectorized index...")
            start = time.time()
            self.index = VectorizedIndex()
            self.index.build(corpus)
            build_time = time.time() - start
            logger.info(f"Built dense index in {build_time:.3f}s ({self.index.num_terms} terms, {self.index.num_docs} docs)")

            # Create vectorized scorer (MaxScore, Numba, and Eager not available for dense)
            self.scorer = VectorizedBM25Scorer(self.index, k1, b)
            self.maxscore_scorer = None
            self.numba_scorer = None
            self.eager_scorer = None

        logger.debug(f"Average document length: {self.index.avg_doc_length:.2f}")

    def save_index(self, filepath: Path):
        """
        Save index to disk for fast loading later.

        Categorical interpretation: Comonadic extraction
        - Serialize the cached structure map
        - Morphism: Index -> SerializedBytes
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required for index persistence. Install with: pip install joblib")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save index and scorer together
        index_data = {
            'index': self.index,
            'scorer': self.scorer,
            'use_sparse': self.use_sparse,
            'corpus_size': len(self.corpus),
            'cache_size': self.query_cache.capacity if self.query_cache else 0
        }

        joblib.dump(index_data, filepath, compress=3)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Index saved to {filepath} ({file_size_mb:.2f} MB)")

    @classmethod
    def load_index(cls, filepath: Path, corpus: DocumentCorpus):
        """
        Load pre-built index from disk.

        Categorical interpretation: Comonadic duplication
        - Deserialize cached structure map
        - Morphism: SerializedBytes -> Index
        """
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib required for index persistence. Install with: pip install joblib")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")

        logger.info(f"Loading index from {filepath}...")
        index_data = joblib.load(filepath)

        # Create instance without rebuilding index
        instance = cls.__new__(cls)
        instance.corpus = corpus
        instance.index = index_data['index']
        instance.scorer = index_data['scorer']
        instance.use_sparse = index_data.get('use_sparse', False)

        # Initialize query cache (default size: 1000)
        cache_size = index_data.get('cache_size', 1000)
        instance.query_cache = LRUCache(capacity=cache_size) if cache_size > 0 else None

        logger.info(f"Index loaded ({index_data.get('corpus_size', 'unknown')} documents)")

        return instance

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Execute optimized search with multi-level caching.

        Scorer preference (fastest to slowest):
        1. Cache hit (instant)
        2. Eager scorer (pre-computed BM25 scores, just slice + sum)
        3. Numba JIT scorer (if available)
        4. MaxScore algorithm (if enabled)
        5. Traditional NumPy/SciPy scorer

        Same categorical structure: Query -> List[SearchResult]
        """
        # Check cache first (comonadic extract)
        if self.query_cache:
            cache_key = f"{query}:{top_k}"
            cached_results = self.query_cache.get(cache_key)
            if cached_results is not None:
                return cached_results

        # Preprocess query
        query_terms = preprocess_text(query)

        if not query_terms:
            return []

        # Priority 1: Eager scorer (fastest - pre-computed scores)
        if self.eager_scorer is not None:
            top_docs = self.eager_scorer.search_top_k(query_terms, top_k)

            results = []
            for rank, (doc_idx, score) in enumerate(top_docs, 1):
                doc_id = self.index.id_to_doc[doc_idx]
                doc = self.corpus.get(doc_id)
                if doc:
                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        rank=rank
                    ))

        # Priority 2: Numba JIT scorer
        elif self.numba_scorer is not None:
            top_docs = self.numba_scorer.search_top_k(query_terms, top_k)

            results = []
            for rank, (doc_idx, score) in enumerate(top_docs, 1):
                doc_id = self.index.id_to_doc[doc_idx]
                doc = self.corpus.get(doc_id)
                if doc:
                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        rank=rank
                    ))

        # Priority 3: MaxScore algorithm (coalgebraic early termination)
        elif self.use_maxscore and self.maxscore_scorer is not None:
            top_docs = self.maxscore_scorer.search_top_k(query_terms, top_k)

            results = []
            for rank, (doc_idx, score) in enumerate(top_docs, 1):
                doc_id = self.index.id_to_doc[doc_idx]
                doc = self.corpus.get(doc_id)
                if doc:
                    results.append(SearchResult(
                        document=doc,
                        score=score,
                        rank=rank
                    ))

        # Priority 4: Traditional coalgebra approach (NumPy/SciPy)
        else:
            state = QueryState(
                query=query,
                query_terms=tuple(query_terms)
            )

            coalgebra = OptimizedBM25SearchCoalgebra(
                corpus=self.corpus,
                index=self.index,
                scorer=self.scorer,
                top_k=top_k
            )

            results = coalgebra.structure_map(state)

        # Cache results (comonadic duplication)
        if self.query_cache:
            cache_key = f"{query}:{top_k}"
            self.query_cache.put(cache_key, results)

        return results

    def get_cache_stats(self) -> Optional[Dict]:
        """Get cache statistics."""
        if self.query_cache:
            return self.query_cache.stats()
        return None

    def clear_cache(self):
        """Clear the query cache."""
        if self.query_cache:
            self.query_cache.clear()

    def add_document(self, doc: Document):
        """
        Add a document to the index incrementally.

        Categorical interpretation: Morphism (Index, Document) -> Index
        - Extends the index structure without full rebuild
        - Updates IDF values (affected morphisms)
        - Clears cache (invalidates cached results)

        Note: For sparse indices, this requires matrix reconstruction.
        For frequent updates, consider batch additions.
        """
        # Clear cache since results will change
        if self.query_cache:
            self.query_cache.clear()

        # Add to corpus
        if doc.id in [d.id for d in self.corpus.documents]:
            raise ValueError(f"Document {doc.id} already exists in corpus")

        # Add document to corpus
        self.corpus = DocumentCorpus(list(self.corpus.documents) + [doc])

        # Rebuild index (for now - could be optimized further)
        # This is still categorical: same morphism, just recomputed
        logger.info(f"Adding document {doc.id} (rebuilding index)...")
        start = time.time()

        if isinstance(self.index, VectorizedIndexSparse):
            self.index = VectorizedIndexSparse()
            self.index.build(self.corpus)
            self.scorer = SparseBM25Scorer(self.index, self.scorer.k1, self.scorer.b)
        else:
            self.index = VectorizedIndex()
            self.index.build(self.corpus)
            self.scorer = VectorizedBM25Scorer(self.index, self.scorer.k1, self.scorer.b)

        rebuild_time = time.time() - start
        logger.info(f"Index rebuilt in {rebuild_time:.3f}s")

    def remove_document(self, doc_id: str):
        """
        Remove a document from the index.

        Categorical interpretation: Morphism (Index, DocID) -> Index
        - Removes document from index structure
        - Updates IDF values
        - Clears cache

        Note: Requires index rebuild. For frequent removals,
        consider marking as deleted and rebuilding periodically.
        """
        # Clear cache
        if self.query_cache:
            self.query_cache.clear()

        # Remove from corpus
        remaining_docs = [d for d in self.corpus.documents if d.id != doc_id]

        if len(remaining_docs) == len(self.corpus.documents):
            raise ValueError(f"Document {doc_id} not found in corpus")

        self.corpus = DocumentCorpus(remaining_docs)

        # Rebuild index
        logger.info(f"Removing document {doc_id} (rebuilding index)...")
        start = time.time()

        if isinstance(self.index, VectorizedIndexSparse):
            self.index = VectorizedIndexSparse()
            self.index.build(self.corpus)
            self.scorer = SparseBM25Scorer(self.index, self.scorer.k1, self.scorer.b)
        else:
            self.index = VectorizedIndex()
            self.index.build(self.corpus)
            self.scorer = VectorizedBM25Scorer(self.index, self.scorer.k1, self.scorer.b)

        rebuild_time = time.time() - start
        logger.info(f"Index rebuilt in {rebuild_time:.3f}s")

    def batch_add_documents(self, docs: List[Document]):
        """
        Add multiple documents efficiently.

        Categorical interpretation: Morphism (Index, List[Document]) -> Index
        - Batches additions for efficiency
        - Single rebuild instead of N rebuilds
        """
        # Clear cache
        if self.query_cache:
            self.query_cache.clear()

        # Check for duplicates
        existing_ids = {d.id for d in self.corpus.documents}
        for doc in docs:
            if doc.id in existing_ids:
                raise ValueError(f"Document {doc.id} already exists")

        # Add all documents
        self.corpus = DocumentCorpus(list(self.corpus.documents) + docs)

        # Single rebuild
        logger.info(f"Adding {len(docs)} documents (rebuilding index)...")
        start = time.time()

        if isinstance(self.index, VectorizedIndexSparse):
            self.index = VectorizedIndexSparse()
            self.index.build(self.corpus)
            self.scorer = SparseBM25Scorer(self.index, self.scorer.k1, self.scorer.b)
        else:
            self.index = VectorizedIndex()
            self.index.build(self.corpus)
            self.scorer = VectorizedBM25Scorer(self.index, self.scorer.k1, self.scorer.b)

        rebuild_time = time.time() - start
        logger.info(f"Index rebuilt in {rebuild_time:.3f}s ({len(self.corpus)} total documents)")


if __name__ == "__main__":
    from vajra_bm25.documents import DocumentCorpus
    from pathlib import Path

    logger.info("="*70)
    logger.info("OPTIMIZED CATEGORICAL BM25")
    logger.info("="*70)

    # Load corpus
    corpus_path = Path("large_corpus.jsonl")
    if not corpus_path.exists():
        logger.error("Run generate_corpus.py first!")
        exit(1)

    logger.info("Loading corpus...")
    corpus = DocumentCorpus.load_jsonl(corpus_path)
    logger.info(f"Loaded {len(corpus)} documents")

    # Build optimized engine
    engine = VajraSearchOptimized(corpus)

    # Test queries
    test_queries = [
        "hypothesis testing statistical significance",
        "neural networks deep learning",
        "matrix eigenvalues",
    ]

    logger.info(f"{'Query':<40} {'Time (ms)':<12} {'Results':<10}")
    logger.info("-" * 70)

    for query in test_queries:
        start = time.time()
        results = engine.search(query, top_k=5)
        elapsed = (time.time() - start) * 1000

        logger.info(f"{query[:38]:<40} {elapsed:<12.3f} {len(results):<10}")

        if results:
            logger.info(f"  Top result: {results[0].document.title[:50]}")
            logger.info(f"  Score: {results[0].score:.3f}")
