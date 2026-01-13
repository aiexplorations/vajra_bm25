"""
Similarity Morphisms: (ℝ^d, ℝ^d) → ℝ

Similarity and distance functions as categorical morphisms.
These measure how close two vectors are in embedding space.

Morphism structure:
    SimilarityMorphism: (ℝ^d × ℝ^d) → ℝ
    - Takes a pair of vectors
    - Returns a scalar similarity/distance score
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from vajra_bm25.categorical import Morphism


# Type alias for vector pair
VectorPair = Tuple[np.ndarray, np.ndarray]


class SimilarityMorphism(Morphism[VectorPair, float], ABC):
    """
    Abstract morphism: (ℝ^d, ℝ^d) → ℝ

    Computes similarity or distance between two vectors.
    Higher values indicate more similarity (for similarity metrics)
    or less similarity (for distance metrics).
    """

    @abstractmethod
    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity/distance between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Scalar similarity or distance score
        """
        pass

    @abstractmethod
    def score_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute scores between query and multiple vectors.

        Args:
            query: Query vector (d,)
            vectors: Matrix of vectors (n, d)

        Returns:
            Array of scores (n,)
        """
        pass

    def apply(self, pair: VectorPair) -> float:
        """Morphism interface"""
        return self.score(pair[0], pair[1])

    @property
    @abstractmethod
    def is_distance(self) -> bool:
        """True if lower values mean more similar (distance metric)"""
        pass


@dataclass
class CosineSimilarity(SimilarityMorphism):
    """
    Cosine similarity: dot(a, b) / (||a|| × ||b||)

    Range: [-1, 1] for general vectors, [0, 1] for normalized vectors
    Higher values = more similar

    For normalized vectors, this equals the dot product.
    """

    eps: float = 1e-8  # Numerical stability

    @property
    def is_distance(self) -> bool:
        return False

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return float(dot / (norm_a * norm_b + self.eps))

    def score_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and multiple vectors"""
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + self.eps)

        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm = vectors / (norms + self.eps)

        # Dot product gives cosine similarity for normalized vectors
        return (vectors_norm @ query_norm).astype(np.float32)

    def score_batch_normalized(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """
        Optimized batch scoring for pre-normalized vectors.

        When vectors are already L2-normalized, cosine similarity
        is just the dot product.
        """
        return (vectors @ query).astype(np.float32)


@dataclass
class L2Distance(SimilarityMorphism):
    """
    L2 (Euclidean) distance: ||a - b||₂

    Range: [0, ∞)
    Lower values = more similar

    This is a distance metric, so is_distance=True.
    """

    @property
    def is_distance(self) -> bool:
        return True

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute L2 distance between two vectors"""
        diff = a - b
        return float(np.sqrt(np.dot(diff, diff)))

    def score_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute L2 distances between query and multiple vectors"""
        diff = vectors - query
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)


@dataclass
class SquaredL2Distance(SimilarityMorphism):
    """
    Squared L2 distance: ||a - b||₂²

    Avoids the sqrt computation for faster comparisons.
    Monotonic with L2, so valid for ranking.
    """

    @property
    def is_distance(self) -> bool:
        return True

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute squared L2 distance"""
        diff = a - b
        return float(np.dot(diff, diff))

    def score_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute squared L2 distances"""
        diff = vectors - query
        return np.sum(diff * diff, axis=1).astype(np.float32)


@dataclass
class InnerProduct(SimilarityMorphism):
    """
    Inner (dot) product: a · b

    Range: (-∞, ∞)
    Higher values = more similar

    For normalized vectors, equivalent to cosine similarity.
    For unnormalized vectors, captures both direction and magnitude.
    """

    @property
    def is_distance(self) -> bool:
        return False

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute inner product"""
        return float(np.dot(a, b))

    def score_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute inner products"""
        return (vectors @ query).astype(np.float32)


# Numba-accelerated versions (optional, for HNSW)
_numba_available = False
try:
    import numba

    _numba_available = True

    @numba.jit(nopython=True, fastmath=True)
    def _l2_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
        """Numba-accelerated L2 distance"""
        diff = a - b
        return np.sqrt(np.dot(diff, diff))

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _l2_distance_batch_numba(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Numba-accelerated batch L2 distance"""
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float32)
        for i in numba.prange(n):
            diff = query - vectors[i]
            distances[i] = np.sqrt(np.dot(diff, diff))
        return distances

    @numba.jit(nopython=True, fastmath=True)
    def _cosine_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
        """Numba-accelerated cosine distance (1 - similarity)"""
        dot = np.dot(a, b)
        norm_a = np.sqrt(np.dot(a, a))
        norm_b = np.sqrt(np.dot(b, b))
        return 1.0 - dot / (norm_a * norm_b + 1e-8)

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _cosine_distance_batch_numba(
        query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated batch cosine distance"""
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float32)
        norm_q = np.sqrt(np.dot(query, query))

        for i in numba.prange(n):
            dot = np.dot(query, vectors[i])
            norm_v = np.sqrt(np.dot(vectors[i], vectors[i]))
            distances[i] = 1.0 - dot / (norm_q * norm_v + 1e-8)
        return distances

    @numba.jit(nopython=True, fastmath=True)
    def _inner_product_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
        """Negative inner product (for max inner product search)"""
        return -np.dot(a, b)

    @numba.jit(nopython=True, fastmath=True, parallel=True)
    def _inner_product_distance_batch_numba(
        query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Numba-accelerated batch negative inner product"""
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float32)
        for i in numba.prange(n):
            distances[i] = -np.dot(query, vectors[i])
        return distances

except ImportError:
    pass


def get_distance_function(metric: str, use_numba: bool = True):
    """
    Get the appropriate distance function for a metric.

    Args:
        metric: One of "cosine", "l2", "ip" (inner product)
        use_numba: Whether to use Numba-accelerated versions if available

    Returns:
        Tuple of (single_distance_fn, batch_distance_fn)
    """
    if use_numba and _numba_available:
        if metric == "cosine":
            return _cosine_distance_numba, _cosine_distance_batch_numba
        elif metric == "l2":
            return _l2_distance_numba, _l2_distance_batch_numba
        elif metric == "ip":
            return _inner_product_distance_numba, _inner_product_distance_batch_numba

    # Fallback to numpy implementations
    if metric == "cosine":
        scorer = CosineSimilarity()
        # Convert similarity to distance for HNSW
        def single_fn(a, b):
            return 1.0 - scorer.score(a, b)

        def batch_fn(q, vs):
            return 1.0 - scorer.score_batch(q, vs)

        return single_fn, batch_fn

    elif metric == "l2":
        scorer = L2Distance()
        return scorer.score, scorer.score_batch

    elif metric == "ip":
        scorer = InnerProduct()
        # Negate for distance (lower = better)
        def single_fn(a, b):
            return -scorer.score(a, b)

        def batch_fn(q, vs):
            return -scorer.score_batch(q, vs)

        return single_fn, batch_fn

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'cosine', 'l2', or 'ip'")
