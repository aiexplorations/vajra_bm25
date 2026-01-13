"""
Flat Vector Index: Exact Brute-Force Search

The simplest vector index: stores all vectors and computes exact
distances to the query at search time.

Use cases:
- Small datasets (< 10K vectors)
- Accuracy baseline for approximate methods
- When you need 100% recall

Complexity:
- Add: O(1)
- Search: O(n × d) where n = num vectors, d = dimension
- Space: O(n × d)
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import pickle

from vajra_bm25.vector.index import VectorIndex, VectorSearchResult
from vajra_bm25.vector.scorer import (
    CosineSimilarity,
    L2Distance,
    InnerProduct,
    get_distance_function,
)


class FlatVectorIndex(VectorIndex):
    """
    Brute-force exact nearest neighbor search.

    Stores all vectors in a numpy array and computes exact distances
    at query time. Simple, accurate, but O(n) per query.

    Usage:
        index = FlatVectorIndex(dimension=384, metric="cosine")
        index.add(["doc1", "doc2"], vectors)
        results = index.search(query_vector, k=10)
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        normalize: bool = True,
    ):
        """
        Initialize flat vector index.

        Args:
            dimension: Vector dimensionality
            metric: Distance metric ("cosine", "l2", or "ip")
            normalize: Whether to L2-normalize vectors on add (for cosine)
        """
        self._dimension = dimension
        self.metric = metric
        self.normalize = normalize and metric == "cosine"

        # Storage
        self._ids: List[str] = []
        self._vectors: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        # Scorer
        if metric == "cosine":
            self._scorer = CosineSimilarity()
        elif metric == "l2":
            self._scorer = L2Distance()
        elif metric == "ip":
            self._scorer = InnerProduct()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._ids)

    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add vectors to the index.

        Args:
            ids: Unique identifiers
            vectors: (n, d) array of vectors
            metadata: Optional metadata for each vector
        """
        vectors = vectors.astype(np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vectors have dimension {vectors.shape[1]}, "
                f"expected {self._dimension}"
            )

        # Normalize if needed
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            vectors = vectors / norms

        # Add to storage
        start_idx = len(self._ids)
        for i, id_ in enumerate(ids):
            if id_ in self._id_to_idx:
                raise ValueError(f"Duplicate ID: {id_}")
            self._id_to_idx[id_] = start_idx + i
            self._ids.append(id_)

            if metadata and i < len(metadata):
                self._metadata[id_] = metadata[i]

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """
        Find k nearest neighbors.

        Args:
            query: Query vector (d,)
            k: Number of results

        Returns:
            List of VectorSearchResult sorted by relevance
        """
        if self._vectors is None or len(self._ids) == 0:
            return []

        query = query.astype(np.float32)

        # Normalize query if needed
        if self.normalize:
            norm = np.linalg.norm(query)
            if norm > 1e-8:
                query = query / norm

        # Compute scores
        scores = self._scorer.score_batch(query, self._vectors)

        # For distance metrics, lower is better
        # For similarity metrics, higher is better
        if self._scorer.is_distance:
            # Get k smallest distances
            if k < len(scores):
                top_k_idx = np.argpartition(scores, k)[:k]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])]
            else:
                top_k_idx = np.argsort(scores)
            # Convert distance to similarity-like score (negate)
            result_scores = -scores[top_k_idx]
        else:
            # Get k largest similarities
            if k < len(scores):
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(scores)[::-1]
            result_scores = scores[top_k_idx]

        # Build results
        results = []
        for idx, score in zip(top_k_idx[:k], result_scores[:k]):
            id_ = self._ids[idx]
            results.append(
                VectorSearchResult(
                    id=id_,
                    score=float(score),
                    vector=self._vectors[idx].copy(),
                    metadata=self._metadata.get(id_, {}),
                )
            )

        return results

    def search_batch(
        self, queries: np.ndarray, k: int
    ) -> List[List[VectorSearchResult]]:
        """
        Batch search for multiple queries.

        More efficient than individual searches due to matrix operations.
        """
        if self._vectors is None or len(self._ids) == 0:
            return [[] for _ in range(len(queries))]

        queries = queries.astype(np.float32)

        # Normalize queries if needed
        if self.normalize:
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            queries = queries / norms

        # Compute all scores at once (matrix multiply for cosine/ip)
        if self.metric in ("cosine", "ip"):
            # (n_queries, n_vectors)
            all_scores = queries @ self._vectors.T
            if self.metric == "cosine":
                pass  # Already similarity
            # For ip, scores are already correct
        else:
            # L2: compute pairwise distances
            # (n_queries, n_vectors)
            diff = queries[:, np.newaxis, :] - self._vectors[np.newaxis, :, :]
            all_scores = -np.sqrt(np.sum(diff * diff, axis=2))

        # Get top-k for each query
        results = []
        for i, scores in enumerate(all_scores):
            if k < len(scores):
                top_k_idx = np.argpartition(scores, -k)[-k:]
                top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            else:
                top_k_idx = np.argsort(scores)[::-1]

            query_results = []
            for idx in top_k_idx[:k]:
                id_ = self._ids[idx]
                query_results.append(
                    VectorSearchResult(
                        id=id_,
                        score=float(scores[idx]),
                        vector=self._vectors[idx].copy(),
                        metadata=self._metadata.get(id_, {}),
                    )
                )
            results.append(query_results)

        return results

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get vector by ID"""
        idx = self._id_to_idx.get(id)
        if idx is not None and self._vectors is not None:
            return self._vectors[idx].copy()
        return None

    def remove(self, ids: List[str]) -> int:
        """
        Remove vectors from index.

        Note: This is O(n) as it requires rebuilding the array.
        For frequent removals, consider a different index type.
        """
        indices_to_remove = []
        for id_ in ids:
            idx = self._id_to_idx.get(id_)
            if idx is not None:
                indices_to_remove.append(idx)

        if not indices_to_remove:
            return 0

        # Create mask of indices to keep
        keep_mask = np.ones(len(self._ids), dtype=bool)
        keep_mask[indices_to_remove] = False

        # Filter vectors
        if self._vectors is not None:
            self._vectors = self._vectors[keep_mask]
            if len(self._vectors) == 0:
                self._vectors = None

        # Rebuild ID mappings
        new_ids = []
        new_id_to_idx = {}
        for i, (keep, id_) in enumerate(zip(keep_mask, self._ids)):
            if keep:
                new_id_to_idx[id_] = len(new_ids)
                new_ids.append(id_)
            else:
                self._metadata.pop(id_, None)

        self._ids = new_ids
        self._id_to_idx = new_id_to_idx

        return len(indices_to_remove)

    def save(self, path: str) -> None:
        """Save index to disk"""
        data = {
            "dimension": self._dimension,
            "metric": self.metric,
            "normalize": self.normalize,
            "ids": self._ids,
            "vectors": self._vectors,
            "metadata": self._metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "FlatVectorIndex":
        """Load index from disk"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls(
            dimension=data["dimension"],
            metric=data["metric"],
            normalize=data["normalize"],
        )
        index._ids = data["ids"]
        index._vectors = data["vectors"]
        index._id_to_idx = {id_: i for i, id_ in enumerate(data["ids"])}
        index._metadata = data.get("metadata", {})

        return index

    def __repr__(self) -> str:
        return (
            f"FlatVectorIndex(dimension={self._dimension}, "
            f"metric={self.metric!r}, size={self.size})"
        )
