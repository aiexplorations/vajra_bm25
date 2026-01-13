"""
Vector Index Interface

Abstract interface for vector indices. All index implementations
(Flat, HNSW, etc.) implement this interface.

The index is a morphism from query vectors to ranked results:
    VectorIndex: ℝ^d × ℕ → List[VectorSearchResult]
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VectorSearchResult:
    """
    Result from vector index search.

    Attributes:
        id: Document/vector identifier
        score: Similarity score (higher = more similar for similarity metrics)
        vector: The actual vector (optional, for debugging)
        metadata: Additional metadata associated with the vector
    """

    id: str
    score: float
    vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"VectorSearchResult(id={self.id!r}, score={self.score:.4f})"


class VectorIndex(ABC):
    """
    Abstract vector index interface.

    A vector index is a morphism: Query × k → List[Result]
    It maps a query vector and desired count to ranked results.

    All index implementations (Flat, HNSW, IVF, etc.) implement this interface.
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add vectors to the index.

        Args:
            ids: List of unique identifiers for each vector
            vectors: (n, d) array of vectors to add
            metadata: Optional list of metadata dicts for each vector
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> List[VectorSearchResult]:
        """
        Find k nearest neighbors to query vector.

        Args:
            query: Query vector (d,)
            k: Number of results to return

        Returns:
            List of k VectorSearchResult, sorted by score (best first)
        """
        pass

    def search_batch(
        self, queries: np.ndarray, k: int
    ) -> List[List[VectorSearchResult]]:
        """
        Batch search for multiple queries.

        Default implementation calls search() for each query.
        Subclasses can override for more efficient batch processing.

        Args:
            queries: (n, d) array of query vectors
            k: Number of results per query

        Returns:
            List of result lists, one per query
        """
        return [self.search(q, k) for q in queries]

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors in the index"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of vectors in the index"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist index to disk.

        Args:
            path: File path to save to
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "VectorIndex":
        """
        Load index from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded index instance
        """
        pass

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by ID.

        Default implementation returns None (not all indices support this).
        Subclasses can override to provide vector retrieval.

        Args:
            id: Vector identifier

        Returns:
            The vector, or None if not found/supported
        """
        return None

    def remove(self, ids: List[str]) -> int:
        """
        Remove vectors from the index.

        Default implementation raises NotImplementedError.
        Not all index types support removal.

        Args:
            ids: List of IDs to remove

        Returns:
            Number of vectors actually removed
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support vector removal"
        )

    def __len__(self) -> int:
        return self.size

    def __contains__(self, id: str) -> bool:
        """Check if an ID exists in the index"""
        return self.get_vector(id) is not None
