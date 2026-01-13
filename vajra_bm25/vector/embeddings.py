"""
Embedding Morphisms: Text → ℝ^d

Embeddings as categorical morphisms that map text to vector space.
These are the fundamental transformations for semantic search.

Morphism structure:
    EmbeddingMorphism: T → ℝ^d
    - Maps objects of type T to d-dimensional real vectors
    - Preserves semantic structure (similar texts → nearby vectors)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, TypeVar, Optional, Union
from dataclasses import dataclass
import numpy as np

from vajra_bm25.categorical import Morphism


T = TypeVar("T")


class EmbeddingMorphism(Morphism[T, np.ndarray], ABC):
    """
    Abstract morphism: T → ℝ^d (embedding space)

    Maps objects to dense vector representations.
    The vector space structure enables similarity computations.
    """

    @abstractmethod
    def embed(self, item: T) -> np.ndarray:
        """
        Embed a single item into vector space.

        Args:
            item: Object to embed

        Returns:
            d-dimensional numpy array
        """
        pass

    @abstractmethod
    def embed_batch(self, items: List[T]) -> np.ndarray:
        """
        Batch embedding for efficiency.

        Args:
            items: List of objects to embed

        Returns:
            (n, d) numpy array where n = len(items)
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality"""
        pass

    def apply(self, a: T) -> np.ndarray:
        """Morphism interface: apply embedding"""
        return self.embed(a)


class TextEmbeddingMorphism(EmbeddingMorphism[str]):
    """
    Text → Embedding using sentence-transformers.

    Maps natural language text to dense vectors that capture semantic meaning.
    Similar sentences are mapped to nearby vectors in the embedding space.

    Usage:
        embedder = TextEmbeddingMorphism("all-MiniLM-L6-v2")
        vector = embedder.embed("Hello world")
        vectors = embedder.embed_batch(["Hello", "World"])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize text embedding morphism.

        Args:
            model_name: Name of sentence-transformers model
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            device: Device to run model on ("cpu", "cuda", "mps")
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._dimension: Optional[int] = None
        self._device = device

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for TextEmbeddingMorphism. "
                    "Install with: pip install sentence-transformers"
                )

            self._model = SentenceTransformer(self.model_name, device=self._device)
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Get embedding dimension (loads model if needed)"""
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text

        Returns:
            d-dimensional embedding vector
        """
        self._load_model()
        embedding = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch embed multiple texts.

        Args:
            texts: List of input texts

        Returns:
            (n, d) array of embeddings
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)


class PrecomputedEmbeddingMorphism(EmbeddingMorphism[str]):
    """
    Lookup morphism for pre-computed embeddings.

    When embeddings are already computed (e.g., loaded from file),
    this morphism provides efficient lookup without recomputation.

    Usage:
        embeddings = {"doc1": np.array([...]), "doc2": np.array([...])}
        embedder = PrecomputedEmbeddingMorphism(embeddings)
        vector = embedder.embed("doc1")
    """

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        """
        Initialize with pre-computed embeddings.

        Args:
            embeddings: Dictionary mapping IDs to embedding vectors
        """
        if not embeddings:
            raise ValueError("embeddings dictionary cannot be empty")

        self._embeddings = embeddings
        first_vector = next(iter(embeddings.values()))
        self._dimension = first_vector.shape[0]

        # Validate all embeddings have same dimension
        for id_, vec in embeddings.items():
            if vec.shape[0] != self._dimension:
                raise ValueError(
                    f"Embedding for '{id_}' has dimension {vec.shape[0]}, "
                    f"expected {self._dimension}"
                )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, item_id: str) -> np.ndarray:
        """
        Look up embedding by ID.

        Args:
            item_id: ID of the item

        Returns:
            Pre-computed embedding vector

        Raises:
            KeyError: If item_id not in embeddings
        """
        if item_id not in self._embeddings:
            raise KeyError(f"No embedding for ID: {item_id}")
        return self._embeddings[item_id].astype(np.float32)

    def embed_batch(self, item_ids: List[str]) -> np.ndarray:
        """
        Batch lookup of embeddings.

        Args:
            item_ids: List of IDs

        Returns:
            (n, d) array of embeddings
        """
        return np.vstack([self.embed(id_) for id_ in item_ids]).astype(np.float32)

    def add(self, item_id: str, embedding: np.ndarray) -> None:
        """Add a new embedding to the lookup table."""
        if embedding.shape[0] != self._dimension:
            raise ValueError(
                f"Embedding has dimension {embedding.shape[0]}, "
                f"expected {self._dimension}"
            )
        self._embeddings[item_id] = embedding.astype(np.float32)

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._embeddings

    def __len__(self) -> int:
        return len(self._embeddings)


class IdentityEmbeddingMorphism(EmbeddingMorphism[np.ndarray]):
    """
    Identity morphism for pre-embedded data.

    When data is already in vector form, this morphism passes it through unchanged.
    Useful for composing with other morphisms or when vectors come from external sources.
    """

    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] != self._dimension:
            raise ValueError(
                f"Vector has dimension {vector.shape[0]}, expected {self._dimension}"
            )
        return vector.astype(np.float32)

    def embed_batch(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Vectors have dimension {vectors.shape[1]}, expected {self._dimension}"
            )
        return vectors.astype(np.float32)
