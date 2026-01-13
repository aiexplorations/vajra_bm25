"""
Hybrid Search: BM25 + Vector Fusion

Combines keyword-based BM25 search with semantic vector search
using score fusion techniques.

Why hybrid search?
- BM25 excels at exact keyword matches
- Vector search captures semantic similarity
- Combined: best of both worlds

Fusion methods:
- Reciprocal Rank Fusion (RRF): Robust, parameter-free
- Linear combination: Simple weighted average
- Relative Score Fusion: Normalizes scores before combining

Usage:
    from vajra_bm25 import VajraSearchOptimized
    from vajra_bm25.vector import VajraVectorSearch, HybridSearchEngine

    hybrid = HybridSearchEngine(
        bm25_engine=bm25_engine,
        vector_engine=vector_engine,
        alpha=0.5,  # BM25 weight
        method="rrf"
    )
    results = hybrid.search("query", top_k=10)
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from vajra_bm25.documents import Document
from vajra_bm25.search import SearchResult


class FusionMethod(Enum):
    """Score fusion methods"""

    RRF = "rrf"  # Reciprocal Rank Fusion
    LINEAR = "linear"  # Linear combination
    RSF = "rsf"  # Relative Score Fusion


@dataclass
class HybridSearchResult:
    """Result from hybrid search with component scores"""

    document: Document
    score: float
    rank: int
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    vector_rank: Optional[int] = None


class HybridSearchEngine:
    """
    Hybrid search combining BM25 and vector search.

    Uses score fusion to combine results from both retrieval methods.
    The default method (RRF) is robust and doesn't require score normalization.

    Usage:
        hybrid = HybridSearchEngine(bm25_engine, vector_engine)
        results = hybrid.search("machine learning papers", top_k=10)
    """

    def __init__(
        self,
        bm25_engine,  # VajraSearchOptimized or similar
        vector_engine,  # VajraVectorSearch
        alpha: float = 0.5,
        method: Literal["rrf", "linear", "rsf"] = "rrf",
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search engine.

        Args:
            bm25_engine: BM25 search engine (VajraSearchOptimized)
            vector_engine: Vector search engine (VajraVectorSearch)
            alpha: Weight for BM25 in fusion (0-1). Higher = more BM25 influence.
            method: Fusion method ("rrf", "linear", "rsf")
            rrf_k: Constant for RRF formula (default 60)
        """
        self.bm25 = bm25_engine
        self.vector = vector_engine
        self.alpha = alpha
        self.method = FusionMethod(method)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining BM25 and vector results.

        Args:
            query: Search query
            top_k: Number of results to return
            bm25_weight: Override default alpha (BM25 weight)

        Returns:
            List of SearchResult sorted by fused score
        """
        alpha = bm25_weight if bm25_weight is not None else self.alpha

        # Get results from both engines (fetch more for better fusion)
        fetch_k = top_k * 2
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        vector_results = self.vector.search(query, top_k=fetch_k)

        # Fuse results
        if self.method == FusionMethod.RRF:
            fused = self._rrf_fusion(bm25_results, vector_results, alpha)
        elif self.method == FusionMethod.LINEAR:
            fused = self._linear_fusion(bm25_results, vector_results, alpha)
        else:
            fused = self._rsf_fusion(bm25_results, vector_results, alpha)

        # Sort by fused score and take top_k
        sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        # Convert to SearchResult
        return [
            SearchResult(
                document=item["document"],
                score=item["score"],
                rank=i + 1,
            )
            for i, item in enumerate(sorted_results[:top_k])
        ]

    def search_detailed(
        self,
        query: str,
        top_k: int = 10,
        bm25_weight: Optional[float] = None,
    ) -> List[HybridSearchResult]:
        """
        Hybrid search with detailed component scores.

        Returns HybridSearchResult which includes both BM25 and vector
        scores/ranks for analysis.
        """
        alpha = bm25_weight if bm25_weight is not None else self.alpha

        fetch_k = top_k * 2
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        vector_results = self.vector.search(query, top_k=fetch_k)

        # Build detailed results
        results_map: Dict[str, Dict] = {}

        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.document.id
            results_map[doc_id] = {
                "document": result.document,
                "bm25_score": result.score,
                "bm25_rank": rank,
                "vector_score": None,
                "vector_rank": None,
            }

        # Add vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document.id
            if doc_id in results_map:
                results_map[doc_id]["vector_score"] = result.score
                results_map[doc_id]["vector_rank"] = rank
            else:
                results_map[doc_id] = {
                    "document": result.document,
                    "bm25_score": None,
                    "bm25_rank": None,
                    "vector_score": result.score,
                    "vector_rank": rank,
                }

        # Compute fused scores
        if self.method == FusionMethod.RRF:
            for doc_id, item in results_map.items():
                bm25_contrib = 0
                vector_contrib = 0

                if item["bm25_rank"]:
                    bm25_contrib = alpha / (self.rrf_k + item["bm25_rank"])
                if item["vector_rank"]:
                    vector_contrib = (1 - alpha) / (self.rrf_k + item["vector_rank"])

                item["score"] = bm25_contrib + vector_contrib
        else:
            # Linear/RSF fusion
            self._add_linear_scores(results_map, alpha, bm25_results, vector_results)

        # Sort and convert
        sorted_items = sorted(
            results_map.values(), key=lambda x: x["score"], reverse=True
        )

        return [
            HybridSearchResult(
                document=item["document"],
                score=item["score"],
                rank=i + 1,
                bm25_score=item["bm25_score"],
                bm25_rank=item["bm25_rank"],
                vector_score=item["vector_score"],
                vector_rank=item["vector_rank"],
            )
            for i, item in enumerate(sorted_items[:top_k])
        ]

    def _rrf_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        alpha: float,
    ) -> Dict[str, Dict]:
        """
        Reciprocal Rank Fusion.

        RRF(d) = sum over all rankings r: 1 / (k + rank_r(d))

        With weighted version:
        RRF(d) = alpha * 1/(k + bm25_rank) + (1-alpha) * 1/(k + vector_rank)
        """
        results: Dict[str, Dict] = {}

        # Add BM25 contributions
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.document.id
            rrf_score = alpha / (self.rrf_k + rank)

            if doc_id in results:
                results[doc_id]["score"] += rrf_score
            else:
                results[doc_id] = {"document": result.document, "score": rrf_score}

        # Add vector contributions
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document.id
            rrf_score = (1 - alpha) / (self.rrf_k + rank)

            if doc_id in results:
                results[doc_id]["score"] += rrf_score
            else:
                results[doc_id] = {"document": result.document, "score": rrf_score}

        return results

    def _linear_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        alpha: float,
    ) -> Dict[str, Dict]:
        """
        Linear combination of normalized scores.

        score(d) = alpha * norm_bm25(d) + (1-alpha) * norm_vector(d)

        Scores are min-max normalized to [0, 1].
        """
        results: Dict[str, Dict] = {}

        # Normalize BM25 scores
        bm25_scores = [r.score for r in bm25_results]
        bm25_min = min(bm25_scores) if bm25_scores else 0
        bm25_max = max(bm25_scores) if bm25_scores else 1
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

        # Normalize vector scores
        vector_scores = [r.score for r in vector_results]
        vector_min = min(vector_scores) if vector_scores else 0
        vector_max = max(vector_scores) if vector_scores else 1
        vector_range = vector_max - vector_min if vector_max > vector_min else 1

        # Add BM25 contributions
        for result in bm25_results:
            doc_id = result.document.id
            norm_score = (result.score - bm25_min) / bm25_range
            weighted = alpha * norm_score

            results[doc_id] = {"document": result.document, "score": weighted}

        # Add vector contributions
        for result in vector_results:
            doc_id = result.document.id
            norm_score = (result.score - vector_min) / vector_range
            weighted = (1 - alpha) * norm_score

            if doc_id in results:
                results[doc_id]["score"] += weighted
            else:
                results[doc_id] = {"document": result.document, "score": weighted}

        return results

    def _rsf_fusion(
        self,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
        alpha: float,
    ) -> Dict[str, Dict]:
        """
        Relative Score Fusion.

        Similar to linear but normalizes relative to the best score:
        norm_score = score / max_score
        """
        results: Dict[str, Dict] = {}

        # Get max scores
        bm25_max = max((r.score for r in bm25_results), default=1)
        vector_max = max((r.score for r in vector_results), default=1)

        # Avoid division by zero
        bm25_max = bm25_max if bm25_max > 0 else 1
        vector_max = vector_max if vector_max > 0 else 1

        # Add BM25 contributions
        for result in bm25_results:
            doc_id = result.document.id
            norm_score = result.score / bm25_max
            weighted = alpha * norm_score

            results[doc_id] = {"document": result.document, "score": weighted}

        # Add vector contributions
        for result in vector_results:
            doc_id = result.document.id
            norm_score = result.score / vector_max
            weighted = (1 - alpha) * norm_score

            if doc_id in results:
                results[doc_id]["score"] += weighted
            else:
                results[doc_id] = {"document": result.document, "score": weighted}

        return results

    def _add_linear_scores(
        self,
        results_map: Dict[str, Dict],
        alpha: float,
        bm25_results: List[SearchResult],
        vector_results: List[SearchResult],
    ) -> None:
        """Helper to add linear scores to results_map"""
        # Get score ranges
        bm25_scores = [r.score for r in bm25_results]
        bm25_min = min(bm25_scores) if bm25_scores else 0
        bm25_max = max(bm25_scores) if bm25_scores else 1
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

        vector_scores = [r.score for r in vector_results]
        vector_min = min(vector_scores) if vector_scores else 0
        vector_max = max(vector_scores) if vector_scores else 1
        vector_range = vector_max - vector_min if vector_max > vector_min else 1

        # Compute scores
        for doc_id, item in results_map.items():
            score = 0

            if item["bm25_score"] is not None:
                norm = (item["bm25_score"] - bm25_min) / bm25_range
                score += alpha * norm

            if item["vector_score"] is not None:
                norm = (item["vector_score"] - vector_min) / vector_range
                score += (1 - alpha) * norm

            item["score"] = score

    def __repr__(self) -> str:
        return (
            f"HybridSearchEngine(method={self.method.value!r}, "
            f"alpha={self.alpha})"
        )
