"""
Advanced optimizations: Parallel processing and batch queries

Additional categorical optimizations:
- Parallel coalgebra unfolding (concurrent morphism application)
- Batch query processing (functor over query lists)
"""

import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from vajra_bm25.optimized import (
    VajraSearchOptimized,
    SearchResult,
    QueryState,
    VectorizedBM25Scorer
)
from vajra_bm25.text_processing import preprocess_text
from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("parallel")


class VajraSearchParallel(VajraSearchOptimized):
    """
    Parallel Vajra BM25 with concurrent unfolding.

    Vajra (Sanskrit: vajra, "thunderbolt/diamond") parallel implementation.

    Categorical interpretation:
    - Parallel composition: Apply morphisms concurrently
    - Maintains commutativity: order-independent scoring
    - Coalgebra unfolding happens in parallel

    Optimized for batch query processing with multi-core parallelism.
    """

    def __init__(self, corpus, k1=1.5, b=0.75, max_workers=4, use_sparse=False, cache_size=1000):
        super().__init__(corpus, k1, b, use_sparse=use_sparse, cache_size=cache_size)
        self.max_workers = max_workers

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[SearchResult]]:
        """
        Batch query processing (parallel).

        Categorical interpretation:
        Functor application over query list:
        F: List[Query] -> List[List[SearchResult]]

        Each query unfolds independently (coalgebra)
        Parallel execution respects categorical independence
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.search, query, top_k): i
                for i, query in enumerate(queries)
            }

            results = [None] * len(queries)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


if __name__ == "__main__":
    from vajra_bm25.documents import DocumentCorpus
    from pathlib import Path

    logger.info("="*70)
    logger.info("PARALLEL OPTIMIZED CATEGORICAL BM25")
    logger.info("="*70)

    corpus_path = Path("large_corpus.jsonl")
    if corpus_path.exists():
        corpus = DocumentCorpus.load_jsonl(corpus_path)
        logger.info(f"Loaded {len(corpus)} documents")
    else:
        from vajra_bm25.documents import create_sample_corpus
        corpus = create_sample_corpus()
        logger.info(f"Using sample corpus with {len(corpus)} documents")

    # Build parallel engine
    engine = VajraSearchParallel(corpus, max_workers=4)

    # Test batch queries
    queries = [
        "hypothesis testing",
        "neural networks",
        "matrix algebra",
        "data preprocessing",
        "gradient descent",
    ] * 4  # 20 queries total

    logger.info(f"Batch processing {len(queries)} queries...")

    start = time.time()
    results = engine.search_batch(queries, top_k=5)
    elapsed = time.time() - start

    logger.info(f"Completed in {elapsed:.3f}s")
    logger.info(f"  Throughput: {len(queries)/elapsed:.1f} queries/second")
    logger.info(f"  Avg per query: {elapsed*1000/len(queries):.2f}ms")
