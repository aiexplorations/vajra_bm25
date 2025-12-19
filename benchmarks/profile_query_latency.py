#!/usr/bin/env python3
"""
Profile query latency to identify bottlenecks in Vajra vs BM25S.
"""

import time
import numpy as np
from pathlib import Path
from collections import Counter
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vajra_bm25 import DocumentCorpus, VajraSearchOptimized, preprocess_text
from vajra_bm25.optimized import SearchResult
from vajra_bm25.logging_config import get_logger

logger = get_logger("profile_query")

# Load a smaller corpus for quick profiling
CORPUS_SIZE = 200000  # Full 200K for accurate profiling
CACHE_DIR = Path.home() / "Github" / "ir_benchmark_data"


def profile_query_breakdown(engine, query: str, top_k: int = 10):
    """Break down query execution time into components."""
    times = {}

    # 1. Query preprocessing
    start = time.perf_counter()
    query_terms = preprocess_text(query)
    times["preprocessing"] = time.perf_counter() - start

    if not query_terms:
        return times, []

    # 2. Get term IDs
    start = time.perf_counter()
    term_ids = [engine.index.term_to_id[t] for t in query_terms if t in engine.index.term_to_id]
    times["term_lookup"] = time.perf_counter() - start

    if not term_ids:
        return times, []

    # 3. Candidate retrieval (sparse matrix operations)
    start = time.perf_counter()
    candidates = np.asarray(
        (engine.index.term_doc_matrix[term_ids, :].sum(axis=0) > 0)
    ).flatten()
    times["candidate_retrieval"] = time.perf_counter() - start

    # 4. Get TF matrix slice
    start = time.perf_counter()
    tf_matrix = engine.index.term_doc_matrix[term_ids, :]
    tf_dense = tf_matrix.toarray()
    times["tf_extraction"] = time.perf_counter() - start

    # 5. BM25 scoring
    start = time.perf_counter()
    query_idfs = engine.index.idf_cache[term_ids]
    norm_factors = 1.0 - engine.scorer.b + engine.scorer.b * (
        engine.index.doc_lengths / engine.index.avg_doc_length
    )
    numerator = tf_dense * (engine.scorer.k1 + 1)
    denominator = tf_dense + engine.scorer.k1 * norm_factors
    denominator = np.where(denominator == 0, 1e-10, denominator)
    term_scores = query_idfs[:, np.newaxis] * (numerator / denominator)
    doc_scores = term_scores.sum(axis=0)
    doc_scores = doc_scores * candidates
    times["bm25_scoring"] = time.perf_counter() - start

    # 6. Top-k selection
    start = time.perf_counter()
    if top_k >= len(doc_scores):
        top_indices = np.argsort(doc_scores)[::-1]
    else:
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(doc_scores[top_indices])[::-1]]
    times["top_k_selection"] = time.perf_counter() - start

    # 7. Result construction
    start = time.perf_counter()
    results = []
    for idx in top_indices[:top_k]:
        if doc_scores[idx] > 0:
            results.append((int(idx), float(doc_scores[idx])))
    times["result_construction"] = time.perf_counter() - start

    return times, results


def profile_full_search_breakdown(engine, query: str, top_k: int = 10):
    """Profile the full search() method including all overheads."""
    times = {}

    # 1. Query preprocessing
    start = time.perf_counter()
    query_terms = preprocess_text(query)
    times["preprocessing"] = time.perf_counter() - start

    if not query_terms:
        return times

    # 2. Get candidate documents
    start = time.perf_counter()
    candidate_mask = engine.index.get_candidate_docs_vectorized(query_terms)
    times["candidate_retrieval"] = time.perf_counter() - start

    if not candidate_mask.any():
        return times

    # 3. BM25 scoring
    start = time.perf_counter()
    scores = engine.scorer.score_batch(query_terms, candidate_mask)
    times["bm25_scoring"] = time.perf_counter() - start

    # 4. Top-k selection
    start = time.perf_counter()
    top_docs = engine.scorer.get_top_k(scores, top_k)
    times["top_k_selection"] = time.perf_counter() - start

    # 5. Document retrieval (corpus.get for each result)
    start = time.perf_counter()
    results = []
    for rank, (doc_idx, score) in enumerate(top_docs, 1):
        doc_id = engine.index.id_to_doc[doc_idx]
        doc = engine.corpus.get(doc_id)
        if doc:
            results.append(SearchResult(
                document=doc,
                score=score,
                rank=rank
            ))
    times["result_construction"] = time.perf_counter() - start

    return times


def main():
    logger.info("=" * 70)
    logger.info("QUERY LATENCY PROFILING")
    logger.info("=" * 70)

    # Load corpus
    wiki_file = CACHE_DIR / "wikipedia" / f"wikipedia_200000.jsonl"
    logger.info(f"Loading corpus from {wiki_file}...")

    corpus = DocumentCorpus.load_jsonl(wiki_file)
    # Take subset for faster profiling
    subset = DocumentCorpus(list(corpus.documents)[:CORPUS_SIZE])
    logger.info(f"Using {len(subset)} documents for profiling")

    # Build engine WITH NO CACHE to get accurate query timing
    logger.info("Building Vajra engine (no cache)...")
    start = time.time()
    engine = VajraSearchOptimized(subset, cache_size=0)  # Disable cache
    logger.info(f"Built in {time.time()-start:.2f}s")

    # Generate unique test queries from corpus content to avoid any caching effects
    random.seed(42)
    sample_docs = random.sample(list(subset.documents), 100)
    test_queries = []
    for doc in sample_docs:
        # Use content fragments as queries
        if doc.content and len(doc.content) > 50:
            fragment = doc.content[50:120]  # Different fragment than stored
            words = fragment.split()[:5]
            if len(words) >= 3:
                test_queries.append(" ".join(words))
    test_queries = test_queries[:50]  # Use 50 unique queries
    logger.info(f"Generated {len(test_queries)} unique test queries")

    # No warmup to get cold query performance

    # Profile queries using full breakdown
    logger.info("\n" + "=" * 70)
    logger.info("PROFILING QUERY BREAKDOWN (200K corpus)")
    logger.info("=" * 70)

    all_times = {
        "preprocessing": [],
        "candidate_retrieval": [],
        "bm25_scoring": [],
        "top_k_selection": [],
        "result_construction": [],
    }

    n_runs = 10  # Fewer runs since we have more queries and bigger corpus
    for query in test_queries:
        for _ in range(n_runs):
            times = profile_full_search_breakdown(engine, query)
            for k, v in times.items():
                if k in all_times:
                    all_times[k].append(v)

    # Print results
    logger.info(f"\nResults over {len(test_queries) * n_runs} queries:")
    logger.info("-" * 50)

    total_time = 0
    for component, timings in all_times.items():
        if timings:
            avg_ms = np.mean(timings) * 1000
            total_time += avg_ms
            pct = 100 * avg_ms / sum(np.mean(t) * 1000 for t in all_times.values() if t)
            logger.info(f"{component:<25} {avg_ms:>8.3f}ms ({pct:>5.1f}%)")

    logger.info("-" * 50)
    logger.info(f"{'TOTAL':<25} {total_time:>8.3f}ms")

    # Compare with full search()
    logger.info("\n" + "=" * 70)
    logger.info("FULL SEARCH() METHOD TIMING")
    logger.info("=" * 70)

    latencies = []
    for query in test_queries:
        for _ in range(n_runs):
            start = time.perf_counter()
            engine.search(query, top_k=10)
            latencies.append(time.perf_counter() - start)

    avg_latency = np.mean(latencies) * 1000
    p50_latency = np.percentile(latencies, 50) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000

    logger.info(f"Average latency:  {avg_latency:.3f}ms")
    logger.info(f"P50 latency:      {p50_latency:.3f}ms")
    logger.info(f"P99 latency:      {p99_latency:.3f}ms")

    # Identify biggest bottleneck
    if all_times:
        bottleneck = max(all_times.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
        logger.info(f"\nBiggest bottleneck: {bottleneck[0]} ({np.mean(bottleneck[1])*1000:.3f}ms)")

    # Also compare BM25S if available
    logger.info("\n" + "=" * 70)
    logger.info("BM25S COMPARISON")
    logger.info("=" * 70)

    try:
        import bm25s
        corpus_texts = [doc.title + " " + doc.content for doc in subset.documents]
        doc_ids = [doc.id for doc in subset.documents]

        logger.info("Building BM25S index...")
        start = time.time()
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        logger.info(f"Built in {time.time()-start:.2f}s")

        # Time BM25S queries
        bm25s_latencies = []
        for query in test_queries:
            for _ in range(n_runs):
                start = time.perf_counter()
                query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
                results_obj, scores = retriever.retrieve(query_tokens, k=10, n_threads=1, show_progress=False)
                bm25s_latencies.append(time.perf_counter() - start)

        bm25s_avg = np.mean(bm25s_latencies) * 1000
        bm25s_p50 = np.percentile(bm25s_latencies, 50) * 1000
        bm25s_p99 = np.percentile(bm25s_latencies, 99) * 1000

        logger.info(f"BM25S Average latency:  {bm25s_avg:.3f}ms")
        logger.info(f"BM25S P50 latency:      {bm25s_p50:.3f}ms")
        logger.info(f"BM25S P99 latency:      {bm25s_p99:.3f}ms")

        logger.info(f"\nVajra/BM25S ratio: {avg_latency/bm25s_avg:.2f}x slower")

    except ImportError:
        logger.info("BM25S not available for comparison")


if __name__ == "__main__":
    main()
