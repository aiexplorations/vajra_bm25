#!/usr/bin/env python3
"""
Fair no-cache comparison between Vajra and BM25S.

This script addresses the valid feedback that Vajra's LRU cache gives it
an unfair advantage in benchmarks with repeated queries.

Key differences from standard benchmark:
1. cache_size=0 for Vajra (no query result caching)
2. Each query runs only once (no warm cache from repeated runs)
3. Fresh engine instances for each measurement
"""

import time
import statistics
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vajra_bm25 import VajraSearchOptimized, DocumentCorpus, Document
from vajra_bm25.text_processing import preprocess_text

# Optional imports
try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False


def load_beir_scifact():
    """Load BEIR SciFact dataset for testing."""
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        print("Install BEIR: pip install beir")
        return None, None

    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = Path.home() / ".cache" / "beir"
    data_path = util.download_and_unzip(url, str(out_dir))

    corpus_beir, queries_beir, qrels = GenericDataLoader(data_path).load(split="test")

    # Convert to DocumentCorpus
    corpus = DocumentCorpus()
    for doc_id, doc in corpus_beir.items():
        corpus.add(Document(id=doc_id, title=doc.get("title", ""), content=doc.get("text", "")))

    # Get query list with relevance info
    queries = []
    for qid, text in queries_beir.items():
        relevant = list(qrels.get(qid, {}).keys())
        queries.append({"id": qid, "text": text, "relevant": relevant})

    return corpus, queries


def load_wikipedia(max_docs: int = 10000):
    """Load Wikipedia subset."""
    wiki_file = Path.home() / "Github" / "ir_benchmark_data" / "wikipedia" / f"wikipedia_{max_docs}.jsonl"

    if not wiki_file.exists():
        print(f"Wikipedia file not found: {wiki_file}")
        print("Using BEIR SciFact instead...")
        return load_beir_scifact()

    corpus = DocumentCorpus.load(str(wiki_file))

    # Use standard IR test queries
    queries = [
        {"id": "q1", "text": "machine learning algorithms", "relevant": []},
        {"id": "q2", "text": "climate change effects", "relevant": []},
        {"id": "q3", "text": "quantum computing applications", "relevant": []},
        {"id": "q4", "text": "artificial intelligence ethics", "relevant": []},
        {"id": "q5", "text": "neural network architecture", "relevant": []},
        {"id": "q6", "text": "renewable energy sources", "relevant": []},
        {"id": "q7", "text": "genetic engineering CRISPR", "relevant": []},
        {"id": "q8", "text": "space exploration Mars", "relevant": []},
        {"id": "q9", "text": "blockchain cryptocurrency", "relevant": []},
        {"id": "q10", "text": "deep learning natural language processing", "relevant": []},
        {"id": "q11", "text": "protein folding prediction", "relevant": []},
        {"id": "q12", "text": "autonomous vehicles self-driving", "relevant": []},
        {"id": "q13", "text": "internet of things smart home", "relevant": []},
        {"id": "q14", "text": "cybersecurity data breaches", "relevant": []},
        {"id": "q15", "text": "augmented reality virtual reality", "relevant": []},
        {"id": "q16", "text": "5G wireless network technology", "relevant": []},
        {"id": "q17", "text": "gene therapy treatment", "relevant": []},
        {"id": "q18", "text": "robotics automation manufacturing", "relevant": []},
        {"id": "q19", "text": "nanotechnology materials science", "relevant": []},
        {"id": "q20", "text": "ocean pollution microplastics", "relevant": []},
    ]

    return corpus, queries


def benchmark_vajra_no_cache(corpus, queries, top_k=10):
    """Benchmark Vajra with cache disabled."""
    print("\n=== Vajra (cache_size=0) ===")

    # Build index
    start = time.time()
    engine = VajraSearchOptimized(
        corpus,
        use_sparse=True,
        use_eager=True,
        cache_size=0  # NO CACHE
    )
    build_time = time.time() - start
    print(f"Build time: {build_time:.2f}s")

    # Warm-up (single query, discarded)
    engine.search("warm up query", top_k=top_k)

    # Run each query ONCE
    latencies = []
    for q in queries:
        start = time.time()
        results = engine.search(q["text"], top_k=top_k)
        latencies.append((time.time() - start) * 1000)  # ms

    print(f"Queries: {len(queries)}")
    print(f"Mean latency: {statistics.mean(latencies):.3f} ms")
    print(f"Median latency: {statistics.median(latencies):.3f} ms")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.3f} ms")
    print(f"Min/Max: {min(latencies):.3f} / {max(latencies):.3f} ms")
    print(f"QPS: {1000 / statistics.mean(latencies):.0f}")

    return {
        "engine": "vajra (no cache)",
        "build_time": build_time,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "qps": 1000 / statistics.mean(latencies),
        "latencies": latencies,
    }


def benchmark_vajra_with_cache(corpus, queries, top_k=10):
    """Benchmark Vajra with cache enabled (for comparison)."""
    print("\n=== Vajra (cache_size=1000, fresh queries) ===")

    engine = VajraSearchOptimized(
        corpus,
        use_sparse=True,
        use_eager=True,
        cache_size=1000  # Default cache
    )

    # Run each query ONCE (cache won't help since queries are unique)
    latencies = []
    for q in queries:
        start = time.time()
        results = engine.search(q["text"], top_k=top_k)
        latencies.append((time.time() - start) * 1000)

    print(f"Mean latency: {statistics.mean(latencies):.3f} ms")
    print(f"Median latency: {statistics.median(latencies):.3f} ms")

    # Show cache stats (should be all misses)
    stats = engine.get_cache_stats()
    print(f"Cache stats: {stats}")

    return {
        "engine": "vajra (with cache, unique queries)",
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
    }


def benchmark_bm25s(corpus, queries, top_k=10):
    """Benchmark BM25S (no query caching by default)."""
    if not BM25S_AVAILABLE:
        print("\n=== BM25S: Not installed ===")
        return None

    print("\n=== BM25S ===")

    # Build index
    texts = []
    doc_ids = []
    for doc in corpus:
        texts.append(f"{doc.title or ''} {doc.content or ''}")
        doc_ids.append(doc.id)

    import os
    os.environ["BM25S_DISABLE_PROGRESS"] = "1"

    start = time.time()
    retriever = bm25s.BM25()
    tokenized = bm25s.tokenize(texts, show_progress=False)
    retriever.index(tokenized, show_progress=False)
    build_time = time.time() - start
    print(f"Build time: {build_time:.2f}s")

    # Warm-up
    retriever.retrieve(bm25s.tokenize(["warm up query"], show_progress=False), k=top_k, show_progress=False)

    # Run each query ONCE
    latencies = []
    for q in queries:
        start = time.time()
        tokenized_query = bm25s.tokenize([q["text"]], show_progress=False)
        results = retriever.retrieve(tokenized_query, k=top_k, show_progress=False)
        latencies.append((time.time() - start) * 1000)

    print(f"Queries: {len(queries)}")
    print(f"Mean latency: {statistics.mean(latencies):.3f} ms")
    print(f"Median latency: {statistics.median(latencies):.3f} ms")
    print(f"P95 latency: {sorted(latencies)[int(len(latencies) * 0.95)]:.3f} ms")
    print(f"Min/Max: {min(latencies):.3f} / {max(latencies):.3f} ms")
    print(f"QPS: {1000 / statistics.mean(latencies):.0f}")

    return {
        "engine": "bm25s",
        "build_time": build_time,
        "mean_latency_ms": statistics.mean(latencies),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "qps": 1000 / statistics.mean(latencies),
        "latencies": latencies,
    }


def benchmark_bm25s_with_lru(corpus, queries, top_k=10):
    """Benchmark BM25S with LRU cache wrapper (fair comparison with Vajra cached)."""
    if not BM25S_AVAILABLE:
        return None

    from functools import lru_cache

    print("\n=== BM25S + LRU Cache ===")

    texts = []
    doc_ids = []
    for doc in corpus:
        texts.append(f"{doc.title or ''} {doc.content or ''}")
        doc_ids.append(doc.id)

    retriever = bm25s.BM25()
    tokenized = bm25s.tokenize(texts, show_progress=False)
    retriever.index(tokenized, show_progress=False)

    # Create cached search function
    @lru_cache(maxsize=1000)
    def cached_search(query_text: str):
        tokenized_query = bm25s.tokenize([query_text], show_progress=False)
        results, scores = retriever.retrieve(tokenized_query, k=top_k, return_as="tuple")
        return tuple(doc_ids[i] for i in results[0])

    # Run queries twice to measure cached vs uncached
    print("First pass (cache cold):")
    latencies_cold = []
    for q in queries:
        start = time.time()
        results = cached_search(q["text"])
        latencies_cold.append((time.time() - start) * 1000)
    print(f"  Mean: {statistics.mean(latencies_cold):.3f} ms")

    print("Second pass (cache warm):")
    latencies_warm = []
    for q in queries:
        start = time.time()
        results = cached_search(q["text"])
        latencies_warm.append((time.time() - start) * 1000)
    print(f"  Mean: {statistics.mean(latencies_warm):.3f} ms")

    return {
        "engine": "bm25s + lru_cache",
        "cold_latency_ms": statistics.mean(latencies_cold),
        "warm_latency_ms": statistics.mean(latencies_warm),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fair no-cache BM25 comparison")
    parser.add_argument("--dataset", choices=["scifact", "wiki-10k", "wiki-100k", "wiki-200k"],
                        default="wiki-100k", help="Dataset to use")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results")
    args = parser.parse_args()

    print("=" * 60)
    print("FAIR NO-CACHE COMPARISON: Vajra vs BM25S")
    print("=" * 60)
    print(f"\nDataset: {args.dataset}")

    # Load dataset
    if args.dataset == "scifact":
        corpus, queries = load_beir_scifact()
    else:
        max_docs = {
            "wiki-10k": 10000,
            "wiki-100k": 100000,
            "wiki-200k": 200000,
        }[args.dataset]
        corpus, queries = load_wikipedia(max_docs)

    if corpus is None:
        print("Failed to load dataset")
        return

    print(f"Corpus size: {len(corpus):,} documents")
    print(f"Queries: {len(queries)}")

    # Run benchmarks
    results = []

    # Vajra without cache
    results.append(benchmark_vajra_no_cache(corpus, queries, args.top_k))

    # Vajra with cache (but unique queries, so cache doesn't help)
    results.append(benchmark_vajra_with_cache(corpus, queries, args.top_k))

    # BM25S (no cache)
    bm25s_result = benchmark_bm25s(corpus, queries, args.top_k)
    if bm25s_result:
        results.append(bm25s_result)

    # BM25S with LRU cache
    bm25s_cached = benchmark_bm25s_with_lru(corpus, queries, args.top_k)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (No Cache Comparison)")
    print("=" * 60)
    print(f"\n{'Engine':<30} {'Mean (ms)':<12} {'Median (ms)':<12} {'QPS':<10}")
    print("-" * 64)

    for r in results:
        if r and "qps" in r:
            print(f"{r['engine']:<30} {r['mean_latency_ms']:<12.3f} {r['median_latency_ms']:<12.3f} {r['qps']:<10.0f}")

    if bm25s_result and results[0]:
        speedup = bm25s_result["mean_latency_ms"] / results[0]["mean_latency_ms"]
        print(f"\nVajra speedup over BM25S (no cache): {speedup:.1f}x")


if __name__ == "__main__":
    main()
