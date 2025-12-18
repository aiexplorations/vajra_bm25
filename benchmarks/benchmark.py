#!/usr/bin/env python3
"""
Vajra BM25 Benchmark: Compare against rank-bm25 and BM25S

Tests performance and ranking quality across different corpus sizes.
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Any

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False

from vajra_bm25 import (
    DocumentCorpus,
    VajraSearch,
    VajraSearchOptimized,
    preprocess_text,
    BM25Parameters,
)


class RankBM25Wrapper:
    """Wrapper around rank-bm25 for fair comparison."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.tokenized_docs = []
        for doc in corpus:
            full_text = doc.title + " " + doc.content
            tokens = preprocess_text(full_text)
            self.tokenized_docs.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10):
        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        scored_docs = [(scores[i], i) for i in range(len(scores))]
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        results = []
        for rank, (score, idx) in enumerate(scored_docs[:top_k], 1):
            results.append({
                'doc_id': self.doc_ids[idx],
                'score': score,
                'rank': rank
            })
        return results


class BM25SWrapper:
    """Wrapper around BM25S for fair comparison."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]

        # Tokenize documents
        corpus_texts = []
        for doc in corpus:
            full_text = doc.title + " " + doc.content
            corpus_texts.append(full_text)

        # BM25S uses its own tokenizer
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")

        # Create and index
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search(self, query: str, top_k: int = 10):
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stopwords="en")

        # Retrieve
        results_obj, scores = self.retriever.retrieve(query_tokens, k=top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(results_obj[0], scores[0]), 1):
            results.append({
                'doc_id': self.doc_ids[idx],
                'score': float(score),
                'rank': rank
            })
        return results


def calculate_recall_at_k_vajra(vajra_results, baseline_results, k=10) -> float:
    """Calculate recall@k for Vajra results against baseline."""
    baseline_ids = set(r['doc_id'] for r in baseline_results[:k])
    vajra_ids = set(r.document.id for r in vajra_results[:k])

    if not baseline_ids:
        return 1.0

    overlap = len(baseline_ids & vajra_ids)
    return overlap / len(baseline_ids)


def calculate_recall_at_k_dict(results, baseline_results, k=10) -> float:
    """Calculate recall@k for dict results against baseline."""
    baseline_ids = set(r['doc_id'] for r in baseline_results[:k])
    result_ids = set(r['doc_id'] for r in results[:k])

    if not baseline_ids:
        return 1.0

    overlap = len(baseline_ids & result_ids)
    return overlap / len(baseline_ids)


def run_benchmark(corpus_path: Path, queries: List[str], num_runs: int = 3) -> Dict[str, Any]:
    """Run benchmark on a corpus."""

    print(f"\n{'='*80}")
    print(f"Benchmarking: {corpus_path.name}")
    print(f"{'='*80}")

    # Load corpus
    print("Loading corpus...")
    corpus = DocumentCorpus.load_jsonl(corpus_path)
    print(f"Corpus size: {len(corpus)} documents")

    # Build engines
    print("\nBuilding search engines...")

    # Vajra base
    start = time.time()
    vajra_base = VajraSearch(corpus)
    vajra_base_build = time.time() - start

    # Vajra optimized
    start = time.time()
    vajra_opt = VajraSearchOptimized(corpus)
    vajra_opt_build = time.time() - start

    # rank-bm25
    start = time.time()
    rank_bm25 = RankBM25Wrapper(corpus)
    rank_bm25_build = time.time() - start

    # BM25S
    bm25s_engine = None
    bm25s_build = 0
    if BM25S_AVAILABLE:
        start = time.time()
        bm25s_engine = BM25SWrapper(corpus)
        bm25s_build = time.time() - start

    print(f"\nBuild times:")
    print(f"  Vajra (base):      {vajra_base_build*1000:.1f}ms")
    print(f"  Vajra (optimized): {vajra_opt_build*1000:.1f}ms")
    print(f"  rank-bm25:         {rank_bm25_build*1000:.1f}ms")
    if BM25S_AVAILABLE:
        print(f"  BM25S:             {bm25s_build*1000:.1f}ms")

    # Run queries
    print(f"\nRunning {len(queries)} queries x {num_runs} runs each...")

    vajra_base_times = []
    vajra_opt_times = []
    rank_bm25_times = []
    bm25s_times = []
    recalls_base = []
    recalls_opt = []
    recalls_bm25s = []

    for query in queries:
        for _ in range(num_runs):
            # rank-bm25 (baseline)
            start = time.time()
            baseline_results = rank_bm25.search(query, top_k=10)
            rank_bm25_times.append(time.time() - start)

            # Vajra base
            start = time.time()
            vajra_base_results = vajra_base.search(query, top_k=10)
            vajra_base_times.append(time.time() - start)

            # Vajra optimized
            start = time.time()
            vajra_opt_results = vajra_opt.search(query, top_k=10)
            vajra_opt_times.append(time.time() - start)

            # BM25S
            if BM25S_AVAILABLE and bm25s_engine:
                start = time.time()
                bm25s_results = bm25s_engine.search(query, top_k=10)
                bm25s_times.append(time.time() - start)
                recalls_bm25s.append(calculate_recall_at_k_dict(bm25s_results, baseline_results, 10))

            # Calculate recall
            recalls_base.append(calculate_recall_at_k_vajra(vajra_base_results, baseline_results, 10))
            recalls_opt.append(calculate_recall_at_k_vajra(vajra_opt_results, baseline_results, 10))

    # Calculate statistics
    results = {
        'corpus_size': len(corpus),
        'num_queries': len(queries),
        'rank_bm25': {
            'avg_latency_ms': statistics.mean(rank_bm25_times) * 1000,
            'p50_latency_ms': statistics.median(rank_bm25_times) * 1000,
            'p95_latency_ms': sorted(rank_bm25_times)[int(len(rank_bm25_times) * 0.95)] * 1000,
        },
        'vajra_base': {
            'avg_latency_ms': statistics.mean(vajra_base_times) * 1000,
            'p50_latency_ms': statistics.median(vajra_base_times) * 1000,
            'p95_latency_ms': sorted(vajra_base_times)[int(len(vajra_base_times) * 0.95)] * 1000,
            'recall_at_10': statistics.mean(recalls_base) * 100,
            'speedup': statistics.mean(rank_bm25_times) / statistics.mean(vajra_base_times) if vajra_base_times else 0,
        },
        'vajra_optimized': {
            'avg_latency_ms': statistics.mean(vajra_opt_times) * 1000,
            'p50_latency_ms': statistics.median(vajra_opt_times) * 1000,
            'p95_latency_ms': sorted(vajra_opt_times)[int(len(vajra_opt_times) * 0.95)] * 1000,
            'recall_at_10': statistics.mean(recalls_opt) * 100,
            'speedup': statistics.mean(rank_bm25_times) / statistics.mean(vajra_opt_times) if vajra_opt_times else 0,
        },
    }

    if BM25S_AVAILABLE and bm25s_times:
        results['bm25s'] = {
            'avg_latency_ms': statistics.mean(bm25s_times) * 1000,
            'p50_latency_ms': statistics.median(bm25s_times) * 1000,
            'p95_latency_ms': sorted(bm25s_times)[int(len(bm25s_times) * 0.95)] * 1000,
            'recall_at_10': statistics.mean(recalls_bm25s) * 100,
            'speedup': statistics.mean(rank_bm25_times) / statistics.mean(bm25s_times) if bm25s_times else 0,
        }

    # Print results
    print(f"\nResults:")
    print(f"{'Engine':<20} {'Avg (ms)':<12} {'P50 (ms)':<12} {'Speedup':<12} {'Recall@10':<10}")
    print("-" * 80)

    rb = results['rank_bm25']
    print(f"{'rank-bm25':<20} {rb['avg_latency_ms']:<12.2f} {rb['p50_latency_ms']:<12.2f} {'1.0x':<12} {'baseline':<10}")

    vb = results['vajra_base']
    print(f"{'Vajra (base)':<20} {vb['avg_latency_ms']:<12.2f} {vb['p50_latency_ms']:<12.2f} {vb['speedup']:.1f}x{'':<8} {vb['recall_at_10']:.1f}%")

    vo = results['vajra_optimized']
    print(f"{'Vajra (optimized)':<20} {vo['avg_latency_ms']:<12.2f} {vo['p50_latency_ms']:<12.2f} {vo['speedup']:.1f}x{'':<8} {vo['recall_at_10']:.1f}%")

    if 'bm25s' in results:
        bs = results['bm25s']
        print(f"{'BM25S':<20} {bs['avg_latency_ms']:<12.2f} {bs['p50_latency_ms']:<12.2f} {bs['speedup']:.1f}x{'':<8} {bs['recall_at_10']:.1f}%")

    return results


def main():
    print("=" * 80)
    print("VAJRA BM25 BENCHMARK")
    print("Comparing: rank-bm25, Vajra, BM25S")
    print("=" * 80)

    if not RANK_BM25_AVAILABLE:
        print("\nError: rank-bm25 not installed. Run: pip install rank-bm25")
        return

    if not BM25S_AVAILABLE:
        print("\nWarning: BM25S not installed. Run: pip install bm25s")
        print("Continuing without BM25S benchmarks...\n")

    # Test queries
    queries = [
        "hypothesis testing statistical significance",
        "neural networks deep learning backpropagation",
        "matrix eigenvalues linear algebra",
        "database indexing query optimization",
        "sorting algorithm time complexity",
        "regression analysis machine learning",
        "gradient descent optimization",
        "data preprocessing normalization",
        "probability distribution sampling",
        "classification clustering algorithms",
    ]

    # Data directory
    data_dir = Path("/Users/rajesh/Github/state_dynamic_modeling/data")

    # Run benchmarks on different corpus sizes
    all_results = []

    for corpus_file in ["corpus_1k.jsonl", "corpus_10k.jsonl", "corpus_50k.jsonl", "corpus_100k.jsonl"]:
        corpus_path = data_dir / corpus_file
        if corpus_path.exists():
            results = run_benchmark(corpus_path, queries)
            all_results.append(results)
        else:
            print(f"\nSkipping {corpus_file} (not found)")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    has_bm25s = any('bm25s' in r for r in all_results)

    if has_bm25s:
        print(f"\n{'Corpus':<10} {'rank-bm25':<12} {'Vajra Opt':<12} {'BM25S':<12} {'Vajra Speedup':<15} {'BM25S Speedup':<15}")
        print("-" * 90)

        for r in all_results:
            corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
            rb_ms = r['rank_bm25']['avg_latency_ms']
            vo_ms = r['vajra_optimized']['avg_latency_ms']
            vo_speedup = r['vajra_optimized']['speedup']

            if 'bm25s' in r:
                bs_ms = r['bm25s']['avg_latency_ms']
                bs_speedup = r['bm25s']['speedup']
                print(f"{corpus_label:<10} {rb_ms:<12.2f} {vo_ms:<12.2f} {bs_ms:<12.2f} {vo_speedup:.1f}x{'':<10} {bs_speedup:.1f}x")
            else:
                print(f"{corpus_label:<10} {rb_ms:<12.2f} {vo_ms:<12.2f} {'N/A':<12} {vo_speedup:.1f}x{'':<10} {'N/A':<15}")
    else:
        print(f"\n{'Corpus':<12} {'rank-bm25 (ms)':<15} {'Vajra Opt (ms)':<15} {'Speedup':<10} {'Recall@10':<10}")
        print("-" * 70)

        for r in all_results:
            corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
            rb_ms = r['rank_bm25']['avg_latency_ms']
            vo_ms = r['vajra_optimized']['avg_latency_ms']
            speedup = r['vajra_optimized']['speedup']
            recall = r['vajra_optimized']['recall_at_10']
            print(f"{corpus_label:<12} {rb_ms:<15.2f} {vo_ms:<15.2f} {speedup:<10.1f}x {recall:<10.1f}%")

    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON (Recall@10 vs rank-bm25 baseline)")
    print("=" * 80)

    print(f"\n{'Corpus':<10} {'Vajra Base':<15} {'Vajra Opt':<15} {'BM25S':<15}")
    print("-" * 60)

    for r in all_results:
        corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
        vb_recall = r['vajra_base']['recall_at_10']
        vo_recall = r['vajra_optimized']['recall_at_10']
        bs_recall = r.get('bm25s', {}).get('recall_at_10', 0)

        if bs_recall > 0:
            print(f"{corpus_label:<10} {vb_recall:<15.1f}% {vo_recall:<15.1f}% {bs_recall:<15.1f}%")
        else:
            print(f"{corpus_label:<10} {vb_recall:<15.1f}% {vo_recall:<15.1f}% {'N/A':<15}")


if __name__ == "__main__":
    main()
