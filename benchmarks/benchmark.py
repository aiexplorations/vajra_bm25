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
    VajraSearchParallel,
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

    def __init__(self, corpus: DocumentCorpus, n_threads: int = 0):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.n_threads = n_threads

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

        # Retrieve with threading support
        results_obj, scores = self.retriever.retrieve(
            query_tokens, k=top_k, n_threads=self.n_threads, show_progress=False
        )

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

    # Vajra parallel (uses optimized internally)
    start = time.time()
    vajra_parallel = VajraSearchParallel(corpus, max_workers=4)
    vajra_parallel_build = time.time() - start

    # rank-bm25
    start = time.time()
    rank_bm25 = RankBM25Wrapper(corpus)
    rank_bm25_build = time.time() - start

    # BM25S (single-threaded)
    bm25s_engine = None
    bm25s_build = 0
    if BM25S_AVAILABLE:
        start = time.time()
        bm25s_engine = BM25SWrapper(corpus, n_threads=0)
        bm25s_build = time.time() - start

    # BM25S parallel (all cores)
    bm25s_parallel_engine = None
    bm25s_parallel_build = 0
    if BM25S_AVAILABLE:
        start = time.time()
        bm25s_parallel_engine = BM25SWrapper(corpus, n_threads=-1)
        bm25s_parallel_build = time.time() - start

    print(f"\nBuild times:")
    print(f"  Vajra (base):      {vajra_base_build*1000:.1f}ms")
    print(f"  Vajra (optimized): {vajra_opt_build*1000:.1f}ms")
    print(f"  Vajra (parallel):  {vajra_parallel_build*1000:.1f}ms")
    print(f"  rank-bm25:         {rank_bm25_build*1000:.1f}ms")
    if BM25S_AVAILABLE:
        print(f"  BM25S:             {bm25s_build*1000:.1f}ms")
        print(f"  BM25S (parallel):  {bm25s_parallel_build*1000:.1f}ms")

    # Run queries
    print(f"\nRunning {len(queries)} queries x {num_runs} runs each...")

    vajra_base_times = []
    vajra_opt_times = []
    vajra_parallel_times = []
    rank_bm25_times = []
    bm25s_times = []
    bm25s_parallel_times = []
    recalls_base = []
    recalls_opt = []
    recalls_parallel = []
    recalls_bm25s = []
    recalls_bm25s_parallel = []

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

            # Vajra parallel (single query - uses same optimized backend)
            start = time.time()
            vajra_parallel_results = vajra_parallel.search(query, top_k=10)
            vajra_parallel_times.append(time.time() - start)

            # BM25S (single-threaded)
            if BM25S_AVAILABLE and bm25s_engine:
                start = time.time()
                bm25s_results = bm25s_engine.search(query, top_k=10)
                bm25s_times.append(time.time() - start)
                recalls_bm25s.append(calculate_recall_at_k_dict(bm25s_results, baseline_results, 10))

            # BM25S parallel
            if BM25S_AVAILABLE and bm25s_parallel_engine:
                start = time.time()
                bm25s_parallel_results = bm25s_parallel_engine.search(query, top_k=10)
                bm25s_parallel_times.append(time.time() - start)
                recalls_bm25s_parallel.append(calculate_recall_at_k_dict(bm25s_parallel_results, baseline_results, 10))

            # Calculate recall
            recalls_base.append(calculate_recall_at_k_vajra(vajra_base_results, baseline_results, 10))
            recalls_opt.append(calculate_recall_at_k_vajra(vajra_opt_results, baseline_results, 10))
            recalls_parallel.append(calculate_recall_at_k_vajra(vajra_parallel_results, baseline_results, 10))

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
        'vajra_parallel': {
            'avg_latency_ms': statistics.mean(vajra_parallel_times) * 1000,
            'p50_latency_ms': statistics.median(vajra_parallel_times) * 1000,
            'p95_latency_ms': sorted(vajra_parallel_times)[int(len(vajra_parallel_times) * 0.95)] * 1000,
            'recall_at_10': statistics.mean(recalls_parallel) * 100,
            'speedup': statistics.mean(rank_bm25_times) / statistics.mean(vajra_parallel_times) if vajra_parallel_times else 0,
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

    if BM25S_AVAILABLE and bm25s_parallel_times:
        results['bm25s_parallel'] = {
            'avg_latency_ms': statistics.mean(bm25s_parallel_times) * 1000,
            'p50_latency_ms': statistics.median(bm25s_parallel_times) * 1000,
            'p95_latency_ms': sorted(bm25s_parallel_times)[int(len(bm25s_parallel_times) * 0.95)] * 1000,
            'recall_at_10': statistics.mean(recalls_bm25s_parallel) * 100,
            'speedup': statistics.mean(rank_bm25_times) / statistics.mean(bm25s_parallel_times) if bm25s_parallel_times else 0,
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

    vp = results['vajra_parallel']
    print(f"{'Vajra (parallel)':<20} {vp['avg_latency_ms']:<12.2f} {vp['p50_latency_ms']:<12.2f} {vp['speedup']:.1f}x{'':<8} {vp['recall_at_10']:.1f}%")

    if 'bm25s' in results:
        bs = results['bm25s']
        print(f"{'BM25S':<20} {bs['avg_latency_ms']:<12.2f} {bs['p50_latency_ms']:<12.2f} {bs['speedup']:.1f}x{'':<8} {bs['recall_at_10']:.1f}%")

    if 'bm25s_parallel' in results:
        bsp = results['bm25s_parallel']
        print(f"{'BM25S (parallel)':<20} {bsp['avg_latency_ms']:<12.2f} {bsp['p50_latency_ms']:<12.2f} {bsp['speedup']:.1f}x{'':<8} {bsp['recall_at_10']:.1f}%")

    return results


def main():
    print("=" * 80)
    print("VAJRA BM25 BENCHMARK")
    print("Comparing: rank-bm25, Vajra, BM25S, BM25S (parallel)")
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
    print("SUMMARY TABLE (Latency in ms)")
    print("=" * 80)

    has_bm25s = any('bm25s' in r for r in all_results)
    has_bm25s_parallel = any('bm25s_parallel' in r for r in all_results)

    # Print header based on available engines
    header = f"{'Corpus':<8} {'rank-bm25':<11} {'Vajra Opt':<11} {'Vajra Par':<11}"
    if has_bm25s:
        header += f" {'BM25S':<11}"
    if has_bm25s_parallel:
        header += f" {'BM25S Par':<11}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
        rb_ms = r['rank_bm25']['avg_latency_ms']
        vo_ms = r['vajra_optimized']['avg_latency_ms']
        vp_ms = r['vajra_parallel']['avg_latency_ms']

        line = f"{corpus_label:<8} {rb_ms:<11.2f} {vo_ms:<11.2f} {vp_ms:<11.2f}"

        if has_bm25s:
            bs_ms = r.get('bm25s', {}).get('avg_latency_ms', 0)
            line += f" {bs_ms:<11.2f}" if bs_ms else f" {'N/A':<11}"

        if has_bm25s_parallel:
            bsp_ms = r.get('bm25s_parallel', {}).get('avg_latency_ms', 0)
            line += f" {bsp_ms:<11.2f}" if bsp_ms else f" {'N/A':<11}"

        print(line)

    # Speedup table
    print("\n" + "=" * 80)
    print("SPEEDUP vs rank-bm25")
    print("=" * 80)

    header2 = f"{'Corpus':<8} {'Vajra Opt':<12} {'Vajra Par':<12}"
    if has_bm25s:
        header2 += f" {'BM25S':<12}"
    if has_bm25s_parallel:
        header2 += f" {'BM25S Par':<12}"
    print(f"\n{header2}")
    print("-" * len(header2))

    for r in all_results:
        corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
        vo_speedup = r['vajra_optimized']['speedup']
        vp_speedup = r['vajra_parallel']['speedup']

        line = f"{corpus_label:<8} {vo_speedup:.1f}x{'':<7} {vp_speedup:.1f}x{'':<7}"

        if has_bm25s:
            bs_speedup = r.get('bm25s', {}).get('speedup', 0)
            line += f" {bs_speedup:.1f}x{'':<7}" if bs_speedup else f" {'N/A':<12}"

        if has_bm25s_parallel:
            bsp_speedup = r.get('bm25s_parallel', {}).get('speedup', 0)
            line += f" {bsp_speedup:.1f}x{'':<7}" if bsp_speedup else f" {'N/A':<12}"

        print(line)

    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON (Recall@10 vs rank-bm25 baseline)")
    print("=" * 80)

    # Build header dynamically
    header3 = f"{'Corpus':<8} {'Vajra Base':<12} {'Vajra Opt':<12} {'Vajra Par':<12}"
    if has_bm25s:
        header3 += f" {'BM25S':<12}"
    if has_bm25s_parallel:
        header3 += f" {'BM25S Par':<12}"
    print(f"\n{header3}")
    print("-" * len(header3))

    for r in all_results:
        corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
        vb_recall = r['vajra_base']['recall_at_10']
        vo_recall = r['vajra_optimized']['recall_at_10']
        vp_recall = r['vajra_parallel']['recall_at_10']

        line = f"{corpus_label:<8} {vb_recall:.1f}%{'':<6} {vo_recall:.1f}%{'':<6} {vp_recall:.1f}%{'':<6}"

        if has_bm25s:
            bs_recall = r.get('bm25s', {}).get('recall_at_10', 0)
            line += f" {bs_recall:.1f}%{'':<6}" if bs_recall else f" {'N/A':<12}"

        if has_bm25s_parallel:
            bsp_recall = r.get('bm25s_parallel', {}).get('recall_at_10', 0)
            line += f" {bsp_recall:.1f}%{'':<6}" if bsp_recall else f" {'N/A':<12}"

        print(line)


def run_batch_benchmark(corpus_path: Path, queries: List[str], num_runs: int = 3) -> Dict[str, Any]:
    """Run batch processing benchmark."""

    print(f"\n{'='*80}")
    print(f"BATCH BENCHMARK: {corpus_path.name}")
    print(f"{'='*80}")

    corpus = DocumentCorpus.load_jsonl(corpus_path)
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Batch size: {len(queries)} queries")

    # Build engines
    vajra_opt = VajraSearchOptimized(corpus)
    vajra_parallel = VajraSearchParallel(corpus, max_workers=4)

    # Sequential processing (optimized)
    seq_times = []
    for _ in range(num_runs):
        start = time.time()
        for query in queries:
            vajra_opt.search(query, top_k=10)
        seq_times.append(time.time() - start)

    # Batch parallel processing
    batch_times = []
    for _ in range(num_runs):
        start = time.time()
        vajra_parallel.search_batch(queries, top_k=10)
        batch_times.append(time.time() - start)

    seq_avg = statistics.mean(seq_times)
    batch_avg = statistics.mean(batch_times)
    speedup = seq_avg / batch_avg if batch_avg > 0 else 0

    print(f"\nResults ({len(queries)} queries):")
    print(f"  Sequential:     {seq_avg*1000:.1f}ms total ({seq_avg*1000/len(queries):.2f}ms per query)")
    print(f"  Parallel batch: {batch_avg*1000:.1f}ms total ({batch_avg*1000/len(queries):.2f}ms per query)")
    print(f"  Batch speedup:  {speedup:.1f}x")
    print(f"  Throughput:     {len(queries)/batch_avg:.1f} queries/sec")

    return {
        'corpus_size': len(corpus),
        'batch_size': len(queries),
        'sequential_ms': seq_avg * 1000,
        'parallel_ms': batch_avg * 1000,
        'batch_speedup': speedup,
        'throughput_qps': len(queries) / batch_avg,
    }


if __name__ == "__main__":
    main()

    # Run batch benchmarks
    print("\n" + "=" * 80)
    print("BATCH PROCESSING BENCHMARKS")
    print("=" * 80)

    data_dir = Path("/Users/rajesh/Github/state_dynamic_modeling/data")
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
    ] * 5  # 50 queries for batch test

    batch_results = []
    for corpus_file in ["corpus_10k.jsonl", "corpus_50k.jsonl", "corpus_100k.jsonl"]:
        corpus_path = data_dir / corpus_file
        if corpus_path.exists():
            result = run_batch_benchmark(corpus_path, queries)
            batch_results.append(result)

    # Batch summary
    if batch_results:
        print("\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)
        print(f"\n{'Corpus':<12} {'Sequential':<15} {'Parallel':<15} {'Batch Speedup':<15} {'Throughput':<15}")
        print("-" * 75)

        for r in batch_results:
            corpus_label = f"{r['corpus_size']//1000}K" if r['corpus_size'] >= 1000 else str(r['corpus_size'])
            print(f"{corpus_label:<12} {r['sequential_ms']:.1f}ms{'':<8} {r['parallel_ms']:.1f}ms{'':<8} {r['batch_speedup']:.1f}x{'':<10} {r['throughput_qps']:.0f} q/s")
