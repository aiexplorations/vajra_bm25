#!/usr/bin/env python3
"""
Vajra BM25 Unified Profiler

Profiles various aspects of Vajra BM25 performance:
- index-build: Index building time breakdown
- query-latency: Query execution time breakdown
- comparison: Compare Vajra vs other engines

Usage:
    python profiler.py --mode index-build --corpus /path/to/corpus.jsonl
    python profiler.py --mode query-latency --corpus /path/to/corpus.jsonl
    python profiler.py --mode comparison --corpus /path/to/corpus.jsonl
"""

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Progress display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# BM25 implementations
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

try:
    from scipy.sparse import lil_matrix, coo_matrix
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Vajra
sys.path.insert(0, str(Path(__file__).parent.parent))
from vajra_bm25 import DocumentCorpus, VajraSearchOptimized, preprocess_text
from vajra_bm25.optimized import SearchResult


# =============================================================================
# Display Helpers
# =============================================================================

def print_header(title: str):
    if RICH_AVAILABLE:
        console.print(Panel(title, style="bold blue", box=box.DOUBLE))
    else:
        print("=" * 70)
        print(title)
        print("=" * 70)


def print_section(title: str):
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
        console.print("-" * 50)
    else:
        print(f"\n{title}")
        print("-" * 50)


def print_timing(label: str, seconds: float, percentage: float = None):
    if RICH_AVAILABLE:
        pct_str = f" ({percentage:.1f}%)" if percentage else ""
        console.print(f"  {label:<35} [green]{seconds:>8.2f}s[/green]{pct_str}")
    else:
        pct_str = f" ({percentage:.1f}%)" if percentage else ""
        print(f"  {label:<35} {seconds:>8.2f}s{pct_str}")


def print_metric(label: str, value: str):
    if RICH_AVAILABLE:
        console.print(f"  {label:<35} [yellow]{value}[/yellow]")
    else:
        print(f"  {label:<35} {value}")


def print_bottleneck(phase: str, seconds: float, percentage: float):
    if RICH_AVAILABLE:
        console.print(f"\n[bold red]BOTTLENECK:[/bold red] {phase} ({seconds:.2f}s, {percentage:.1f}%)")
    else:
        print(f"\nBOTTLENECK: {phase} ({seconds:.2f}s, {percentage:.1f}%)")


# =============================================================================
# Index Build Profiling
# =============================================================================

def profile_index_build(corpus: DocumentCorpus) -> Dict[str, float]:
    """
    Profile each phase of index building.

    Phases:
    1. Tokenization and vocabulary building
    2. Term ID assignment
    3. Sparse matrix construction
    4. IDF computation
    """
    print_header("INDEX BUILD PROFILING")
    print_metric("Corpus size", f"{len(corpus):,} documents")
    print_metric("Scipy available", str(SCIPY_AVAILABLE))

    timings = {}

    # Phase 1: Tokenization
    print_section("Phase 1: Tokenization")
    phase1_start = time.time()

    term_set = set()
    doc_term_counts = []
    doc_lengths = []

    for doc_idx, doc in enumerate(corpus):
        full_text = doc.title + " " + doc.content
        terms = preprocess_text(full_text)

        term_counts = Counter(terms)
        term_set.update(term_counts.keys())
        doc_term_counts.append(term_counts)
        doc_lengths.append(len(terms))

        if (doc_idx + 1) % 50000 == 0:
            elapsed = time.time() - phase1_start
            rate = (doc_idx + 1) / elapsed
            print_metric(f"  Progress", f"{doc_idx + 1:,} docs ({rate:.0f} docs/sec)")

    phase1_time = time.time() - phase1_start
    timings['tokenization'] = phase1_time

    num_docs = len(corpus)
    num_terms = len(term_set)

    print_timing("Tokenization", phase1_time)
    print_metric("Unique terms", f"{num_terms:,}")
    print_metric("Throughput", f"{num_docs/phase1_time:.0f} docs/sec")

    # Phase 2: Term ID assignment
    print_section("Phase 2: Term ID Assignment")
    phase2_start = time.time()

    term_to_id = {term: idx for idx, term in enumerate(sorted(term_set))}

    phase2_time = time.time() - phase2_start
    timings['term_id_assignment'] = phase2_time
    print_timing("Term ID assignment", phase2_time)

    # Phase 3: Sparse matrix construction
    print_section("Phase 3: Sparse Matrix Construction")
    phase3_start = time.time()

    if SCIPY_AVAILABLE:
        # Use COO format for fast construction
        rows = []
        cols = []
        data = []

        for doc_idx, term_counts in enumerate(doc_term_counts):
            for term, count in term_counts.items():
                term_id = term_to_id[term]
                rows.append(term_id)
                cols.append(doc_idx)
                data.append(count)

        coo_build_time = time.time() - phase3_start
        print_timing("COO data collection", coo_build_time)

        # Convert to CSR
        csr_start = time.time()
        coo = coo_matrix((data, (rows, cols)), shape=(num_terms, num_docs), dtype=np.float32)
        term_doc_matrix = coo.tocsr()
        csr_time = time.time() - csr_start

        timings['coo_to_csr_conversion'] = csr_time
        print_timing("COO to CSR conversion", csr_time)

        # Matrix stats
        nnz = term_doc_matrix.nnz
        sparsity = 1.0 - (nnz / (num_terms * num_docs))
        memory_mb = (term_doc_matrix.data.nbytes +
                     term_doc_matrix.indices.nbytes +
                     term_doc_matrix.indptr.nbytes) / (1024**2)

        print_metric("Matrix shape", f"{num_terms:,} x {num_docs:,}")
        print_metric("Non-zero entries", f"{nnz:,}")
        print_metric("Sparsity", f"{sparsity*100:.2f}%")
        print_metric("Memory", f"{memory_mb:.1f} MB")
    else:
        print_metric("Status", "Scipy not available - skipping matrix build")
        term_doc_matrix = None

    phase3_time = time.time() - phase3_start
    timings['sparse_matrix_build'] = phase3_time
    print_timing("Total matrix build", phase3_time)

    # Phase 4: IDF computation
    print_section("Phase 4: IDF Computation")
    phase4_start = time.time()

    if SCIPY_AVAILABLE and term_doc_matrix is not None:
        doc_freqs = np.asarray((term_doc_matrix > 0).sum(axis=1)).flatten()
        idf_cache = np.log((num_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0)

        print_metric("IDF range", f"[{idf_cache.min():.3f}, {idf_cache.max():.3f}]")

    phase4_time = time.time() - phase4_start
    timings['idf_computation'] = phase4_time
    print_timing("IDF computation", phase4_time)

    # Summary
    print_section("TIMING SUMMARY")
    total_time = sum(timings.values())

    for phase, duration in sorted(timings.items(), key=lambda x: -x[1]):
        percentage = (duration / total_time) * 100
        print_timing(phase, duration, percentage)

    print_timing("TOTAL", total_time)
    print_metric("Overall throughput", f"{num_docs/total_time:.0f} docs/sec")

    # Bottleneck
    max_phase = max(timings.items(), key=lambda x: x[1])
    print_bottleneck(max_phase[0], max_phase[1], (max_phase[1]/total_time)*100)

    return timings


# =============================================================================
# Query Latency Profiling
# =============================================================================

def profile_query_latency(corpus: DocumentCorpus, num_queries: int = 50) -> Dict[str, List[float]]:
    """
    Profile query execution time breakdown.

    Phases:
    1. Query preprocessing
    2. Candidate retrieval
    3. BM25 scoring
    4. Top-k selection
    5. Result construction
    """
    import random

    print_header("QUERY LATENCY PROFILING")
    print_metric("Corpus size", f"{len(corpus):,} documents")

    # Build engine
    print_section("Building Index")
    start = time.time()
    engine = VajraSearchOptimized(corpus, cache_size=0)  # No cache for accurate timing
    build_time = time.time() - start
    print_timing("Index build time", build_time)

    # Generate test queries
    print_section("Generating Test Queries")
    random.seed(42)
    sample_docs = random.sample(list(corpus.documents), min(num_queries * 2, len(corpus)))

    queries = []
    for doc in sample_docs:
        if len(queries) >= num_queries:
            break
        if doc.content and len(doc.content) > 50:
            fragment = doc.content[50:120]
            words = fragment.split()[:5]
            if len(words) >= 3:
                queries.append(" ".join(words))

    print_metric("Test queries", str(len(queries)))

    # Profile query breakdown
    print_section("Profiling Query Phases")

    all_times = {
        "preprocessing": [],
        "candidate_retrieval": [],
        "bm25_scoring": [],
        "top_k_selection": [],
        "result_construction": [],
    }

    n_runs = 5
    for query in queries:
        for _ in range(n_runs):
            times = _profile_single_query(engine, query)
            for k, v in times.items():
                if k in all_times:
                    all_times[k].append(v)

    # Print results
    print_section("QUERY PHASE BREAKDOWN")
    total_time = 0

    for component, timings in sorted(all_times.items(), key=lambda x: -np.mean(x[1]) if x[1] else 0):
        if timings:
            avg_ms = np.mean(timings) * 1000
            total_time += avg_ms
            pct = 100 * avg_ms / sum(np.mean(t) * 1000 for t in all_times.values() if t)
            print_timing(component, avg_ms / 1000, pct)

    # Full search timing
    print_section("Full Search() Timing")

    latencies = []
    for query in queries:
        for _ in range(n_runs):
            start = time.perf_counter()
            engine.search(query, top_k=10)
            latencies.append(time.perf_counter() - start)

    avg_latency = np.mean(latencies) * 1000
    p50_latency = np.percentile(latencies, 50) * 1000
    p95_latency = np.percentile(latencies, 95) * 1000
    p99_latency = np.percentile(latencies, 99) * 1000

    print_metric("Average latency", f"{avg_latency:.3f}ms")
    print_metric("P50 latency", f"{p50_latency:.3f}ms")
    print_metric("P95 latency", f"{p95_latency:.3f}ms")
    print_metric("P99 latency", f"{p99_latency:.3f}ms")
    print_metric("Queries/sec", f"{1000/avg_latency:.0f}")

    # Bottleneck
    if all_times:
        bottleneck = max(all_times.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
        print_bottleneck(bottleneck[0], np.mean(bottleneck[1]),
                        100 * np.mean(bottleneck[1]) / sum(np.mean(t) for t in all_times.values() if t))

    return all_times


def _profile_single_query(engine, query: str, top_k: int = 10) -> Dict[str, float]:
    """Profile a single query execution."""
    times = {}

    # 1. Preprocessing
    start = time.perf_counter()
    query_terms = preprocess_text(query)
    times["preprocessing"] = time.perf_counter() - start

    if not query_terms:
        return times

    # 2. Candidate retrieval
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

    # 5. Result construction
    start = time.perf_counter()
    results = []
    for rank, (doc_idx, score) in enumerate(top_docs, 1):
        doc_id = engine.index.id_to_doc[doc_idx]
        doc = engine.corpus.get(doc_id)
        if doc:
            results.append(SearchResult(document=doc, score=score, rank=rank))
    times["result_construction"] = time.perf_counter() - start

    return times


# =============================================================================
# Comparison Profiling
# =============================================================================

def profile_comparison(corpus: DocumentCorpus, num_queries: int = 50) -> Dict[str, Dict[str, float]]:
    """Compare Vajra vs other BM25 implementations."""
    import random

    print_header("ENGINE COMPARISON PROFILING")
    print_metric("Corpus size", f"{len(corpus):,} documents")

    results = {}

    # Generate test queries
    random.seed(42)
    sample_docs = random.sample(list(corpus.documents), min(num_queries * 2, len(corpus)))
    queries = []
    for doc in sample_docs:
        if len(queries) >= num_queries:
            break
        if doc.title and len(doc.title.split()) >= 2:
            queries.append(doc.title)

    print_metric("Test queries", str(len(queries)))

    # Vajra
    print_section("Vajra (Optimized)")
    start = time.time()
    vajra = VajraSearchOptimized(corpus)
    vajra_build = time.time() - start

    vajra_latencies = []
    for query in queries:
        start = time.perf_counter()
        vajra.search(query, top_k=10)
        vajra_latencies.append(time.perf_counter() - start)

    results["vajra"] = {
        "build_time_s": vajra_build,
        "avg_latency_ms": np.mean(vajra_latencies) * 1000,
        "p50_latency_ms": np.percentile(vajra_latencies, 50) * 1000,
        "p99_latency_ms": np.percentile(vajra_latencies, 99) * 1000,
    }

    print_timing("Build time", vajra_build)
    print_metric("Avg latency", f"{results['vajra']['avg_latency_ms']:.3f}ms")
    print_metric("Throughput", f"{len(corpus)/vajra_build:.0f} docs/sec")

    # rank-bm25
    if RANK_BM25_AVAILABLE and len(corpus) <= 100000:
        print_section("rank-bm25")

        start = time.time()
        tokenized = [preprocess_text(doc.title + " " + doc.content) for doc in corpus]
        doc_ids = [doc.id for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        rank_build = time.time() - start

        rank_latencies = []
        for query in queries:
            start = time.perf_counter()
            query_tokens = preprocess_text(query)
            scores = bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
            rank_latencies.append(time.perf_counter() - start)

        results["rank-bm25"] = {
            "build_time_s": rank_build,
            "avg_latency_ms": np.mean(rank_latencies) * 1000,
            "p50_latency_ms": np.percentile(rank_latencies, 50) * 1000,
            "p99_latency_ms": np.percentile(rank_latencies, 99) * 1000,
        }

        print_timing("Build time", rank_build)
        print_metric("Avg latency", f"{results['rank-bm25']['avg_latency_ms']:.3f}ms")
        print_metric("Vajra speedup", f"{results['rank-bm25']['avg_latency_ms']/results['vajra']['avg_latency_ms']:.1f}x")
    elif RANK_BM25_AVAILABLE:
        print_section("rank-bm25")
        print_metric("Status", "Skipped (corpus > 100K, too slow)")

    # BM25S
    if BM25S_AVAILABLE:
        print_section("BM25S")

        start = time.time()
        corpus_texts = [doc.title + " " + doc.content for doc in corpus]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        bm25s_build = time.time() - start

        doc_ids = [doc.id for doc in corpus]
        bm25s_latencies = []
        for query in queries:
            start = time.perf_counter()
            query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
            results_obj, scores = retriever.retrieve(query_tokens, k=10, n_threads=1, show_progress=False)
            bm25s_latencies.append(time.perf_counter() - start)

        results["bm25s"] = {
            "build_time_s": bm25s_build,
            "avg_latency_ms": np.mean(bm25s_latencies) * 1000,
            "p50_latency_ms": np.percentile(bm25s_latencies, 50) * 1000,
            "p99_latency_ms": np.percentile(bm25s_latencies, 99) * 1000,
        }

        print_timing("Build time", bm25s_build)
        print_metric("Avg latency", f"{results['bm25s']['avg_latency_ms']:.3f}ms")
        print_metric("Vajra vs BM25S", f"{results['vajra']['avg_latency_ms']/results['bm25s']['avg_latency_ms']:.2f}x")

    # Summary table
    print_section("COMPARISON SUMMARY")

    if RICH_AVAILABLE:
        table = Table(box=box.ROUNDED)
        table.add_column("Engine", style="cyan")
        table.add_column("Build (s)", justify="right")
        table.add_column("Avg Latency (ms)", justify="right")
        table.add_column("P99 Latency (ms)", justify="right")

        for engine, metrics in results.items():
            table.add_row(
                engine,
                f"{metrics['build_time_s']:.2f}",
                f"{metrics['avg_latency_ms']:.3f}",
                f"{metrics['p99_latency_ms']:.3f}",
            )

        console.print(table)
    else:
        print(f"{'Engine':<15} {'Build (s)':<12} {'Avg (ms)':<15} {'P99 (ms)':<12}")
        print("-" * 55)
        for engine, metrics in results.items():
            print(f"{engine:<15} {metrics['build_time_s']:<12.2f} "
                  f"{metrics['avg_latency_ms']:<15.3f} {metrics['p99_latency_ms']:<12.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vajra BM25 Unified Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  index-build    Profile index building phases (tokenization, matrix build, etc.)
  query-latency  Profile query execution phases (preprocessing, scoring, etc.)
  comparison     Compare Vajra vs rank-bm25 and BM25S

Examples:
  python profiler.py --mode index-build --corpus /path/to/corpus.jsonl
  python profiler.py --mode query-latency --dataset wiki-200k
  python profiler.py --mode comparison --corpus /path/to/corpus.jsonl --max-docs 50000
        """
    )

    parser.add_argument(
        '--mode', required=True,
        choices=['index-build', 'query-latency', 'comparison'],
        help="Profiling mode"
    )
    parser.add_argument(
        '--corpus', type=Path,
        help="Path to JSONL corpus file"
    )
    parser.add_argument(
        '--dataset',
        choices=['wiki-100k', 'wiki-200k', 'wiki-500k', 'wiki-1m'],
        help="Use a Wikipedia dataset instead of --corpus"
    )
    parser.add_argument(
        '--cache-dir', type=Path,
        default=Path.home() / "Github" / "ir_benchmark_data",
        help="Cache directory for datasets"
    )
    parser.add_argument(
        '--max-docs', type=int,
        help="Limit number of documents to load"
    )
    parser.add_argument(
        '--num-queries', type=int, default=50,
        help="Number of test queries (for query-latency and comparison modes)"
    )
    parser.add_argument(
        '--output', type=Path,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Load corpus
    if args.dataset:
        max_docs = int(args.dataset.replace('wiki-', '').replace('k', '000').replace('m', '000000'))
        wiki_file = args.cache_dir / "wikipedia" / f"wikipedia_{max_docs}.jsonl"
        if not wiki_file.exists():
            print(f"ERROR: Wikipedia file not found: {wiki_file}")
            print(f"Run: python benchmarks/download_wikipedia.py --max-docs {max_docs}")
            sys.exit(1)
        corpus = DocumentCorpus.load_jsonl(wiki_file)
    elif args.corpus:
        if not args.corpus.exists():
            print(f"ERROR: Corpus file not found: {args.corpus}")
            sys.exit(1)
        corpus = DocumentCorpus.load_jsonl(args.corpus)
    else:
        print("ERROR: Either --corpus or --dataset required")
        sys.exit(1)

    # Apply max-docs limit
    if args.max_docs and len(corpus) > args.max_docs:
        corpus = DocumentCorpus(list(corpus.documents)[:args.max_docs])

    # Run profiling
    if args.mode == 'index-build':
        results = profile_index_build(corpus)
    elif args.mode == 'query-latency':
        results = profile_query_latency(corpus, args.num_queries)
    elif args.mode == 'comparison':
        results = profile_comparison(corpus, args.num_queries)

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert numpy types to Python types for JSON
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
