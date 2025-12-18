#!/usr/bin/env python3
"""
Vajra BM25 Benchmark on Standard IR Datasets

Benchmarks Vajra against rank-bm25 and BM25S on standard IR evaluation datasets:
- BEIR SciFact (scientific fact verification)
- BEIR NFCorpus (biomedical information retrieval)
- Wikipedia (100K, 1M, 5M documents)
- MS MARCO passage ranking
- Natural Questions

Requirements:
    pip install beir rank-bm25 bm25s vajra-bm25[optimized] ir-datasets
"""

import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import sys

# Standard IR evaluation
try:
    import ir_datasets
    IR_DATASETS_AVAILABLE = True
except ImportError:
    IR_DATASETS_AVAILABLE = False
    print("Warning: ir_datasets not installed. Run: pip install ir-datasets")

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False
    print("Warning: beir not installed. Run: pip install beir")

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

from vajra_bm25 import (
    Document,
    DocumentCorpus,
    VajraSearch,
    VajraSearchOptimized,
    VajraSearchParallel,
    preprocess_text,
)

# Wikipedia downloader
try:
    from download_wikipedia import WikipediaDownloader
    WIKIPEDIA_DOWNLOADER_AVAILABLE = True
except ImportError:
    WIKIPEDIA_DOWNLOADER_AVAILABLE = False
    print("Warning: Wikipedia downloader not available. Run from benchmarks directory.")

# Set up logging
sys.path.insert(0, str(Path(__file__).parent.parent))
from vajra_bm25.logging_config import get_logger
logger = get_logger("benchmark")


@dataclass
class EvalQuery:
    """Query with relevance judgments."""
    query_id: str
    text: str
    relevant_docs: List[str]  # List of relevant doc IDs


def load_msmarco_dev(max_docs: int = 100000, max_queries: int = 500) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """
    Load MS MARCO passage ranking dev set.

    Args:
        max_docs: Maximum number of documents to load
        max_queries: Maximum number of queries to evaluate

    Returns:
        Tuple of (corpus, queries with relevance judgments)
    """
    if not IR_DATASETS_AVAILABLE:
        raise ImportError("ir_datasets required. Run: pip install ir-datasets")

    print("Loading MS MARCO passage dev set...")
    dataset = ir_datasets.load("msmarco-passage/dev/small")

    # Load documents
    print(f"Loading up to {max_docs} documents...")
    documents = []
    doc_count = 0
    for doc in dataset.docs_iter():
        if doc_count >= max_docs:
            break
        documents.append(Document(
            id=doc.doc_id,
            title="",  # MS MARCO passages don't have titles
            content=doc.text
        ))
        doc_count += 1
        if doc_count % 10000 == 0:
            print(f"  Loaded {doc_count} documents...")

    corpus = DocumentCorpus(documents)
    print(f"Loaded {len(corpus)} documents")

    # Build set of loaded doc IDs for filtering
    loaded_doc_ids = set(doc.id for doc in documents)

    # Load queries with qrels
    print(f"Loading up to {max_queries} queries with relevance judgments...")
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = []
        if qrel.relevance > 0 and qrel.doc_id in loaded_doc_ids:
            qrels[qrel.query_id].append(qrel.doc_id)

    queries = []
    query_count = 0
    for query in dataset.queries_iter():
        if query_count >= max_queries:
            break
        if query.query_id in qrels and len(qrels[query.query_id]) > 0:
            queries.append(EvalQuery(
                query_id=query.query_id,
                text=query.text,
                relevant_docs=qrels[query.query_id]
            ))
            query_count += 1

    print(f"Loaded {len(queries)} queries with relevance judgments")
    return corpus, queries


def load_beir_dataset(dataset_name: str = "scifact", max_docs: int = None) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """
    Load a BEIR benchmark dataset.

    Args:
        dataset_name: Name of BEIR dataset (scifact, nfcorpus, arguana, etc.)
        max_docs: Maximum number of documents (None for all)

    Returns:
        Tuple of (corpus, queries with relevance judgments)
    """
    if not BEIR_AVAILABLE:
        raise ImportError("beir required. Run: pip install beir")

    print(f"Loading BEIR dataset: {dataset_name}...")

    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = Path("./datasets") / dataset_name

    if not data_path.exists():
        print(f"Downloading {dataset_name}...")
        data_path = util.download_and_unzip(url, "./datasets")

    # Load using BEIR's data loader
    corpus_data, queries_data, qrels_data = GenericDataLoader(data_folder=str(data_path)).load(split="test")

    # Convert to Vajra format
    documents = []
    doc_count = 0
    for doc_id, doc_content in corpus_data.items():
        if max_docs and doc_count >= max_docs:
            break
        title = doc_content.get("title", "")
        text = doc_content.get("text", "")
        documents.append(Document(
            id=doc_id,
            title=title,
            content=text
        ))
        doc_count += 1

    corpus = DocumentCorpus(documents)
    print(f"Loaded {len(corpus)} documents")

    # Build set of loaded doc IDs
    loaded_doc_ids = set(doc.id for doc in documents)

    # Convert queries
    queries = []
    for query_id, query_text in queries_data.items():
        if query_id in qrels_data:
            relevant_docs = [
                doc_id for doc_id, relevance in qrels_data[query_id].items()
                if relevance > 0 and doc_id in loaded_doc_ids
            ]
            if relevant_docs:
                queries.append(EvalQuery(
                    query_id=query_id,
                    text=query_text,
                    relevant_docs=relevant_docs
                ))

    print(f"Loaded {len(queries)} queries with relevance judgments")
    return corpus, queries


def load_wikipedia(
    max_docs: int = 100000,
    cache_dir: Optional[Path] = None,
    sample_queries: bool = True
) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """
    Load Wikipedia corpus for benchmarking.

    Args:
        max_docs: Number of documents to load (100K, 1M, or 5M)
        cache_dir: Cache directory (default: ~/Github/ir_benchmark_data)
        sample_queries: Generate sample queries from documents (True) or load from qrels (False)

    Returns:
        Tuple of (corpus, queries)
    """
    if not WIKIPEDIA_DOWNLOADER_AVAILABLE:
        raise ImportError("Wikipedia downloader not available. Ensure download_wikipedia.py is in benchmarks/")

    print(f"Loading Wikipedia corpus ({max_docs:,} documents)...")

    # Initialize downloader
    if cache_dir is None:
        cache_dir = Path.home() / "Github" / "ir_benchmark_data"

    downloader = WikipediaDownloader(cache_dir=cache_dir)

    # Download/load corpus
    if not downloader.is_cached(max_docs):
        print(f"Wikipedia corpus not cached. Downloading {max_docs:,} documents...")
        print("This may take a while on first run...")
        downloader.download_and_cache(max_docs=max_docs)

    corpus = downloader.load_corpus(max_docs=max_docs)
    print(f"✓ Loaded {len(corpus):,} Wikipedia documents")

    # Generate or load queries
    if sample_queries:
        # Generate synthetic queries from document titles and content
        print("Generating sample queries from Wikipedia articles...")
        queries = _generate_wikipedia_queries(corpus, num_queries=min(500, len(corpus) // 100))
    else:
        # Try to load actual Wikipedia qrels if available
        print("Loading Wikipedia qrels...")
        queries = _load_wikipedia_qrels(corpus)

    print(f"✓ Generated {len(queries)} test queries")
    return corpus, queries


def _generate_wikipedia_queries(corpus: DocumentCorpus, num_queries: int = 500) -> List[EvalQuery]:
    """
    Generate synthetic queries from Wikipedia articles.

    Strategy:
    1. Sample documents uniformly
    2. Extract key phrases from titles and first sentences
    3. Create queries that should retrieve the source document
    """
    import random

    documents = list(corpus.documents)
    random.seed(42)  # Reproducible

    # Sample documents
    sample_docs = random.sample(documents, min(num_queries, len(documents)))

    queries = []
    for idx, doc in enumerate(sample_docs):
        # Create query from title (most distinctive signal)
        if doc.title and len(doc.title.strip()) > 3:
            # Use title as query
            query_text = doc.title.strip()

            # Add first sentence if title is very short
            if len(query_text.split()) < 3 and doc.content:
                first_sentence = doc.content.split('.')[0].strip()
                if len(first_sentence) < 200:  # Reasonable length
                    query_text = f"{query_text} {first_sentence}"

            queries.append(EvalQuery(
                query_id=f"wiki_q{idx}",
                text=query_text,
                relevant_docs=[doc.id]
            ))

    return queries


def _load_wikipedia_qrels(corpus: DocumentCorpus) -> List[EvalQuery]:
    """
    Load actual Wikipedia qrels if available via ir-datasets.

    Falls back to synthetic queries if not available.
    """
    if not IR_DATASETS_AVAILABLE:
        print("  ir-datasets not available, using synthetic queries")
        return _generate_wikipedia_queries(corpus)

    try:
        # Try to load Wikipedia dataset with queries
        dataset = ir_datasets.load('wiki/en-2019')

        # Build doc ID set for filtering
        doc_ids = set(doc.id for doc in corpus.documents)

        # Load queries with qrels
        queries = []
        # Note: Most Wikipedia dumps don't come with queries/qrels
        # This is a placeholder for when they do
        print("  No qrels available for Wikipedia, using synthetic queries")
        return _generate_wikipedia_queries(corpus)

    except Exception as e:
        print(f"  Could not load Wikipedia qrels: {e}")
        print("  Using synthetic queries")
        return _generate_wikipedia_queries(corpus)


def load_natural_questions(max_docs: int = 50000, max_queries: int = 500) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """
    Load Natural Questions dataset for retrieval.

    Args:
        max_docs: Maximum number of documents to load
        max_queries: Maximum number of queries

    Returns:
        Tuple of (corpus, queries with relevance judgments)
    """
    if not IR_DATASETS_AVAILABLE:
        raise ImportError("ir_datasets required. Run: pip install ir-datasets")

    print("Loading Natural Questions dataset...")

    # Use the DPH-w100 version which has full Wikipedia for realistic retrieval
    try:
        dataset = ir_datasets.load("dph-w100/natural-questions/dev")
    except Exception:
        # Fallback to standard NQ if DPH version not available
        print("DPH version not available, using standard NQ...")
        dataset = ir_datasets.load("natural-questions/dev")

    # Load documents
    print(f"Loading up to {max_docs} documents...")
    documents = []
    doc_count = 0
    for doc in dataset.docs_iter():
        if doc_count >= max_docs:
            break
        documents.append(Document(
            id=doc.doc_id,
            title=getattr(doc, 'title', ''),
            content=doc.text if hasattr(doc, 'text') else str(doc)
        ))
        doc_count += 1
        if doc_count % 10000 == 0:
            print(f"  Loaded {doc_count} documents...")

    corpus = DocumentCorpus(documents)
    print(f"Loaded {len(corpus)} documents")

    # Build set of loaded doc IDs
    loaded_doc_ids = set(doc.id for doc in documents)

    # Load queries with qrels
    print(f"Loading up to {max_queries} queries...")
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = []
        if qrel.relevance > 0 and qrel.doc_id in loaded_doc_ids:
            qrels[qrel.query_id].append(qrel.doc_id)

    queries = []
    query_count = 0
    for query in dataset.queries_iter():
        if query_count >= max_queries:
            break
        if query.query_id in qrels and len(qrels[query.query_id]) > 0:
            queries.append(EvalQuery(
                query_id=query.query_id,
                text=query.text,
                relevant_docs=qrels[query.query_id]
            ))
            query_count += 1

    print(f"Loaded {len(queries)} queries with relevance judgments")
    return corpus, queries


# Evaluation metrics

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Calculate Recall@k."""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Calculate Precision@k."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 10) -> float:
    """Calculate NDCG@k (binary relevance)."""
    import math

    relevant_set = set(relevant_ids)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# Wrapper classes

class RankBM25Wrapper:
    """Wrapper around rank-bm25."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.tokenized_docs = []
        for doc in corpus:
            full_text = doc.title + " " + doc.content
            tokens = preprocess_text(full_text)
            self.tokenized_docs.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        scored_docs = [(scores[i], i) for i in range(len(scores))]
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [self.doc_ids[idx] for _, idx in scored_docs[:top_k]]


class BM25SWrapper:
    """Wrapper around BM25S."""

    def __init__(self, corpus: DocumentCorpus, n_threads: int = 1):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.n_threads = n_threads

        corpus_texts = [doc.title + " " + doc.content for doc in corpus]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)

        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results_obj, scores = self.retriever.retrieve(
            query_tokens, k=top_k, n_threads=self.n_threads, show_progress=False
        )
        return [self.doc_ids[idx] for idx in results_obj[0]]


class VajraWrapper:
    """Wrapper around Vajra."""

    def __init__(self, corpus: DocumentCorpus, variant: str = "parallel"):
        self.corpus = corpus
        if variant == "base":
            self.engine = VajraSearch(corpus)
        elif variant == "optimized":
            self.engine = VajraSearchOptimized(corpus)
        else:
            self.engine = VajraSearchParallel(corpus, max_workers=8)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        results = self.engine.search(query, top_k=top_k)
        return [r.document.id for r in results]


def evaluate_engine(engine, queries: List[EvalQuery], top_k: int = 10) -> Dict[str, float]:
    """Evaluate a search engine on queries with relevance judgments."""

    recall_scores = []
    precision_scores = []
    mrr_scores = []
    ndcg_scores = []
    latencies = []

    for query in queries:
        start = time.time()
        retrieved_ids = engine.search(query.text, top_k=top_k)
        latencies.append(time.time() - start)

        recall_scores.append(recall_at_k(retrieved_ids, query.relevant_docs, k=top_k))
        precision_scores.append(precision_at_k(retrieved_ids, query.relevant_docs, k=top_k))
        mrr_scores.append(mrr(retrieved_ids, query.relevant_docs))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, query.relevant_docs, k=top_k))

    return {
        f"Recall@{top_k}": statistics.mean(recall_scores) * 100,
        f"Precision@{top_k}": statistics.mean(precision_scores) * 100,
        "MRR": statistics.mean(mrr_scores) * 100,
        f"NDCG@{top_k}": statistics.mean(ndcg_scores) * 100,
        "Avg Latency (ms)": statistics.mean(latencies) * 1000,
        "P50 Latency (ms)": statistics.median(latencies) * 1000,
    }


def run_benchmark(dataset_name: str, corpus: DocumentCorpus, queries: List[EvalQuery]):
    """Run benchmark on a dataset."""

    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK: {dataset_name}")
    logger.info(f"Corpus: {len(corpus)} documents, Queries: {len(queries)}")
    logger.info(f"{'='*80}")

    results = {}

    # Build engines
    logger.info("\nBuilding search engines...")

    if RANK_BM25_AVAILABLE:
        logger.info("  Building rank-bm25...")
        start = time.time()
        rank_bm25 = RankBM25Wrapper(corpus)
        logger.info(f"    Built in {(time.time()-start)*1000:.1f}ms")

    logger.info("  Building Vajra (Optimized)...")
    start = time.time()
    vajra_opt = VajraWrapper(corpus, variant="optimized")
    logger.info(f"    Built in {(time.time()-start)*1000:.1f}ms")

    logger.info("  Building Vajra (Parallel)...")
    start = time.time()
    vajra_par = VajraWrapper(corpus, variant="parallel")
    logger.info(f"    Built in {(time.time()-start)*1000:.1f}ms")

    if BM25S_AVAILABLE:
        logger.info("  Building BM25S...")
        start = time.time()
        bm25s_engine = BM25SWrapper(corpus, n_threads=1)
        logger.info(f"    Built in {(time.time()-start)*1000:.1f}ms")

        logger.info("  Building BM25S (Parallel, 8 threads)...")
        start = time.time()
        bm25s_parallel = BM25SWrapper(corpus, n_threads=8)
        logger.info(f"    Built in {(time.time()-start)*1000:.1f}ms")

    # Evaluate
    logger.info("\nEvaluating...")

    if RANK_BM25_AVAILABLE:
        logger.info("  Evaluating rank-bm25...")
        results["rank-bm25"] = evaluate_engine(rank_bm25, queries)

    logger.info("  Evaluating Vajra (Optimized)...")
    results["Vajra (Optimized)"] = evaluate_engine(vajra_opt, queries)

    logger.info("  Evaluating Vajra (Parallel, 8 workers)...")
    results["Vajra (Parallel)"] = evaluate_engine(vajra_par, queries)

    if BM25S_AVAILABLE:
        logger.info("  Evaluating BM25S...")
        results["BM25S"] = evaluate_engine(bm25s_engine, queries)

        logger.info("  Evaluating BM25S (Parallel, 8 threads)...")
        results["BM25S (Parallel)"] = evaluate_engine(bm25s_parallel, queries)

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {dataset_name}")
    logger.info(f"{'='*80}")

    # Header
    metrics = ["Recall@10", "NDCG@10", "MRR", "Avg Latency (ms)"]
    header = f"{'Engine':<20}"
    for metric in metrics:
        header += f" {metric:<15}"
    logger.info(f"\n{header}")
    logger.info("-" * (20 + 15 * len(metrics)))

    # Results
    for engine_name, engine_results in results.items():
        row = f"{engine_name:<20}"
        for metric in metrics:
            value = engine_results.get(metric, 0)
            if "Latency" in metric:
                row += f" {value:<15.2f}"
            else:
                row += f" {value:<15.1f}"
        logger.info(row)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Vajra BM25 on standard IR datasets"
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['beir-scifact', 'beir-nfcorpus', 'wiki-100k', 'wiki-1m', 'wiki-5m', 'msmarco', 'nq', 'all'],
        default=['beir-scifact'],
        help="Datasets to benchmark (default: beir-scifact)"
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path.home() / "Github" / "ir_benchmark_data",
        help="Cache directory for downloaded datasets"
    )

    args = parser.parse_args()

    # Expand 'all' option
    if 'all' in args.datasets:
        args.datasets = ['beir-scifact', 'beir-nfcorpus', 'wiki-100k', 'wiki-1m', 'wiki-5m']

    logger.info("=" * 80)
    logger.info("VAJRA BM25 BENCHMARK ON STANDARD IR DATASETS")
    logger.info("=" * 80)
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("=" * 80)

    all_results = {}

    # BEIR datasets
    if 'beir-scifact' in args.datasets and BEIR_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading BEIR SciFact...")
            corpus, queries = load_beir_dataset("scifact")
            all_results["BEIR/SciFact"] = run_benchmark("BEIR/SciFact", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading SciFact: {e}")

    if 'beir-nfcorpus' in args.datasets and BEIR_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading BEIR NFCorpus...")
            corpus, queries = load_beir_dataset("nfcorpus")
            all_results["BEIR/NFCorpus"] = run_benchmark("BEIR/NFCorpus", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading NFCorpus: {e}")

    # Wikipedia datasets
    if 'wiki-100k' in args.datasets and WIKIPEDIA_DOWNLOADER_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading Wikipedia 100K...")
            corpus, queries = load_wikipedia(max_docs=100000, cache_dir=args.cache_dir)
            all_results["Wikipedia/100K"] = run_benchmark("Wikipedia/100K", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading Wikipedia 100K: {e}")
            import traceback
            traceback.print_exc()

    if 'wiki-1m' in args.datasets and WIKIPEDIA_DOWNLOADER_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading Wikipedia 1M...")
            corpus, queries = load_wikipedia(max_docs=1000000, cache_dir=args.cache_dir)
            all_results["Wikipedia/1M"] = run_benchmark("Wikipedia/1M", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading Wikipedia 1M: {e}")
            import traceback
            traceback.print_exc()

    if 'wiki-5m' in args.datasets and WIKIPEDIA_DOWNLOADER_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading Wikipedia 5M...")
            corpus, queries = load_wikipedia(max_docs=5000000, cache_dir=args.cache_dir)
            all_results["Wikipedia/5M"] = run_benchmark("Wikipedia/5M", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading Wikipedia 5M: {e}")
            import traceback
            traceback.print_exc()

    # MS MARCO
    if 'msmarco' in args.datasets and IR_DATASETS_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading MS MARCO...")
            corpus, queries = load_msmarco_dev(max_docs=100000, max_queries=500)
            all_results["MS MARCO/100K"] = run_benchmark("MS MARCO/100K", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading MS MARCO: {e}")

    # Natural Questions
    if 'nq' in args.datasets and IR_DATASETS_AVAILABLE:
        try:
            logger.info("\n" + "=" * 80)
            logger.info("Loading Natural Questions...")
            corpus, queries = load_natural_questions(max_docs=50000, max_queries=500)
            all_results["Natural Questions/50K"] = run_benchmark("Natural Questions/50K", corpus, queries)
        except Exception as e:
            logger.error(f"Error loading Natural Questions: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY ACROSS ALL DATASETS")
    logger.info("=" * 80)

    if all_results:
        engines = set()
        for dataset_results in all_results.values():
            engines.update(dataset_results.keys())

        header = f"{'Dataset':<25}"
        for engine in sorted(engines):
            header += f" {engine:<20}"
        logger.info(f"\n{header}")
        logger.info("-" * (25 + 20 * len(engines)))

        for dataset_name, dataset_results in all_results.items():
            row = f"{dataset_name:<25}"
            for engine in sorted(engines):
                if engine in dataset_results:
                    ndcg = dataset_results[engine].get("NDCG@10", 0)
                    row += f" NDCG@10: {ndcg:.1f}%{'':<5}"
                else:
                    row += f" {'N/A':<20}"
            logger.info(row)

    logger.info("\n" + "=" * 80)
    logger.info("Benchmark complete!")
    logger.info("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
