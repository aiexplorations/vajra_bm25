#!/usr/bin/env python3
"""
Vajra BM25 Unified Benchmark

Benchmarks Vajra against rank-bm25 and BM25S on various datasets with:
- Progress display with status indicators
- Automatic output to results file
- Index persistence to avoid expensive rebuilds

Usage:
    python benchmark.py --datasets wiki-200k wiki-500k
    python benchmark.py --datasets beir-scifact --engines vajra bm25s
    python benchmark.py --corpus /path/to/corpus.jsonl --output results/custom.json
"""

import argparse
import hashlib
import json
import os
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Progress display
RICH_AVAILABLE = False
TQDM_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    pass

if not RICH_AVAILABLE:
    try:
        from tqdm import tqdm
        TQDM_AVAILABLE = True
    except ImportError:
        pass

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

# Tantivy (Rust-based search engine)
try:
    import tantivy
    TANTIVY_AVAILABLE = True
except ImportError:
    TANTIVY_AVAILABLE = False

# Pyserini (Lucene wrapper via Anserini) - requires Java 11+
# NOTE: Import is deferred to avoid JVM conflicts with multiprocessing fork
# The JVM doesn't handle fork() well on macOS, causing SIGSEGV crashes
PYSERINI_AVAILABLE = False  # Will be set True lazily when needed

def setup_jvm_environment():
    """Configure JVM environment for Pyserini on macOS.

    Must be called before importing pyserini to ensure JVM can be found.
    """
    import subprocess
    import platform

    if platform.system() != "Darwin":
        return  # Only needed on macOS

    # Skip if already configured
    if os.environ.get("JVM_PATH"):
        return

    try:
        # Get Java home from macOS java_home utility
        result = subprocess.run(
            ["/usr/libexec/java_home"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            os.environ["JAVA_HOME"] = java_home
            # Set JVM_PATH for pyjnius
            jvm_path = os.path.join(java_home, "lib", "server", "libjvm.dylib")
            if os.path.exists(jvm_path):
                os.environ["JVM_PATH"] = jvm_path
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # Java not installed or java_home not available

# BEIR for standard datasets
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False

# Vajra
sys.path.insert(0, str(Path(__file__).parent.parent))
from vajra_bm25 import (
    Document,
    DocumentCorpus,
    VajraSearchOptimized,
    VajraSearchParallel,
    preprocess_text,
)

# Persistence
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# =============================================================================
# Console and Progress Display
# =============================================================================

if RICH_AVAILABLE:
    console = Console()

    def print_status(msg: str, style: str = ""):
        console.print(msg, style=style)

    def print_error(msg: str):
        console.print(f"[red]ERROR:[/red] {msg}")

    def print_success(msg: str):
        console.print(f"[green]{msg}[/green]")

    def print_header(title: str):
        console.print(Panel(title, style="bold blue", box=box.DOUBLE))

else:
    def print_status(msg: str, style: str = ""):
        print(msg)

    def print_error(msg: str):
        print(f"ERROR: {msg}")

    def print_success(msg: str):
        print(msg)

    def print_header(title: str):
        print("=" * 70)
        print(title)
        print("=" * 70)


class BenchmarkProgress:
    """Unified progress display for benchmarks."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.progress = None
        self.current_task = None

    def __enter__(self):
        if self.quiet:
            return self
        if RICH_AVAILABLE:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            )
            self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: int) -> Any:
        if self.quiet:
            return None
        if RICH_AVAILABLE and self.progress:
            return self.progress.add_task(description, total=total)
        elif TQDM_AVAILABLE:
            return tqdm(total=total, desc=description, ncols=80)
        return None

    def update(self, task_id: Any, advance: int = 1, description: str = None):
        if self.quiet or task_id is None:
            return
        if RICH_AVAILABLE and self.progress:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = description
            self.progress.update(task_id, **kwargs)
        elif TQDM_AVAILABLE and hasattr(task_id, 'update'):
            task_id.update(advance)
            if description:
                task_id.set_description(description)

    def complete(self, task_id: Any):
        if self.quiet or task_id is None:
            return
        if TQDM_AVAILABLE and hasattr(task_id, 'close'):
            task_id.close()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EvalQuery:
    """Query with relevance judgments."""
    query_id: str
    text: str
    relevant_docs: List[str]


@dataclass
class SearchResultSample:
    """A single search result with document details."""
    rank: int
    doc_id: str
    title: str
    snippet: str  # First 200 chars of content
    is_relevant: bool  # Whether this doc is in the relevance judgments


@dataclass
class QuerySample:
    """A sample query with its search results."""
    query_id: str
    query_text: str
    latency_ms: float
    results: List[SearchResultSample]
    relevant_docs: List[str]  # Ground truth relevant docs
    recall_at_10: float
    ndcg_at_10: float


@dataclass
class BenchmarkResult:
    """Results from a single engine benchmark."""
    engine: str
    build_time_s: float
    index_loaded: bool  # True if loaded from cache
    # Single query metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    queries_per_sec: float
    # Batch query metrics (optional)
    batch_latency_ms: Optional[float] = None  # Total time for batch
    batch_qps: Optional[float] = None  # Queries per second in batch mode
    # Sample queries with results
    sample_queries: Optional[List[QuerySample]] = None


@dataclass
class DatasetResult:
    """Results for an entire dataset."""
    dataset: str
    corpus_size: int
    num_queries: int
    timestamp: str
    results: List[BenchmarkResult]


# =============================================================================
# Index Persistence
# =============================================================================

class IndexCache:
    """Manages persistent index storage."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _corpus_hash(self, corpus: DocumentCorpus) -> str:
        """Generate a hash for the corpus based on document IDs and content length."""
        h = hashlib.md5()
        for doc in corpus.documents[:100]:  # Sample first 100 docs
            h.update(doc.id.encode())
            h.update(str(len(doc.content)).encode())
        h.update(str(len(corpus)).encode())
        return h.hexdigest()[:16]

    def _index_path(self, engine: str, corpus_hash: str) -> Path:
        return self.cache_dir / f"{engine}_{corpus_hash}.idx"

    def get(self, engine: str, corpus: DocumentCorpus) -> Optional[Any]:
        """Load index from cache if available."""
        corpus_hash = self._corpus_hash(corpus)
        path = self._index_path(engine, corpus_hash)

        if not path.exists():
            return None

        try:
            if engine == "bm25s":
                # BM25S has its own load mechanism
                return bm25s.BM25.load(str(path), load_corpus=False)
            elif JOBLIB_AVAILABLE:
                return joblib.load(path)
            else:
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print_status(f"  Cache load failed: {e}", style="yellow")
            return None

    def save(self, engine: str, corpus: DocumentCorpus, index: Any) -> bool:
        """Save index to cache."""
        corpus_hash = self._corpus_hash(corpus)
        path = self._index_path(engine, corpus_hash)

        try:
            if engine == "bm25s":
                index.save(str(path))
            elif JOBLIB_AVAILABLE:
                joblib.dump(index, path, compress=3)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(index, f)
            return True
        except Exception as e:
            print_status(f"  Cache save failed: {e}", style="yellow")
            return False

    def clear(self):
        """Clear all cached indexes."""
        import shutil
        for f in self.cache_dir.glob("*.idx"):
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()


# =============================================================================
# Dataset Loaders
# =============================================================================

def load_beir_dataset(dataset_name: str) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """Load a BEIR benchmark dataset."""
    if not BEIR_AVAILABLE:
        raise ImportError("BEIR not installed. Run: pip install beir")

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = Path("./datasets") / dataset_name

    if not data_path.exists():
        print_status(f"  Downloading {dataset_name}...")
        data_path = util.download_and_unzip(url, "./datasets")

    corpus_data, queries_data, qrels_data = GenericDataLoader(
        data_folder=str(data_path)
    ).load(split="test")

    documents = [
        Document(
            id=doc_id,
            title=doc.get("title", ""),
            content=doc.get("text", "")
        )
        for doc_id, doc in corpus_data.items()
    ]
    corpus = DocumentCorpus(documents)

    loaded_doc_ids = set(doc.id for doc in documents)
    queries = []
    for query_id, query_text in queries_data.items():
        if query_id in qrels_data:
            relevant_docs = [
                doc_id for doc_id, rel in qrels_data[query_id].items()
                if rel > 0 and doc_id in loaded_doc_ids
            ]
            if relevant_docs:
                queries.append(EvalQuery(query_id, query_text, relevant_docs))

    return corpus, queries


def load_wikipedia(max_docs: int, cache_dir: Path) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """Load Wikipedia corpus."""
    import random

    wiki_file = cache_dir / "wikipedia" / f"wikipedia_{max_docs}.jsonl"
    if not wiki_file.exists():
        raise FileNotFoundError(
            f"Wikipedia corpus not found: {wiki_file}\n"
            f"Run: python benchmarks/download_wikipedia.py --max-docs {max_docs}"
        )

    corpus = DocumentCorpus.load_jsonl(wiki_file)

    # Generate synthetic queries from document titles or content
    random.seed(42)
    sample_size = min(500, len(corpus) // 100)
    sample_docs = random.sample(list(corpus.documents), min(sample_size * 2, len(corpus)))

    queries = []
    for idx, doc in enumerate(sample_docs):
        if len(queries) >= sample_size:
            break

        query_text = None

        # Try title first
        if doc.title and len(doc.title.strip()) > 3:
            query_text = doc.title.strip()
        # Fall back to content fragment (3-6 words from middle of document)
        elif doc.content and len(doc.content) > 100:
            # Take a fragment from the middle of the document
            mid = len(doc.content) // 2
            fragment = doc.content[mid:mid+200]
            # Find word boundaries
            words = fragment.split()
            if len(words) >= 4:
                # Take 3-6 words
                query_text = " ".join(words[1:min(6, len(words)-1)])

        if query_text and len(query_text.split()) >= 3:
            queries.append(EvalQuery(
                query_id=f"wiki_q{idx}",
                text=query_text,
                relevant_docs=[doc.id]
            ))

    return corpus, queries


def load_custom_corpus(corpus_path: Path, queries_path: Optional[Path] = None
                       ) -> Tuple[DocumentCorpus, List[EvalQuery]]:
    """Load a custom JSONL corpus."""
    import random

    corpus = DocumentCorpus.load_jsonl(corpus_path)

    if queries_path and queries_path.exists():
        # Load queries from file
        queries = []
        with open(queries_path) as f:
            for line in f:
                data = json.loads(line)
                queries.append(EvalQuery(
                    query_id=data["id"],
                    text=data["text"],
                    relevant_docs=data.get("relevant_docs", [])
                ))
    else:
        # Generate synthetic queries
        random.seed(42)
        sample_size = min(100, len(corpus) // 10)
        sample_docs = random.sample(list(corpus.documents), min(sample_size * 2, len(corpus)))

        queries = []
        for idx, doc in enumerate(sample_docs):
            if len(queries) >= sample_size:
                break
            if doc.title and len(doc.title.split()) >= 2:
                queries.append(EvalQuery(
                    query_id=f"q{idx}",
                    text=doc.title,
                    relevant_docs=[doc.id]
                ))

    return corpus, queries


# =============================================================================
# Evaluation Metrics
# =============================================================================

def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 10) -> float:
    import math
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant_set
    )
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# Engine Wrappers
# =============================================================================

class VajraEngine:
    """Wrapper for Vajra search engines."""

    def __init__(self, corpus: DocumentCorpus, parallel: bool = False, workers: int = 8, cache_size: int = 0):
        self.corpus = corpus
        self.parallel = parallel
        self.workers = workers
        self.cache_size = cache_size
        self.engine = None
        self.supports_batch = parallel  # Only parallel version has search_batch

    def build(self):
        if self.parallel:
            self.engine = VajraSearchParallel(self.corpus, max_workers=self.workers, cache_size=self.cache_size)
        else:
            self.engine = VajraSearchOptimized(self.corpus, cache_size=self.cache_size)

    def clear_cache(self):
        """Clear query cache if available."""
        if hasattr(self.engine, 'clear_cache'):
            self.engine.clear_cache()

    def search(self, query: str, top_k: int = 10) -> List[str]:
        results = self.engine.search(query, top_k=top_k)
        return [r.document.id for r in results]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """Batch search - uses parallel execution if available."""
        if self.parallel and hasattr(self.engine, 'search_batch'):
            batch_results = self.engine.search_batch(queries, top_k=top_k)
            return [[r.document.id for r in results] for results in batch_results]
        else:
            # Fallback to sequential
            return [self.search(q, top_k) for q in queries]


class RankBM25Engine:
    """Wrapper for rank-bm25."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.bm25 = None
        self.supports_batch = False

    def build(self):
        tokenized = []
        for doc in self.corpus:
            text = doc.title + " " + doc.content
            tokenized.append(preprocess_text(text))
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        query_tokens = preprocess_text(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.doc_ids[i] for i in top_indices]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """Sequential batch (no parallelism)."""
        return [self.search(q, top_k) for q in queries]


class BM25SEngine:
    """Wrapper for BM25S."""

    def __init__(self, corpus: DocumentCorpus, n_threads: int = 1):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.n_threads = n_threads
        self.retriever = None
        self.supports_batch = True  # BM25S supports native batch retrieval

    def build(self):
        corpus_texts = [doc.title + " " + doc.content for doc in self.corpus]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        query_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        results, _ = self.retriever.retrieve(
            query_tokens, k=top_k, n_threads=self.n_threads, show_progress=False
        )
        return [self.doc_ids[idx] for idx in results[0]]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """Batch search - BM25S supports native batch retrieval."""
        query_tokens = bm25s.tokenize(queries, stopwords="en", show_progress=False)
        results, _ = self.retriever.retrieve(
            query_tokens, k=top_k, n_threads=self.n_threads, show_progress=False
        )
        return [[self.doc_ids[idx] for idx in row] for row in results]


class TantivyEngine:
    """Wrapper for Tantivy (Rust-based search engine)."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.index = None
        self.searcher = None
        self.supports_batch = False  # No native batch support

    def build(self):
        # Create schema with id, title, content fields
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("title", stored=False)
        schema_builder.add_text_field("content", stored=False)
        schema_builder.add_unsigned_field("doc_idx", stored=True)
        schema = schema_builder.build()

        # Create in-memory index
        self.index = tantivy.Index(schema)

        # Index all documents
        writer = self.index.writer()
        for idx, doc in enumerate(self.corpus):
            writer.add_document(tantivy.Document(
                title=doc.title or "",
                content=doc.content or "",
                doc_idx=idx,
            ))
        writer.commit()

        # Create searcher
        self.index.reload()
        self.searcher = self.index.searcher()

    def search(self, query: str, top_k: int = 10) -> List[str]:
        try:
            # Use index.parse_query() - searches all text fields by default
            parsed_query = self.index.parse_query(query, ["title", "content"])
            results = self.searcher.search(parsed_query, top_k).hits
            doc_ids = []
            for score, doc_address in results:
                doc = self.searcher.doc(doc_address)
                doc_idx = doc.get_first("doc_idx")
                if doc_idx is not None:
                    doc_ids.append(self.doc_ids[doc_idx])
            return doc_ids
        except Exception:
            # Query parsing can fail for special characters
            return []

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """Sequential batch (no native batch support)."""
        return [self.search(q, top_k) for q in queries]


class PyseriniEngine:
    """Wrapper for Pyserini (Lucene via Anserini)."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.doc_ids = [doc.id for doc in corpus]
        self.searcher = None
        self.index_dir = None
        self.supports_batch = True  # Pyserini supports batch search
        self.LuceneSearcher = None  # Lazy import

    def build(self):
        import tempfile
        import subprocess

        # Configure JVM environment before importing pyserini
        setup_jvm_environment()

        # Lazy import to avoid JVM conflicts with multiprocessing
        from pyserini.search.lucene import LuceneSearcher
        self.LuceneSearcher = LuceneSearcher

        # Create temp directory for index
        self.index_dir = tempfile.mkdtemp(prefix="pyserini_idx_")
        docs_file = Path(self.index_dir) / "docs.jsonl"

        # Write documents in Pyserini JSONL format
        with open(docs_file, 'w') as f:
            for doc in self.corpus:
                doc_obj = {
                    "id": doc.id,
                    "contents": f"{doc.title or ''} {doc.content or ''}"
                }
                f.write(json.dumps(doc_obj) + "\n")

        index_path = Path(self.index_dir) / "index"

        # Build Lucene index using pyserini command-line tool
        cmd = [
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(Path(self.index_dir)),
            "--index", str(index_path),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw",
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Pyserini indexing failed: {e.stderr.decode()}")

        # Initialize searcher with BM25 scoring
        self.searcher = self.LuceneSearcher(str(index_path))
        self.searcher.set_bm25(k1=1.5, b=0.75)

    def search(self, query: str, top_k: int = 10) -> List[str]:
        hits = self.searcher.search(query, k=top_k)
        return [hit.docid for hit in hits]

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        """Batch search using Pyserini's batch_search."""
        # Pyserini batch_search expects dict of {qid: query_text}
        qids = [f"q{i}" for i in range(len(queries))]
        query_dict = dict(zip(qids, queries))

        results = self.searcher.batch_search(query_dict, qids, k=top_k, threads=1)

        return [[hit.docid for hit in results[qid]] for qid in qids]

    def __del__(self):
        # Cleanup temp directory
        if self.index_dir and Path(self.index_dir).exists():
            import shutil
            try:
                shutil.rmtree(self.index_dir)
            except Exception:
                pass


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    dataset_name: str,
    corpus: DocumentCorpus,
    queries: List[EvalQuery],
    engines: List[str],
    index_cache: Optional[IndexCache],
    progress: BenchmarkProgress,
    workers: int = 8,
) -> DatasetResult:
    """Run benchmark on a dataset."""

    results = []
    n_engines = len(engines)

    # Main task for this dataset
    main_task = progress.add_task(
        f"[{dataset_name}] Initializing...",
        total=n_engines * 3  # Build + Single + Batch for each engine
    )

    for engine_name in engines:
        # Check availability
        if engine_name == "rank-bm25" and not RANK_BM25_AVAILABLE:
            print_status(f"  Skipping rank-bm25 (not installed)")
            progress.update(main_task, advance=3)
            continue
        if engine_name.startswith("bm25s") and not BM25S_AVAILABLE:
            print_status(f"  Skipping {engine_name} (not installed)")
            progress.update(main_task, advance=3)
            continue
        if engine_name == "tantivy" and not TANTIVY_AVAILABLE:
            print_status(f"  Skipping tantivy (not installed: pip install tantivy)")
            progress.update(main_task, advance=3)
            continue
        if engine_name == "pyserini":
            # Lazy check for pyserini - only import when explicitly requested
            # This avoids JVM conflicts with multiprocessing fork
            try:
                setup_jvm_environment()  # Configure JVM before import
                from pyserini.search.lucene import LuceneSearcher
            except (ImportError, Exception) as e:
                print_status(f"  Skipping pyserini (requires Java 11+: {e})")
                progress.update(main_task, advance=3)
                continue

        # Skip rank-bm25 for large corpora (too slow)
        if engine_name == "rank-bm25" and len(corpus) > 50000:
            print_status(f"  Skipping rank-bm25 (corpus > 50K, too slow)")
            progress.update(main_task, advance=3)
            continue

        # Build or load index
        progress.update(main_task, description=f"[{dataset_name}] Building {engine_name}...")

        # Use common cache keys for engines that share indexes
        # vajra and vajra-parallel use the same VajraSearchOptimized index
        # bm25s and bm25s-parallel use the same BM25S index
        # tantivy and pyserini don't support caching yet (directory-based indexes)
        if engine_name in ("vajra", "vajra-parallel"):
            cache_key = "vajra"
        elif engine_name in ("bm25s", "bm25s-parallel"):
            cache_key = "bm25s"
        elif engine_name in ("tantivy", "pyserini"):
            cache_key = None  # No caching for these engines
        else:
            cache_key = engine_name

        index_loaded = False
        cached_index = None
        if index_cache and cache_key:
            cached_index = index_cache.get(cache_key, corpus)
            if cached_index:
                index_loaded = True

        # Create engine with configurable workers
        # Note: cache_size=0 for fair comparison (no query result caching)
        if engine_name == "vajra":
            engine = VajraEngine(corpus, parallel=False, workers=workers, cache_size=0)
        elif engine_name == "vajra-parallel":
            engine = VajraEngine(corpus, parallel=True, workers=workers, cache_size=0)
        elif engine_name == "rank-bm25":
            engine = RankBM25Engine(corpus)
        elif engine_name == "bm25s":
            engine = BM25SEngine(corpus, n_threads=1)
        elif engine_name == "bm25s-parallel":
            engine = BM25SEngine(corpus, n_threads=workers)
        elif engine_name == "tantivy":
            engine = TantivyEngine(corpus)
        elif engine_name == "pyserini":
            engine = PyseriniEngine(corpus)
        else:
            print_error(f"Unknown engine: {engine_name}")
            progress.update(main_task, advance=3)
            continue

        # Build index
        if cached_index and cache_key == "bm25s":
            engine.retriever = cached_index
            build_time = 0.0
        elif cached_index and cache_key == "vajra":
            engine.engine = cached_index
            build_time = 0.0
        else:
            start = time.time()
            engine.build()
            build_time = time.time() - start

            # Cache the index (skip for engines that don't support caching)
            if index_cache and cache_key and not index_loaded:
                if cache_key == "bm25s":
                    index_cache.save(cache_key, corpus, engine.retriever)
                elif cache_key == "vajra":
                    index_cache.save(cache_key, corpus, engine.engine)

        progress.update(main_task, advance=1)

        # === Batch Query Evaluation ===
        progress.update(main_task, description=f"[{dataset_name}] Batch queries {engine_name}...")

        batch_latency_ms = None
        batch_qps = None

        # Run batch evaluation once (no caching, fair comparison)
        query_texts = [q.text for q in queries]
        start = time.time()
        batch_results = engine.search_batch(query_texts, top_k=10)
        total_batch_time = time.time() - start

        batch_latency_ms = (total_batch_time / len(queries)) * 1000  # Per-query latency
        batch_qps = len(queries) / total_batch_time

        progress.update(main_task, advance=1)

        # === Single Query Evaluation ===
        # Note: Query caching is disabled (cache_size=0) for fair comparison
        progress.update(main_task, description=f"[{dataset_name}] Single queries {engine_name}...")

        latencies = []
        recall_scores = []
        ndcg_scores = []
        mrr_scores = []

        # Collect sample queries (first 5 queries with their full results)
        sample_queries = []
        num_samples = min(5, len(queries))

        for idx, query in enumerate(queries):
            start = time.time()
            retrieved = engine.search(query.text, top_k=10)
            query_latency = time.time() - start
            latencies.append(query_latency)

            query_recall = recall_at_k(retrieved, query.relevant_docs, 10)
            query_ndcg = ndcg_at_k(retrieved, query.relevant_docs, 10)
            recall_scores.append(query_recall)
            ndcg_scores.append(query_ndcg)
            mrr_scores.append(mrr(retrieved, query.relevant_docs))

            # Capture sample query details for first N queries
            if idx < num_samples:
                relevant_set = set(query.relevant_docs)
                result_samples = []
                for rank, doc_id in enumerate(retrieved[:10], 1):
                    doc = corpus.get(doc_id)
                    if doc:
                        # Create snippet from content (first 200 chars)
                        content = doc.content or ""
                        snippet = content[:200].replace("\n", " ").strip()
                        if len(content) > 200:
                            snippet += "..."
                        result_samples.append(SearchResultSample(
                            rank=rank,
                            doc_id=doc_id,
                            title=doc.title or "(no title)",
                            snippet=snippet,
                            is_relevant=doc_id in relevant_set,
                        ))

                sample_queries.append(QuerySample(
                    query_id=query.query_id,
                    query_text=query.text,
                    latency_ms=query_latency * 1000,
                    results=result_samples,
                    relevant_docs=query.relevant_docs,
                    recall_at_10=query_recall * 100,
                    ndcg_at_10=query_ndcg * 100,
                ))

        avg_latency = statistics.mean(latencies) * 1000

        progress.update(main_task, advance=1)

        results.append(BenchmarkResult(
            engine=engine_name,
            build_time_s=build_time,
            index_loaded=index_loaded,
            avg_latency_ms=avg_latency,
            p50_latency_ms=statistics.median(latencies) * 1000,
            p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] * 1000,
            recall_at_10=statistics.mean(recall_scores) * 100,
            ndcg_at_10=statistics.mean(ndcg_scores) * 100,
            mrr=statistics.mean(mrr_scores) * 100,
            queries_per_sec=1000 / avg_latency if avg_latency > 0 else 0,
            batch_latency_ms=batch_latency_ms,
            batch_qps=batch_qps,
            sample_queries=sample_queries,
        ))

    progress.complete(main_task)

    return DatasetResult(
        dataset=dataset_name,
        corpus_size=len(corpus),
        num_queries=len(queries),
        timestamp=datetime.now().isoformat(),
        results=results,
    )


# =============================================================================
# Output Formatting
# =============================================================================

def format_results_table(result: DatasetResult) -> str:
    """Format results as a text table."""
    lines = []
    lines.append("=" * 110)
    lines.append(f"RESULTS: {result.dataset}")
    lines.append(f"Corpus: {result.corpus_size:,} documents | Queries: {result.num_queries}")
    lines.append("=" * 110)
    lines.append("")

    # Header
    header = (f"{'Engine':<18} {'Build (s)':<12} {'Single (ms)':<12} {'Batch (ms)':<12} "
              f"{'Recall@10':<10} {'NDCG@10':<10} {'QPS':<8} {'Batch QPS':<10}")
    lines.append(header)
    lines.append("-" * 110)

    for r in result.results:
        build_str = f"{r.build_time_s:.2f}" if not r.index_loaded else "(cached)"
        batch_str = f"{r.batch_latency_ms:.4f}" if r.batch_latency_ms else "-"
        batch_qps_str = f"{r.batch_qps:.0f}" if r.batch_qps else "-"
        lines.append(
            f"{r.engine:<18} {build_str:<12} {r.avg_latency_ms:<12.3f} {batch_str:<12} "
            f"{r.recall_at_10:<10.1f} {r.ndcg_at_10:<10.1f} {r.queries_per_sec:<8.0f} {batch_qps_str:<10}"
        )

    lines.append("")
    return "\n".join(lines)


def display_results_rich(result: DatasetResult):
    """Display results using rich tables."""
    if not RICH_AVAILABLE:
        print(format_results_table(result))
        return

    table = Table(
        title=f"{result.dataset} ({result.corpus_size:,} docs, {result.num_queries} queries)",
        box=box.ROUNDED,
    )

    table.add_column("Engine", style="cyan")
    table.add_column("Build (s)", justify="right")
    table.add_column("Single (ms)", justify="right")
    table.add_column("Batch (ms)", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("Batch QPS", justify="right")

    for r in result.results:
        build_str = f"{r.build_time_s:.2f}" if not r.index_loaded else "[dim](cached)[/dim]"
        batch_str = f"{r.batch_latency_ms:.4f}" if r.batch_latency_ms else "-"
        batch_qps_str = f"{r.batch_qps:.0f}" if r.batch_qps else "-"
        table.add_row(
            r.engine,
            build_str,
            f"{r.avg_latency_ms:.3f}",
            batch_str,
            f"{r.recall_at_10:.1f}%",
            f"{r.ndcg_at_10:.1f}%",
            f"{r.queries_per_sec:.0f}",
            batch_qps_str,
        )

    console.print(table)
    console.print()


def save_results(results: List[DatasetResult], output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark_run": datetime.now().isoformat(),
        "datasets": [asdict(r) for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def append_to_log(results: List[DatasetResult], log_path: Path):
    """Append formatted results to log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'a') as f:
        f.write(f"\n{'#' * 90}\n")
        f.write(f"# Benchmark Run: {datetime.now().isoformat()}\n")
        f.write(f"{'#' * 90}\n\n")

        for result in results:
            f.write(format_results_table(result))
            f.write("\n")


# =============================================================================
# Main
# =============================================================================

DATASET_CHOICES = [
    'beir-scifact', 'beir-nfcorpus',
    'wiki-100k', 'wiki-200k', 'wiki-500k', 'wiki-1m',
    'custom',
]

ENGINE_CHOICES = ['vajra', 'vajra-parallel', 'bm25s', 'bm25s-parallel', 'rank-bm25', 'tantivy', 'pyserini']


def main():
    parser = argparse.ArgumentParser(
        description="Vajra BM25 Unified Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --datasets wiki-200k
  python benchmark.py --datasets beir-scifact beir-nfcorpus --engines vajra bm25s
  python benchmark.py --corpus /path/to/data.jsonl
  python benchmark.py --datasets wiki-1m --no-cache  # Rebuild indexes
        """
    )

    parser.add_argument(
        '--datasets', nargs='+', choices=DATASET_CHOICES, default=['beir-scifact'],
        help="Datasets to benchmark (default: beir-scifact)"
    )
    parser.add_argument(
        '--corpus', type=Path,
        help="Path to custom JSONL corpus (use with --datasets custom)"
    )
    parser.add_argument(
        '--queries', type=Path,
        help="Path to custom queries JSONL (optional, for --datasets custom)"
    )
    parser.add_argument(
        '--engines', nargs='+', choices=ENGINE_CHOICES,
        default=['vajra', 'vajra-parallel', 'bm25s', 'bm25s-parallel'],
        help="Engines to benchmark"
    )
    parser.add_argument(
        '--output', type=Path, default=Path('results/benchmark_results.json'),
        help="Output JSON file path (default: results/benchmark_results.json)"
    )
    parser.add_argument(
        '--log', type=Path, default=Path('results/benchmark.log'),
        help="Output log file path (default: results/benchmark.log)"
    )
    parser.add_argument(
        '--cache-dir', type=Path,
        default=Path.home() / "Github" / "ir_benchmark_data",
        help="Directory for dataset cache"
    )
    parser.add_argument(
        '--index-cache-dir', type=Path, default=Path('.index_cache'),
        help="Directory for index cache (default: .index_cache)"
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help="Disable index caching (rebuild all indexes)"
    )
    parser.add_argument(
        '--clear-cache', action='store_true',
        help="Clear index cache before running"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Suppress progress display"
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help="Number of workers for parallel engines (default: 8)"
    )

    args = parser.parse_args()

    # Setup
    print_header("VAJRA BM25 BENCHMARK")

    # Index cache
    index_cache = None
    if not args.no_cache:
        index_cache = IndexCache(args.index_cache_dir)
        if args.clear_cache:
            print_status("Clearing index cache...")
            index_cache.clear()

    # Load and benchmark each dataset
    all_results = []

    with BenchmarkProgress(quiet=args.quiet) as progress:
        for dataset in args.datasets:
            try:
                print_status(f"\nLoading {dataset}...", style="bold")

                if dataset == 'custom':
                    if not args.corpus:
                        print_error("--corpus required with --datasets custom")
                        continue
                    corpus, queries = load_custom_corpus(args.corpus, args.queries)
                    dataset_name = args.corpus.stem

                elif dataset.startswith('beir-'):
                    beir_name = dataset.replace('beir-', '')
                    corpus, queries = load_beir_dataset(beir_name)
                    dataset_name = f"BEIR/{beir_name}"

                elif dataset.startswith('wiki-'):
                    max_docs = int(dataset.replace('wiki-', '').replace('k', '000').replace('m', '000000'))
                    corpus, queries = load_wikipedia(max_docs, args.cache_dir)
                    dataset_name = f"Wikipedia/{dataset.replace('wiki-', '').upper()}"

                else:
                    print_error(f"Unknown dataset: {dataset}")
                    continue

                print_status(f"  Loaded {len(corpus):,} documents, {len(queries)} queries")

                # Run benchmark
                result = run_benchmark(
                    dataset_name, corpus, queries,
                    args.engines, index_cache, progress,
                    workers=args.workers
                )
                all_results.append(result)

                # Display results
                if not args.quiet:
                    display_results_rich(result)

            except Exception as e:
                print_error(f"Failed to benchmark {dataset}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    if all_results:
        save_results(all_results, args.output)
        append_to_log(all_results, args.log)

        print_success(f"\nResults saved to: {args.output}")
        print_success(f"Log appended to: {args.log}")

        # Final summary
        if not args.quiet and RICH_AVAILABLE:
            console.print("\n[bold]Summary:[/bold]")
            for r in all_results:
                fastest = min(r.results, key=lambda x: x.avg_latency_ms)
                console.print(
                    f"  {r.dataset}: Best latency = {fastest.avg_latency_ms:.2f}ms ({fastest.engine})"
                )


if __name__ == "__main__":
    main()
