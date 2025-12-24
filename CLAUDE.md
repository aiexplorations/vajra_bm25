# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vajra BM25 is a high-performance BM25 search engine using Category Theory abstractions. It provides vectorized implementations with sparse matrix support, achieving up to 291x speedup over rank-bm25 at 100K documents.

**Package name:** `vajra-bm25` (on PyPI)
**Current version:** 0.2.1

## Development Commands

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test
pytest tests/test_bm25.py::test_vajra_search_query -v

# Run with coverage
pytest --cov=vajra_bm25 --cov-report=html

# Build and publish
pip install build twine
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine upload dist/*
```

## Benchmarking

The unified benchmark script outputs to `results/benchmark_results.json` and `results/benchmark.log` by default. It supports index caching to avoid expensive rebuilds.

```bash
# Install benchmark dependencies
pip install rank-bm25 bm25s beir rich tantivy

# Optional: Pyserini (requires Java 11+)
pip install pyserini

# Run BEIR benchmarks (small, fast)
python benchmarks/benchmark.py --datasets beir-scifact

# Run Wikipedia benchmarks (requires data files)
python benchmarks/benchmark.py --datasets wiki-200k wiki-500k

# Select specific engines
python benchmarks/benchmark.py --datasets wiki-200k --engines vajra bm25s tantivy

# All available engines
python benchmarks/benchmark.py --engines vajra vajra-parallel bm25s bm25s-parallel tantivy pyserini

# Rebuild indexes (ignore cache)
python benchmarks/benchmark.py --datasets wiki-200k --no-cache

# Clear index cache before running
python benchmarks/benchmark.py --datasets wiki-200k --clear-cache

# Custom corpus
python benchmarks/benchmark.py --datasets custom --corpus /path/to/data.jsonl

# Profiling
python benchmarks/profiler.py --mode index-build --dataset wiki-200k
python benchmarks/profiler.py --mode query-latency --dataset wiki-200k
python benchmarks/profiler.py --mode comparison --dataset wiki-100k
```

### Available Engines

| Engine | Description | Notes |
|--------|-------------|-------|
| `vajra` | Vajra single-threaded | Default, fastest single queries |
| `vajra-parallel` | Vajra with thread pool | Best for batch queries |
| `bm25s` | BM25S single-threaded | Fast Python BM25 |
| `bm25s-parallel` | BM25S with threading | Native batch support |
| `tantivy` | Tantivy (Rust) | In-memory index, no caching |
| `pyserini` | Lucene via Pyserini | Requires Java 11+ |
| `rank-bm25` | rank-bm25 baseline | Pure Python, slow |

### Benchmark Outputs

- `results/benchmark_results.json` - Structured JSON with all metrics
- `results/benchmark.log` - Human-readable log (appended each run)
- `.index_cache/` - Cached indexes (speeds up subsequent runs)

## Architecture

### Search Engine Hierarchy

Three search engine classes with increasing performance characteristics:

1. **VajraSearch** (`search.py`) - Base implementation using pure categorical abstractions
2. **VajraSearchOptimized** (`optimized.py`) - Vectorized with sparse matrices, NumPy/SciPy operations
3. **VajraSearchParallel** (`parallel.py`) - Thread-parallel batch processing

### Scoring Strategy Priority (VajraSearchOptimized)

The optimized engine selects scorers in this order:
1. **Eager scorer** - Pre-computed BM25 scores at index time (fastest)
2. **Numba JIT scorer** - Compiled scoring loops (if numba available)
3. **MaxScore algorithm** - Coalgebraic early termination
4. **Traditional NumPy/SciPy** - Vectorized fallback

### Key Modules

- `documents.py` - Document, DocumentCorpus (JSONL persistence)
- `text_processing.py` - Tokenization, stop word removal, preprocessing
- `inverted_index.py` - Base inverted index for VajraSearch
- `optimized.py` - Sparse matrix indices, vectorized scorers, LRU caching
- `categorical/` - Category theory primitives (Morphism, Functor, Coalgebra)

### Category Theory Mapping

| Concept | Implementation |
|---------|----------------|
| Morphism | BM25 scoring: `(Query, Document) → ℝ` |
| Coalgebra | Search unfolding: `QueryState → List[SearchResult]` |
| Functor | List functor for multiple-results semantics |
| Comonad | LRU caching with extract/duplicate |

## Optional Dependencies

```bash
pip install vajra-bm25[optimized]    # numpy, scipy
pip install vajra-bm25[persistence]  # joblib for index save/load
pip install vajra-bm25[all]          # all optional deps
```

Numba provides additional JIT compilation speedup but is not included in package extras - install separately if needed.

## Index Building Optimizations

VectorizedIndexSparse uses:
- Parallel tokenization via multiprocessing
- COO matrix construction (3-5x faster than LIL format)
- Pre-computed term bounds for MaxScore algorithm
- Optional eager score matrix (BM25S approach)

## Benchmark Data

Large-scale benchmark datasets are stored in a separate repo:
```
/Users/rajesh/Github/ir_benchmark_data/
└── wikipedia/
    ├── wikipedia_200000.jsonl   (1.7 GB, 200K docs)
    ├── wikipedia_500000.jsonl   (3.3 GB, 500K docs)
    └── wikipedia_1000000.jsonl  (5.3 GB, 1M docs)
```

## Testing Notes

- Tests use pytest fixtures for sample documents
- Larger corpus fixture (100+ docs) used for sparse matrix testing
- Eager scoring tests verify equivalence with traditional scoring
- All tests should pass with just `pytest tests/ -v`
