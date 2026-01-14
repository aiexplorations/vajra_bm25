# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vajra BM25 is a high-performance BM25 search engine using Category Theory abstractions. It provides vectorized implementations with sparse matrix support, achieving up to 291x speedup over rank-bm25 at 100K documents.

**Package name:** `vajra-bm25` (on PyPI)
**Current version:** 0.4.0

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

Five search engine classes with increasing performance characteristics:

1. **VajraSearch** (`search.py`) - Base BM25 implementation using pure categorical abstractions
2. **VajraSearchOptimized** (`optimized.py`) - Vectorized BM25 with sparse matrices, NumPy/SciPy operations
3. **VajraSearchParallel** (`parallel.py`) - Thread-parallel BM25 batch processing
4. **VajraVectorSearch** (`vector/search.py`) - Semantic vector search with HNSW/Flat indices
5. **HybridSearchEngine** (`vector/hybrid.py`) - BM25 + Vector fusion with RRF/Linear/RSF methods

### Scoring Strategy Priority (VajraSearchOptimized)

The optimized engine selects scorers in this order:
1. **Eager scorer** - Pre-computed BM25 scores at index time (fastest)
2. **Numba JIT scorer** - Compiled scoring loops (if numba available)
3. **MaxScore algorithm** - Coalgebraic early termination
4. **Traditional NumPy/SciPy** - Vectorized fallback

### Key Modules

#### BM25 Search
- `documents.py` - Document, DocumentCorpus (JSONL/PDF persistence)
- `text_processing.py` - Tokenization, stop word removal, preprocessing
- `inverted_index.py` - Base inverted index for VajraSearch
- `optimized.py` - Sparse matrix indices, vectorized scorers, LRU caching
- `parallel.py` - Thread-parallel batch processing
- `cli.py` - Interactive CLI with BM25/Vector/Hybrid modes

#### Vector Search (v0.4.0+)
- `vector/embeddings.py` - EmbeddingMorphism, TextEmbeddingMorphism (sentence-transformers)
- `vector/scorer.py` - SimilarityMorphism (Cosine, L2, InnerProduct)
- `vector/index_flat.py` - FlatVectorIndex (exact brute-force search)
- `vector/hnsw/` - NativeHNSWIndex, HNSWNavigationCoalgebra (approximate search)
- `vector/search.py` - VajraVectorSearch engine
- `vector/hybrid.py` - HybridSearchEngine (BM25 + Vector fusion)

#### Categorical Abstractions
- `categorical/` - Category theory primitives (Morphism, Functor, Coalgebra)

### Category Theory Mapping

| Concept | BM25 Implementation | Vector Implementation |
|---------|-------------------|----------------------|
| Morphism | BM25 scoring: `(Query, Document) → ℝ` | Embeddings: `Text → ℝ^d`<br>Similarity: `(ℝ^d, ℝ^d) → ℝ` |
| Coalgebra | Search unfolding: `QueryState → List[SearchResult]` | HNSW navigation: `HNSWSearchState → List[Candidate]` |
| Functor | List functor for multiple-results semantics | List functor for k-NN results |
| Comonad | LRU caching with extract/duplicate | Query result caching |

## Optional Dependencies

```bash
# BM25 optimizations
pip install vajra-bm25[optimized]    # numpy, scipy
pip install vajra-bm25[persistence]  # joblib for index save/load
pip install vajra-bm25[pdf]          # pypdf for PDF indexing

# Vector search (v0.4.0+)
pip install vajra-bm25[vector]       # numpy, sentence-transformers
pip install vajra-bm25[vector-numba] # adds numba for distance acceleration

# CLI and benchmarking
pip install vajra-bm25[cli]          # rich, beir for interactive CLI
pip install vajra-bm25[benchmark]    # rank-bm25, bm25s, tantivy, beir

# Everything
pip install vajra-bm25[all]          # all core optional deps (BM25 + Vector)
```

### Numba Acceleration
Numba provides JIT compilation speedup for both BM25 scoring and vector distance computation. Install separately or use `[vector-numba]` extra.

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

## CLI Usage

The interactive CLI supports BM25, vector, and hybrid search modes:

```bash
# BM25 search (default)
vajra-search
vajra-search -q "machine learning"
vajra-search --corpus my_docs.jsonl

# Vector search (requires: pip install vajra-bm25[vector])
vajra-search --mode vector
vajra-search --mode vector --model all-MiniLM-L6-v2
vajra-search --mode vector -q "semantic query"

# Hybrid search (BM25 + Vector fusion)
vajra-search --mode hybrid
vajra-search --mode hybrid --alpha 0.5  # 50% BM25, 50% vector
vajra-search --mode hybrid --alpha 0.7  # 70% BM25, 30% vector

# PDF support
vajra-search --corpus document.pdf
vajra-search --corpus ./pdf_folder/

# Options
vajra-search --help
vajra-search --stats  # Show index statistics
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-q, --query` | Single query mode (non-interactive) | Interactive mode |
| `-m, --mode` | Search mode: `bm25`, `vector`, `hybrid` | `bm25` |
| `--model` | Embedding model for vector/hybrid | `all-MiniLM-L6-v2` |
| `--alpha` | BM25 weight for hybrid (0-1) | `0.5` |
| `-c, --corpus` | Path to JSONL/PDF/directory | BEIR SciFact |
| `-d, --dataset` | BEIR dataset: `beir-scifact`, `beir-nfcorpus` | `beir-scifact` |
| `-k, --top-k` | Number of results | `10` |
| `--stats` | Show index stats and exit | - |

## Testing Notes

- Tests use pytest fixtures for sample documents
- Larger corpus fixture (100+ docs) used for sparse matrix testing
- Eager scoring tests verify equivalence with traditional scoring
- Vector search tests cover embeddings, HNSW, flat index, hybrid fusion
- Index persistence tests verify save/load roundtrip (BM25 + eager scorer)
- Test suite: 236 tests passing (including 40 vector search tests)
- All tests should pass with just `pytest tests/ -v`

## Bug Fixes (v0.4.0)

### Eager Scorer Persistence Fix
- Fixed `save_index`/`load_index` to properly save and restore all scorer attributes
- `save_index` now serializes: k1, b, use_maxscore, use_numba, use_eager
- `load_index` now correctly recreates eager_scorer, numba_scorer, maxscore_scorer
- Added comprehensive roundtrip tests for all index modes (traditional, eager, maxscore, numba)
