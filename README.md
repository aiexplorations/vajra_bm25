# Vajra BM25

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vajra** (Sanskrit: वज्र, "thunderbolt") is a high-performance BM25 search engine. It uses Category Theory abstractions to reframe the BM25 algorithm, providing a well-structured API with vectorized implementations. Benchmarks show Vajra is **~1.2-1.3x faster than BM25S** (one of the fastest Python BM25 libraries) while achieving **competitive recall** across datasets.

## What Makes Vajra Different

Vajra implements the standard BM25 ranking algorithm using rigorous mathematical abstractions:

- **Morphisms**: BM25 scoring as a mathematical arrow `(Query, Document) → ℝ`
- **Coalgebras**: Search as state unfolding `QueryState → List[SearchResult]`
- **Functors**: The List functor captures multiple-results semantics

While Vajra BM25 uses the same underlying mathematics of BM25, it uses different vocabulary to describe the search process, and the abstractions are more amenable to experimentation and improvement. The core BM25 formula is identical to other implementations—category theory provides the organizational structure.

## Installation

```bash
# Basic installation (zero dependencies)
pip install vajra-bm25

# With optimizations (NumPy + SciPy for vectorized operations)
pip install vajra-bm25[optimized]

# With index persistence (save/load indices)
pip install vajra-bm25[persistence]

# With CLI (interactive search)
pip install vajra-bm25[cli]

# Everything
pip install vajra-bm25[all]
```

## Interactive CLI

Vajra includes a rich interactive CLI for exploring search:

```bash
# Interactive mode with BEIR SciFact dataset
vajra-search

# Single query mode
vajra-search -q "machine learning algorithms"

# Use custom corpus
vajra-search --corpus my_documents.jsonl

# Show options
vajra-search --help
```

In interactive mode, use `:help` for commands, `:stats` for index info, and `:quit` to exit.

## Quick Start

The Python API for using Vajra BM25 is quite straightforward, and there's currently support for using JSONL document corpuses via the `DocumentCorpus` class.

```python
from vajra_bm25 import VajraSearch, Document, DocumentCorpus

# Create documents
documents = [
    Document(id="1", title="Category Theory", content="Functors preserve structure"),
    Document(id="2", title="Coalgebras", content="Coalgebras model dynamics"),
    Document(id="3", title="Search Algorithms", content="BFS explores level by level"),
]
corpus = DocumentCorpus(documents)

# Create search engine
engine = VajraSearch(corpus)

# Search
results = engine.search("category functors", top_k=5)

for r in results:
    print(f"{r.rank}. {r.document.title} (score: {r.score:.3f})")
```

## Optimized Usage

For larger corpora (1000+ documents), use the optimized version. This optimized version is much faster.

```python
from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

# Load corpus from JSONL
corpus = DocumentCorpus.load_jsonl("corpus.jsonl")

# Create optimized engine
# Automatically uses sparse matrices for >10K documents
engine = VajraSearchOptimized(corpus)

# Search (vectorized, cached)
results = engine.search("neural networks", top_k=10)
```

## Parallel Batch Processing

For high-throughput scenarios, use the parallel batch processing version. This version is much faster and able to return results for multiple queries in parallel. There's obviously the overhead due to parallelism, which may work against the search algorithm, but in cases where we have memory limitations, this may work better than Vajra Search Optimized.

```python
from vajra_bm25 import VajraSearchParallel

engine = VajraSearchParallel(corpus, max_workers=4)

# Process multiple queries in parallel
queries = ["machine learning", "deep learning", "neural networks"]
batch_results = engine.search_batch(queries, top_k=5)
```

## Performance

Benchmarked against BM25 implementations across BEIR and Wikipedia datasets (January 2025):

### BEIR/SciFact (5,183 docs, 300 queries)

| Engine | Latency | Recall@10 | NDCG@10 | QPS |
|--------|---------|-----------|---------|-----|
| **vajra** | **0.14ms** | 78.9% | 67.0% | **7,133** |
| bm25s | 0.18ms | 77.4% | 66.2% | 5,512 |
| tantivy | 0.28ms | 72.5% | 60.0% | 3,539 |

### Wikipedia/200K (200,000 docs, 500 queries)

| Engine | Latency | Recall@10 | NDCG@10 | QPS |
|--------|---------|-----------|---------|-----|
| **vajra** | **0.89ms** | 44.4% | 35.1% | **1,125** |
| bm25s | 1.08ms | 44.6% | 35.2% | 925 |
| tantivy | 6.68ms | 45.6% | 36.4% | 150 |

### Wikipedia/500K (500,000 docs, 500 queries)

| Engine | Latency | Recall@10 | NDCG@10 | QPS |
|--------|---------|-----------|---------|-----|
| **vajra** | **1.89ms** | 49.6% | 36.7% | **529** |
| bm25s | 2.45ms | 49.8% | 37.1% | 409 |
| tantivy | 5.52ms | 51.6% | 38.3% | 181 |

### Wikipedia/1M (1,000,000 docs, 500 queries)

| Engine | Build Time | Latency | Recall@10 | NDCG@10 | QPS |
|--------|------------|---------|-----------|---------|-----|
| **vajra** | 17.0 min | **3.40ms** | 45.6% | 36.3% | **294** |
| bm25s | 11.3 min | 5.44ms | 45.8% | 36.7% | 184 |

**Key observations:**

- Vajra is **~1.3-1.6x faster** than BM25S on single queries across corpus sizes
- Sub-4ms latency even at 1M documents
- Competitive accuracy: within 1% NDCG of BM25S
- Optimized index building with per-document array concatenation

### Caching for Production Workloads

For production systems with repeated queries, Vajra includes built-in LRU caching:

| Scenario | Typical Latency | Explanation |
|----------|-----------------|-------------|
| **First query (cold)** | 0.14-3.5ms | Full BM25 computation (scales with corpus) |
| **Repeated query (cached)** | ~0.001ms | LRU cache hit, near-instant |

Enable caching by setting `cache_size` (default: 1000):

```python
engine = VajraSearchOptimized(corpus, cache_size=1000)
```

### How Vajra Achieves Performance

1. **Eager Score Matrix**: Pre-computes BM25 term-document scores at index time
2. **Sparse Matrices**: Avoids computation on ~99% zeros in the term-document matrix
3. **Vectorized NumPy/SciPy**: Uses SIMD instructions for batch scoring
4. **Numba JIT**: Optional compiled scoring loops for additional speedup
5. **LRU Caching**: Optional query result caching for repeated queries

For detailed benchmark methodology and results, see [docs/benchmarks.md](docs/benchmarks.md).

### Running Benchmarks

The benchmark system includes progress display, automatic file output, and index caching to avoid expensive rebuilds.

```bash
# Install benchmark dependencies
pip install vajra-bm25[optimized] rank-bm25 bm25s beir rich tantivy

# Optional: Pyserini (requires Java 11+)
pip install pyserini

# Quick start: Run BEIR SciFact benchmark (small dataset, ~5K docs)
python benchmarks/benchmark.py --datasets beir-scifact

# Run Wikipedia benchmarks (requires downloading data first)
python benchmarks/download_wikipedia.py --max-docs 200000
python benchmarks/benchmark.py --datasets wiki-200k

# Run multiple datasets
python benchmarks/benchmark.py --datasets beir-scifact beir-nfcorpus wiki-200k wiki-500k

# All engines comparison
python benchmarks/benchmark.py --datasets beir-scifact \
    --engines vajra vajra-parallel bm25s tantivy pyserini

# Force rebuild indexes (skip cache)
python benchmarks/benchmark.py --datasets wiki-200k --no-cache
```

**Output files:**
- `results/benchmark_results.json` - Structured JSON with detailed metrics
- `results/benchmark.log` - Human-readable log (appended each run)
- `.index_cache/` - Cached indexes for faster subsequent runs

**Available datasets:** `beir-scifact`, `beir-nfcorpus`, `wiki-100k`, `wiki-200k`, `wiki-500k`, `wiki-1m`, `custom`

**Available engines:** `vajra`, `vajra-parallel`, `bm25s`, `bm25s-parallel`, `tantivy`, `pyserini`, `rank-bm25`

> **Note:** Pyserini requires Java 11+ installed. On macOS: `brew install openjdk@21`

## JSONL Format

Vajra uses JSONL for corpus persistence:

```jsonl
{"id": "doc1", "title": "First Document", "content": "Content here"}
{"id": "doc2", "title": "Second Document", "content": "More content"}
```

Load and save:

```python
# Save
corpus.save_jsonl("corpus.jsonl")

# Load
corpus = DocumentCorpus.load_jsonl("corpus.jsonl")
```

## BM25 Parameters

```python
from vajra_bm25 import VajraSearch, BM25Parameters

# Custom BM25 parameters
params = BM25Parameters(
    k1=1.5,  # Term frequency saturation (default: 1.5)
    b=0.75   # Length normalization (default: 0.75)
)

engine = VajraSearch(corpus, params=params)
```

## Categorical Abstractions (Advanced)

For users interested in the category theory foundations:

```python
from vajra_bm25 import (
    Morphism, FunctionMorphism, IdentityMorphism,
    Coalgebra, SearchCoalgebra,
    Functor, ListFunctor,
)

# Morphism composition
f = FunctionMorphism(lambda x: x + 1)
g = FunctionMorphism(lambda x: x * 2)
h = f >> g  # h(x) = (x + 1) * 2

# Identity laws
identity = IdentityMorphism()
assert (f >> identity).apply(5) == f.apply(5)  # f . id = f
assert (identity >> f).apply(5) == f.apply(5)  # id . f = f
```

There's a better, more rigorous treatment of the concepts of Category Theory by Bartosz Milewski [here](https://www.youtube.com/watch?v=I8LbkfSSR58&list=PLbgaMIhjbmEnaH_LTkxLI7FMa2HsnawM_).

## API Reference

### Core Classes

- `Document(id, title, content, metadata=None)` - Immutable document
- `DocumentCorpus(documents)` - Collection of documents
- `VajraSearch(corpus, params=None)` - Base search engine
- `VajraSearchOptimized(corpus, k1=1.5, b=0.75)` - Vectorized search
- `VajraSearchParallel(corpus, max_workers=4)` - Parallel batch search

### Search Results

```python
@dataclass
class SearchResult:
    document: Document  # The matched document
    score: float        # BM25 relevance score
    rank: int           # Position in results (1-indexed)
```

## Why Category Theory?

Category theory did not make Vajra fast. The speed comes from NumPy, sparse matrices, and caching. But it gave the code a clean organizational structure.

### How the Abstractions Help

The `categorical/` module provides base classes that derived implementations extend:

- **`Morphism[A, B]`**: Composable transformations with `apply()` and `>>` operator for chaining (`preprocess >> tokenize >> score`)
- **`Coalgebra[X, FX]`**: The "unfolding" pattern via `structure_map()`. BM25 search extends this as `QueryState → List[SearchResult]`
- **`Functor[A, FA]`**: Lifts operations to containers. `ListFunctor` handles multiple results; `MaybeFunctor` handles failure

This creates clear separation: data types, transformations, and search dynamics each have their place. Adding a new scorer or search strategy means implementing a known interface.

### Where Speed Actually Comes From

| Technique | Impact |
|-----------|--------|
| Sparse matrices (CSR) | 6-8x memory reduction |
| Vectorized NumPy | 10-50x speedup |
| Eager pre-computed scores | Sub-millisecond queries |
| LRU query caching | Near-instant repeated queries |

These are engineering techniques. The categorical structure organizes the code; NumPy and SciPy deliver the performance.

## Development

```bash
# Clone repository
git clone https://github.com/aiexplorations/vajra_bm25.git
cd vajra_bm25

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=vajra_bm25 --cov-report=html
```

## Publishing to PyPI

To build and publish a new version of Vajra BM25:

1. **Install build tools**:

   ```bash
   pip install build twine
   ```

2. **Clean previous builds**:

   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

3. **Build the package**:

   ```bash
   python -m build
   ```

   This generates a `.whl` and a `.tar.gz` in the `dist/` directory.

4. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- BM25 algorithm: Robertson & Zaragoza, "The Probabilistic Relevance Framework"
- Category theory foundations: Rutten, "Universal Coalgebra: A Theory of Systems"
- Built and explored in the [State Dynamic Modeling](https://github.com/aiexplorations/state_dynamic_modeling) project
- Inspired by the Category Theory lectures by [Bartosz Milewski](https://bartoszmilewski.com/) which are [here on YouTube](https://www.youtube.com/watch?v=I8LbkfSSR58&list=PLbgaMIhjbmEnaH_LTkxLI7FMa2HsnawM_).
