# Vajra BM25

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vajra** (Sanskrit: वज्र, "thunderbolt") is a high-performance BM25 search engine. It uses Category Theory abstractions to reframe the BM25 algorithm, providing a well-structured API with vectorized implementations. Benchmarks show Vajra is **faster than BM25S** (one of the fastest Python BM25 libraries) at larger corpus sizes while achieving **better recall** on certain datasets.

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

# Everything
pip install vajra-bm25[all]
```

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

Benchmarked on synthetic corpora with 10 queries per run, comparing against rank-bm25, BM25S, and BM25S parallel:

### Speed Comparison

| Corpus Size | rank-bm25 (ms) | Vajra Optimized (ms) | Vajra Parallel (ms) | BM25S (ms) | BM25S parallel (ms) |
| ----------- | -------------- | -------------------- | ------------------- | ---------- | ------------------- |
| 1,000       | 0.63           | 0.04                 | 0.02                | 0.18       | 0.28                |
| 10,000      | 9.11           | 0.13                 | 0.08                | 0.32       | 0.37                |
| 50,000      | 47.14          | 0.47                 | 0.39                | 0.70       | 0.73                |
| 100,000     | 102.19         | 0.44                 | 0.35                | 0.85       | 0.89                |

### Speedup vs rank-bm25

| Corpus Size | Vajra Optimized | Vajra Parallel | BM25S | BM25S parallel |
| ----------- | --------------- | -------------- | ----- | -------------- |
| 1,000       | 17x             | 30x            | 4x    | 2x             |
| 10,000      | 69x             | 119x           | 28x   | 25x            |
| 50,000      | 101x            | 122x           | 68x   | 64x            |
| 100,000     | 230x            | **291x**       | 120x  | 114x           |

### Recall@10 (vs rank-bm25 baseline)

| Corpus Size | Vajra Optimized | Vajra Parallel | BM25S | BM25S parallel |
| ----------- | --------------- | -------------- | ----- | -------------- |
| 1,000       | 99%             | 99%            | 98%   | 98%            |
| 10,000      | 56%             | 56%            | 56%   | 56%            |
| 50,000      | **80%**         | **80%**        | 56%   | 56%            |
| 100,000     | 50%             | 50%            | 50%   | 50%            |

**Key observations:**

- Sub-millisecond query latency at all corpus sizes
- Up to **291x speedup** over rank-bm25 at 100K documents with Vajra Parallel
- Vajra is **faster than both BM25S variants** at all corpus sizes
- **BM25S parallel is slower than single-threaded** for single queries (parallelism overhead)
- Vajra achieves **better recall** at 50K docs (80% vs 56%)
- Recall varies by corpus characteristics (vocabulary overlap, document length distribution)

### Standard IR Benchmarks (BEIR)

Validated on standard information retrieval datasets from the [BEIR benchmark suite](https://github.com/beir-cellar/beir):

| Dataset  | Docs  | Queries | Vajra NDCG@10 | rank-bm25 NDCG@10 | Vajra Speedup    | BM25S NDCG@10  |
| -------- | ----- | ------- | ------------- | ----------------- | ---------------- | -------------- |
| SciFact  | 5,183 | 300     | **67.0%**     | 66.7%             | **49x** (0.18ms) | 66.2% (0.16ms) |
| NFCorpus | 3,633 | 323     | **30.9%**     | 30.9%             | **33x** (0.06ms) | 30.7% (0.14ms) |

With 8 workers, Vajra Parallel achieves **equal or better retrieval quality** than rank-bm25 while being **33-49x faster**. Vajra also outperforms BM25S on **both speed and accuracy** on NFCorpus.

Vajra achieves these speedups through structural optimizations:

1. **Sparse Matrices** (Primary driver at scale): Avoids computation on ~99% zeros in the term-document matrix.
2. **Vectorized NumPy** (10-50x speedup): Uses SIMD instructions for batch scoring candidates.
3. **Partial Sort** (O(n) average): `np.argpartition` avoids sorting 100,000 documents when only 10 are needed.
4. **LRU Caching**: Caches both preprocessing results and full query/top-k pairs.
5. **Thread Parallelism**: Concurrent query execution with `VajraSearchParallel`.

Vajra achieves these speedups through:

- Vectorized NumPy operations
- Pre-computed IDF values
- Sparse matrix representation
- LRU query caching
- Partial sort for top-k
- Thread pool parallelism (VajraSearchParallel)

For detailed benchmark methodology and results, see [docs/benchmarks.md](docs/benchmarks.md).

### Running Benchmarks

To reproduce the benchmark results:

```bash
# Install benchmark dependencies
pip install vajra-bm25[optimized] rank-bm25 bm25s

# Run synthetic corpus benchmarks
python benchmarks/benchmark.py

# Run BEIR standard dataset benchmarks
pip install beir ir-datasets
python benchmarks/benchmark_standard_datasets.py
```

The synthetic benchmarks test on corpora of 1K, 10K, 50K, and 100K documents. The BEIR benchmarks validate on standard IR datasets (SciFact, NFCorpus).

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

Category theory provides:

1. **Unified abstractions** - Same `Coalgebra.structure_map()` interface for graph search and document retrieval
2. **Explicit type signatures** - `BM25: (Query, Document) → ℝ` makes inputs/outputs clear
3. **Composable pipelines** - `preprocess >> score >> rank` as morphism composition

What it doesn't provide:

- Performance improvements (those come from NumPy/sparse matrices)
- Novel algorithms (BM25 is BM25)
- Runtime machinery (it's just well-organized code)

The honest summary: **category theory is a design vocabulary, not a runtime mechanism**.

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
