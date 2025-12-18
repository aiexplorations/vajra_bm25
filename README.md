# Vajra BM25

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Vajra** (Sanskrit: वज्र, "thunderbolt/diamond") is a BM25 search engine. It uses Category Theory abstractions to reframe the BM25 algorithm. It provides a well-structured API and compares favourably with BM25S, which is one of the fastest Python implementations of BM25, specifically favouring better recall while remaining fast because of the vectorized implementations.

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

At 100,000 documents:

| Implementation    | Query Latency | Recall@10 |
| ----------------- | ------------- | --------- |
| rank-bm25         | 133.54 ms     | baseline  |
| Vajra (base)      | 59.14 ms      | 65.0%     |
| Vajra (optimized) | 1.39 ms       | 66.5%     |

Vajra (optimized) achieves **96x speedup** over rank-bm25 through:

- Vectorized NumPy operations
- Pre-computed IDF values
- Sparse matrix representation
- LRU query caching
- Partial sort for top-k

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

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- BM25 algorithm: Robertson & Zaragoza, "The Probabilistic Relevance Framework"
- Category theory foundations: Rutten, "Universal Coalgebra: A Theory of Systems"
- Built and explored in the [State Dynamic Modeling](https://github.com/aiexplorations/state_dynamic_modeling) project
- Inspired by the Category Theory lectures by [Bartosz Milewski](https://bartoszmilewski.com/) which are [here on YouTube](https://www.youtube.com/watch?v=I8LbkfSSR58&list=PLbgaMIhjbmEnaH_LTkxLI7FMa2HsnawM_).
