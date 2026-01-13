# Search Engines

Vajra provides three search engine implementations for different use cases.

## Engine Comparison

| Engine | Best For | Features |
|--------|----------|----------|
| `VajraSearch` | Small corpora (<1K docs) | Pure Python, educational |
| `VajraSearchOptimized` | Production use | Sparse matrices, caching, fastest |
| `VajraSearchParallel` | Batch processing | Multi-threaded, high throughput |

## VajraSearch (Base)

The reference implementation using pure categorical abstractions.

```python
from vajra_bm25 import VajraSearch, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("small_corpus.jsonl")
engine = VajraSearch(corpus)

results = engine.search("query terms", top_k=10)
```

**Characteristics:**

- Pure Python implementation
- Clear categorical structure
- Good for learning and small datasets
- No external dependencies

## VajraSearchOptimized (Recommended)

Production-ready engine with vectorized operations.

```python
from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("large_corpus.jsonl")

engine = VajraSearchOptimized(
    corpus,
    k1=1.5,           # BM25 k1 parameter
    b=0.75,           # BM25 b parameter
    cache_size=1000,  # LRU cache for repeated queries
    use_eager=True,   # Pre-compute score matrix
)

results = engine.search("query terms", top_k=10)
```

**Characteristics:**

- Sparse matrix operations (CSR format)
- NumPy/SciPy vectorization
- Optional Numba JIT compilation
- LRU query caching
- Eager score pre-computation

### Automatic Mode Selection

The engine automatically selects the best mode:

- **< 10K documents**: Dense matrices
- **≥ 10K documents**: Sparse matrices (99%+ memory savings)

### Configuration Options

```python
engine = VajraSearchOptimized(
    corpus,
    # BM25 Parameters
    k1=1.5,              # Term frequency saturation (default: 1.5)
    b=0.75,              # Length normalization (default: 0.75)

    # Performance Options
    use_sparse=True,     # Force sparse matrices (auto-detected)
    use_eager=True,      # Pre-compute BM25 scores at index time
    cache_size=1000,     # Query result cache size (0 to disable)
)
```

## VajraSearchParallel

For high-throughput batch processing.

```python
from vajra_bm25 import VajraSearchParallel, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("corpus.jsonl")

engine = VajraSearchParallel(
    corpus,
    max_workers=4,  # Number of parallel workers
)

# Process multiple queries in parallel
queries = ["machine learning", "deep learning", "neural networks"]
batch_results = engine.search_batch(queries, top_k=10)

for query, results in zip(queries, batch_results):
    print(f"Query: {query}")
    for r in results:
        print(f"  {r.rank}. {r.document.title}")
```

**Characteristics:**

- Thread-based parallelism
- Optimized for batch queries
- Higher throughput, slightly higher latency per query
- Memory overhead from worker threads

## Choosing the Right Engine

```
┌─────────────────────────────────────────────────────────────┐
│                    Which Engine?                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Documents < 1,000?                                          │
│      └── Yes → VajraSearch (simple, educational)             │
│      └── No  ↓                                               │
│                                                              │
│  Need batch processing?                                      │
│      └── Yes → VajraSearchParallel                           │
│      └── No  ↓                                               │
│                                                              │
│  Otherwise → VajraSearchOptimized (recommended)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Comparison

At 500K documents:

| Engine | Single Query | Batch (100 queries) |
|--------|--------------|---------------------|
| VajraSearchOptimized | 1.89ms | 189ms |
| VajraSearchParallel (4 workers) | 2.5ms | 95ms |

Use `VajraSearchOptimized` for single queries, `VajraSearchParallel` for batch workloads.
