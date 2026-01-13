# Search Engines API

Vajra provides three search engine implementations.

## Common Interface

All search engines share a common interface:

```python
# Build index
engine = SearchEngine(corpus, **options)

# Single query
results = engine.search(query, top_k=10)

# Batch queries (if supported)
batch_results = engine.search_batch(queries, top_k=10)
```

## SearchResult

Search results are returned as `SearchResult` objects:

```python
@dataclass
class SearchResult:
    document: Document    # The matched document
    score: float          # BM25 relevance score
    rank: int             # Position in results (1-indexed)
```

### Accessing Results

```python
results = engine.search("machine learning", top_k=5)

for result in results:
    print(f"{result.rank}. {result.document.title}")
    print(f"   Score: {result.score:.4f}")
    print(f"   ID: {result.document.id}")
```

## VajraSearch

Base implementation using categorical abstractions.

```python
from vajra_bm25 import VajraSearch, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("corpus.jsonl")
engine = VajraSearch(corpus)

results = engine.search("neural networks", top_k=10)
```

### Constructor

```python
VajraSearch(
    corpus: DocumentCorpus,
    params: Optional[BM25Parameters] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `DocumentCorpus` | required | Document corpus to index |
| `params` | `BM25Parameters` | `None` | BM25 parameters (k1, b) |

### Using BM25Parameters

```python
from vajra_bm25 import VajraSearch, BM25Parameters

params = BM25Parameters(k1=1.2, b=0.75)
engine = VajraSearch(corpus, params=params)
```

## VajraSearchOptimized

Production-ready engine with vectorized operations.

```python
from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("large_corpus.jsonl")

engine = VajraSearchOptimized(
    corpus,
    k1=1.5,
    b=0.75,
    cache_size=1000,
    use_eager=True
)

results = engine.search("deep learning", top_k=10)
```

### Constructor

```python
VajraSearchOptimized(
    corpus: DocumentCorpus,
    k1: float = 1.5,
    b: float = 0.75,
    use_sparse: Optional[bool] = None,
    use_eager: bool = False,
    cache_size: int = 0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `DocumentCorpus` | required | Document corpus to index |
| `k1` | `float` | `1.5` | Term frequency saturation |
| `b` | `float` | `0.75` | Length normalization |
| `use_sparse` | `bool` | `None` | Force sparse mode (auto-detected) |
| `use_eager` | `bool` | `False` | Pre-compute BM25 scores |
| `cache_size` | `int` | `0` | LRU cache size (0 disables) |

### Methods

#### `search(query: str, top_k: int = 10) -> List[SearchResult]`

Search for documents matching the query.

```python
results = engine.search("transformer architecture", top_k=5)
```

#### `search_batch(queries: List[str], top_k: int = 10) -> List[List[SearchResult]]`

Process multiple queries.

```python
queries = ["machine learning", "neural networks", "deep learning"]
all_results = engine.search_batch(queries, top_k=10)

for query, results in zip(queries, all_results):
    print(f"Query: {query} -> {len(results)} results")
```

#### `save(filepath: Path)`

Save index to disk (requires `vajra-bm25[persistence]`).

```python
engine.save("index.pkl")
```

#### `VajraSearchOptimized.load(filepath: Path) -> VajraSearchOptimized`

Load index from disk.

```python
engine = VajraSearchOptimized.load("index.pkl")
```

### Automatic Mode Selection

The engine automatically chooses optimal settings:

| Corpus Size | Matrix Format | Reason |
|-------------|---------------|--------|
| < 10K docs | Dense | Faster for small corpora |
| ≥ 10K docs | Sparse (CSR) | 99%+ memory savings |

### Scoring Priority

When searching, the engine uses scorers in this order:

1. **Eager scorer** - Pre-computed scores (fastest)
2. **Numba JIT** - Compiled scoring (if available)
3. **MaxScore** - Early termination algorithm
4. **NumPy/SciPy** - Vectorized fallback

## VajraSearchParallel

Thread-parallel engine for batch processing.

```python
from vajra_bm25 import VajraSearchParallel, DocumentCorpus

corpus = DocumentCorpus.load_jsonl("corpus.jsonl")

engine = VajraSearchParallel(
    corpus,
    max_workers=4
)

# Efficient batch processing
queries = ["query1", "query2", "query3", ...]
results = engine.search_batch(queries, top_k=10)
```

### Constructor

```python
VajraSearchParallel(
    corpus: DocumentCorpus,
    max_workers: int = 4,
    **kwargs
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `DocumentCorpus` | required | Document corpus |
| `max_workers` | `int` | `4` | Number of parallel workers |
| `**kwargs` | | | Passed to VajraSearchOptimized |

### When to Use

Use `VajraSearchParallel` when:

- Processing many queries at once
- Query latency is not critical
- CPU has multiple cores available

For single queries, `VajraSearchOptimized` is faster due to lower overhead.

## Performance Comparison

At 500K documents:

| Operation | VajraSearchOptimized | VajraSearchParallel |
|-----------|---------------------|---------------------|
| Single query | 1.89ms | 2.5ms |
| 100 queries (batch) | 189ms | 95ms |

## Index Statistics

Get information about the built index:

```python
engine = VajraSearchOptimized(corpus)

# Access index properties
print(f"Documents: {len(corpus)}")
print(f"Vocabulary size: {len(engine.index.vocabulary)}")
```

## Example: Complete Workflow

```python
from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

# Load corpus
corpus = DocumentCorpus.load("./papers/")

# Build optimized index
engine = VajraSearchOptimized(
    corpus,
    k1=1.5,
    b=0.75,
    use_eager=True,
    cache_size=1000
)

# Save for later
engine.save("papers_index.pkl")

# Search
results = engine.search("attention mechanism", top_k=5)

for r in results:
    print(f"{r.rank}. [{r.score:.2f}] {r.document.title}")
    print(f"   {r.document.content[:100]}...")
```
