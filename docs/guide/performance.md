# Performance Tips

Optimize Vajra for your production workload.

## Quick Wins

### 1. Use VajraSearchOptimized

Always use the optimized engine for production:

```python
# Good
from vajra_bm25 import VajraSearchOptimized
engine = VajraSearchOptimized(corpus)

# Not recommended for production
from vajra_bm25 import VajraSearch
engine = VajraSearch(corpus)  # Much slower
```

### 2. Enable Eager Scoring

Pre-compute BM25 scores at index time:

```python
engine = VajraSearchOptimized(corpus, use_eager=True)
```

This increases index build time but provides sub-millisecond queries.

### 3. Enable Query Caching

For repeated queries:

```python
engine = VajraSearchOptimized(corpus, cache_size=1000)
```

Cache performance:

| Scenario | Latency |
|----------|---------|
| Cold query | 0.14 - 3.5ms |
| Cached query | ~0.001ms |

### 4. Install Numba

For additional JIT compilation speedup:

```bash
pip install numba
```

Vajra automatically detects and uses Numba if available.

## Memory Optimization

### Sparse Matrices

For large corpora, Vajra automatically uses sparse matrices:

```python
# Automatic (>10K docs triggers sparse mode)
engine = VajraSearchOptimized(corpus)

# Force sparse mode
engine = VajraSearchOptimized(corpus, use_sparse=True)
```

Memory comparison at 1M documents:

| Mode | Memory Usage |
|------|--------------|
| Dense | ~30 GB |
| Sparse | ~300 MB |

### Disable Eager Scoring

Trade query speed for memory:

```python
# Lower memory, slower queries
engine = VajraSearchOptimized(corpus, use_eager=False)
```

## Batch Processing

For multiple queries, use batch methods:

```python
# Slower: Individual queries
for query in queries:
    results = engine.search(query, top_k=10)

# Faster: Batch processing
all_results = engine.search_batch(queries, top_k=10)
```

For high throughput, use parallel engine:

```python
from vajra_bm25 import VajraSearchParallel

engine = VajraSearchParallel(corpus, max_workers=4)
batch_results = engine.search_batch(queries, top_k=10)
```

## Index Persistence

Save and reload indexes to avoid rebuilding:

```python
# Save index
engine.save("index.pkl")

# Load index (fast)
engine = VajraSearchOptimized.load("index.pkl")
```

!!! note
    Requires: `pip install vajra-bm25[persistence]`

## Profiling

Identify bottlenecks in your pipeline:

```python
import time

# Profile index building
t0 = time.time()
engine = VajraSearchOptimized(corpus)
print(f"Index build: {time.time() - t0:.2f}s")

# Profile queries
t0 = time.time()
for _ in range(100):
    engine.search("test query", top_k=10)
print(f"100 queries: {time.time() - t0:.3f}s")
print(f"Avg latency: {(time.time() - t0) / 100 * 1000:.2f}ms")
```

## Scaling Guidelines

| Corpus Size | Recommended Setup |
|-------------|-------------------|
| < 10K docs | `VajraSearchOptimized`, default settings |
| 10K - 100K | `VajraSearchOptimized`, sparse mode, eager scoring |
| 100K - 1M | Same + query caching + Numba |
| > 1M | Same + index persistence + consider sharding |

## Hardware Recommendations

| Corpus Size | RAM | CPU |
|-------------|-----|-----|
| 100K docs | 2 GB | Any |
| 500K docs | 4 GB | Multi-core recommended |
| 1M docs | 8 GB | Multi-core recommended |

## Benchmarking Your Setup

```bash
# Run built-in benchmarks
python benchmarks/benchmark.py --datasets beir-scifact --engines vajra
```

See [Benchmarks](../benchmarks.md) for detailed performance data.
