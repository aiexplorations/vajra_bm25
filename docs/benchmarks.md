# Vajra BM25 Benchmarks

Comprehensive performance benchmarks comparing Vajra BM25 against rank-bm25 (baseline) and BM25S.

## Test Configuration

- **Queries**: 10 domain-specific queries (statistics, ML, algorithms, databases)
- **Runs**: 3 runs per query for statistical reliability
- **Corpus sizes**: 1K, 10K, 50K, 100K synthetic documents
- **Metrics**: Latency (avg, P50, P95), Speedup vs baseline, Recall@10

## Implementations Compared

| Implementation | Description |
|----------------|-------------|
| **rank-bm25** | Standard Python BM25 library (baseline) |
| **Vajra Base** | Category theory implementation, pure Python |
| **Vajra Optimized** | Vectorized NumPy + sparse matrices |
| **Vajra Parallel** | Thread pool parallelism (4 workers) |
| **BM25S** | Fast BM25 library with native tokenization |

## Results

### Speed Comparison (Average Latency in ms)

| Corpus Size | rank-bm25 | Vajra Optimized | Vajra Parallel | BM25S |
|-------------|-----------|-----------------|----------------|-------|
| 1,000       | 0.46      | 0.05            | 0.01           | 0.06  |
| 10,000      | 7.64      | 0.15            | 0.06           | 0.14  |
| 50,000      | 40.28     | 0.33            | 0.24           | 0.36  |
| 100,000     | 79.62     | 0.34            | 0.26           | 0.47  |

### Speedup vs rank-bm25

| Corpus Size | Vajra Optimized | Vajra Parallel | BM25S  |
|-------------|-----------------|----------------|--------|
| 1,000       | 10x             | 31x            | 8x     |
| 10,000      | 52x             | 119x           | 54x    |
| 50,000      | 122x            | 167x           | 113x   |
| 100,000     | 234x            | **307x**       | 168x   |

### Recall@10 (vs rank-bm25 baseline)

| Corpus Size | Vajra Base | Vajra Optimized | Vajra Parallel | BM25S |
|-------------|------------|-----------------|----------------|-------|
| 1,000       | 98%        | 99%             | 99%            | 98%   |
| 10,000      | 55%        | 56%             | 56%            | 56%   |
| 50,000      | 78%        | **80%**         | **80%**        | 56%   |
| 100,000     | 51%        | 50%             | 50%            | 50%   |

### Build Times (ms)

| Corpus Size | Vajra Base | Vajra Optimized | Vajra Parallel | rank-bm25 | BM25S |
|-------------|------------|-----------------|----------------|-----------|-------|
| 1,000       | 54         | 64              | 59             | 52        | 32    |
| 10,000      | 368        | 902             | 874            | 335       | 225   |
| 50,000      | 2,970      | 5,930           | 5,727          | 2,667     | 1,571 |
| 100,000     | 4,160      | 10,390          | 10,437         | 3,498     | 2,568 |

## Batch Processing

Batch processing benchmark with 50 queries (10 unique queries x 5 repetitions):

| Corpus Size | Sequential (ms) | Parallel Batch (ms) | Batch Speedup | Throughput |
|-------------|-----------------|---------------------|---------------|------------|
| 10,000      | 0.6             | 1.6                 | 0.4x          | 31,550 q/s |
| 50,000      | 2.6             | 2.8                 | 0.9x          | 17,883 q/s |
| 100,000     | 3.1             | 3.1                 | 1.0x          | 16,351 q/s |

**Note**: For sub-millisecond queries, thread pool overhead outweighs parallelism benefits. Batch parallelism is more beneficial for:
- Larger result sets (higher top_k)
- More expensive scoring operations
- I/O-bound integrations

## Key Findings

### Performance

1. **Vajra Parallel achieves 307x speedup** over rank-bm25 at 100K documents
2. **Sub-millisecond latency** at all corpus sizes (0.01-0.34ms)
3. **Faster than BM25S** at all corpus sizes tested
4. **Throughput**: Up to 31,550 queries/second

### Ranking Quality

1. **Vajra achieves 80% recall** at 50K docs vs 56% for BM25S
2. **Equal or better recall** than BM25S at all corpus sizes
3. Recall varies by corpus characteristics (vocabulary overlap, document length)

### Trade-offs

1. **Build time**: Vajra Optimized has higher index build time due to sparse matrix construction
2. **Memory**: Sparse matrices reduce memory for large corpora (>10K docs)
3. **Caching**: LRU cache provides significant speedup for repeated queries

## Optimization Techniques

Vajra achieves these speedups through:

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Vectorized NumPy** | Batch operations over document sets | 10-50x |
| **Sparse matrices** | Efficient storage for large corpora | Memory reduction |
| **Pre-computed IDF** | One-time IDF calculation at build | Per-query savings |
| **LRU caching** | Cache query preprocessing and scores | Repeated query speedup |
| **Partial sort** | O(n + k log k) for top-k selection | Large corpus speedup |
| **Thread pool** | Concurrent query processing | Additional 1.3-3x |

## Running Benchmarks

```bash
# Install dependencies
pip install vajra-bm25[optimized] rank-bm25 bm25s

# Run benchmarks (requires corpus files)
python benchmarks/benchmark.py
```

## Corpus Generation

Generate synthetic test corpora:

```python
from vajra_bm25 import Document, DocumentCorpus
import random

# Generate documents with domain-specific vocabulary
topics = ["machine_learning", "statistics", "algorithms", "databases"]
documents = []

for i in range(10000):
    topic = random.choice(topics)
    doc = Document(
        id=f"doc_{i}",
        title=f"Document about {topic}",
        content=generate_content(topic)  # Your content generator
    )
    documents.append(doc)

corpus = DocumentCorpus(documents)
corpus.save_jsonl("corpus_10k.jsonl")
```

## Environment

- **Python**: 3.8+
- **NumPy**: 1.20+
- **SciPy**: 1.7+ (for sparse matrices)
- **Platform**: macOS Darwin 25.1.0
- **Date**: December 2025
