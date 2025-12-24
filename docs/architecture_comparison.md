# Architecture Comparison: BM25S vs Vajra BM25

## Overview

This document compares the architectural approaches of [BM25S](https://github.com/xhluca/bm25s) and Vajra BM25 for lexical search. Both libraries implement the BM25 algorithm but use fundamentally different strategies for balancing index build time vs. query latency.

## Core Architectural Difference

| Aspect | BM25S | Vajra BM25 |
|--------|-------|------------|
| **Scoring Strategy** | Eager (pre-computed at index time) | Lazy (computed at query time) + Eager (optional) |
| **Sparse Matrix Format** | CSC (Compressed Sparse Column) | CSR (Compressed Sparse Row) |
| **Query-time Operation** | Slice + Sum | BM25 formula computation |
| **Index Memory** | Higher (stores scores) | Lower (stores TFs) |
| **Index Build Time** | Longer (score computation) | Shorter (TF only) |

## Detailed Architecture

### BM25S: Eager Sparse Scoring

BM25S implements the approach described in the paper "BM25S: Orders of magnitude faster lexical search via eager sparse scoring" ([arXiv:2407.03618](https://arxiv.org/abs/2407.03618)).

**Key Innovation:** Pre-compute all possible BM25 scores at index time and store them in a sparse matrix.

```
Index Time:
  For each (term, document) pair where TF > 0:
    score[term, doc] = IDF[term] * TF_component(TF, doc_length)
  Store in CSC sparse matrix

Query Time:
  1. Select rows for query terms
  2. Sum across rows
  3. Get top-k
```

**Advantages:**
- Extremely fast query time (just matrix slicing + summation)
- Query complexity is O(Q * D_avg) where Q = query terms, D_avg = avg docs per term
- No floating-point BM25 computation at query time

**Disadvantages:**
- Index must be rebuilt if k1/b parameters change
- Higher memory usage (stores float scores vs. integer TFs)
- Longer index build time

### Vajra BM25: Multi-Strategy Approach

Vajra BM25 offers multiple scoring strategies, allowing users to choose the best trade-off:

```
Index Time:
  1. Build term_doc_matrix (TF values) in CSR format
  2. Pre-compute IDF values
  3. Pre-compute document length normalization factors
  4. (Optional) Build score_matrix for eager scoring

Query Time (priority order):
  1. Cache hit (instant)
  2. Eager scorer (if enabled) - slice + sum
  3. Numba JIT scorer - compiled BM25 computation
  4. MaxScore algorithm - early termination
  5. NumPy/SciPy scorer - vectorized BM25
```

**Available Scorers:**

| Scorer | When Used | Approach |
|--------|-----------|----------|
| `EagerSparseBM25Scorer` | `use_eager=True` | Pre-computed scores, slice + sum |
| `NumbaSparseBM25Scorer` | `use_numba=True` | JIT-compiled BM25 scoring |
| `MaxScoreBM25Scorer` | `use_maxscore=True` | Early termination with bounds |
| `SparseBM25Scorer` | Fallback | NumPy/SciPy vectorized |

## Sparse Matrix Format Choice

### BM25S: CSC (Compressed Sparse Column)

```
CSC is optimal for:
- Column-wise operations (selecting all docs for a term)
- Summing across rows (query scoring)
- BM25S query: score_matrix[term_ids, :].sum(axis=0)
```

### Vajra: CSR (Compressed Sparse Row)

```
CSR is optimal for:
- Row-wise operations (iterating through a term's posting list)
- Term-at-a-time scoring
- Vajra query: Iterate through each query term's row
```

For eager scoring in Vajra, CSR is still used because:
1. The same index serves both eager and non-eager scoring
2. CSR row slicing + sum is nearly as efficient as CSC column operations
3. Consistency with the existing codebase

## Performance Comparison

### Theoretical Complexity

| Operation | BM25S | Vajra (Eager) | Vajra (Numba) |
|-----------|-------|---------------|---------------|
| Index Build | O(N * T) | O(N * T) | O(N * T) |
| Score Matrix | O(nnz) | O(nnz) | N/A |
| Query (no top-k) | O(Q * D_avg) | O(Q * D_avg) | O(Q * posting_size) |
| Top-k Selection | O(D) | O(D) | O(D) |

Where:
- N = number of documents
- T = average terms per document
- Q = query terms
- D = total documents
- D_avg = average documents per query term
- nnz = non-zero entries in sparse matrix

### Benchmark Results (Wikipedia 200K)

| Engine | Search Latency | Build Time |
|--------|---------------|------------|
| BM25S | 4.3ms | 250s |
| Vajra (Numba) | 17ms | 249s |
| Vajra (Eager) | TBD | TBD |

*Note: Eager scoring was just implemented. Benchmark results pending.*

## Memory Usage

### BM25S

```
score_matrix: nnz * 4 bytes (float32 scores)
```

### Vajra (with Eager)

```
term_doc_matrix: nnz * 4 bytes (float32 TFs)
score_matrix: nnz * 4 bytes (float32 scores)  # if use_eager=True
idf_cache: T * 4 bytes
norm_factors: D * 4 bytes
max_term_score: T * 4 bytes  # for MaxScore algorithm
```

When eager scoring is enabled, Vajra uses approximately 2x the memory for sparse matrices compared to BM25S, because it stores both TF values (for flexibility) and pre-computed scores.

## Implementation Details

### BM25S Score Computation

From the BM25S paper, scores are computed as:

```python
# At index time:
score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len / avg_len))

# Stored in sparse matrix
```

### Vajra Eager Score Computation

```python
# In VectorizedIndexSparse.build_score_matrix():
tf_component = (tf * k1_plus_1) / (tf + k1 * norm_factors[doc_indices])
score_data = tf_component * idf_for_entries
self.score_matrix = csr_matrix((score_data, indices, indptr), ...)
```

### Query-Time Scoring

```python
# BM25S:
scores = score_matrix[term_ids, :].sum(axis=0).A1

# Vajra Eager:
scores = np.asarray(
    self.index.score_matrix[term_ids, :].sum(axis=0)
).flatten()
```

## When to Use Each Approach

### Use BM25S When:
- Query latency is critical (< 5ms required)
- Parameters (k1, b) are fixed
- Memory is not constrained
- Index rebuild is acceptable

### Use Vajra with Eager Scoring When:
- Query latency is important but flexible parameters needed
- You want both fast queries AND the ability to change k1/b
- You need MaxScore early termination for selective queries

### Use Vajra without Eager Scoring When:
- Memory is constrained
- Index build time is critical
- You frequently change BM25 parameters
- Numba JIT provides acceptable query latency

## Categorical Interpretation

Both BM25S and Vajra can be understood through category theory:

**BM25S:**
- Pre-computation is a functor: `Index -> ScoredIndex`
- Query is a morphism: `ScoredIndex × Query -> Scores`
- The functor "lifts" scoring from query-time to index-time

**Vajra:**
- Index is a coalgebra with structure map: `State -> List[Results]`
- Eager scoring adds a new morphism path
- MaxScore uses coalgebraic guards for early termination

## Future Directions

1. **Numba-compiled eager scoring:** Combine pre-computed scores with JIT compilation
2. **Hybrid approach:** Use eager for common terms, lazy for rare terms
3. **Rust backend:** Native performance for both approaches
4. **Memory-mapped indices:** Handle larger-than-RAM corpora

## References

1. Luca, X. (2024). [BM25S: Orders of magnitude faster lexical search via eager sparse scoring](https://arxiv.org/abs/2407.03618). arXiv:2407.03618.
2. Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval.
3. Kamphuis, C., et al. (2020). Which BM25 Do You Mean? A Large-Scale Reproducibility Study of Scoring Variants.
