# Vajra BM25 Performance Optimization Plan

## Problem Statement

**Current Performance (Wikipedia benchmarks):**
| Metric | Vajra | BM25S | Gap |
|--------|-------|-------|-----|
| Index 200K | 249s | 250s | ~same |
| Index 500K | 982s | 583s | 1.7x slower |
| Search 200K | 17ms | 4ms | 4x slower |
| Search 500K | 27ms | 10ms | 2.7x slower |

**Root Causes:**
1. **Indexing:** Term bounds computation is a Python loop (not Numba-compiled)
2. **Search:** Numba scorer exists but SparseBM25Scorer fallback creates dense matrices
3. **Benchmark:** Builds duplicate indexes for Optimized and Parallel variants

---

## Phase 1: Quick Wins (< 1 day each)

### 1.1 Make Numba Scorer the Primary Path
**File:** `vajra_bm25/optimized.py` (lines 1297-1330)

**Problem:** Search falls through to SparseBM25Scorer which calls `.toarray()` creating 8MB+ dense matrices per query.

**Fix:** Ensure NumbaSparseBM25Scorer is always used when Numba available. The Numba scorer is already implemented and working (we just fixed the top-k bug).

**Changes:**
- Verify `use_numba=True` is default (it is)
- Remove/simplify SparseBM25Scorer code path for large corpora
- Add logging to confirm Numba path is taken

**Expected:** 2-3x search speedup

---

### 1.2 Numba-compile Term Bounds Computation
**File:** `vajra_bm25/optimized.py` (lines 504-530)

**Problem:** `_compute_term_bounds()` uses Python loop through ~1M+ terms:
```python
for term_id in range(self.num_terms):  # Python loop - SLOW
```

**Fix:** Create Numba JIT function with parallel execution:
```python
@njit(cache=True, parallel=True)
def _numba_compute_term_bounds(indptr, indices, data, norm_factors,
                                idf_cache, k1, num_terms):
    max_term_score = np.zeros(num_terms, dtype=np.float32)
    for term_id in prange(num_terms):  # Parallel over terms
        row_start = indptr[term_id]
        row_end = indptr[term_id + 1]
        if row_start == row_end:
            continue
        max_tf_comp = 0.0
        for j in range(row_start, row_end):
            tf = data[j]
            norm = norm_factors[indices[j]]
            tf_comp = (tf * (k1 + 1)) / (tf + k1 * norm)
            if tf_comp > max_tf_comp:
                max_tf_comp = tf_comp
        max_term_score[term_id] = idf_cache[term_id] * max_tf_comp
    return max_term_score
```

**Expected:** 5-10x speedup for term bounds (10-15% of index time)

---

### 1.3 Fix Benchmark Duplicate Index Building
**File:** `benchmarks/benchmark_standard_datasets.py` (lines 673-682)

**Problem:** Benchmark creates both `VajraWrapper(variant="optimized")` and `VajraWrapper(variant="parallel")`, each building their own index.

**Fix:** Share index between variants or skip parallel variant in fast mode.

**Expected:** ~40% reduction in benchmark index build time

---

## Phase 2: Medium-Term Improvements (1-2 days each)

### 2.1 Eager Sparse Scoring (BM25S Approach)
**File:** `vajra_bm25/optimized.py`

**Concept:** Pre-compute BM25 scores at index time, store in sparse matrix. Query time just sums pre-computed values.

**Implementation:**
```python
class EagerSparseIndex(VectorizedIndexSparse):
    def __init__(self):
        super().__init__()
        self.score_matrix: Optional[csr_matrix] = None  # Pre-computed BM25 scores

    def _build_score_matrix(self, k1, b):
        """Pre-compute BM25 scores for all term-doc pairs."""
        # Iterate over existing term_doc_matrix and compute scores
        # Store in new sparse matrix (same sparsity pattern)
```

**Search becomes:**
```python
scores = self.score_matrix[term_ids, :].sum(axis=0).A1  # Just sum!
```

**Expected:** 5-10x search speedup (eliminates all BM25 computation at query time)

---

### 2.2 Compositional Term Score Caching
**File:** `vajra_bm25/optimized.py`

**Concept:** Cache per-term score vectors, compose for multi-term queries (monoid homomorphism).

**Implementation:**
- LRU cache mapping term -> sparse score vector
- Query "machine learning" = cached_scores["machine"] + cached_scores["learning"]
- 10K term cache covers most common terms

**Expected:** 10-100x speedup for repeated terms (warm cache)

---

## Phase 3: Future Improvements

### 3.1 Numba MaxScore Algorithm
- Port `MaxScoreBM25Scorer` to Numba for early termination
- Expected: 2-5x for selective queries

### 3.2 Parallel Numba Scoring
- Enable `parallel=True` in Numba scoring with proper accumulation
- Expected: 2-4x on multi-core

---

## Implementation Order (Search Speed Priority)

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **1** | **2.1 Eager sparse scoring** | **1 day** | **5-10x search** |
| 2 | 1.1 Verify Numba is primary | 1 hr | 2-3x search |
| 3 | 2.2 Compositional caching | 1 day | 10-100x warm |
| 4 | 1.2 Numba term bounds | 2-4 hrs | 5-10% index |
| 5 | 1.3 Fix benchmark duplication | 1-2 hrs | 40% benchmark |

**User Priority:** Search speed first, with pre-computed eager scoring approach.

---

## Files to Modify

1. **`vajra_bm25/optimized.py`** - All core optimizations
   - Lines 504-530: Term bounds computation
   - Lines 1297-1330: Search method scorer selection
   - New: EagerSparseIndex class

2. **`benchmarks/benchmark_standard_datasets.py`** - Fix duplicate index building
   - Lines 673-682: VajraWrapper creation

3. **`tests/test_bm25.py`** - Verify no regressions

---

## Success Criteria

After Phase 1:
- Search latency: < 10ms on 200K docs (currently 17ms)
- Index time: < 900s on 500K docs (currently 982s)

After Phase 2:
- Search latency: < 5ms on 200K docs (match BM25S)
- Warm query cache: < 1ms for repeated terms

---

## Testing Strategy

1. Run existing tests after each change: `pytest tests/`
2. Benchmark comparison: `python benchmarks/benchmark_standard_datasets.py --datasets wiki-200k`
3. Score verification: Compare scores between old/new implementations
