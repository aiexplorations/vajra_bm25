# Plan: Improve Vajra Build Time

## Problem Statement

Vajra's index build time is **2x slower** than BM25S at scale:

| Corpus | Vajra Build | BM25S Build | Gap |
|--------|-------------|-------------|-----|
| 500K docs | 7 min | 4 min | 1.75x |
| 1M docs | 29.5 min | 15.3 min | 1.93x |

This gap widens with corpus size, making Vajra less practical for large-scale indexing.

## Root Cause Analysis

### Build Phase Breakdown (1M docs)

| Phase | Time | % of Total | Bottleneck |
|-------|------|------------|------------|
| COO Matrix Construction | ~10-12 min | 35-40% | Python list appends |
| Eager Score Matrix | ~6 min | 20% | Duplicate matrix work |
| Tokenization | ~6 min | 20% | Regex-based |
| Term Bounds | ~4-5 min | 15% | Python loop over terms |
| Vocabulary Building | ~3 min | 10% | Python set operations |

### Primary Bottleneck: COO Matrix Construction

**Location**: `optimized.py` lines 444-466

```python
rows = []
cols = []
data = []

for doc_idx, term_counts in enumerate(doc_term_counts):
    for term, count in term_counts.items():
        term_id = self.term_to_id[term]
        rows.append(term_id)  # ← 100M+ appends
        cols.append(doc_idx)
        data.append(count)
```

For 1M documents with ~100M non-zero entries, this performs **300M list append operations**, each potentially triggering memory reallocation.

## Proposed Optimizations

### Priority 1: Pre-allocate Arrays for COO Construction

**Impact**: -30-35% build time

**Current** (slow):
```python
rows = []
cols = []
data = []
for doc_idx, term_counts in enumerate(doc_term_counts):
    for term, count in term_counts.items():
        rows.append(term_id)
        cols.append(doc_idx)
        data.append(count)
```

**Proposed** (fast):
```python
# Count total entries first
total_entries = sum(len(tc) for tc in doc_term_counts)

# Pre-allocate
rows = np.empty(total_entries, dtype=np.int32)
cols = np.empty(total_entries, dtype=np.int32)
data = np.empty(total_entries, dtype=np.float32)

# Fill arrays
idx = 0
for doc_idx, term_counts in enumerate(doc_term_counts):
    n = len(term_counts)
    for term, count in term_counts.items():
        rows[idx] = self.term_to_id[term]
        cols[idx] = doc_idx
        data[idx] = count
        idx += 1
```

**Even faster** (Numba):
```python
@njit
def fill_coo_arrays(doc_term_ids, doc_term_counts, rows, cols, data):
    idx = 0
    for doc_idx in range(len(doc_term_ids)):
        for i, term_id in enumerate(doc_term_ids[doc_idx]):
            rows[idx] = term_id
            cols[idx] = doc_idx
            data[idx] = doc_term_counts[doc_idx][i]
            idx += 1
```

### Priority 2: Make Eager Scoring Optional/Lazy

**Impact**: -15-20% build time (when disabled)

**Location**: `optimized.py` lines 1379-1382

**Current**: Always builds score matrix
```python
if self.use_eager:
    self.index.build_score_matrix(k1, b)
```

**Proposed**:
- Add `use_eager=False` as default for large corpora
- Build score matrix lazily on first query
- Or use Numba scorer instead (already fast enough)

```python
def __init__(self, corpus, use_eager="auto", ...):
    # Auto-disable eager for large corpora
    if use_eager == "auto":
        use_eager = len(corpus) < 100000  # Only for <100K docs
```

### Priority 3: Faster Tokenization

**Impact**: -10-15% build time

**Location**: `text_processing.py` line 56

**Current** (regex):
```python
def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return [t for t in tokens if t not in STOP_WORDS]
```

**Proposed** (split-based):
```python
def preprocess_text_fast(text: str) -> List[str]:
    tokens = text.lower().split()
    return [t.strip('.,!?;:()[]{}"\'-') for t in tokens
            if t not in STOP_WORDS and t.isalnum()]
```

**Or Cython/Rust extension** for maximum speed.

### Priority 4: Vectorize Term Bounds Computation

**Impact**: -10-15% build time

**Location**: `optimized.py` lines 516-532

**Current** (Python loop):
```python
for term_id in range(self.num_terms):
    row_start = indptr[term_id]
    row_end = indptr[term_id + 1]
    ...
    self.max_term_score[term_id] = self.idf_cache[term_id] * tf_components.max()
```

**Proposed** (Numba JIT):
```python
@njit(parallel=True)
def compute_term_bounds_numba(indptr, indices, data, doc_norms, idf_cache, k1, max_scores):
    for term_id in prange(len(indptr) - 1):
        row_start = indptr[term_id]
        row_end = indptr[term_id + 1]
        ...
```

### Priority 5: Parallel Vocabulary Building

**Impact**: -5% build time

**Current**: Sequential set operations
```python
term_set = set()
for term_counts in doc_term_counts:
    term_set.update(term_counts.keys())
```

**Proposed**: Parallel merge with reduction
```python
from concurrent.futures import ThreadPoolExecutor

def merge_sets(sets):
    result = set()
    for s in sets:
        result.update(s)
    return result

# Split into chunks and merge in parallel
```

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. [ ] Pre-allocate COO arrays instead of list appends
2. [ ] Make `use_eager=False` default for >100K docs
3. [ ] Add `--fast-build` flag to benchmark script

### Phase 2: Moderate Effort (4-6 hours)
4. [ ] Add Numba-accelerated COO construction
5. [ ] Implement split-based tokenizer option
6. [ ] Vectorize term bounds with Numba

### Phase 3: Significant Effort (1-2 days)
7. [ ] Cython extension for core indexing path
8. [ ] Streaming index construction (don't hold all docs in memory)
9. [ ] Incremental index updates

## Expected Results

| Optimization | Time Saved | Cumulative |
|--------------|------------|------------|
| Pre-allocate arrays | -6 min | 23.5 min |
| Disable eager (large) | -5 min | 18.5 min |
| Fast tokenizer | -3 min | 15.5 min |
| Numba term bounds | -2 min | 13.5 min |

**Target**: Match or beat BM25S build time (~15 min for 1M docs)

## Files to Modify

1. `vajra_bm25/optimized.py` - COO construction, term bounds, eager default
2. `vajra_bm25/text_processing.py` - Fast tokenizer
3. `benchmarks/benchmark.py` - Add `--fast-build` option

## Verification

After each optimization:
1. Run `pytest tests/` to ensure correctness
2. Run `python benchmarks/benchmark.py --datasets wiki-1m --engines vajra`
3. Compare build time vs baseline (29.5 min)
4. Verify query latency unchanged (~3.5ms)
