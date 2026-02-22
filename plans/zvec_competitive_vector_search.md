# Plan: Vajra Vector Search — Closing the Gap with ZVec

## Benchmark Baseline

10k Wikipedia documents, all-MiniLM-L6-v2 (384d), HNSW cosine, M=16, ef_construction=200.

| Metric | ZVec 0.2.0 | Vajra (current) | Gap |
|--------|-----------|-----------------|-----|
| Build time | 0.45 s | 97.77 s | **217×** |
| p50 latency | 0.33 ms | 2.63 ms | **8×** |
| p95 latency | 0.52 ms | 3.77 ms | **7×** |
| Throughput | 2,866 QPS | 374 QPS | **7.7×** |
| Recall@10 | 1.000 | 0.987 | −1.3% |
| Index size | 18.4 MB | 15.7 MB | similar |

ZVec's edge comes from a C++ (Proxima) backend with disk-mmap. Vajra's disadvantages are entirely Python-side and fixable. The gaps are explained by four specific bugs/design choices — not by any fundamental algorithmic difference.

---

## Root Cause Analysis

### Bug 1 — O(N²) array growth during build  [responsible for ~95% of build overhead]

**File:** `vajra_bm25/vector/hnsw/index.py` — `_insert_vector()`

```python
# Called N times during add(). Every call copies ALL existing vectors.
self.graph.vectors = np.vstack([self.graph.vectors, vector])
```

For N=10,000 with 384-dim float32:
- Total bytes copied: Σ(i × 1536 bytes) for i=0..9999 ≈ **73 GB**
- At ~10 GB/s effective numpy throughput in a Python loop: ~7s baseline, plus Python call overhead → the observed 97s

`add()` already receives the full array upfront (`np.ndarray` of shape (N, 384)). The fix is to pre-allocate once and fill by index, not grow incrementally.

---

### Bug 2 — Immutable coalgebra state creates garbage in the search hot path  [responsible for ~70% of query overhead]

**File:** `vajra_bm25/vector/hnsw/coalgebra.py` — `_beam_step()` called by `unfold()`

For each candidate expansion (called ~ef × M ≈ 50 × 16 = 800 times per query):

```python
return [HNSWSearchState(
    candidates=tuple(candidates),     # New tuple allocation
    results=tuple(results),           # New tuple allocation
    visited=frozenset(visited),       # O(|visited|) copy — grows each step
    ...
)]
```

`frozenset(visited | {node})` at step k is O(k). Over 800 steps, that is O(800²/2) = 320,000 element-copy operations per query, plus 800 × 3 Python heap allocations. This is the dominant cost, not the distance computations.

The `HNSWNavigationCoalgebra` is a clean, composable abstraction worth keeping. But the hot path should bypass it.

---

### Bug 3 — Graph edge insertion uses O(M) list membership tests  [build: ~3% overhead]

**File:** `vajra_bm25/vector/hnsw/graph.py` — `add_edge()`

```python
if node_b not in self.layers[level][node_a]:  # O(M) scan every edge add
    self.layers[level][node_a].append(node_b)
```

With M=16 and N=10k, this is 160k O(16) scans during build. Minor, but it is a correctness-safe O(1) swap.

---

### Bug 4 — Recall shortfall at ef_search=50  [−1.3% recall vs ZVec]

ZVec's HNSW uses ef defaults that produce perfect recall at 10k. Vajra defaults to `ef_search=50` which is adequate but not tuned for 384-dim cosine. At 10k, `ef_search=100` likely closes the gap. This should be verified empirically rather than blindly raised.

---

## Implementation Plan

### Phase 1 — Fix Build Time

**Target: < 5s for 10k (from 97s)**

**1a. Pre-allocate vector storage in `add()`**

Refactor `_insert_vector()` to accept an index into a pre-allocated array rather than growing it. The full vector matrix is known at the start of `add()`.

```python
# In NativeHNSWIndex.add():
n_new = len(ids)
existing = 0 if self.graph.vectors is None else len(self.graph.ids)

# Allocate once — zero copies during insertion
if self.graph.vectors is None:
    self.graph.vectors = np.empty((n_new, self._dimension), dtype=np.float32)
else:
    self.graph.vectors = np.concatenate(
        [self.graph.vectors, np.empty((n_new, self._dimension), dtype=np.float32)]
    )

# Normalize in batch (one pass, no per-vector allocation)
if self.metric == "cosine":
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-8)

# Fill positions
for i in range(n_new):
    self.graph.vectors[existing + i] = vectors[i]
    self._insert_node(existing + i, ids[i])   # renamed from _insert_vector
```

`_insert_node(idx, id_)` takes a pre-assigned index; it no longer touches `self.graph.vectors` for storage — only for distance calculations.

**1b. Set-based neighbor lists during build**

In `HNSWGraph`, change `layers[level][node]` from `list` to `set` during construction, converting to a plain list only once after all insertions via `finalize()`. This turns O(M) membership checks into O(1).

The `NativeHNSWIndex.add()` method calls `self.graph.finalize()` as the last step before updating the coalgebra.

**Files:** `hnsw/index.py`, `hnsw/graph.py`

---

### Phase 2 — Fix Query Speed

**Target: p50 < 0.5ms at 10k (from 2.63ms)**

**2a. Imperative beam search bypassing the coalgebra**

Add `_beam_search_fast()` as a private method on `NativeHNSWIndex`. It implements the same HNSW beam search algorithm as the coalgebra, using mutable Python dicts and heapq directly — no object creation per step, no frozenset copies.

```python
def _beam_search_fast(
    self, query: np.ndarray, entry: int, ef: int
) -> List[Tuple[int, float]]:
    """Imperative beam search. Identical semantics to coalgebra unfold, faster."""
    vectors = self.graph.vectors
    get_neighbors = self.graph.get_neighbors

    d0 = float(self._distance_batch(query, vectors[entry : entry + 1])[0])
    candidates = [(d0, entry)]        # min-heap by distance
    results = [(-d0, entry)]          # max-heap (negated), tracks ef best
    visited = {entry}

    while candidates:
        dist, current = heapq.heappop(candidates)
        if dist > -results[0][0]:
            break
        neighbors = get_neighbors(current, 0)
        unvisited = [n for n in neighbors if n not in visited]
        if not unvisited:
            continue
        visited.update(unvisited)
        dists = self._distance_batch(query, vectors[unvisited])
        worst = -results[0][0]
        for n, d in zip(unvisited, dists):
            fd = float(d)
            heapq.heappush(candidates, (fd, n))
            if len(results) < ef:
                heapq.heappush(results, (-fd, n))
            elif fd < worst:
                heapq.heapreplace(results, (-fd, n))
                worst = -results[0][0]

    return sorted([(idx, -d) for d, idx in results], key=lambda x: x[1])
```

Similarly, add `_greedy_search_fast()` for upper-layer traversal during both build and search.

Wire `search()` to use these. The coalgebra is not deleted — it remains the documented public abstraction and continues to be tested.

**2b. Batch distance calls in upper-layer greedy descent**

`_greedy_search_layer()` in `hnsw/index.py` already uses `_distance_batch` on all neighbors at once (good). Verify this is the case after the refactor — the new `_greedy_search_fast()` must preserve this batching.

**Files:** `hnsw/index.py`

---

### Phase 3 — Save/Load and Recall

**Target: Recall@10 ≥ 0.999, load < 10ms**

**3a. Separate vector storage from graph metadata in save/load**

Currently `save()` pickles the full (N, 384) float32 array alongside the graph structure. Pickle is ~2× slower than `np.save()` for large arrays and prevents memory-mapped loading.

```python
def save(self, path: str) -> None:
    base = path.rstrip(".pkl")
    np.save(base + ".vectors.npy", self.graph.vectors)
    with open(base + ".meta.pkl", "wb") as f:
        pickle.dump({...everything except vectors...}, f)

@classmethod
def load(cls, path: str, mmap: bool = False) -> "NativeHNSWIndex":
    base = path.rstrip(".pkl")
    mode = "r" if mmap else None
    vectors = np.load(base + ".vectors.npy", mmap_mode=mode)
    with open(base + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    # reconstruct
```

With `mmap=True`, the vector array is lazily mapped from disk (matching ZVec's architecture), making load time near-zero. For search, the OS page cache warms the frequently accessed pages, same as ZVec.

Backward-compatible: detect old single-file format by checking for `.meta.pkl` existence.

**3b. Tune ef_search default**

Run a sweep at 10k: ef_search ∈ {50, 75, 100, 150} measuring recall@10 and p50 latency. Set the new default to the point where recall@10 = 1.000 with minimum latency cost. Update the benchmark to expose `--ef-search`.

**Files:** `hnsw/index.py`

---

## Expected Outcomes

| Metric | Current | After Ph.1 | After Ph.2 | After Ph.3 |
|--------|---------|-----------|-----------|-----------|
| Build 10k | 97.77 s | ~2–4 s | ~2–4 s | ~2–4 s |
| p50 10k | 2.63 ms | 2.63 ms | ~0.35–0.55 ms | ~0.3–0.5 ms |
| p95 10k | 3.77 ms | 3.77 ms | ~0.7–1.0 ms | ~0.6–0.9 ms |
| QPS 10k | 374 | 374 | ~1,500–2,500 | ~1,800–3,000 |
| Recall@10 | 0.987 | 0.987 | 0.987 | ~1.000 |
| Load time | 0.014 s | 0.014 s | 0.014 s | ~0.003 s |

**Build time**: Phase 1 eliminates the O(N²) bottleneck. The remaining cost is the HNSW graph construction itself — expected 2–4s for 10k in Python, vs ZVec's C++ 0.45s. This is a fair Python-vs-C++ comparison; the algorithmic cost is identical.

**Query speed**: Phase 2 should close to within 1.5–3× of ZVec at 10k. Python `heapq` carries irreducible per-call overhead (~1–2μs per push/pop). With ef=50 and M=16 neighbors, the beam search does ~800 heapq operations ≈ 1.5ms baseline floor in CPython. Beating ZVec's 0.33ms consistently requires either Numba JIT on the inner loop or a C extension — this is a potential Phase 4. At larger datasets (50k+), Vajra's in-memory access pattern may have an edge over ZVec's disk-backed mmap on cold-start queries.

---

## Testing Strategy

After each phase, run in sequence:

1. `pytest ~/Github/vajra_bm25/tests/test_vector.py -v` — all tests must pass; raise recall assertion in `test_recall_vs_flat` from 0.70 to 0.95 after Phase 3
2. `python ~/Github/zvec_vajra_benchmark/benchmark.py --sizes 10000 --engines vajra` — compare directly to the baseline in this document
3. Manual correctness check: verify top-5 results for 3 fixed queries are identical before/after refactor (determinism test)

---

## Files to Modify

| File | Changes |
|------|---------|
| `vajra_bm25/vector/hnsw/index.py` | Pre-alloc, `_insert_node()`, `_beam_search_fast()`, `_greedy_search_fast()`, save/load split |
| `vajra_bm25/vector/hnsw/graph.py` | Set-based neighbor storage during build, `finalize()` method |
| `vajra_bm25/vector/hnsw/coalgebra.py` | No changes (preserved as documented abstraction) |
| `vajra_bm25/vector/hnsw/state.py` | No changes |

Total estimated changes: ~120 lines modified / added across 2 files.

---

## What This Does Not Address (Phase 4 scope)

- **Numba JIT on the beam search inner loop** — would push QPS above ZVec on CPU; requires restructuring the heapq loop into a Numba-compilable form (no Python objects, fixed-size priority queue)
- **Batch query parallelism** — `search_batch()` today is a serial loop; thread-pool parallelism with `concurrent.futures` would give near-linear QPS scaling on multi-core
- **Quantization** — ZVec supports INT8 quantization; Vajra could add FP16 storage to halve memory and improve cache utilization
- **Disk-mapped index at full scale** — the `mmap=True` load path (Phase 3a) is the entry point to this
