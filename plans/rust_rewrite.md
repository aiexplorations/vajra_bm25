# Plan: 10x Query Performance via Rust Extension

## Overview

**Objective:** Achieve 10x query latency improvement for 1M+ document corpora using Rust with PyO3 bindings.

**Branch:** `feature/rust-backend`

**Version:** 0.3.0 (major feature addition with new optional dependency)

### Current State

| Metric | Value |
|--------|-------|
| Performance | 0.35-0.44ms at 100K docs (230-291x faster than rank-bm25) |
| Hot path | BM25 scoring (70-80% of query time) in Numba JIT |
| Stack | Pure Python + NumPy/SciPy + Numba |

### Target

| Metric | Target |
|--------|--------|
| Latency at 100K docs | ~0.04ms (10x improvement) |
| Latency at 1M docs | <0.5ms |
| Ranking | Exact BM25 (no approximations) |
| Optimization focus | Both single query latency AND batch throughput |

---

## Phase 0: Branch Setup and Prerequisites

### Checklist

- [ ] **0.1 Create feature branch**
  ```bash
  git checkout -b feature/rust-backend
  git push -u origin feature/rust-backend
  ```

- [ ] **0.2 Install Rust toolchain**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup default stable
  rustup target add x86_64-apple-darwin aarch64-apple-darwin  # macOS
  ```

- [ ] **0.3 Install maturin**
  ```bash
  pip install maturin
  ```

- [ ] **0.4 Verify current test suite passes**
  ```bash
  source venv/bin/activate
  pytest tests/ -v
  ```

- [ ] **0.5 Document baseline performance**
  - Run `benchmarks/profile_query_latency.py` on 100K and 1M corpora
  - Save results to `benchmarks/baseline_v0.2.1.json`

---

## Phase 1: Rust Crate Setup

### Checklist

- [ ] **1.1 Create directory structure**
  ```
  rust/
  ├── Cargo.toml
  ├── src/
  │   ├── lib.rs
  │   ├── sparse_matrix.rs
  │   ├── bm25_scorer.rs
  │   ├── top_k.rs
  │   └── batch.rs
  └── benches/
      └── bm25_bench.rs
  ```

- [ ] **1.2 Create `rust/Cargo.toml`**
  ```toml
  [package]
  name = "vajra_bm25_core"
  version = "0.1.0"
  edition = "2021"

  [lib]
  name = "vajra_bm25_core"
  crate-type = ["cdylib"]

  [dependencies]
  pyo3 = { version = "0.22", features = ["extension-module", "abi3-py38"] }
  numpy = "0.22"
  rayon = "1.10"
  memmap2 = "0.9"
  bytemuck = { version = "1.15", features = ["derive"] }

  [dev-dependencies]
  criterion = "0.5"

  [[bench]]
  name = "bm25_bench"
  harness = false

  [profile.release]
  lto = true
  codegen-units = 1
  opt-level = 3
  ```

- [ ] **1.3 Create minimal `rust/src/lib.rs`**
  - PyO3 module skeleton
  - Empty `RustBM25Index` struct with `#[pyclass]`
  - Verify builds with `maturin develop`

- [ ] **1.4 Verify Rust builds and imports in Python**
  ```bash
  cd rust && maturin develop --release
  python -c "from vajra_bm25_core import RustBM25Index; print('OK')"
  ```

### Details

The Rust crate uses:
- **pyo3**: Python bindings with stable ABI (abi3-py38 for broad compatibility)
- **numpy**: Zero-copy array access from Python
- **rayon**: Work-stealing parallelism for batch queries
- **memmap2**: Memory-mapped files for 1M+ doc indices
- **bytemuck**: Safe transmutation for SIMD-friendly data layouts

---

## Phase 2: Data Structures

### Checklist

- [ ] **2.1 Implement `TermRow` struct** (`sparse_matrix.rs`)
  ```rust
  #[repr(C)]
  #[derive(Clone, Copy, Pod, Zeroable)]
  pub struct TermRow {
      pub start: u32,      // Offset into data/indices arrays
      pub len: u16,        // Number of non-zeros in this row
      pub _pad: u16,       // Alignment padding
      pub idf: f32,        // Precomputed IDF for this term
  }
  ```

- [ ] **2.2 Implement `BM25SparseMatrix` struct**
  ```rust
  pub struct BM25SparseMatrix {
      // Struct-of-arrays for cache efficiency
      pub term_rows: Vec<TermRow>,      // One per vocabulary term
      pub doc_indices: Vec<u32>,         // Column indices (doc IDs)
      pub term_freqs: Vec<f32>,          // Term frequencies

      // BM25 precomputed values (per document)
      pub doc_norms: Vec<f32>,           // k1 * (1 - b + b * doc_len / avg_len)

      // Metadata
      pub num_docs: u32,
      pub num_terms: u32,
      pub k1: f32,
      pub b: f32,
      pub k1_plus_1: f32,                // Precomputed k1 + 1
  }
  ```

- [ ] **2.3 Implement `from_scipy_csr()` constructor**
  - Accept numpy arrays: `indptr`, `indices`, `data`, `doc_lengths`, `idf_values`
  - Convert SciPy CSR format to optimized internal format
  - Precompute `doc_norms[d] = k1 * (1 - b + b * len[d] / avg_len)`
  - Precompute `k1_plus_1 = k1 + 1`

- [ ] **2.4 Add unit tests for data structure conversion**
  - Verify CSR conversion preserves all non-zero entries
  - Verify precomputed values match Python computation

### Details

**Key optimization:** Precomputing `doc_norms` at index build time transforms the BM25 formula:

**Before (6 ops per term-doc):**
```
score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len / avg))
```

**After (3 ops per term-doc):**
```
score = idf * (tf * k1_plus_1) / (tf + doc_norms[d])
```

This saves 50% of arithmetic operations in the innermost loop.

---

## Phase 3: BM25 Scoring Kernel

### Checklist

- [ ] **3.1 Implement sequential scorer** (`bm25_scorer.rs`)
  ```rust
  pub fn score_single_query(
      matrix: &BM25SparseMatrix,
      term_ids: &[u32],
      scores: &mut [f32],
  ) {
      // Zero scores
      scores.fill(0.0);

      // For each query term
      for &term_id in term_ids {
          let row = &matrix.term_rows[term_id as usize];
          let start = row.start as usize;
          let end = start + row.len as usize;

          // Score all docs containing this term
          for j in start..end {
              let doc_idx = matrix.doc_indices[j] as usize;
              let tf = matrix.term_freqs[j];
              let norm = matrix.doc_norms[doc_idx];

              scores[doc_idx] += row.idf * (tf * matrix.k1_plus_1) / (tf + norm);
          }
      }
  }
  ```

- [ ] **3.2 Add correctness tests**
  - Generate test cases with known BM25 scores
  - Compare Rust output to Python/Numba output within 1e-5 tolerance

- [ ] **3.3 Implement SIMD-optimized scorer** (after correctness verified)
  ```rust
  use std::simd::{f32x8, Simd};

  // Process 8 term-doc pairs at once
  // - Load 8 TFs into f32x8
  // - Gather 8 doc_norms (non-contiguous access)
  // - Vectorized: idf * (tf * k1_plus_1) / (tf + norms)
  // - Scatter-add to scores
  ```

- [ ] **3.4 Benchmark SIMD vs scalar**
  - Use Criterion.rs for micro-benchmarks
  - Measure on various query lengths (1, 3, 5, 10 terms)
  - Measure on various posting list lengths

### Details

**SIMD considerations:**
- The scatter-add to `scores[]` is the bottleneck (non-contiguous writes)
- Modern CPUs handle gather/scatter, but it's slower than contiguous SIMD
- Alternative: Sort postings by doc_id first, then SIMD score contiguous chunks
- Trade-off: Sorting overhead vs SIMD benefit (benchmark both)

---

## Phase 4: Top-K Selection

### Checklist

- [ ] **4.1 Implement partial sort** (`top_k.rs`)
  ```rust
  pub fn top_k_partial_sort(
      scores: &[f32],
      k: usize,
  ) -> Vec<(u32, f32)> {
      // Filter non-zero scores (candidates only)
      let mut candidates: Vec<(u32, f32)> = scores.iter()
          .enumerate()
          .filter(|(_, &s)| s > 0.0)
          .map(|(i, &s)| (i as u32, s))
          .collect();

      if candidates.len() <= k {
          candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
          return candidates;
      }

      // Partition around k-th element (O(n) average)
      let k_idx = candidates.len() - k;
      candidates.select_nth_unstable_by(k_idx, |a, b| a.1.partial_cmp(&b.1).unwrap());

      // Sort only top-k
      let mut top_k = candidates[k_idx..].to_vec();
      top_k.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
      top_k
  }
  ```

- [ ] **4.2 Add tests for top-k**
  - Test with k > candidates (return all)
  - Test with k = 1 (find max)
  - Test with ties in scores
  - Verify ordering is descending by score

- [ ] **4.3 Benchmark against NumPy argpartition**
  - Compare Rust `select_nth_unstable` vs Python `np.argpartition`

---

## Phase 5: PyO3 Bindings

### Checklist

- [ ] **5.1 Implement `RustBM25Index` Python class** (`lib.rs`)
  ```rust
  #[pyclass]
  pub struct RustBM25Index {
      inner: BM25SparseMatrix,
  }

  #[pymethods]
  impl RustBM25Index {
      #[new]
      fn new(
          indptr: PyReadonlyArray1<i64>,
          indices: PyReadonlyArray1<i32>,
          data: PyReadonlyArray1<f32>,
          doc_lengths: PyReadonlyArray1<i32>,
          idf_values: PyReadonlyArray1<f32>,
          k1: f32,
          b: f32,
      ) -> PyResult<Self>;

      /// Return raw scores for all documents
      fn score_query<'py>(
          &self,
          py: Python<'py>,
          term_ids: PyReadonlyArray1<i64>,
      ) -> PyResult<Bound<'py, PyArray1<f32>>>;

      /// Return top-k (doc_id, score) pairs
      fn search(
          &self,
          term_ids: PyReadonlyArray1<i64>,
          top_k: usize,
      ) -> PyResult<Vec<(u32, f32)>>;

      /// Batch search for multiple queries
      fn search_batch(
          &self,
          queries: Vec<PyReadonlyArray1<i64>>,
          top_k: usize,
      ) -> PyResult<Vec<Vec<(u32, f32)>>>;
  }
  ```

- [ ] **5.2 Create type stubs** (`vajra_bm25/_rust.pyi`)
  ```python
  from typing import List, Tuple
  import numpy as np
  from numpy.typing import NDArray

  class RustBM25Index:
      def __init__(
          self,
          indptr: NDArray[np.int64],
          indices: NDArray[np.int32],
          data: NDArray[np.float32],
          doc_lengths: NDArray[np.int32],
          idf_values: NDArray[np.float32],
          k1: float,
          b: float,
      ) -> None: ...

      def score_query(self, term_ids: NDArray[np.int64]) -> NDArray[np.float32]: ...
      def search(self, term_ids: NDArray[np.int64], top_k: int) -> List[Tuple[int, float]]: ...
      def search_batch(self, queries: List[NDArray[np.int64]], top_k: int) -> List[List[Tuple[int, float]]]: ...
  ```

- [ ] **5.3 Test Python bindings**
  - Verify type conversions work correctly
  - Test with empty queries, single term, many terms
  - Test error handling (invalid term IDs, negative k)

---

## Phase 6: Python Integration

### Checklist

- [ ] **6.1 Update `vajra_bm25/optimized.py`**
  ```python
  # Add at top of file
  try:
      from vajra_bm25.vajra_bm25_core import RustBM25Index
      RUST_AVAILABLE = True
  except ImportError:
      RUST_AVAILABLE = False
      RustBM25Index = None
  ```

- [ ] **6.2 Add Rust backend to `SparseBM25Scorer`**
  ```python
  class SparseBM25Scorer:
      def __init__(self, index: VectorizedIndexSparse, k1=1.5, b=0.75):
          self.index = index
          self.k1 = k1
          self.b = b
          self._rust_index = None

          if RUST_AVAILABLE:
              self._init_rust_backend()

      def _init_rust_backend(self):
          try:
              csr = self.index.term_doc_matrix
              self._rust_index = RustBM25Index(
                  indptr=csr.indptr.astype(np.int64),
                  indices=csr.indices.astype(np.int32),
                  data=csr.data.astype(np.float32),
                  doc_lengths=self.index.doc_lengths.astype(np.int32),
                  idf_values=self.index.idf_cache.astype(np.float32),
                  k1=self.k1,
                  b=self.b,
              )
              logger.info("Rust BM25 backend initialized")
          except Exception as e:
              logger.warning(f"Failed to initialize Rust backend: {e}")
              self._rust_index = None
  ```

- [ ] **6.3 Update `score_batch()` to use Rust**
  ```python
  def score_batch(self, query_terms, doc_mask):
      term_ids = [self.index.term_to_id[t] for t in query_terms
                  if t in self.index.term_to_id]

      if not term_ids:
          return np.zeros(self.index.num_docs, dtype=np.float32)

      # Use Rust if available
      if self._rust_index is not None:
          term_ids_arr = np.array(term_ids, dtype=np.int64)
          return self._rust_index.score_query(term_ids_arr)

      # Fallback to Numba/NumPy
      return self._score_batch_python(term_ids, doc_mask)
  ```

- [ ] **6.4 Update `VajraSearchParallel` to use Rust batch API**
  - Modify `search_batch()` to call `_rust_index.search_batch()`
  - Keep ThreadPoolExecutor fallback for non-Rust path

- [ ] **6.5 Add `use_rust` parameter to constructors**
  - Allow explicit enable/disable of Rust backend
  - Default: auto-detect (use if available)

- [ ] **6.6 Update logging to indicate backend**
  - Log which backend is active (Rust/Numba/NumPy)
  - Include backend in benchmark output

---

## Phase 7: Build System Overhaul

### Checklist

- [ ] **7.1 Update `pyproject.toml` for maturin**
  ```toml
  [build-system]
  requires = ["maturin>=1.4,<2.0"]
  build-backend = "maturin"

  [project]
  name = "vajra-bm25"
  version = "0.3.0"
  # ... existing metadata ...

  [project.optional-dependencies]
  optimized = ["numpy>=1.20.0", "scipy>=1.7.0"]
  persistence = ["joblib>=1.0.0"]
  rust = []  # Rust extension included in wheel
  dev = [
      "pytest>=7.0",
      "pytest-cov>=4.0",
      "rank-bm25>=0.2.2",
      "maturin>=1.4",
  ]
  all = [
      "numpy>=1.20.0",
      "scipy>=1.7.0",
      "joblib>=1.0.0",
  ]

  [tool.maturin]
  python-source = "."
  module-name = "vajra_bm25.vajra_bm25_core"
  manifest-path = "rust/Cargo.toml"
  features = ["pyo3/extension-module"]
  strip = true
  ```

- [ ] **7.2 Create `Makefile` for common operations**
  ```makefile
  .PHONY: dev build test bench clean

  dev:
  	maturin develop --release

  build:
  	maturin build --release

  test:
  	pytest tests/ -v

  bench:
  	python benchmarks/profile_query_latency.py

  clean:
  	rm -rf target/ dist/ *.egg-info
  ```

- [ ] **7.3 Update `.gitignore`**
  ```
  # Rust
  target/
  Cargo.lock

  # Maturin
  *.so
  *.pyd
  *.dll
  ```

- [ ] **7.4 Create `rust-toolchain.toml`**
  ```toml
  [toolchain]
  channel = "stable"
  components = ["rustfmt", "clippy"]
  ```

- [ ] **7.5 Update README with new installation instructions**
  - Document Rust backend as optional enhancement
  - Explain fallback behavior (Numba → NumPy)
  - Add performance comparison table

---

## Phase 8: CI/CD for Multi-Platform Wheels

### Checklist

- [ ] **8.1 Create `.github/workflows/build.yml`**
  ```yaml
  name: Build and Test

  on:
    push:
      branches: [main, feature/rust-backend]
    pull_request:
      branches: [main]

  jobs:
    test-python:
      runs-on: ubuntu-latest
      strategy:
        matrix:
          python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ matrix.python-version }}
        - uses: dtolnay/rust-toolchain@stable
        - name: Build and test
          run: |
            pip install maturin pytest numpy scipy
            maturin develop --release
            pytest tests/ -v

    build-wheels:
      runs-on: ${{ matrix.os }}
      strategy:
        matrix:
          os: [ubuntu-latest, macos-latest, windows-latest]
          python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: ${{ matrix.python-version }}
        - uses: dtolnay/rust-toolchain@stable
        - name: Build wheel
          run: |
            pip install maturin
            maturin build --release
        - uses: actions/upload-artifact@v4
          with:
            name: wheel-${{ matrix.os }}-${{ matrix.python-version }}
            path: target/wheels/*.whl
  ```

- [ ] **8.2 Create `.github/workflows/release.yml`**
  ```yaml
  name: Release

  on:
    push:
      tags: ['v*']

  jobs:
    build-wheels:
      # ... same as above but with PyPI upload

    publish:
      needs: build-wheels
      runs-on: ubuntu-latest
      steps:
        - uses: actions/download-artifact@v4
        - name: Publish to PyPI
          uses: pypa/gh-action-pypi-publish@release/v1
          with:
            packages-dir: wheels/
  ```

- [ ] **8.3 Test wheel builds locally**
  ```bash
  # Build for current platform
  maturin build --release

  # Install and verify
  pip install target/wheels/vajra_bm25-*.whl
  python -c "from vajra_bm25 import VajraSearchOptimized; print('OK')"
  ```

- [ ] **8.4 Create sdist (source distribution)**
  - For users who want to compile from source
  - Requires Rust toolchain on user machine

---

## Phase 9: Testing and Validation

### Checklist

- [ ] **9.1 Create `tests/test_rust_backend.py`**
  ```python
  import pytest
  import numpy as np
  from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

  @pytest.fixture
  def corpus_10k():
      # Load or generate 10K doc corpus
      pass

  def test_rust_python_score_equivalence(corpus_10k):
      """Verify Rust produces identical scores to Python."""
      engine = VajraSearchOptimized(corpus_10k)

      # Get Python scores (disable Rust)
      engine.scorer._rust_index = None
      py_scores = engine.scorer.score_batch(["test", "query"], None)

      # Get Rust scores
      engine.scorer._init_rust_backend()
      rust_scores = engine.scorer.score_batch(["test", "query"], None)

      # Compare
      np.testing.assert_allclose(py_scores, rust_scores, rtol=1e-5)

  def test_rust_search_results_match(corpus_10k):
      """Verify search results match between backends."""
      pass

  def test_rust_batch_search(corpus_10k):
      """Verify batch search produces correct results."""
      pass

  def test_rust_fallback_when_unavailable():
      """Verify graceful fallback when Rust is not available."""
      pass
  ```

- [ ] **9.2 Add benchmark comparisons**
  - Update `benchmarks/profile_query_latency.py`
  - Add Rust vs Numba vs NumPy comparison
  - Generate comparison charts

- [ ] **9.3 Run full BEIR benchmark suite**
  - Verify ranking quality unchanged (NDCG, Recall)
  - Measure speedup on standard datasets

- [ ] **9.4 Memory profiling**
  - Compare memory usage: Rust index vs Python index
  - Profile on 1M doc corpus

---

## Phase 10: Memory-Mapped Indices (1M+ Scale)

### Checklist

- [ ] **10.1 Design serialization format**
  ```
  index.vajra (binary file)
  ├── Header (32 bytes)
  │   ├── magic: u32 = 0x56414A52 ("VAJR")
  │   ├── version: u32 = 1
  │   ├── num_terms: u32
  │   ├── num_docs: u32
  │   ├── k1: f32
  │   ├── b: f32
  │   └── offsets for each section
  ├── TermRows section (12 bytes × num_terms)
  ├── DocIndices section (4 bytes × nnz)
  ├── TermFreqs section (4 bytes × nnz)
  └── DocNorms section (4 bytes × num_docs)
  ```

- [ ] **10.2 Implement `save()` and `load()` in Rust**
  - Use memory-mapping for large sections
  - Keep TermRows in memory (small, frequently accessed)

- [ ] **10.3 Add Python API for persistence**
  ```python
  # Save
  engine = VajraSearchOptimized(corpus)
  engine.save("index.vajra")

  # Load (memory-mapped, instant)
  engine = VajraSearchOptimized.load("index.vajra")
  ```

- [ ] **10.4 Benchmark load times**
  - Compare: pickle vs joblib vs mmap
  - Target: <1s load time for 1M doc index

---

## Phase 11: Documentation and Release

### Checklist

- [ ] **11.1 Update README.md**
  - New performance numbers
  - Rust backend documentation
  - Installation options (with/without Rust)

- [ ] **11.2 Update CHANGELOG.md**
  ```markdown
  ## [0.3.0] - 2024-XX-XX

  ### Added
  - Rust backend for 6-10x faster query performance
  - Memory-mapped indices for 1M+ document scale
  - Multi-platform wheel builds (Linux, macOS, Windows)

  ### Changed
  - Build system switched from setuptools to maturin
  - Minimum Python version remains 3.8
  ```

- [ ] **11.3 Create GitHub release**
  - Tag: `v0.3.0`
  - Attach wheel artifacts
  - Release notes

- [ ] **11.4 Publish to PyPI**
  ```bash
  # Automated via GitHub Actions on tag push
  git tag v0.3.0
  git push origin v0.3.0
  ```

---

## Files Summary

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Switch to maturin, bump to 0.3.0 |
| `vajra_bm25/optimized.py` | Add Rust backend integration |
| `vajra_bm25/parallel.py` | Use Rust batch API |
| `benchmarks/profile_query_latency.py` | Add Rust comparison |
| `.gitignore` | Add Rust/maturin patterns |
| `README.md` | Update docs and benchmarks |

### Files to Create

| File | Purpose |
|------|---------|
| `rust/Cargo.toml` | Rust crate configuration |
| `rust/src/lib.rs` | PyO3 module entry point |
| `rust/src/sparse_matrix.rs` | BM25SparseMatrix data structure |
| `rust/src/bm25_scorer.rs` | SIMD-optimized BM25 scoring |
| `rust/src/top_k.rs` | Floyd-Rivest partial sort |
| `rust/src/batch.rs` | Rayon parallel batch processing |
| `rust/benches/bm25_bench.rs` | Criterion benchmarks |
| `rust/rust-toolchain.toml` | Rust version pinning |
| `vajra_bm25/_rust.pyi` | Type stubs for Rust module |
| `tests/test_rust_backend.py` | Correctness tests |
| `.github/workflows/build.yml` | CI for testing |
| `.github/workflows/release.yml` | CD for PyPI publishing |
| `Makefile` | Development commands |

---

## Expected Performance

| Optimization | Single Query | Batch (100 queries) |
|--------------|-------------|---------------------|
| Current (Numba) | 0.44ms | 35ms |
| + Rust baseline | 0.20ms (2x) | 15ms (2x) |
| + SIMD | 0.08ms (5x) | 6ms (6x) |
| + Precomputation | 0.06ms (7x) | 4.5ms (8x) |
| + Rayon parallel | 0.06ms | 1.5ms (23x) |

**Conservative target: 6-10x single query, 15-20x batch throughput**

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SIMD portability | Use `std::simd` (portable) or feature-gate platform-specific intrinsics |
| Build complexity | Provide pre-built wheels; fallback to Python if Rust unavailable |
| Correctness regression | Extensive test suite comparing Rust to Python output |
| Memory usage increase | Monitor with benchmarks; mmap for large indices |
| CI/CD failures | Test on all platforms before release; use matrix builds |
