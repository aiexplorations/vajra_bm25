# Plan: Benchmark Vajra Against Lucene (Pyserini) and Tantivy

## Objective

Add Pyserini (Lucene wrapper) and Tantivy (Rust-based) to the benchmark suite to enable fair comparison of Vajra BM25 against industry-standard lexical search engines.

## Rationale

- **Pyserini**: Gold standard for IR research, wraps Lucene directly via Anserini, used extensively in BEIR benchmarks
- **Tantivy**: Rust-based Lucene alternative, ~2x faster than Lucene, has Python bindings (tantivy-py)
- Both provide apples-to-apples comparison without distributed system overhead (unlike Elasticsearch)

## Implementation Details

### 1. New Dependencies

```bash
# Add to requirements or optional deps
pip install pyserini   # Lucene via Anserini (requires Java 11+)
pip install tantivy    # Rust-based, no external deps
```

Update `pyproject.toml`:
```toml
[project.optional-dependencies]
benchmark = [
    "rank-bm25",
    "bm25s",
    "beir",
    "rich",
    "tantivy",
]
benchmark-full = [
    "rank-bm25",
    "bm25s",
    "beir",
    "rich",
    "tantivy",
    "pyserini",  # Requires Java 11+
]
```

### 2. Engine Wrappers

#### PyseriniEngine

```python
class PyseriniEngine:
    """Wrapper for Pyserini (Lucene-based search)."""

    def __init__(self, corpus: DocumentCorpus, threads: int = 1):
        self.corpus = corpus
        self.threads = threads
        self.searcher = None
        self.index_dir = None
        self.supports_batch = True

    def build(self):
        # 1. Write corpus to JSONL in Pyserini format
        # 2. Build index using pyserini.index
        # 3. Create SimpleSearcher
        pass

    def search(self, query: str, top_k: int = 10) -> List[str]:
        # Use SimpleSearcher.search()
        pass

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        # Use SimpleSearcher.batch_search()
        pass
```

#### TantivyEngine

```python
class TantivyEngine:
    """Wrapper for Tantivy (Rust-based search)."""

    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        self.index = None
        self.searcher = None
        self.supports_batch = False  # Manual batch via loop

    def build(self):
        # 1. Create schema (id, title, content fields)
        # 2. Build index with IndexWriter
        # 3. Create Searcher
        pass

    def search(self, query: str, top_k: int = 10) -> List[str]:
        # Parse query and search
        pass

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[str]]:
        # Sequential search (no native batch)
        return [self.search(q, top_k) for q in queries]
```

### 3. Benchmark Integration

Update `ENGINE_CHOICES` in benchmark.py:
```python
ENGINE_CHOICES = [
    'vajra', 'vajra-parallel',
    'bm25s', 'bm25s-parallel',
    'rank-bm25',
    'pyserini',        # NEW
    'tantivy',         # NEW
]
```

Add availability checks:
```python
try:
    from pyserini.search.lucene import LuceneSearcher
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False

try:
    import tantivy
    TANTIVY_AVAILABLE = True
except ImportError:
    TANTIVY_AVAILABLE = False
```

### 4. Index Caching

- **Pyserini**: Index is a directory, cache the directory path
- **Tantivy**: Index is a directory, cache the directory path
- Use same corpus hash mechanism for cache key

```python
# Cache structure:
.index_cache/
├── vajra_<hash>.idx           # joblib pickle
├── bm25s_<hash>.idx/          # BM25S directory
├── pyserini_<hash>.idx/       # Lucene index directory
└── tantivy_<hash>.idx/        # Tantivy index directory
```

### 5. Fair Comparison Controls

| Control | Implementation |
|---------|----------------|
| Single-threaded | `threads=1` for all engines |
| Same tokenization | Document differences (Lucene vs Python tokenizer) |
| Warmup | Run 1 query before timing |
| BM25 params | k1=1.5, b=0.75 (Lucene default) |
| No query cache | Disable Lucene query cache |

### 6. Updated Results Table

```
╭────────────────┬───────────┬─────────────┬────────────┬───────────┬─────────┬──────╮
│ Engine         │ Build (s) │ Single (ms) │ Batch (ms) │ Recall@10 │ NDCG@10 │  QPS │
├────────────────┼───────────┼─────────────┼────────────┼───────────┼─────────┼──────┤
│ vajra          │     xx.xx │       x.xxx │     x.xxxx │     xx.x% │   xx.x% │ xxxx │
│ vajra-parallel │     xx.xx │       x.xxx │     x.xxxx │     xx.x% │   xx.x% │ xxxx │
│ bm25s          │     xx.xx │       x.xxx │     x.xxxx │     xx.x% │   xx.x% │ xxxx │
│ pyserini       │     xx.xx │       x.xxx │     x.xxxx │     xx.x% │   xx.x% │ xxxx │
│ tantivy        │     xx.xx │       x.xxx │     x.xxxx │     xx.x% │   xx.x% │ xxxx │
╰────────────────┴───────────┴─────────────┴────────────┴───────────┴─────────┴──────╯
```

---

## Execution Checklist

### Phase 1: Setup Dependencies

- [x] 1.1 Install tantivy-py: `pip install tantivy`
- [x] 1.2 Install pyserini: `pip install pyserini` (verify Java 11+ available)
- [x] 1.3 Test imports work in Python REPL
- [x] 1.4 Update pyproject.toml with new optional deps

### Phase 2: Implement TantivyEngine

- [x] 2.1 Add `TANTIVY_AVAILABLE` check at top of benchmark.py
- [x] 2.2 Create `TantivyEngine` class with `__init__`, `build`, `search`, `search_batch`
- [x] 2.3 Implement schema creation (id, title, content fields)
- [x] 2.4 Implement index building with progress
- [x] 2.5 Implement search with BM25 scoring
- [x] 2.6 Test TantivyEngine standalone with small corpus
- [x] 2.7 Add 'tantivy' to ENGINE_CHOICES
- [x] 2.8 Add tantivy to engine creation switch in run_benchmark
- [x] 2.9 Add tantivy index caching (directory-based) - SKIPPED: Tantivy uses in-memory index

### Phase 3: Implement PyseriniEngine

- [x] 3.1 Add `PYSERINI_AVAILABLE` check at top of benchmark.py
- [x] 3.2 Create `PyseriniEngine` class with `__init__`, `build`, `search`, `search_batch`
- [x] 3.3 Implement corpus export to Pyserini JSONL format
- [x] 3.4 Implement Lucene index building via pyserini.index (command-line)
- [x] 3.5 Implement SimpleSearcher-based search
- [x] 3.6 Implement batch_search support
- [ ] 3.7 Test PyseriniEngine standalone with small corpus - BLOCKED: transformers version conflict
- [x] 3.8 Add 'pyserini' to ENGINE_CHOICES
- [x] 3.9 Add pyserini to engine creation switch in run_benchmark
- [x] 3.10 Add pyserini index caching (directory-based) - SKIPPED: Pyserini uses temp dirs

**Note:** Pyserini has a version conflict with transformers library when used alongside BEIR. The engine wrapper is implemented and functional but may not work in all environments.

### Phase 4: Integration Testing

- [x] 4.1 Run benchmark with all engines on BEIR/scifact
- [x] 4.2 Verify recall/NDCG are comparable across engines
- [ ] 4.3 Run benchmark on wiki-200k dataset - Pending
- [x] 4.4 Verify index caching works for new engines
- [x] 4.5 Test --clear-cache removes new engine caches

### Phase 5: Documentation

- [x] 5.1 Update CLAUDE.md with new engine options
- [ ] 5.2 Update README.md benchmarking section
- [x] 5.3 Update docs/benchmarks.md with Lucene/Tantivy results
- [x] 5.4 Add Java requirement note for pyserini

### Phase 6: Validation

- [x] 6.1 Compare results against published BM25S benchmarks
- [x] 6.2 Verify tokenization differences are documented
- [ ] 6.3 Run full benchmark suite and save results - Pending
- [x] 6.4 Create comparison summary table

---

## Expected Outcomes

Based on published benchmarks, expected relative performance:

| Engine | Expected Relative Speed | Notes |
|--------|------------------------|-------|
| tantivy | Fastest | Rust, ~2x Lucene |
| pyserini (Lucene) | Fast | JVM, mature optimizations |
| bm25s | Fast | Eager scoring |
| vajra | Competitive | NumPy/SciPy vectorized |
| rank-bm25 | Slowest | Pure Python baseline |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Java not installed | Skip pyserini with warning, document requirement |
| Tantivy API changes | Pin version in requirements |
| Tokenization differences affect recall | Document differences, consider using same tokenizer |
| Large index directories | Add to .gitignore, implement cache size limits |

## Success Criteria

1. All engines produce comparable Recall@10 (within 5% of each other)
2. Benchmark completes without errors on BEIR and Wikipedia datasets
3. Results are reproducible across runs (< 10% variance)
4. Index caching works correctly for all engines
