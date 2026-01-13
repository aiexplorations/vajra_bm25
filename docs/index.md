# Vajra BM25

**Vajra** (Sanskrit: वज्र, "thunderbolt") is a high-performance BM25 search engine built with Category Theory abstractions.

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Fast__

    ---

    **1.3-1.6x faster** than BM25S with sub-4ms latency at 1M documents

-   :material-file-document-multiple:{ .lg .middle } __Flexible Formats__

    ---

    Index **JSONL** and **PDF** documents out of the box

-   :material-api:{ .lg .middle } __Clean API__

    ---

    Simple Python API with **categorical abstractions** for extensibility

-   :material-console:{ .lg .middle } __Interactive CLI__

    ---

    Rich command-line interface for exploring search

</div>

## Quick Example

```python
from vajra_bm25 import DocumentCorpus, VajraSearchOptimized

# Load documents (JSONL, PDF, or directory)
corpus = DocumentCorpus.load("./my_documents/")

# Build search index
engine = VajraSearchOptimized(corpus)

# Search
results = engine.search("machine learning algorithms", top_k=10)

for r in results:
    print(f"{r.rank}. {r.document.title} (score: {r.score:.3f})")
```

## Installation

```bash
pip install vajra-bm25[all]
```

Or install specific features:

```bash
pip install vajra-bm25              # Basic (zero dependencies)
pip install vajra-bm25[optimized]   # NumPy/SciPy optimizations
pip install vajra-bm25[pdf]         # PDF support
pip install vajra-bm25[cli]         # Interactive CLI
```

## Performance

Benchmarked on Wikipedia (1M documents, 500 queries):

| Engine | Build Time | Latency | QPS |
|--------|------------|---------|-----|
| **Vajra** | 17.0 min | **3.40ms** | **294** |
| BM25S | 11.3 min | 5.44ms | 184 |

See [Benchmarks](benchmarks.md) for detailed results.

## Why Category Theory?

Vajra uses categorical abstractions to organize code:

- **Morphisms**: Composable scoring functions `(Query, Doc) → Score`
- **Coalgebras**: Search as state unfolding `State → List[Result]`
- **Functors**: Container transformations for multi-result semantics

These abstractions don't make Vajra fast (NumPy and sparse matrices do), but they provide a clean, extensible structure. Learn more in [Category Theory](category-theory.md).

## License

MIT License - see [LICENSE](https://github.com/aiexplorations/vajra_bm25/blob/main/LICENSE) for details.
