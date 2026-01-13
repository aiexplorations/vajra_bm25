# Quick Start

This guide will get you searching in under 5 minutes.

## Basic Usage

### Creating Documents

```python
from vajra_bm25 import Document, DocumentCorpus, VajraSearch

# Create documents manually
documents = [
    Document(id="1", title="Introduction to ML", content="Machine learning is..."),
    Document(id="2", title="Deep Learning", content="Neural networks are..."),
    Document(id="3", title="NLP Basics", content="Natural language processing..."),
]

corpus = DocumentCorpus(documents)
```

### Building the Index

```python
# Create search engine (builds index automatically)
engine = VajraSearch(corpus)
```

### Searching

```python
results = engine.search("machine learning neural networks", top_k=5)

for r in results:
    print(f"{r.rank}. {r.document.title}")
    print(f"   Score: {r.score:.3f}")
    print(f"   Content: {r.document.content[:100]}...")
    print()
```

## Loading from Files

### JSONL Files

```python
from vajra_bm25 import DocumentCorpus, VajraSearchOptimized

# Load from JSONL
corpus = DocumentCorpus.load_jsonl("documents.jsonl")

# Use optimized engine for better performance
engine = VajraSearchOptimized(corpus)
results = engine.search("your query here", top_k=10)
```

JSONL format:
```jsonl
{"id": "doc1", "title": "First Doc", "content": "Content here..."}
{"id": "doc2", "title": "Second Doc", "content": "More content..."}
```

### PDF Files

```python
# Single PDF
corpus = DocumentCorpus.load_pdf("research_paper.pdf")

# Directory of PDFs
corpus = DocumentCorpus.load_pdf_directory("./papers/")

# Auto-detect format
corpus = DocumentCorpus.load("./my_data/")  # Works with JSONL, PDF, or directories
```

## Search Results

Each result contains:

```python
@dataclass
class SearchResult:
    document: Document  # The matched document
    score: float        # BM25 relevance score (higher = more relevant)
    rank: int           # Position in results (1-indexed)
```

## Next Steps

- [CLI Usage](cli.md) - Interactive command-line search
- [Search Engines](../guide/engines.md) - Choose the right engine for your use case
- [Performance Tips](../guide/performance.md) - Optimize for production
