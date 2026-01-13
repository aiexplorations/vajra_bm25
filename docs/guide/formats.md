# Document Formats

Vajra supports multiple document formats for flexible data ingestion.

## JSONL Format

The primary format for structured documents.

### File Structure

Each line is a JSON object with required fields:

```jsonl
{"id": "doc1", "title": "Document Title", "content": "Full text content here"}
{"id": "doc2", "title": "Another Doc", "content": "More content", "metadata": {"author": "Jane"}}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique document identifier |
| `title` | Yes | Document title |
| `content` | Yes | Full text content (searchable) |
| `metadata` | No | Optional dictionary of metadata |

### Loading JSONL

```python
from vajra_bm25 import DocumentCorpus

# Load from file
corpus = DocumentCorpus.load_jsonl("documents.jsonl")

# Save corpus to file
corpus.save_jsonl("output.jsonl")
```

## PDF Format

Index and search PDF documents directly.

!!! note "Installation"
    PDF support requires: `pip install vajra-bm25[pdf]`

### Single PDF

```python
corpus = DocumentCorpus.load_pdf("research_paper.pdf")
```

The document will have:

- `id`: Filename without extension (e.g., "research_paper")
- `title`: PDF title metadata or filename
- `content`: Extracted text from all pages
- `metadata`: `{"source": "path", "format": "pdf", "pages": N, "author": "..."}`

### Custom Document ID

```python
corpus = DocumentCorpus.load_pdf("paper.pdf", doc_id="custom_id")
```

### Directory of PDFs

```python
# All PDFs in directory
corpus = DocumentCorpus.load_pdf_directory("./papers/")

# Recursive (include subdirectories)
corpus = DocumentCorpus.load_pdf_directory("./papers/", recursive=True)
```

### PDF Metadata

Vajra automatically extracts PDF metadata:

```python
corpus = DocumentCorpus.load_pdf("paper.pdf")
doc = corpus.documents[0]

print(doc.metadata)
# {
#     "source": "/path/to/paper.pdf",
#     "format": "pdf",
#     "pages": 12,
#     "author": "John Smith"
# }
```

## Auto-Detection

The `load()` method automatically detects format:

```python
# Auto-detects based on path
corpus = DocumentCorpus.load("data.jsonl")      # JSONL file
corpus = DocumentCorpus.load("paper.pdf")        # PDF file
corpus = DocumentCorpus.load("./papers/")        # Directory of PDFs
```

### Explicit Format

Override auto-detection with the `format` parameter:

```python
# Force JSONL parsing on non-standard extension
corpus = DocumentCorpus.load("data.txt", format="jsonl")

# Force PDF directory mode
corpus = DocumentCorpus.load("./mixed_folder/", format="pdf_dir")
```

## Creating Documents Programmatically

```python
from vajra_bm25 import Document, DocumentCorpus

# Create individual documents
doc1 = Document(
    id="unique_id",
    title="My Document",
    content="This is the searchable content...",
    metadata={"category": "tutorial", "year": 2024}
)

# Build corpus
corpus = DocumentCorpus([doc1, doc2, doc3])

# Or add incrementally
corpus = DocumentCorpus()
corpus.add(doc1)
corpus.add(doc2)
```

## Best Practices

1. **Use unique IDs**: Ensure each document has a unique `id`
2. **Content quality**: Put searchable text in `content`, not `title`
3. **PDF text extraction**: Works best with text-based PDFs (not scanned images)
4. **Large corpora**: Use JSONL for faster loading vs many individual PDFs
