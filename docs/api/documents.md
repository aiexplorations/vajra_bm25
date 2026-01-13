# Documents API

Core data structures for documents and corpora.

## Document

An immutable document object.

```python
from vajra_bm25 import Document

doc = Document(
    id="unique_id",
    title="Document Title",
    content="The searchable content of the document...",
    metadata={"author": "Jane Doe", "year": 2024}
)
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique document identifier |
| `title` | `str` | Document title |
| `content` | `str` | Full text content (searchable) |
| `metadata` | `Optional[Dict[str, Any]]` | Optional metadata dictionary |

### Methods

#### `to_dict() -> Dict[str, Any]`

Serialize document to a dictionary.

```python
data = doc.to_dict()
# {'id': 'unique_id', 'title': '...', 'content': '...', 'metadata': {...}}
```

#### `Document.from_dict(data: Dict) -> Document`

Create a document from a dictionary.

```python
doc = Document.from_dict({
    "id": "doc1",
    "title": "My Doc",
    "content": "Content here"
})
```

### Immutability

Documents are frozen dataclasses (immutable). This preserves identity through transformations and enables safe caching.

```python
doc.title = "New Title"  # Raises FrozenInstanceError
```

## DocumentCorpus

A collection of documents with persistence support.

```python
from vajra_bm25 import DocumentCorpus, Document

# Create empty corpus
corpus = DocumentCorpus()

# Create from list
corpus = DocumentCorpus([doc1, doc2, doc3])
```

### Methods

#### `add(doc: Document)`

Add a document to the corpus.

```python
corpus.add(Document(id="new", title="New Doc", content="..."))
```

#### `get(doc_id: str) -> Optional[Document]`

Retrieve a document by ID.

```python
doc = corpus.get("doc1")
if doc:
    print(doc.title)
```

#### `__len__() -> int`

Get corpus size.

```python
print(f"Corpus has {len(corpus)} documents")
```

#### `__iter__()`

Iterate over documents.

```python
for doc in corpus:
    print(doc.title)
```

### JSONL Persistence

#### `save_jsonl(filepath: Path)`

Save corpus to JSONL file.

```python
corpus.save_jsonl("my_corpus.jsonl")
```

#### `DocumentCorpus.load_jsonl(filepath: Path) -> DocumentCorpus`

Load corpus from JSONL file.

```python
corpus = DocumentCorpus.load_jsonl("my_corpus.jsonl")
```

### PDF Loading

!!! note "Installation"
    PDF support requires: `pip install vajra-bm25[pdf]`

#### `DocumentCorpus.load_pdf(filepath: Path, doc_id: Optional[str] = None) -> DocumentCorpus`

Load a single PDF as a document.

```python
corpus = DocumentCorpus.load_pdf("research_paper.pdf")

# Custom ID
corpus = DocumentCorpus.load_pdf("paper.pdf", doc_id="my_paper")
```

#### `DocumentCorpus.load_pdf_directory(dirpath: Path, recursive: bool = False) -> DocumentCorpus`

Load all PDFs from a directory.

```python
# Single directory
corpus = DocumentCorpus.load_pdf_directory("./papers/")

# Include subdirectories
corpus = DocumentCorpus.load_pdf_directory("./papers/", recursive=True)
```

### Auto-Detection

#### `DocumentCorpus.load(path: Path, format: Optional[str] = None) -> DocumentCorpus`

Load corpus with automatic format detection.

```python
# Auto-detect based on path
corpus = DocumentCorpus.load("data.jsonl")    # JSONL file
corpus = DocumentCorpus.load("paper.pdf")      # Single PDF
corpus = DocumentCorpus.load("./papers/")      # PDF directory

# Explicit format
corpus = DocumentCorpus.load("data.txt", format="jsonl")
corpus = DocumentCorpus.load("./folder/", format="pdf_dir")
```

| Format | Description |
|--------|-------------|
| `jsonl` | JSONL file |
| `pdf` | Single PDF file |
| `pdf_dir` | Directory of PDFs |

## Categorical Interpretation

In category theory terms:

- **Document** is an object in our category
- **DocumentCorpus** is a collection of objects
- **load/save** are morphisms between the file system and memory representations

```
File System ←→ Memory
    ↑            ↑
  load()      save()
    │            │
    └── Morphisms ─┘
```

Documents are immutable (frozen) because objects in a category should preserve their identity through transformations.
