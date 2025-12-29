# Vajra BM25: Use Cases and Integration Opportunities

## Executive Summary

This plan explores integration opportunities for Vajra BM25 across document stores, AI/ML pipelines, and application architectures. The research reveals three primary opportunity areas:

1. **Object Store Search Layer** - S3/MinIO have zero text search capability; Vajra fills this gap
2. **RAG Hybrid Retrieval** - BM25 + vector search is the emerging standard for LLM applications
3. **Lightweight Search Microservice** - Alternative to Elasticsearch for resource-constrained environments

## Research Sources

- [Milvus: Role of BM25 in Full-Text Search](https://milvus.io/ai-quick-reference/what-is-the-role-of-bm25-in-fulltext-search)
- [LlamaIndex BM25 Retriever Documentation](https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/)
- [LangChain BM25 Retriever](https://python.langchain.com/docs/integrations/retrievers/bm25/)
- [Optimizing RAG with Hybrid Search & Reranking](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking)
- [Hybrid Retrieval BM25 + FAISS in RAG](https://www.chitika.com/hybrid-retrieval-rag/)
- [MinIO + MeiliSearch Full Text Search](https://blog.min.io/master-full-text-search-with-meilisearch-on-minio/)
- [LangChain + OpenAI + S3 Loader](https://blog.min.io/langchain-openai-s3-loader/)
- [Meilisearch: Elasticsearch Alternatives](https://www.meilisearch.com/blog/elasticsearch-alternatives)
- [ZincSearch: Lightweight Elasticsearch Alternative](https://github.com/zincsearch/zincsearch)
- [MinerU: PDF to Markdown for LLM](https://github.com/opendatalab/MinerU)
- [FastAPI Microservice Patterns](https://python.plainenglish.io/fastapi-microservice-patterns-3052c1241019)

---

## Part 1: Market Landscape

### Document Database Search Capabilities

| System | Built-in Text Search | Technology | Vajra Opportunity |
|--------|---------------------|------------|-------------------|
| **MongoDB** | Yes | Text indexes, Atlas Search (Lucene) | Limited - already has search |
| **S3 / MinIO** | No | Object store only | **High** - zero search capability |
| **DynamoDB** | No | Key-value only | Medium - needs OpenSearch integration |
| **Firestore** | No | Field matching only | Medium - needs Algolia/Typesense |
| **CouchDB** | Limited | MapReduce views | Medium - no native BM25 |
| **Redis** | Yes | RediSearch module | Limited - already has search |
| **PostgreSQL** | Yes | pg_trgm, tsvector | Limited - decent full-text |

### Lightweight Search Engine Alternatives

| Engine | Language | RAM Usage | Python Support | Notes |
|--------|----------|-----------|----------------|-------|
| **Vajra** | Python | In-memory | Native | Category theory design, sub-ms latency |
| **Whoosh** | Python | In-memory | Native | Pure Python, no longer maintained |
| **Sonic** | Rust | ~30MB | Client lib | Fast, schema-less |
| **ZincSearch** | Go | <100MB | REST API | Elasticsearch-compatible API |
| **Meilisearch** | Rust | Low | Official SDK | Developer-friendly, typo tolerance |
| **Typesense** | C++ | In-memory | Client lib | Speed-focused, production-ready |

### Vajra's Competitive Position

**Strengths:**
- Pure Python (no external service)
- Sub-millisecond cached queries
- Category theory abstractions enable experimentation
- 180,000-800,000 QPS at scale

**Limitations:**
- Single-node, RAM-bound
- No real-time updates (rebuild index)
- No built-in persistence (joblib optional)

---

## Part 2: Integration Opportunities

### 2.1 Object Store Search Layer (S3/MinIO)

**Problem:** S3 and MinIO are pure object stores with no text search. Users must integrate external search systems.

**Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MinIO     │────▶│   Sync/ETL  │────▶│   Vajra     │
│  (storage)  │     │   Worker    │     │  (search)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │         ┌─────────────┐               │
       └────────▶│   FastAPI   │◀──────────────┘
                 │   Gateway   │
                 └─────────────┘
```

**Implementation Components:**

1. **vajra-minio-sync** - Background worker that:
   - Listens to MinIO bucket notifications
   - Extracts text from PDFs/docs using PyMuPDF or Unstructured
   - Updates Vajra index incrementally (or full rebuild)
   - Persists index to MinIO itself

2. **vajra-search-api** - FastAPI microservice that:
   - Exposes `/search` endpoint
   - Returns document IDs and presigned URLs for S3 objects
   - Supports filtering by bucket/prefix

**Code Sketch:**
```python
# vajra_minio_sync.py
from minio import Minio
from vajra_bm25 import VajraSearchOptimized, Document, DocumentCorpus
import pymupdf

def sync_bucket(client: Minio, bucket: str) -> VajraSearchOptimized:
    docs = []
    for obj in client.list_objects(bucket, recursive=True):
        if obj.object_name.endswith('.pdf'):
            data = client.get_object(bucket, obj.object_name)
            text = extract_pdf_text(data.read())
            docs.append(Document(id=obj.object_name, title=obj.object_name, content=text))

    corpus = DocumentCorpus(docs)
    return VajraSearchOptimized(corpus)
```

**Delivery:**
- PyPI package: `vajra-minio`
- Docker image: `ghcr.io/aiexplorations/vajra-minio-search`

---

### 2.2 RAG Hybrid Retrieval (LlamaIndex / LangChain)

**Problem:** Pure vector search misses exact keyword matches. BM25 + vector hybrid outperforms either alone.

**Current State:**
- LlamaIndex uses `llama-index-retrievers-bm25` (wraps rank-bm25)
- LangChain uses `BM25Retriever` (wraps rank-bm25)
- Both are slow on large corpora

**Opportunity:** Replace rank-bm25 with Vajra for 100-1000x speedup.

**Architecture:**
```
                    ┌─────────────────┐
                    │  Query Router   │
                    └────────┬────────┘
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
    ┌──────▼──────┐                    ┌───────▼───────┐
    │   Vajra     │                    │  Vector DB    │
    │  (BM25)     │                    │  (Semantic)   │
    └──────┬──────┘                    └───────┬───────┘
           │                                   │
           └─────────────────┬─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Score Fusion   │
                    │  (RRF / Linear) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Reranker      │
                    │  (Optional)     │
                    └─────────────────┘
```

**Implementation Components:**

1. **llama-index-retrievers-vajra** - Drop-in replacement for llama-index-retrievers-bm25
   ```python
   from llama_index.retrievers.vajra import VajraRetriever

   retriever = VajraRetriever.from_defaults(
       nodes=nodes,
       similarity_top_k=10
   )
   ```

2. **langchain-vajra** - Drop-in replacement for BM25Retriever
   ```python
   from langchain_vajra import VajraRetriever

   retriever = VajraRetriever.from_documents(documents, k=10)
   ```

3. **HybridRetriever** - Combined BM25 + vector with RRF fusion
   ```python
   from vajra_bm25.rag import HybridRetriever

   retriever = HybridRetriever(
       bm25_retriever=vajra_retriever,
       vector_retriever=chroma_retriever,
       fusion_method="rrf",  # or "linear"
       alpha=0.5  # BM25 weight for linear fusion
   )
   ```

**Benchmark Target:**
| Metric | rank-bm25 | Vajra | Improvement |
|--------|-----------|-------|-------------|
| 10K docs | 50ms | 0.5ms | 100x |
| 100K docs | 500ms | 1ms | 500x |
| 1M docs | 5s | 5ms | 1000x |

**Delivery:**
- PyPI packages: `llama-index-retrievers-vajra`, `langchain-vajra`
- Blog post: "1000x Faster RAG with Vajra BM25"

---

### 2.3 FastAPI Search Microservice

**Problem:** Many apps need search but don't want Elasticsearch complexity.

**Architecture:**
```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Gateway                    │
├─────────────────────────────────────────────────────┤
│  /search           POST  { query, top_k, filters }  │
│  /index            POST  { documents }              │
│  /index/{id}       DELETE                           │
│  /stats            GET                              │
│  /health           GET                              │
└─────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │    VajraService     │
              │  - In-memory index  │
              │  - LRU cache        │
              │  - Periodic persist │
              └─────────────────────┘
```

**Implementation Components:**

1. **vajra-server** - Standalone search microservice
   ```python
   # main.py
   from fastapi import FastAPI
   from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

   app = FastAPI(title="Vajra Search Server")
   engine: VajraSearchOptimized = None

   @app.post("/search")
   async def search(query: str, top_k: int = 10):
       results = engine.search(query, top_k=top_k)
       return [{"id": r.document.id, "score": r.score} for r in results]

   @app.post("/index")
   async def index(documents: list[dict]):
       global engine
       corpus = DocumentCorpus([Document(**d) for d in documents])
       engine = VajraSearchOptimized(corpus)
       return {"indexed": len(documents)}
   ```

2. **Docker Compose stack**
   ```yaml
   services:
     vajra:
       image: ghcr.io/aiexplorations/vajra-server
       ports:
         - "8000:8000"
       volumes:
         - ./data:/data
       environment:
         - VAJRA_PERSIST_PATH=/data/index.pkl
   ```

3. **Client SDK**
   ```python
   from vajra_client import VajraClient

   client = VajraClient("http://localhost:8000")
   client.index([{"id": "1", "title": "Doc", "content": "..."}])
   results = client.search("query", top_k=5)
   ```

**Delivery:**
- PyPI package: `vajra-server`
- Docker image: `ghcr.io/aiexplorations/vajra-server`

---

### 2.4 Document Processing Pipeline

**Problem:** Raw documents (PDF, DOCX, HTML) need text extraction before indexing.

**Architecture:**
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source  │───▶│ Extract  │───▶│  Index   │───▶│  Search  │
│  (S3)    │    │ (Tika)   │    │ (Vajra)  │    │  (API)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**Implementation Components:**

1. **vajra-ingest** - Document ingestion pipeline
   ```python
   from vajra_bm25.ingest import DocumentPipeline

   pipeline = DocumentPipeline(
       extractors={
           ".pdf": PyMuPDFExtractor(),
           ".docx": DocxExtractor(),
           ".html": BeautifulSoupExtractor(),
       },
       chunker=SentenceChunker(max_length=512),
       engine=VajraSearchOptimized
   )

   pipeline.ingest_directory("/path/to/documents")
   ```

2. **Supported formats:**
   - PDF (PyMuPDF, pdfplumber)
   - DOCX (python-docx)
   - HTML (BeautifulSoup)
   - Markdown (mistune)
   - Plain text

3. **Chunking strategies:**
   - Fixed-size windows with overlap
   - Sentence-based chunking
   - Semantic chunking (paragraph boundaries)

**Delivery:**
- PyPI package: `vajra-ingest`
- CLI: `vajra-ingest --source /docs --output index.pkl`

---

## Part 3: Prioritized Roadmap

### Phase 1: RAG Integration (High Impact, Low Effort)

**Goal:** Replace rank-bm25 in LlamaIndex/LangChain ecosystems.

| Task | Effort | Impact |
|------|--------|--------|
| Create `langchain-vajra` package | 2 days | High |
| Create `llama-index-retrievers-vajra` package | 2 days | High |
| Publish benchmark comparison blog | 1 day | Medium |
| Submit PR to LangChain docs | 1 day | Medium |

**Deliverables:**
- Two PyPI packages
- Blog post with benchmark results
- Documentation PRs

### Phase 2: Search Microservice (Medium Impact, Medium Effort)

**Goal:** Provide standalone search server for teams not wanting Elasticsearch.

| Task | Effort | Impact |
|------|--------|--------|
| Create `vajra-server` FastAPI app | 3 days | High |
| Add persistence layer (joblib) | 1 day | Medium |
| Create Docker image | 1 day | Medium |
| Write deployment docs (k8s, docker-compose) | 1 day | Medium |
| Add OpenAPI schema and client SDK | 2 days | Medium |

**Deliverables:**
- `vajra-server` package
- Docker image
- Client SDK
- Deployment examples

### Phase 3: Object Store Integration (Medium Impact, Higher Effort)

**Goal:** Enable search over S3/MinIO document stores.

| Task | Effort | Impact |
|------|--------|--------|
| Create `vajra-minio` sync worker | 3 days | Medium |
| Add PDF/DOCX extraction | 2 days | Medium |
| Implement bucket notification listener | 2 days | Medium |
| Create combined Docker stack | 1 day | Low |
| Write integration tutorial | 1 day | Medium |

**Deliverables:**
- `vajra-minio` package
- Docker Compose stack
- Tutorial blog post

### Phase 4: Document Pipeline (Lower Priority)

**Goal:** Unified ingestion for various document formats.

| Task | Effort | Impact |
|------|--------|--------|
| Create `vajra-ingest` package | 4 days | Medium |
| Support PDF, DOCX, HTML, MD | 3 days | Medium |
| Add chunking strategies | 2 days | Medium |
| CLI interface | 1 day | Low |

**Deliverables:**
- `vajra-ingest` package
- CLI tool

---

## Part 4: Technical Considerations

### Scalability Limits

| Corpus Size | RAM Required | Index Build Time | Query Latency |
|-------------|--------------|------------------|---------------|
| 10K docs | ~100 MB | <1s | <1ms |
| 100K docs | ~500 MB | ~5s | <1ms |
| 500K docs | ~2 GB | ~30s | <5ms |
| 1M docs | ~4 GB | ~60s | ~10ms |

**Recommendation:** Position Vajra for <1M document corpora. For larger scale, recommend Typesense, Meilisearch, or Elasticsearch.

### Persistence Strategy

Current: Optional joblib serialization
Proposed: Add SQLite-based persistence for incremental updates

```python
engine = VajraSearchOptimized(corpus, persist_path="index.db")
engine.add_document(new_doc)  # Incremental update
engine.remove_document(doc_id)  # Incremental delete
```

### Real-time Updates

Current: Full rebuild required
Proposed: Incremental index updates using differential sparse matrices

```python
# Incremental API
engine.add_documents([doc1, doc2])  # Add without full rebuild
engine.update_document(doc_id, new_content)  # Update existing
engine.delete_document(doc_id)  # Remove from index
```

---

## Part 5: Competitive Positioning

### Target Use Cases

| Use Case | Vajra Fit | Alternatives |
|----------|-----------|--------------|
| RAG hybrid retrieval | Excellent | rank-bm25 (slow) |
| Small corpus search (<100K) | Excellent | Whoosh (unmaintained) |
| Embedded search in Python app | Excellent | None direct |
| Large corpus search (>1M) | Poor | Meilisearch, Typesense |
| Real-time updates | Poor | Elasticsearch, Meilisearch |
| Distributed search | Poor | Elasticsearch, OpenSearch |

### Messaging

**Tagline:** "Sub-millisecond BM25 search for Python applications"

**Key differentiators:**
1. 1000x faster than rank-bm25 on large corpora
2. Pure Python, zero infrastructure
3. Category theory design enables extensibility
4. Drop-in replacement for existing BM25 retrievers

---

## Appendix: Code Examples

### A1: LangChain Integration

```python
from langchain_vajra import VajraRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create retriever from documents
documents = loader.load()
retriever = VajraRetriever.from_documents(documents, k=5)

# Use in RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

answer = qa.invoke("What is the refund policy?")
```

### A2: LlamaIndex Integration

```python
from llama_index.core import VectorStoreIndex
from llama_index.retrievers.vajra import VajraRetriever
from llama_index.core.retrievers import QueryFusionRetriever

# Create hybrid retriever
vector_retriever = index.as_retriever(similarity_top_k=5)
bm25_retriever = VajraRetriever.from_defaults(nodes=nodes, similarity_top_k=5)

hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    mode="reciprocal_rerank"
)

nodes = hybrid_retriever.retrieve("query")
```

### A3: MinIO Integration

```python
from vajra_minio import VajraMinioSync
from minio import Minio

# Initialize
client = Minio("minio:9000", access_key="...", secret_key="...")
sync = VajraMinioSync(client, bucket="documents")

# Full sync
engine = sync.build_index()

# Listen for changes
sync.watch(on_change=lambda: engine.rebuild())

# Search
results = engine.search("contract terms", top_k=10)
for r in results:
    url = client.presigned_get_object("documents", r.document.id)
    print(f"{r.score:.3f} {url}")
```

---

## Summary

Vajra BM25 has clear opportunities in:

1. **RAG ecosystems** - Replace slow rank-bm25 in LlamaIndex/LangChain
2. **Search microservices** - Lightweight alternative to Elasticsearch
3. **Object store search** - Fill the gap in S3/MinIO

The RAG integration path offers the highest impact with lowest effort and should be prioritized first. The category theory foundation provides a unique selling point for developers interested in clean abstractions.
