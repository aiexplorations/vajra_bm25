# Praval + Vajra BM25 Integration Plan

## Objective

Enable Praval agents with fast keyword search capabilities using Vajra BM25 as a native capability. The integration should feel natural within Praval's existing patterns:

```python
# Goal: One-line search like chat()
from praval import agent, chat, search

@agent("researcher", responds_to=["query"])
def researcher(spore):
    # Keyword search as easy as LLM calls
    docs = search("machine learning papers", top_k=5)
    analysis = chat(f"Summarize these findings: {docs}")
    return {"analysis": analysis}
```

## Praval Architecture Summary

### Existing Patterns

| Pattern | Module | Description |
|---------|--------|-------------|
| `@agent` | `decorators.py` | Decorator to define agents with message routing |
| `@tool` | `tools.py` | Decorator to register tools for agent use |
| `chat()` | `decorators.py` | LLM call within agent context |
| `broadcast()` | `decorators.py` | Send message to other agents |
| `BaseStorageProvider` | `storage/base_provider.py` | Abstract base for storage backends |
| `MemoryManager` | `memory/` | Multi-layer memory with vector search |

### Relevant Storage Types (Already Defined)

```python
class StorageType(Enum):
    SEARCH = "search"      # <-- Vajra fits here!
    VECTOR = "vector"      # Qdrant, for embeddings
    DOCUMENT = "document"  # MongoDB, CouchDB
```

### Current Search Capability Gap

- **Vector search**: Available via Qdrant/ChromaDB in memory system
- **Keyword search**: Not available - this is what Vajra provides

## Integration Approaches

### Approach 1: `@tool` Decorator (Simplest)

Register Vajra as a shared tool that any agent can use.

**Implementation:**
```python
# praval_vajra/tools.py
from praval import tool
from vajra_bm25 import VajraSearchOptimized, DocumentCorpus

# Global engine instance
_vajra_engine = None

def init_vajra(corpus: DocumentCorpus):
    """Initialize the Vajra search engine."""
    global _vajra_engine
    _vajra_engine = VajraSearchOptimized(corpus)

@tool("keyword_search", shared=True, category="search")
def keyword_search(query: str, top_k: int = 10) -> list:
    """
    Search documents using BM25 keyword ranking.

    Args:
        query: Search query terms
        top_k: Number of results to return

    Returns:
        List of search results with document ID, title, score
    """
    if _vajra_engine is None:
        raise RuntimeError("Vajra not initialized. Call init_vajra() first.")

    results = _vajra_engine.search(query, top_k=top_k)
    return [
        {"id": r.document.id, "title": r.document.title, "score": r.score}
        for r in results
    ]
```

**Usage:**
```python
from praval import agent, chat, start_agents
from praval_vajra.tools import init_vajra, keyword_search
from vajra_bm25 import DocumentCorpus

# Initialize with corpus
corpus = DocumentCorpus.load_jsonl("papers.jsonl")
init_vajra(corpus)

@agent("researcher", responds_to=["research_request"])
def researcher(spore):
    topic = spore.knowledge["topic"]

    # Use Vajra search as a tool
    results = keyword_search(topic, top_k=5)

    # Pass to LLM for analysis
    context = "\n".join([f"- {r['title']}" for r in results])
    analysis = chat(f"Based on these papers:\n{context}\nSummarize the key findings.")

    return {"analysis": analysis, "sources": results}
```

**Pros:**
- Minimal code changes
- Uses existing `@tool` infrastructure
- Works immediately

**Cons:**
- Global state for engine
- No integration with memory system
- Manual initialization

---

### Approach 2: Storage Provider (Consistent with Praval Patterns)

Create `VajraSearchProvider` extending `BaseStorageProvider`.

**Implementation:**
```python
# praval/storage/providers/vajra_provider.py
from typing import Any, Dict, Union
from vajra_bm25 import VajraSearchOptimized, Document, DocumentCorpus

from ..base_provider import (
    BaseStorageProvider, StorageMetadata, StorageType,
    StorageResult, DataReference
)

class VajraSearchProvider(BaseStorageProvider):
    """
    Vajra BM25 search provider for Praval storage system.

    Provides fast keyword search with BM25 ranking.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.engine = None
        self.corpus = None
        self.documents = {}  # id -> Document mapping

    def _create_metadata(self) -> StorageMetadata:
        return StorageMetadata(
            name=self.name,
            description="Vajra BM25 keyword search engine",
            storage_type=StorageType.SEARCH,
            version="0.3.0",
            supports_async=True,
            supports_search=True,
            supports_indexing=True,
            required_config=[],
            optional_config=["k1", "b", "cache_size"]
        )

    async def connect(self) -> bool:
        """Initialize empty index."""
        self.documents = {}
        self.is_connected = True
        return True

    async def disconnect(self):
        """Clear index."""
        self.engine = None
        self.corpus = None
        self.documents = {}
        self.is_connected = False

    async def store(self, resource: str, data: Any, **kwargs) -> StorageResult:
        """
        Add documents to the search index.

        Args:
            resource: Collection/index name
            data: Document or list of documents
                  Each document should have: id, title, content
        """
        if isinstance(data, dict):
            data = [data]

        for doc_data in data:
            doc = Document(
                id=doc_data.get("id", str(len(self.documents))),
                title=doc_data.get("title", ""),
                content=doc_data.get("content", "")
            )
            self.documents[doc.id] = doc

        # Rebuild index
        self._rebuild_index()

        return StorageResult(
            success=True,
            data={"indexed": len(data)},
            data_reference=DataReference(
                provider=self.name,
                storage_type=StorageType.SEARCH,
                resource_id=resource
            )
        )

    async def retrieve(self, resource: str, **kwargs) -> StorageResult:
        """Get document by ID."""
        doc_id = kwargs.get("id", resource)
        doc = self.documents.get(doc_id)

        if doc:
            return StorageResult(
                success=True,
                data={"id": doc.id, "title": doc.title, "content": doc.content}
            )
        return StorageResult(success=False, error=f"Document {doc_id} not found")

    async def query(self, resource: str, query: Union[str, Dict], **kwargs) -> StorageResult:
        """
        Execute BM25 search query.

        Args:
            resource: Index name (unused, single index)
            query: Search query string or {"text": "...", "top_k": 10}
        """
        if self.engine is None:
            return StorageResult(success=False, error="Index is empty")

        if isinstance(query, dict):
            query_text = query.get("text", "")
            top_k = query.get("top_k", 10)
        else:
            query_text = query
            top_k = kwargs.get("top_k", 10)

        results = self.engine.search(query_text, top_k=top_k)

        return StorageResult(
            success=True,
            data=[
                {
                    "id": r.document.id,
                    "title": r.document.title,
                    "content": r.document.content[:500],
                    "score": r.score,
                    "rank": r.rank
                }
                for r in results
            ],
            metadata={"query": query_text, "top_k": top_k, "hits": len(results)}
        )

    async def delete(self, resource: str, **kwargs) -> StorageResult:
        """Remove document from index."""
        doc_id = kwargs.get("id", resource)

        if doc_id in self.documents:
            del self.documents[doc_id]
            self._rebuild_index()
            return StorageResult(success=True)

        return StorageResult(success=False, error=f"Document {doc_id} not found")

    def _rebuild_index(self):
        """Rebuild Vajra index from current documents."""
        if not self.documents:
            self.engine = None
            self.corpus = None
            return

        self.corpus = DocumentCorpus(list(self.documents.values()))
        k1 = self.config.get("k1", 1.5)
        b = self.config.get("b", 0.75)
        cache_size = self.config.get("cache_size", 1000)

        self.engine = VajraSearchOptimized(
            self.corpus, k1=k1, b=b, cache_size=cache_size
        )
```

**Usage:**
```python
from praval.storage import get_storage_registry
from praval.storage.providers.vajra_provider import VajraSearchProvider

# Register provider
registry = get_storage_registry()
vajra = VajraSearchProvider("vajra", {"k1": 1.5, "b": 0.75})
registry.register_provider(vajra)

# Use in agent
@agent("researcher", responds_to=["research_request"])
async def researcher(spore):
    # Store documents
    await vajra.store("papers", [
        {"id": "1", "title": "ML Paper", "content": "..."},
        {"id": "2", "title": "AI Paper", "content": "..."}
    ])

    # Search
    result = await vajra.query("papers", spore.knowledge["topic"], top_k=5)
    return {"papers": result.data}
```

**Pros:**
- Consistent with Praval's storage pattern
- Async interface
- Provider lifecycle management
- Can be registered/discovered

**Cons:**
- More code
- Async overhead for in-memory operations
- Index rebuild on every store

---

### Approach 3: Native `search()` Function (Best UX)

Add `search()` as a first-class function like `chat()`, backed by a configurable search backend.

**Implementation:**

```python
# praval/search.py
"""
Praval Search Module - Keyword and Hybrid Search for Agents

Provides search() function for agents, similar to chat() for LLM calls.
Supports multiple backends with Vajra BM25 as the default.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
import threading

@dataclass
class SearchResult:
    """A single search result."""
    id: str
    title: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any] = None

@dataclass
class SearchConfig:
    """Configuration for search backend."""
    backend: str = "vajra"  # "vajra", "elasticsearch", "meilisearch"
    k1: float = 1.5
    b: float = 0.75
    cache_size: int = 1000
    default_top_k: int = 10

# Thread-local storage for search context
_search_context = threading.local()

class SearchBackend:
    """Abstract search backend interface."""

    def index(self, documents: List[Dict]) -> int:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

class VajraBackend(SearchBackend):
    """Vajra BM25 search backend."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.engine = None
        self.corpus = None

        # Lazy import to avoid dependency if not used
        try:
            from vajra_bm25 import VajraSearchOptimized, Document, DocumentCorpus
            self._VajraSearchOptimized = VajraSearchOptimized
            self._Document = Document
            self._DocumentCorpus = DocumentCorpus
        except ImportError:
            raise ImportError(
                "Vajra BM25 not installed. Run: pip install vajra-bm25[optimized]"
            )

    def index(self, documents: List[Dict]) -> int:
        """Index documents for search."""
        docs = [
            self._Document(
                id=str(d.get("id", i)),
                title=d.get("title", ""),
                content=d.get("content", "")
            )
            for i, d in enumerate(documents)
        ]

        self.corpus = self._DocumentCorpus(docs)
        self.engine = self._VajraSearchOptimized(
            self.corpus,
            k1=self.config.k1,
            b=self.config.b,
            cache_size=self.config.cache_size
        )

        return len(docs)

    def search(self, query: str, top_k: int) -> List[SearchResult]:
        """Execute BM25 search."""
        if self.engine is None:
            raise RuntimeError("No documents indexed. Call index() first.")

        results = self.engine.search(query, top_k=top_k)

        return [
            SearchResult(
                id=r.document.id,
                title=r.document.title,
                content=r.document.content,
                score=r.score,
                rank=r.rank,
                metadata=r.document.metadata
            )
            for r in results
        ]

    def clear(self):
        """Clear the index."""
        self.engine = None
        self.corpus = None

# Global search manager
class SearchManager:
    """Manages search backends and configuration."""

    def __init__(self):
        self.backends: Dict[str, SearchBackend] = {}
        self.default_config = SearchConfig()
        self.default_backend = "vajra"

    def configure(self, config: Union[SearchConfig, Dict]):
        """Configure the search system."""
        if isinstance(config, dict):
            config = SearchConfig(**config)
        self.default_config = config

    def get_backend(self, name: str = None) -> SearchBackend:
        """Get or create a search backend."""
        name = name or self.default_backend

        if name not in self.backends:
            if name == "vajra":
                self.backends[name] = VajraBackend(self.default_config)
            else:
                raise ValueError(f"Unknown search backend: {name}")

        return self.backends[name]

    def index(self, documents: List[Dict], backend: str = None) -> int:
        """Index documents in the specified backend."""
        return self.get_backend(backend).index(documents)

    def search(self, query: str, top_k: int = None, backend: str = None) -> List[SearchResult]:
        """Search across indexed documents."""
        if top_k is None:
            top_k = self.default_config.default_top_k
        return self.get_backend(backend).search(query, top_k)

# Global instance
_search_manager = SearchManager()

# Public API
def configure_search(config: Union[SearchConfig, Dict]):
    """Configure the Praval search system."""
    _search_manager.configure(config)

def index_documents(documents: List[Dict], backend: str = None) -> int:
    """
    Index documents for search.

    Args:
        documents: List of dicts with id, title, content keys
        backend: Backend name (default: "vajra")

    Returns:
        Number of documents indexed
    """
    return _search_manager.index(documents, backend)

def search(query: str, top_k: int = 10, backend: str = None) -> List[SearchResult]:
    """
    Search indexed documents using keyword matching.

    This is the main search function for agents, analogous to chat() for LLM calls.

    Args:
        query: Search query
        top_k: Number of results to return
        backend: Backend to use (default: "vajra")

    Returns:
        List of SearchResult objects

    Example:
        @agent("researcher", responds_to=["query"])
        def researcher(spore):
            results = search(spore.knowledge["topic"], top_k=5)
            context = "\\n".join([r.title for r in results])
            analysis = chat(f"Summarize: {context}")
            return {"analysis": analysis}
    """
    return _search_manager.search(query, top_k, backend)

def get_search_manager() -> SearchManager:
    """Get the global search manager."""
    return _search_manager
```

**Usage:**
```python
from praval import agent, chat, broadcast, start_agents
from praval.search import search, index_documents, configure_search

# Configure and index
configure_search({"k1": 1.5, "b": 0.75, "default_top_k": 10})
index_documents([
    {"id": "1", "title": "ML Basics", "content": "..."},
    {"id": "2", "title": "Deep Learning", "content": "..."}
])

@agent("researcher", responds_to=["research_query"])
def researcher(spore):
    # Search is as easy as chat!
    results = search(spore.knowledge["topic"], top_k=5)

    context = "\n".join([f"- {r.title}: {r.content[:100]}" for r in results])
    analysis = chat(f"Based on these papers:\n{context}\nProvide key insights.")

    broadcast({"type": "analysis_complete", "findings": analysis})
    return {"analysis": analysis, "sources": [r.id for r in results]}

start_agents(researcher, initial_data={"type": "research_query", "topic": "neural networks"})
```

**Pros:**
- Best developer experience
- Matches Praval's `chat()` pattern
- Clean, minimal API
- Supports multiple backends

**Cons:**
- New module to maintain
- Global state management

---

### Approach 4: Hybrid Search in Memory System

Combine Vajra BM25 with existing vector search for hybrid retrieval.

**Implementation:**
```python
# praval/memory/hybrid_search.py
from typing import List, Dict, Any, Optional
from .memory_types import MemoryEntry, MemoryQuery, MemorySearchResult
from .long_term_memory import LongTermMemory

class HybridSearchMemory:
    """
    Combines vector search (semantic) with BM25 (keyword) for better retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    def __init__(
        self,
        long_term_memory: LongTermMemory,
        bm25_weight: float = 0.5
    ):
        self.ltm = long_term_memory
        self.bm25_weight = bm25_weight
        self.vajra_engine = None

        # Lazy import
        try:
            from vajra_bm25 import VajraSearchOptimized, Document, DocumentCorpus
            self._vajra_available = True
            self._VajraSearchOptimized = VajraSearchOptimized
            self._Document = Document
            self._DocumentCorpus = DocumentCorpus
        except ImportError:
            self._vajra_available = False

    def _build_bm25_index(self, entries: List[MemoryEntry]):
        """Build BM25 index from memory entries."""
        if not self._vajra_available:
            return

        docs = [
            self._Document(
                id=entry.id,
                title="",
                content=entry.content
            )
            for entry in entries
        ]

        corpus = self._DocumentCorpus(docs)
        self.vajra_engine = self._VajraSearchOptimized(corpus)

    def hybrid_search(
        self,
        query: MemoryQuery,
        rebuild_index: bool = False
    ) -> MemorySearchResult:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            rebuild_index: Whether to rebuild BM25 index

        Returns:
            Combined search results
        """
        # Get vector search results
        vector_results = self.ltm.search(query)

        if not self._vajra_available or not self.vajra_engine:
            return vector_results

        # Get BM25 results
        bm25_results = self.vajra_engine.search(query.query_text, top_k=query.limit * 2)
        bm25_ids = {r.document.id: i for i, r in enumerate(bm25_results)}

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        fused_scores = {}

        # Add vector scores
        for i, (entry, score) in enumerate(zip(vector_results.entries, vector_results.scores)):
            rrf_score = (1 - self.bm25_weight) * (1 / (k + i + 1))
            fused_scores[entry.id] = {"entry": entry, "score": rrf_score}

        # Add BM25 scores
        for entry_id, rank in bm25_ids.items():
            rrf_score = self.bm25_weight * (1 / (k + rank + 1))
            if entry_id in fused_scores:
                fused_scores[entry_id]["score"] += rrf_score
            # Note: BM25 might find docs not in vector results

        # Sort by fused score
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:query.limit]

        return MemorySearchResult(
            entries=[r["entry"] for r in sorted_results],
            scores=[r["score"] for r in sorted_results],
            query=query,
            total_found=len(sorted_results)
        )
```

**Pros:**
- Combines semantic + keyword search
- Better retrieval quality
- Integrates with existing memory

**Cons:**
- More complex
- Requires both systems running
- Index synchronization needed

---

## Recommended Implementation Order

### Phase 1: Tool Integration (Immediate Value)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Create `praval-vajra` package | 2 days | PyPI package |
| Implement `@tool` wrapper | 1 day | `praval_vajra.tools` |
| Add corpus loading utilities | 1 day | JSONL, BEIR loaders |
| Write tests | 1 day | >80% coverage |
| Documentation | 1 day | README, examples |

**Deliverables:**
- `pip install praval-vajra`
- Works with existing Praval installations

### Phase 2: Storage Provider

| Task | Effort | Deliverable |
|------|--------|-------------|
| Implement `VajraSearchProvider` | 2 days | PR to Praval |
| Add to storage registry | 1 day | Auto-discovery |
| Integration tests | 1 day | Test suite |

**Deliverables:**
- `pip install praval[search]`
- `from praval.storage.providers import VajraSearchProvider`

### Phase 3: Native `search()` Function

| Task | Effort | Deliverable |
|------|--------|-------------|
| Create `praval/search.py` | 2 days | Core module |
| Add to Praval exports | 1 day | `from praval import search` |
| Backend abstraction | 2 days | Support multiple backends |
| Documentation | 1 day | Guides, examples |

**Deliverables:**
- `from praval import search, index_documents`
- Matches `chat()` ergonomics

### Phase 4: Hybrid Memory Search

| Task | Effort | Deliverable |
|------|--------|-------------|
| Implement `HybridSearchMemory` | 3 days | Memory extension |
| RRF fusion logic | 1 day | Score combination |
| Index sync strategy | 2 days | Consistency handling |
| Benchmarks | 1 day | Quality metrics |

**Deliverables:**
- Improved retrieval quality
- Optional hybrid mode for memory system

---

## Package Structure

```
praval-vajra/
├── praval_vajra/
│   ├── __init__.py
│   ├── tools.py           # @tool wrapper
│   ├── provider.py        # VajraSearchProvider
│   ├── search.py          # search() function
│   └── hybrid.py          # Hybrid search utilities
├── tests/
│   ├── test_tools.py
│   ├── test_provider.py
│   └── test_search.py
├── examples/
│   ├── basic_search.py
│   ├── multi_agent_search.py
│   └── hybrid_rag.py
├── pyproject.toml
└── README.md
```

**pyproject.toml:**
```toml
[project]
name = "praval-vajra"
version = "0.1.0"
description = "Vajra BM25 search integration for Praval agents"
dependencies = [
    "praval>=0.7.0",
    "vajra-bm25[optimized]>=0.3.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
```

---

## Example: Complete RAG Agent

```python
"""
Research Agent with Hybrid Search
Combines BM25 keyword search with LLM analysis.
"""
from praval import agent, chat, broadcast, start_agents, get_reef
from praval.search import search, index_documents, configure_search
from vajra_bm25 import DocumentCorpus

# Load and index research papers
corpus = DocumentCorpus.load_jsonl("papers.jsonl")
index_documents([
    {"id": doc.id, "title": doc.title, "content": doc.content}
    for doc in corpus
])

@agent("query_router", responds_to=["user_query"])
def query_router(spore):
    """Route queries to appropriate specialists."""
    query = spore.knowledge["query"]

    # Quick keyword search to determine topic
    results = search(query, top_k=3)
    topics = set()
    for r in results:
        if "machine learning" in r.content.lower():
            topics.add("ml")
        if "neural" in r.content.lower():
            topics.add("deep_learning")

    broadcast({
        "type": "research_task",
        "query": query,
        "topics": list(topics),
        "initial_docs": [r.id for r in results]
    })

@agent("researcher", responds_to=["research_task"])
def researcher(spore):
    """Deep research on the topic."""
    query = spore.knowledge["query"]

    # Comprehensive search
    results = search(query, top_k=10)

    context = "\n\n".join([
        f"**{r.title}**\n{r.content[:500]}..."
        for r in results
    ])

    analysis = chat(f"""
    Based on these research papers:

    {context}

    Provide a comprehensive analysis of: {query}
    Include key findings, methodologies, and conclusions.
    """)

    broadcast({
        "type": "analysis_complete",
        "query": query,
        "analysis": analysis,
        "sources": [{"id": r.id, "title": r.title, "score": r.score} for r in results]
    })

@agent("summarizer", responds_to=["analysis_complete"])
def summarizer(spore):
    """Create executive summary."""
    analysis = spore.knowledge["analysis"]
    sources = spore.knowledge["sources"]

    summary = chat(f"""
    Create a concise executive summary (3-5 bullet points) from this analysis:

    {analysis}

    Sources used: {len(sources)} papers
    """)

    print(f"\n📋 Executive Summary:\n{summary}")
    print(f"\n📚 Sources: {', '.join([s['title'][:30] for s in sources[:3]])}...")

    return {"summary": summary, "sources": sources}

# Run the system
start_agents(
    query_router, researcher, summarizer,
    initial_data={"type": "user_query", "query": "transformer architectures in NLP"}
)
get_reef().wait_for_completion()
get_reef().shutdown()
```

---

## Summary

| Approach | Complexity | UX | Integration Depth |
|----------|------------|-----|-------------------|
| `@tool` wrapper | Low | Good | Shallow |
| Storage Provider | Medium | Good | Deep |
| Native `search()` | Medium | Excellent | Medium |
| Hybrid Memory | High | Good | Deep |

**Recommendation:** Start with Phase 1 (`@tool` wrapper) for immediate value, then progress to Phase 3 (native `search()`) for the best developer experience. Phase 4 (hybrid) is optional for advanced use cases.
