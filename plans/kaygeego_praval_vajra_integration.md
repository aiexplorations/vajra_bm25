# KayGeeGo + Praval + Vajra Integration Plan

## Overview

Integrate three systems to create a powerful knowledge graph platform:

| Component | Language | Role |
|-----------|----------|------|
| **kay-gee-go** | Go | Graph storage (Neo4j), 3D visualization, concurrent building |
| **Praval** | Python | Multi-agent intelligence for KG construction/enrichment |
| **Vajra BM25** | Python | Fast keyword search over concepts and relationships |

## Current State

### kay-gee-go (v0.6.1)
```
┌─────────────────────────────────────────────────────────┐
│                     kay-gee-go                          │
├─────────────┬─────────────────┬─────────────────────────┤
│ kg-builder  │   kg-enricher   │     kg-frontend         │
│ (Go)        │   (Go)          │     (Go + JS)           │
│             │                 │                         │
│ • BFS       │ • Continuous    │ • 3D visualization      │
│ • LLM calls │ • Batch mining  │ • Search UI             │
│ • Workers   │ • Low-conn seed │ • Stats dashboard       │
└──────┬──────┴────────┬────────┴───────────┬─────────────┘
       │               │                    │
       └───────────────┴────────────────────┘
                       │
              ┌────────▼────────┐
              │     Neo4j       │
              │  (Graph DB)     │
              └─────────────────┘
```

### Praval Knowledge Graph Miner
```python
# praval-examples/knowledge_graph_miner.py
@agent("explorer", responds_to=["concept_request"])
def discover_concepts(spore):
    related = chat(f"List 3 concepts related to '{concept}'")
    return {"type": "discovery", "found": concepts}

@agent("relationship_explorer", responds_to=["discovery"])
def explore_relationships(spore):
    relationship = chat(f"How is '{a}' related to '{b}'?")
    graph["edges"].append({"source": a, "target": b, "relationship": rel})
```

### Vajra BM25
- Fast keyword search (180,000-800,000 QPS)
- Sub-millisecond latency with caching
- Python-native, no external service

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Unified KG Platform                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Python Layer (Praval)                      │  │
│  │                                                               │  │
│  │   @agent("explorer")     @agent("enricher")    @agent("qa")  │  │
│  │   discover_concepts()    mine_relationships()  answer_query()│  │
│  │         │                       │                    │        │  │
│  │         └───────────┬───────────┘                    │        │  │
│  │                     │                                │        │  │
│  │              ┌──────▼──────┐                  ┌──────▼──────┐ │  │
│  │              │   Vajra     │                  │   Vajra     │ │  │
│  │              │  (Indexer)  │                  │  (Search)   │ │  │
│  │              └──────┬──────┘                  └──────┬──────┘ │  │
│  │                     │                                │        │  │
│  └─────────────────────┼────────────────────────────────┼────────┘  │
│                        │                                │           │
│  ┌─────────────────────▼────────────────────────────────▼────────┐  │
│  │                    Bridge API (FastAPI)                       │  │
│  │  POST /concepts      GET /search       POST /relationships    │  │
│  └─────────────────────┬────────────────────────────────┬────────┘  │
│                        │                                │           │
│  ┌─────────────────────▼────────────────────────────────▼────────┐  │
│  │                    Go Layer (kay-gee-go)                      │  │
│  │                                                               │  │
│  │   kg-builder          kg-enricher         kg-frontend         │  │
│  │   (optional)          (optional)          (3D viz)            │  │
│  │                                                               │  │
│  └─────────────────────────────┬─────────────────────────────────┘  │
│                                │                                    │
│                       ┌────────▼────────┐                           │
│                       │     Neo4j       │                           │
│                       │  (Graph Store)  │                           │
│                       └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Integration Approaches

### Approach 1: Vajra as Search Sidecar (Simplest)

Add Vajra as a fast search index alongside Neo4j.

**Architecture:**
```
                    ┌─────────────┐
     User Query ───▶│   Vajra     │───▶ Fast concept lookup
                    │  (BM25)     │     (0.01ms cached)
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Neo4j     │───▶ Graph traversal
                    │  (Cypher)   │     (relationship queries)
                    └─────────────┘
```

**Implementation:**

```python
# kaygeego_vajra/search.py
from vajra_bm25 import VajraSearchOptimized, Document, DocumentCorpus
from neo4j import GraphDatabase

class KGSearch:
    """Fast keyword search over Knowledge Graph concepts."""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.vajra = None

    def sync_from_neo4j(self):
        """Pull all concepts from Neo4j and index in Vajra."""
        with self.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN c.id, c.name, c.description")

            docs = [
                Document(
                    id=record["c.id"],
                    title=record["c.name"],
                    content=f"{record['c.name']}. {record['c.description'] or ''}"
                )
                for record in result
            ]

        corpus = DocumentCorpus(docs)
        self.vajra = VajraSearchOptimized(corpus)
        return len(docs)

    def search(self, query: str, top_k: int = 10) -> list:
        """Fast BM25 search over concepts."""
        if self.vajra is None:
            raise RuntimeError("Index not built. Call sync_from_neo4j() first.")

        return self.vajra.search(query, top_k=top_k)

    def get_relationships(self, concept_id: str) -> list:
        """Get relationships for a concept from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {id: $id})-[r]->(target:Concept)
                RETURN type(r) as type, r.description as description,
                       target.id as target_id, target.name as target_name
            """, id=concept_id)

            return [dict(record) for record in result]
```

**Usage:**
```python
# Search concepts, then traverse graph
kg = KGSearch("bolt://localhost:7687", "neo4j", "password")
kg.sync_from_neo4j()

# Fast keyword search
concepts = kg.search("machine learning", top_k=5)

# Graph traversal for relationships
for c in concepts:
    relationships = kg.get_relationships(c.document.id)
    print(f"{c.document.title}: {len(relationships)} relationships")
```

---

### Approach 2: Praval Agents for KG Construction

Replace kay-gee-go's Go workers with Praval's Python agents.

**Why?**
- Praval has native LLM integration (`chat()`)
- Multi-agent coordination is simpler
- Python ecosystem (Vajra, embeddings, etc.)
- kay-gee-go's Go workers can still handle Neo4j I/O

**Architecture:**
```
┌────────────────────────────────────────────┐
│           Praval Agent System              │
│                                            │
│  @agent("explorer")    @agent("connector") │
│  discover_concepts()   mine_relationships()│
│         │                     │            │
│         └─────────┬───────────┘            │
│                   │                        │
│           ┌───────▼───────┐                │
│           │  Neo4j Client │                │
│           │  (Python)     │                │
│           └───────┬───────┘                │
└───────────────────┼────────────────────────┘
                    │
           ┌────────▼────────┐
           │     Neo4j       │◀──── kg-frontend (3D viz)
           │  (Graph Store)  │
           └─────────────────┘
```

**Implementation:**

```python
# kaygeego_praval/agents.py
from praval import agent, chat, broadcast, start_agents
from neo4j import GraphDatabase

# Neo4j connection
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Shared state
graph_state = {
    "concepts": set(),
    "processed": set(),
    "target_size": 100
}

@agent("explorer", channel="kg", responds_to=["explore_request"])
def discover_concepts(spore):
    """Discover related concepts using LLM."""
    concept = spore.knowledge.get("concept")

    if not concept or concept in graph_state["processed"]:
        return

    # Use Praval's native LLM call
    response = chat(f"""
    List 5 concepts closely related to '{concept}'.
    For each, provide: name, relationship type, brief description.
    Format: name | relationship | description
    """)

    # Parse and create in Neo4j
    for line in response.strip().split("\n"):
        parts = line.split("|")
        if len(parts) >= 2:
            name = parts[0].strip()
            relationship = parts[1].strip()

            create_concept_and_relationship(concept, name, relationship)
            graph_state["concepts"].add(name)

    graph_state["processed"].add(concept)

    return {
        "type": "discovery",
        "source": concept,
        "found": list(graph_state["concepts"] - graph_state["processed"])
    }

@agent("curator", channel="kg", responds_to=["discovery"])
def coordinate_exploration(spore):
    """Coordinate exploration and trigger next concepts."""
    if len(graph_state["concepts"]) >= graph_state["target_size"]:
        return {"type": "complete", "size": len(graph_state["concepts"])}

    unexplored = graph_state["concepts"] - graph_state["processed"]
    if unexplored:
        next_concept = unexplored.pop()
        broadcast({"type": "explore_request", "concept": next_concept})

def create_concept_and_relationship(source: str, target: str, relationship: str):
    """Create concept and relationship in Neo4j."""
    with driver.session() as session:
        session.run("""
            MERGE (s:Concept {name: $source})
            MERGE (t:Concept {name: $target})
            MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(t)
        """, source=source, target=target, rel_type=relationship)

def build_kg(seed: str, target_size: int = 100):
    """Build knowledge graph using Praval agents."""
    graph_state["target_size"] = target_size
    graph_state["concepts"].add(seed)

    start_agents(
        discover_concepts, coordinate_exploration,
        initial_data={"type": "explore_request", "concept": seed},
        channel="kg"
    )
```

---

### Approach 3: Full Integration - Praval + Vajra + kay-gee-go

Complete platform with all three systems working together.

**Components:**

1. **kaygeego-bridge** (FastAPI) - Bridge between Python and Go
2. **Praval agents** - Intelligent KG construction
3. **Vajra search** - Fast concept retrieval
4. **kay-gee-go frontend** - 3D visualization (unchanged)
5. **Neo4j** - Graph storage (unchanged)

**New Package: `kaygeego-praval`**

```
kaygeego-praval/
├── kaygeego_praval/
│   ├── __init__.py
│   ├── agents/
│   │   ├── explorer.py      # Concept discovery agent
│   │   ├── enricher.py      # Relationship mining agent
│   │   └── qa.py            # Question answering agent
│   ├── search/
│   │   ├── vajra_index.py   # Vajra integration
│   │   └── hybrid.py        # BM25 + graph hybrid search
│   ├── bridge/
│   │   ├── api.py           # FastAPI bridge
│   │   └── neo4j_client.py  # Neo4j operations
│   └── cli.py               # CLI interface
├── tests/
├── pyproject.toml
└── README.md
```

**Bridge API:**

```python
# kaygeego_praval/bridge/api.py
from fastapi import FastAPI
from vajra_bm25 import VajraSearchOptimized
from neo4j import GraphDatabase

app = FastAPI(title="KayGeeGo Bridge")

# Initialize
vajra_index = None
neo4j_driver = None

@app.on_event("startup")
async def startup():
    global vajra_index, neo4j_driver
    neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    vajra_index = build_vajra_index(neo4j_driver)

@app.get("/search")
async def search(query: str, top_k: int = 10, include_relationships: bool = True):
    """
    Hybrid search: Vajra BM25 for concepts + Neo4j for relationships.
    """
    # Fast concept search via Vajra
    concepts = vajra_index.search(query, top_k=top_k)

    results = []
    for c in concepts:
        result = {
            "id": c.document.id,
            "name": c.document.title,
            "score": c.score
        }

        if include_relationships:
            result["relationships"] = get_relationships(neo4j_driver, c.document.id)

        results.append(result)

    return {"results": results, "query": query}

@app.post("/build")
async def build_kg(seed: str, target_size: int = 100):
    """
    Trigger Praval agents to build KG from seed concept.
    """
    from kaygeego_praval.agents import build_kg
    stats = build_kg(seed, target_size)

    # Rebuild Vajra index after construction
    global vajra_index
    vajra_index = build_vajra_index(neo4j_driver)

    return {"status": "complete", "stats": stats}

@app.post("/sync-index")
async def sync_index():
    """
    Sync Vajra index with Neo4j.
    """
    global vajra_index
    vajra_index = build_vajra_index(neo4j_driver)
    return {"status": "synced", "concepts": len(vajra_index.corpus)}
```

**Docker Compose (extended):**

```yaml
# docker-compose.yml (additions to kay-gee-go)
services:
  # Existing kay-gee-go services...
  neo4j:
    image: neo4j:5.15.0
    # ...

  kg-frontend:
    # ...

  # New Python services
  praval-bridge:
    build: ./kaygeego-praval
    ports:
      - "8001:8001"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j
    command: uvicorn kaygeego_praval.bridge.api:app --host 0.0.0.0 --port 8001
```

---

## Use Case: Intelligent Q&A over Knowledge Graph

Combining all three systems for RAG-style Q&A:

```python
# kaygeego_praval/agents/qa.py
from praval import agent, chat
from kaygeego_praval.search import hybrid_search

@agent("qa_agent", responds_to=["user_question"])
def answer_question(spore):
    """Answer questions using KG as context."""
    question = spore.knowledge["question"]

    # 1. Fast keyword search via Vajra
    relevant_concepts = vajra_search(question, top_k=10)

    # 2. Graph expansion via Neo4j
    context = []
    for concept in relevant_concepts:
        # Get 1-hop relationships
        relationships = get_relationships(concept.id)
        context.append({
            "concept": concept.title,
            "relationships": relationships
        })

    # 3. LLM synthesis via Praval
    context_str = format_context(context)
    answer = chat(f"""
    Based on this knowledge graph context:

    {context_str}

    Answer the question: {question}

    Cite specific concepts and relationships from the context.
    """)

    return {
        "question": question,
        "answer": answer,
        "sources": [c["concept"] for c in context]
    }
```

---

## Implementation Phases

### Phase 1: Vajra Search Sidecar (3 days)

| Task | Effort |
|------|--------|
| Create `kaygeego-vajra` package | 1 day |
| Neo4j → Vajra sync script | 0.5 day |
| FastAPI search endpoint | 0.5 day |
| Integration tests | 1 day |

**Deliverable:** Fast BM25 search over KG concepts

### Phase 2: Praval Agent Integration (5 days)

| Task | Effort |
|------|--------|
| Port kg-builder logic to Praval agents | 2 days |
| Port kg-enricher logic to Praval agents | 1 day |
| Neo4j Python client | 1 day |
| Tests and documentation | 1 day |

**Deliverable:** Python-based KG construction using Praval

### Phase 3: Bridge API (3 days)

| Task | Effort |
|------|--------|
| FastAPI bridge service | 1 day |
| Docker integration | 1 day |
| Frontend API updates | 1 day |

**Deliverable:** Unified REST API for Python + Go components

### Phase 4: Intelligent Q&A (2 days)

| Task | Effort |
|------|--------|
| Q&A agent implementation | 1 day |
| Hybrid search (Vajra + Neo4j) | 1 day |

**Deliverable:** RAG-style Q&A over knowledge graph

---

## Benefits of Integration

| Capability | Before | After |
|------------|--------|-------|
| Concept search | Neo4j text index (~10ms) | Vajra BM25 (~0.01ms cached) |
| KG construction | Go workers + Ollama | Praval agents + any LLM |
| Agent coordination | Manual Go concurrency | Praval `@agent` decorator |
| Hybrid retrieval | Not available | Vajra + Neo4j combined |
| Python ecosystem | Not available | Full access |

---

## Summary

The integration creates a **polyglot knowledge graph platform**:

- **Go (kay-gee-go)**: Graph storage, 3D visualization, high-performance I/O
- **Python (Praval)**: Intelligent agents, LLM orchestration, easy extensibility
- **Python (Vajra)**: Sub-millisecond keyword search, caching

This combines the strengths of each system while maintaining the existing kay-gee-go frontend and Neo4j storage.
