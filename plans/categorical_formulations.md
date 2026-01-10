# Categorical Formulations for Search and Information Retrieval

## Overview

This document explores creative applications of category theory to search, indexing, and information retrieval problems beyond the core BM25 and vector search implementations.

The goal is to identify where categorical abstractions provide:
1. **Cleaner interfaces** through universal properties
2. **Composability** through morphism composition
3. **Correctness guarantees** through algebraic laws
4. **Novel algorithms** inspired by categorical constructions

## Current Categorical Foundations in Vajra

| Abstraction | Current Use | Mathematical Meaning |
|-------------|-------------|---------------------|
| **Morphism** | BM25 scoring: `(Query, Doc) → Score` | Arrow between objects |
| **Coalgebra** | Search unfolding: `State → F(State)` | State machine with observations |
| **Functor** | `ListFunctor` for multi-result semantics | Structure-preserving map |
| **Comonad** | LRU cache with `extract`/`duplicate` | Context-dependent computation |

---

## 1. Embeddings as Functors (Structure-Preserving Maps)

### The Insight

An embedding should preserve semantic structure. Categorically, this means:
- **Objects**: Words, sentences, documents (text category)
- **Morphisms**: Semantic relationships (synonymy, entailment, etc.)
- **Functor**: Embedding maps that preserve these relationships

```
Text Category                    Vector Space Category
     A ──────f────→ B                E(A) ────E(f)────→ E(B)
     │              │                 │                  │
   synonym      entails            close              close
     │              │                 │                  │
     ↓              ↓                 ↓                  ↓
     A' ─────f'───→ B'               E(A') ───E(f')───→ E(B')
```

### Implementation

```python
# categorical/embedding_functor.py

class SemanticCategory:
    """Category where objects are texts and morphisms are semantic relations"""

    @staticmethod
    def synonym(a: str, b: str) -> bool:
        """Morphism: a ≈ b (synonymy)"""
        pass

    @staticmethod
    def entails(a: str, b: str) -> bool:
        """Morphism: a → b (entailment)"""
        pass


class EmbeddingFunctor(Functor[str, np.ndarray]):
    """
    Functor from Text category to Vector Space category.

    Laws that should hold:
    1. If synonym(a, b), then cosine(E(a), E(b)) > τ_synonym
    2. If entails(a, b), then E(a) · direction(entailment) ≈ E(b)
    3. E(compose(f, g)) = E(f) ∘ E(g)  (functor composition)
    """

    def __init__(self, embedder: EmbeddingMorphism):
        self.embedder = embedder

    def map_object(self, text: str) -> np.ndarray:
        """Map text to vector"""
        return self.embedder.embed(text)

    def map_morphism(self, relation: str) -> Callable[[np.ndarray], np.ndarray]:
        """
        Map semantic relation to vector space transformation.

        E.g., "is-a" relation maps to a learned linear transformation.
        """
        # Could use relation-specific learned transforms
        pass


class ContrastiveEmbeddingFunctor(EmbeddingFunctor):
    """
    Embedding trained with contrastive loss to preserve categorical structure.

    Loss = -log(exp(sim(E(a), E(b⁺))) / Σ exp(sim(E(a), E(b⁻))))

    Where:
    - b⁺ are positive examples (related via morphism)
    - b⁻ are negative examples (no morphism)
    """
    pass
```

### Applications
- **Evaluation**: Test embedding quality by checking functor laws
- **Training**: Use categorical constraints as auxiliary loss
- **Debugging**: Identify where embeddings fail to preserve structure

---

## 2. Query Refinement as a Monad

### The Insight

Interactive search with relevance feedback is inherently monadic:
- **Type**: `Query → M[Results]` where `M` captures the iterative refinement context
- **bind**: Incorporate user feedback to produce new query
- **return**: Initial query with no feedback

```
           query₀
              │
              ↓
         ┌─────────┐
         │ Search  │──→ results₀
         └─────────┘
              │
         user feedback
              │
              ↓
           query₁ = bind(query₀, feedback)
              │
              ↓
         ┌─────────┐
         │ Search  │──→ results₁
         └─────────┘
              │
             ...
```

### Implementation

```python
# categorical/refinement_monad.py

@dataclass
class RefinementContext:
    """Monadic context for query refinement"""
    query: str
    positive_docs: List[str]    # User marked as relevant
    negative_docs: List[str]    # User marked as not relevant
    iteration: int


class QueryRefinementMonad(Monad[RefinementContext]):
    """
    Monad for interactive query refinement.

    Laws:
    1. return a >>= f  ≡  f a                    (left identity)
    2. m >>= return    ≡  m                      (right identity)
    3. (m >>= f) >>= g ≡  m >>= (λx. f x >>= g)  (associativity)
    """

    def return_(self, query: str) -> RefinementContext:
        """Lift initial query into refinement context"""
        return RefinementContext(
            query=query,
            positive_docs=[],
            negative_docs=[],
            iteration=0
        )

    def bind(
        self,
        ctx: RefinementContext,
        refine: Callable[[RefinementContext], RefinementContext]
    ) -> RefinementContext:
        """Apply refinement function to context"""
        return refine(ctx)

    def rocchio_refinement(self, ctx: RefinementContext, embedder: EmbeddingMorphism) -> str:
        """
        Rocchio algorithm as monadic bind operation.

        q' = α·q + β·mean(positive) - γ·mean(negative)
        """
        α, β, γ = 1.0, 0.75, 0.15

        q_vec = embedder.embed(ctx.query)

        if ctx.positive_docs:
            pos_vecs = embedder.embed_batch(ctx.positive_docs)
            pos_centroid = pos_vecs.mean(axis=0)
        else:
            pos_centroid = np.zeros_like(q_vec)

        if ctx.negative_docs:
            neg_vecs = embedder.embed_batch(ctx.negative_docs)
            neg_centroid = neg_vecs.mean(axis=0)
        else:
            neg_centroid = np.zeros_like(q_vec)

        refined_vec = α * q_vec + β * pos_centroid - γ * neg_centroid

        # Convert back to query (e.g., find nearest terms)
        return self._vector_to_query(refined_vec)


class RelevanceFeedbackPipeline:
    """Compose refinement steps monadically"""

    def __init__(self, search_engine, embedder):
        self.monad = QueryRefinementMonad()
        self.search = search_engine
        self.embedder = embedder

    def run(self, initial_query: str, max_iterations: int = 3) -> List[SearchResult]:
        """Run refinement loop"""
        ctx = self.monad.return_(initial_query)

        for i in range(max_iterations):
            results = self.search.search(ctx.query)

            # Get user feedback (simulate or actual)
            feedback = self._get_feedback(results)

            # Monadic bind: incorporate feedback
            ctx = self.monad.bind(ctx, lambda c: RefinementContext(
                query=self.monad.rocchio_refinement(c, self.embedder),
                positive_docs=c.positive_docs + feedback.positive,
                negative_docs=c.negative_docs + feedback.negative,
                iteration=c.iteration + 1
            ))

        return self.search.search(ctx.query)
```

### Applications
- **Relevance feedback**: Rocchio, pseudo-relevance feedback
- **Conversational search**: Multi-turn refinement
- **Active learning**: Query-by-committee with user in loop

---

## 3. Hybrid Search as Coproduct (Disjoint Union)

### The Insight

Combining BM25 and vector search can be viewed as a coproduct (categorical sum):
- Both produce `List[Result]` but from different "source categories"
- Coproduct provides universal way to combine with injections

```
BM25 Results ──inj₁──→ Combined ←──inj₂── Vector Results
                          │
                          │ [fusion, fusion]
                          ↓
                    Final Results
```

### Implementation

```python
# categorical/coproduct_fusion.py

class SearchResultCoproduct:
    """
    Coproduct of BM25 and Vector search results.

    Universal property: Any function that handles both BM25 and Vector
    results factors through the coproduct.
    """

    @dataclass
    class Injected:
        """Tagged result indicating source"""
        result: SearchResult
        source: str  # "bm25" or "vector"
        rank: int

    @staticmethod
    def inj_bm25(results: List[SearchResult]) -> List['SearchResultCoproduct.Injected']:
        """Injection from BM25 results"""
        return [
            SearchResultCoproduct.Injected(r, "bm25", i)
            for i, r in enumerate(results, 1)
        ]

    @staticmethod
    def inj_vector(results: List[SearchResult]) -> List['SearchResultCoproduct.Injected']:
        """Injection from Vector results"""
        return [
            SearchResultCoproduct.Injected(r, "vector", i)
            for i, r in enumerate(results, 1)
        ]

    @staticmethod
    def fold(
        injected: List['SearchResultCoproduct.Injected'],
        bm25_handler: Callable[[SearchResult, int], float],
        vector_handler: Callable[[SearchResult, int], float]
    ) -> List[Tuple[SearchResult, float]]:
        """
        Universal morphism out of coproduct.

        Any fusion strategy is defined by how it handles each source.
        """
        scored = []
        for inj in injected:
            if inj.source == "bm25":
                score = bm25_handler(inj.result, inj.rank)
            else:
                score = vector_handler(inj.result, inj.rank)
            scored.append((inj.result, score))
        return scored


class CategoricalHybridSearch:
    """Hybrid search using coproduct construction"""

    def __init__(self, bm25_engine, vector_engine):
        self.bm25 = bm25_engine
        self.vector = vector_engine
        self.coproduct = SearchResultCoproduct()

    def search(self, query: str, top_k: int = 10, fusion: str = "rrf") -> List[SearchResult]:
        # Get results from both sources
        bm25_results = self.bm25.search(query, top_k=top_k * 2)
        vector_results = self.vector.search(query, top_k=top_k * 2)

        # Inject into coproduct
        injected = (
            self.coproduct.inj_bm25(bm25_results) +
            self.coproduct.inj_vector(vector_results)
        )

        # Select fusion handlers based on strategy
        if fusion == "rrf":
            k = 60
            bm25_handler = lambda r, rank: 1.0 / (k + rank)
            vector_handler = lambda r, rank: 1.0 / (k + rank)
        elif fusion == "linear":
            bm25_handler = lambda r, rank: 0.5 * r.score
            vector_handler = lambda r, rank: 0.5 * r.score
        elif fusion == "bm25_priority":
            bm25_handler = lambda r, rank: 2.0 / (k + rank)
            vector_handler = lambda r, rank: 1.0 / (k + rank)

        # Fold through coproduct
        scored = self.coproduct.fold(injected, bm25_handler, vector_handler)

        # Aggregate by document and sort
        doc_scores = {}
        for result, score in scored:
            doc_id = result.document.id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score

        # Return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        return [...]  # Build final results
```

---

## 4. Filtered Search as Pullback

### The Insight

Combining vector similarity with metadata filtering is a pullback:

```
            Filtered Results
               /       \
              /         \
             /           \
            ↓             ↓
    Vector Matches ──→ All Docs ←── Metadata Matches
```

The pullback gives documents that satisfy BOTH constraints.

### Implementation

```python
# categorical/pullback_filter.py

class FilteredSearchPullback:
    """
    Pullback construction for filtered vector search.

    Pullback of:
    - f: VectorMatches → AllDocs (inclusion of vector search results)
    - g: MetadataMatches → AllDocs (inclusion of metadata filter results)
    """

    def __init__(self, vector_index: VectorIndex, metadata_index: MetadataIndex):
        self.vector = vector_index
        self.metadata = metadata_index

    def search(
        self,
        query_vector: np.ndarray,
        filter_predicate: Callable[[Document], bool],
        k: int
    ) -> List[SearchResult]:
        """
        Compute pullback: docs that match both vector query AND filter.

        Implementation strategies:
        1. Pre-filter: Apply metadata filter first, then vector search on subset
        2. Post-filter: Vector search first, then filter results
        3. In-filter: Integrate filter into HNSW traversal (advanced)
        """

        # Strategy depends on selectivity
        filter_selectivity = self.metadata.estimate_selectivity(filter_predicate)

        if filter_selectivity < 0.1:  # Highly selective filter
            # Pre-filter: get candidate IDs first
            candidate_ids = self.metadata.filter(filter_predicate)
            # Vector search only on candidates
            return self.vector.search_subset(query_vector, candidate_ids, k)
        else:
            # Post-filter: vector search then filter
            results = self.vector.search(query_vector, k=k * 10)  # Over-fetch
            filtered = [r for r in results if filter_predicate(r.document)]
            return filtered[:k]

    def pullback_square(self):
        """
        Verify pullback universal property:

        For any Z with morphisms to both VectorMatches and MetadataMatches
        that agree on AllDocs, there's a unique morphism Z → FilteredResults.
        """
        pass
```

---

## 5. Index Construction as F-Algebra (Catamorphism)

### The Insight

Building an index from documents is a fold (catamorphism):
- **F-Algebra**: `F(Index) → Index` where F captures "add one document"
- **Catamorphism**: Fold over document stream to produce final index

```
Documents: [d₁, d₂, d₃, ...]
               │
               │ cata(add_doc)
               ↓
           Final Index
```

### Implementation

```python
# categorical/index_algebra.py

class IndexAlgebra:
    """
    F-Algebra for index construction.

    F(X) = 1 + Doc × X  (either empty or a document paired with rest)

    Algebra: F(Index) → Index
    - nil: 1 → Index           (empty index)
    - cons: Doc × Index → Index (add document to index)
    """

    @abstractmethod
    def nil(self) -> Index:
        """Create empty index"""
        pass

    @abstractmethod
    def cons(self, doc: Document, index: Index) -> Index:
        """Add document to index"""
        pass


class VectorIndexAlgebra(IndexAlgebra):
    """Algebra for building vector index"""

    def __init__(self, embedder: EmbeddingMorphism):
        self.embedder = embedder

    def nil(self) -> VectorIndex:
        return FlatVectorIndex(dimension=self.embedder.dimension)

    def cons(self, doc: Document, index: VectorIndex) -> VectorIndex:
        embedding = self.embedder.embed(doc.content)
        index.add([doc.id], embedding.reshape(1, -1))
        return index


def cata(algebra: IndexAlgebra, documents: Iterable[Document]) -> Index:
    """
    Catamorphism: fold documents into index using algebra.

    This is the unique morphism from initial algebra (list of docs)
    to the carrier (Index) that respects the algebra structure.
    """
    index = algebra.nil()
    for doc in documents:
        index = algebra.cons(doc, index)
    return index


# Usage
algebra = VectorIndexAlgebra(embedder)
index = cata(algebra, corpus.documents)
```

### Benefits
- **Streaming**: Process documents one at a time (constant memory for logic)
- **Parallelism**: Algebras can be combined (parallel fold with monoid)
- **Incremental**: Same algebra for initial build and updates

---

## 6. Distributed Search as Product (Sharding)

### The Insight

A distributed index is a product of shard indices:
- **Product**: `Index = Shard₁ × Shard₂ × ... × Shardₙ`
- **Projections**: `πᵢ: Index → Shardᵢ`
- **Universal property**: Queries factor through the product

```
                    Query
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Shard₁      Shard₂      Shard₃
          │           │           │
          └───────────┼───────────┘
                      ↓
                   Merge
                      │
                      ↓
                   Results
```

### Implementation

```python
# categorical/product_sharding.py

class ShardedIndex:
    """
    Product of shard indices with categorical structure.

    Satisfies: For any index X with morphisms to all shards,
    there's a unique morphism X → ShardedIndex.
    """

    def __init__(self, shards: List[VectorIndex], shard_fn: Callable[[str], int]):
        self.shards = shards
        self.shard_fn = shard_fn  # Determines which shard for a doc
        self.n_shards = len(shards)

    def projection(self, i: int) -> VectorIndex:
        """πᵢ: Product → Shardᵢ"""
        return self.shards[i]

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add vectors, routing to appropriate shards"""
        # Group by shard
        shard_groups = defaultdict(list)
        for id_, vec in zip(ids, vectors):
            shard_idx = self.shard_fn(id_)
            shard_groups[shard_idx].append((id_, vec))

        # Add to each shard
        for shard_idx, items in shard_groups.items():
            ids_batch = [item[0] for item in items]
            vecs_batch = np.array([item[1] for item in items])
            self.shards[shard_idx].add(ids_batch, vecs_batch)

    def search(self, query: np.ndarray, k: int) -> List[SearchResult]:
        """
        Search all shards (parallel) and merge.

        Uses product universal property: query factors through all projections.
        """
        from concurrent.futures import ThreadPoolExecutor

        def search_shard(shard):
            return shard.search(query, k)

        with ThreadPoolExecutor(max_workers=self.n_shards) as executor:
            shard_results = list(executor.map(search_shard, self.shards))

        # Merge: take top-k across all shards
        all_results = [r for results in shard_results for r in results]
        all_results.sort(key=lambda r: -r.score)
        return all_results[:k]


class ConsistentHashingShard:
    """Shard function using consistent hashing (preserves locality)"""

    def __init__(self, n_shards: int, replicas: int = 100):
        self.ring = self._build_ring(n_shards, replicas)

    def __call__(self, doc_id: str) -> int:
        h = hash(doc_id)
        # Find first shard on ring after hash
        return self._find_shard(h)
```

---

## 7. Multi-Modal Search as Natural Transformation

### The Insight

Transforming between modalities (text ↔ image) while preserving search structure is a natural transformation:

```
      Text Category                    Image Category
           │                                │
      F (embed)                        G (embed)
           │                                │
           ↓                                ↓
    Text Vectors ─────────η────────→ Image Vectors
                  (natural transform)
```

### Implementation

```python
# categorical/natural_transform.py

class ModalityFunctor(Protocol[T]):
    """Functor from modality to vector space"""

    def embed(self, item: T) -> np.ndarray:
        ...


class CrossModalNaturalTransformation:
    """
    Natural transformation between modality embeddings.

    η: F ⇒ G where F, G are embedding functors

    Naturality: η_B ∘ F(f) = G(f) ∘ η_A for any f: A → B

    In practice: A linear map that aligns embedding spaces.
    """

    def __init__(
        self,
        source_functor: ModalityFunctor,
        target_functor: ModalityFunctor,
        alignment_matrix: np.ndarray
    ):
        self.source = source_functor
        self.target = target_functor
        self.W = alignment_matrix  # Learned alignment

    def component(self, source_embedding: np.ndarray) -> np.ndarray:
        """
        η_X: F(X) → G(X)

        Transform source embedding to target space.
        """
        return self.W @ source_embedding

    def check_naturality(self, x, y, relation) -> bool:
        """
        Verify naturality square commutes:

        F(x) ──F(rel)──→ F(y)
          │               │
         η_x             η_y
          │               │
          ↓               ↓
        G(x) ──G(rel)──→ G(y)
        """
        # This is a test, not a guarantee
        pass


class CLIPNaturalTransform(CrossModalNaturalTransformation):
    """
    CLIP-based cross-modal alignment.

    CLIP already produces aligned embeddings, so the natural
    transformation is (approximately) identity.
    """

    def __init__(self, clip_model):
        self.clip = clip_model
        # Identity alignment since CLIP aligns during training
        dim = clip_model.embed_dim
        super().__init__(
            source_functor=CLIPTextEmbedder(clip_model),
            target_functor=CLIPImageEmbedder(clip_model),
            alignment_matrix=np.eye(dim)
        )
```

---

## 8. Similarity as Enriched Category (Lawvere Metric Spaces)

### The Insight

Similarity search operates in enriched categories where hom-sets are metric values:
- **Objects**: Documents
- **Hom(A, B)**: Distance from A to B (a real number, not a set)
- **Composition**: Triangle inequality `d(A,C) ≤ d(A,B) + d(B,C)`

This is a **Lawvere metric space** - a category enriched over `([0,∞], ≥, +)`.

### Implementation

```python
# categorical/enriched_category.py

class LawvereMetricSpace:
    """
    Category enriched over extended reals [0, ∞].

    Generalized metric where:
    - d(x, x) = 0 (identity)
    - d(x, z) ≤ d(x, y) + d(y, z) (composition/triangle)
    - d(x, y) may ≠ d(y, x) (asymmetric allowed)

    Many similarity measures fit this framework.
    """

    def __init__(self, distance_fn: Callable[[Any, Any], float]):
        self.distance = distance_fn

    def hom(self, a, b) -> float:
        """Hom-object: distance from a to b"""
        return self.distance(a, b)

    def identity(self, a) -> float:
        """Identity: d(a, a) = 0"""
        return 0.0

    def compose(self, a, b, c) -> float:
        """
        Composition gives triangle inequality:
        d(a, c) ≤ d(a, b) + d(b, c)
        """
        return self.hom(a, b) + self.hom(b, c)

    def verify_triangle(self, a, b, c) -> bool:
        """Check triangle inequality holds"""
        return self.hom(a, c) <= self.compose(a, b, c)


class ApproximateMetricSpace(LawvereMetricSpace):
    """
    Metric space where triangle inequality holds approximately.

    Common with learned embeddings:
    d(a, c) ≤ d(a, b) + d(b, c) + ε

    The ε bounds how "non-categorical" the embedding is.
    """

    def __init__(self, distance_fn, epsilon: float = 0.1):
        super().__init__(distance_fn)
        self.epsilon = epsilon

    def verify_approximate_triangle(self, a, b, c) -> bool:
        return self.hom(a, c) <= self.compose(a, b, c) + self.epsilon

    def measure_categoricity(self, samples: List[Tuple]) -> float:
        """
        Measure how well the space satisfies categorical laws.

        Returns average triangle violation.
        """
        violations = []
        for a, b, c in samples:
            violation = max(0, self.hom(a, c) - self.compose(a, b, c))
            violations.append(violation)
        return np.mean(violations)
```

---

## 9. Persistent Homology for Embedding Analysis

### The Insight

Use algebraic topology (with categorical foundations) to understand embedding space structure:
- **Simplicial complex**: Built from nearest neighbor graph
- **Homology**: Detects clusters (H₀), loops (H₁), voids (H₂)
- **Persistence**: Track features across distance scales

### Implementation Sketch

```python
# categorical/topological_analysis.py

class EmbeddingTopology:
    """
    Topological analysis of embedding space using persistent homology.

    Helps understand:
    - Cluster structure (connected components)
    - Semantic loops (e.g., antonym cycles)
    - Coverage gaps (voids in concept space)
    """

    def __init__(self, embeddings: np.ndarray, ids: List[str]):
        self.embeddings = embeddings
        self.ids = ids

    def rips_complex(self, epsilon: float) -> 'SimplicialComplex':
        """
        Build Vietoris-Rips complex at scale epsilon.

        Vertices: embedding points
        Edges: pairs within distance epsilon
        Triangles: triples where all pairs are edges
        """
        pass

    def persistence_diagram(self, max_dim: int = 2) -> 'PersistenceDiagram':
        """
        Compute persistent homology across all scales.

        Returns: birth-death pairs for topological features
        """
        pass

    def significant_features(self, persistence_threshold: float) -> Dict:
        """
        Identify topologically significant structures.

        - Long-lived H₀: Robust clusters
        - Long-lived H₁: Semantic cycles
        """
        pass
```

---

## 10. Query Optimization as Free Monad

### The Insight

Complex query execution can be modeled as a free monad:
- **DSL**: Query operations (search, filter, join, rerank)
- **Interpreter**: Different execution strategies
- **Optimization**: Rewrite DSL before interpreting

```python
# categorical/query_dsl.py

class QueryF:
    """Functor for query DSL"""
    pass

@dataclass
class Search(QueryF):
    query: str
    k: int
    next: Callable  # Continuation

@dataclass
class Filter(QueryF):
    predicate: Callable
    next: Callable

@dataclass
class Rerank(QueryF):
    reranker: Callable
    next: Callable

@dataclass
class Return(QueryF):
    results: List


class FreeQuery:
    """
    Free monad over QueryF.

    Allows building query plans that can be:
    1. Interpreted directly
    2. Optimized then interpreted
    3. Analyzed for cost estimation
    """

    @staticmethod
    def search(query: str, k: int) -> 'FreeQuery':
        return FreeQuery(Search(query, k, lambda r: Return(r)))

    @staticmethod
    def filter(predicate: Callable) -> Callable[['FreeQuery'], 'FreeQuery']:
        return lambda fq: FreeQuery(Filter(predicate, fq))

    def bind(self, f: Callable) -> 'FreeQuery':
        """Monadic bind"""
        pass


class QueryOptimizer:
    """Optimize query plans before execution"""

    def push_filters_down(self, query: FreeQuery) -> FreeQuery:
        """Move filters before search when possible"""
        pass

    def fuse_rerankers(self, query: FreeQuery) -> FreeQuery:
        """Combine consecutive rerank operations"""
        pass


class QueryInterpreter:
    """Execute query plans"""

    def interpret(self, query: FreeQuery, engine: SearchEngine) -> List[SearchResult]:
        """Run the query against an engine"""
        match query.op:
            case Search(q, k, next):
                results = engine.search(q, k)
                return self.interpret(next(results), engine)
            case Filter(pred, next):
                # ... etc
                pass
            case Return(results):
                return results
```

---

## Summary: Where Category Theory Helps

| Problem | Categorical Tool | Benefit |
|---------|-----------------|---------|
| Embeddings | Functor | Structure preservation guarantees |
| Relevance feedback | Monad | Clean composition of iterations |
| Hybrid search | Coproduct | Universal fusion interface |
| Filtered search | Pullback | Precise constraint intersection |
| Index building | F-Algebra | Streaming + incremental |
| Distributed search | Product | Clean sharding abstraction |
| Multi-modal | Natural transformation | Aligned cross-modal search |
| Similarity | Enriched category | Proper distance axioms |
| Topology | Homology | Embedding structure analysis |
| Query planning | Free monad | DSL with optimization |

---

## Next Steps

1. **Implement core abstractions** in `vajra_bm25/categorical/`
2. **Add tests** verifying categorical laws hold
3. **Benchmark** to ensure abstractions don't hurt performance
4. **Document** the mathematical foundations for users
