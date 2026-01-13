# Categorical Abstractions API

Vajra implements BM25 search using Category Theory primitives. These abstractions provide mathematical rigor and enable compositional optimizations.

## Overview

| Abstraction | Purpose | Location |
|-------------|---------|----------|
| `Morphism` | Function composition | `categorical/morphism.py` |
| `Functor` | Structure-preserving maps | `categorical/functor.py` |
| `Coalgebra` | State-based unfolding | `categorical/coalgebra.py` |
| `Comonad` | Context-aware caching | `categorical/comonad.py` |

## Morphism

A composable function between types.

```python
from vajra_bm25.categorical import Morphism

# Define morphisms
tokenize = Morphism(lambda text: text.lower().split())
count = Morphism(lambda tokens: len(tokens))

# Compose morphisms
word_count = tokenize >> count

result = word_count("Hello World")  # 2
```

### Class Definition

```python
class Morphism(Generic[A, B]):
    def __init__(self, f: Callable[[A], B]):
        self.f = f

    def __call__(self, x: A) -> B:
        return self.f(x)

    def __rshift__(self, other: 'Morphism[B, C]') -> 'Morphism[A, C]':
        """Compose morphisms: f >> g = g ∘ f"""
        return Morphism(lambda x: other(self(x)))
```

### BM25 as Morphism

BM25 scoring is a morphism from (Query, Document) pairs to scores:

```
score: (Query × Document) → ℝ
```

```python
from vajra_bm25.categorical import Morphism

# BM25 scoring morphism
bm25_score = Morphism(lambda pair: compute_bm25(pair[0], pair[1]))

# Apply to query-document pairs
scores = [bm25_score((query, doc)) for doc in documents]
```

## Functor

A structure-preserving map between categories.

```python
from vajra_bm25.categorical import ListFunctor

# Apply function to all elements, preserving list structure
functor = ListFunctor()
results = functor.fmap(lambda x: x * 2, [1, 2, 3])  # [2, 4, 6]
```

### Class Definition

```python
class Functor(Generic[F]):
    def fmap(self, f: Callable[[A], B], fa: F[A]) -> F[B]:
        """Apply function while preserving structure"""
        raise NotImplementedError
```

### SearchResultFunctor

Used to transform search results while preserving ranking:

```python
class SearchResultFunctor(Functor):
    def fmap(self, f, results):
        """Transform results while preserving rank structure"""
        return [
            SearchResult(
                document=f(r.document),
                score=r.score,
                rank=r.rank
            )
            for r in results
        ]
```

## Coalgebra

Models search as state-based unfolding.

```python
from vajra_bm25.categorical import SearchCoalgebra

# Define initial state
state = QueryState(query="machine learning", corpus=corpus)

# Unfold to get results
coalgebra = SearchCoalgebra(scorer=bm25_scorer)
results = coalgebra.unfold(state, top_k=10)
```

### Coalgebraic Search

Search is modeled as an "unfold" operation:

```
unfold: State → List[Result]
```

The state transitions are:

1. **Initial**: Query string + Corpus
2. **Tokenized**: Query tokens + Document candidates
3. **Scored**: Ranked (score, document) pairs
4. **Final**: Top-k results

```python
@dataclass
class QueryState:
    query: str
    corpus: DocumentCorpus
    candidates: Optional[List[Document]] = None
    scores: Optional[Dict[str, float]] = None

class SearchCoalgebra:
    def step(self, state: QueryState) -> Tuple[List[SearchResult], QueryState]:
        """Single step of search unfolding"""
        if state.scores is None:
            # Compute scores
            scores = self.scorer(state.query, state.candidates)
            return [], QueryState(
                query=state.query,
                corpus=state.corpus,
                candidates=state.candidates,
                scores=scores
            )
        else:
            # Return top-k
            results = self.select_top_k(state.scores)
            return results, state  # Terminal state

    def unfold(self, state: QueryState, top_k: int) -> List[SearchResult]:
        """Unfold until terminal state"""
        results = []
        while not self.is_terminal(state):
            new_results, state = self.step(state)
            results.extend(new_results)
        return results[:top_k]
```

### MaxScore as Coalgebra

The MaxScore algorithm uses coalgebraic early termination:

```python
class MaxScoreCoalgebra:
    def step(self, state):
        """Early termination when upper bound < current min"""
        if state.upper_bound < state.threshold:
            return state.results, state  # Terminal
        # Continue scoring
        ...
```

## Comonad

Context-aware computation with extract and duplicate operations.

```python
from vajra_bm25.categorical import CacheComonad

# Create cached computation
cache = CacheComonad(compute_expensive_result)

# Extract current value
result = cache.extract()

# Duplicate to get cache of caches (for nested contexts)
nested = cache.duplicate()
```

### LRU Cache as Comonad

Query caching follows comonadic structure:

```python
class LRUCacheComonad:
    def __init__(self, cache_size: int):
        self.cache = OrderedDict()
        self.max_size = cache_size

    def extract(self, key: str) -> Optional[Any]:
        """Get cached value if present"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def extend(self, f: Callable, key: str) -> Any:
        """Compute with caching context"""
        cached = self.extract(key)
        if cached is not None:
            return cached
        result = f(key)
        self._store(key, result)
        return result
```

## Monoid Homomorphism

BM25 scoring preserves monoid structure:

```
Queries: (Terms*, ⊕, ε)  -- Free monoid over terms
Scores:  (ℝ, +, 0)       -- Additive monoid

score(q₁ ⊕ q₂) = score(q₁) + score(q₂)
```

This enables:

- **Compositional caching**: Cache term-level scores
- **Upper bounds**: Pre-compute max possible score per term
- **Parallelization**: Score terms independently, combine results

```python
def score_query(query_terms, document):
    """Exploit monoid homomorphism for efficiency"""
    return sum(
        term_score_cache.get(term, document)
        for term in query_terms
    )
```

## Practical Usage

Most users won't interact with categorical primitives directly. They're used internally for:

1. **Code organization**: Clear separation of concerns
2. **Compositional optimization**: Enable algebraic transformations
3. **Correctness**: Mathematical guarantees about behavior
4. **Caching**: Comonadic structure for efficient memoization

### Direct Usage (Advanced)

```python
from vajra_bm25.categorical import Morphism, ListFunctor

# Custom scoring pipeline
preprocess = Morphism(lambda q: q.lower().strip())
tokenize = Morphism(lambda q: q.split())
score = Morphism(lambda tokens: compute_score(tokens))

pipeline = preprocess >> tokenize >> score

# Apply to queries
results = ListFunctor().fmap(pipeline, queries)
```

## Further Reading

- [Category Theory Guide](../category-theory.md) - Conceptual introduction
- [Performance Tips](../guide/performance.md) - How these enable optimizations
- [Benchmarks](../benchmarks.md) - Performance impact of categorical design
