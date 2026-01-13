# Category Theory in Vajra

Vajra uses Category Theory as a principled foundation for its search engine design. This guide explains the key concepts and how they map to search operations.

## Why Category Theory?

Category theory provides:

1. **Compositionality**: Build complex operations from simple pieces
2. **Correctness**: Mathematical guarantees about behavior
3. **Optimization**: Algebraic laws enable automatic transformations
4. **Abstraction**: Separate "what" from "how"

## Core Concepts

### Objects and Morphisms

In category theory, we work with **objects** (things) and **morphisms** (transformations between things).

```
Objects:    Query, Document, Score, SearchResult
Morphisms:  tokenize, score, rank
```

In Vajra:

- **Documents** are objects
- **BM25 scoring** is a morphism: `(Query, Document) → Score`
- **Search** is a morphism: `Query → List[SearchResult]`

### Composition

Morphisms compose: if `f: A → B` and `g: B → C`, then `g ∘ f: A → C`.

```python
# Text processing pipeline
preprocess = tokenize >> remove_stopwords >> stem
query_tokens = preprocess(query)
```

This enables:

- Reusable components
- Clear data flow
- Easy testing

## The Search Category

Vajra defines a category for search:

```
Objects:
  - Query (search query string)
  - TokenizedQuery (list of tokens)
  - Document (searchable document)
  - Score (relevance score in ℝ)
  - SearchResult (document + score + rank)

Morphisms:
  - tokenize: Query → TokenizedQuery
  - score: (TokenizedQuery, Document) → Score
  - rank: List[Score] → List[SearchResult]
  - search: Query → List[SearchResult] (composite)
```

### Identity and Composition Laws

Every category must satisfy:

1. **Identity**: `id ∘ f = f = f ∘ id`
2. **Associativity**: `(h ∘ g) ∘ f = h ∘ (g ∘ f)`

These laws ensure that pipelines behave predictably.

## Functors: Structure Preservation

A **functor** maps between categories while preserving structure.

### List Functor

The List functor lifts single-element operations to work on lists:

```python
# Single document scoring
score_doc: Document → Score

# Lifted to lists (via functor)
score_docs: List[Document] → List[Score]
```

This is why we can score multiple documents efficiently:

```python
# Vectorized scoring is a functor application
scores = functor.fmap(score_function, documents)
```

### SearchResult Functor

Transforms results while preserving ranking structure:

```python
# Add snippets to results without changing ranks
results_with_snippets = result_functor.fmap(add_snippet, results)
```

## Coalgebras: Search as Unfolding

A **coalgebra** models processes that unfold step by step.

### Search Unfolding

Search is naturally coalgebraic:

```
State₀ (query)
    ↓ step
State₁ (tokens + candidates)
    ↓ step
State₂ (scored documents)
    ↓ step
State₃ (ranked results) ← terminal
```

```python
class SearchCoalgebra:
    def step(self, state):
        if is_terminal(state):
            return state.results
        new_state = process(state)
        return step(new_state)
```

### MaxScore Algorithm

The MaxScore early termination algorithm is coalgebraic:

1. Start with all documents as candidates
2. Score documents by term
3. Check upper bound vs current threshold
4. **Terminate early** if no candidates can beat threshold

```python
def maxscore_step(state):
    if state.upper_bound < state.min_threshold:
        return state.results, TERMINAL  # Early exit!
    # Continue scoring...
```

This coalgebraic structure enables principled early termination.

## Comonads: Caching with Context

A **comonad** provides context-aware computation.

### LRU Cache as Comonad

Query caching follows comonadic structure:

```python
class Cache:
    def extract(self, key):
        """Get value from cache context"""
        return self.cache.get(key)

    def extend(self, f, key):
        """Compute in caching context"""
        cached = self.extract(key)
        if cached:
            return cached
        result = f(key)
        self.store(key, result)
        return result
```

The comonadic laws ensure:

- **Consistent extraction**: Same key → same value
- **Compositional caching**: Nested caches work correctly

## Monoid Homomorphisms

BM25 scoring is a **monoid homomorphism**.

### Monoids

A monoid is a set with:
- Binary operation `⊕`
- Identity element `ε`
- Associativity: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`

### BM25 Scoring Monoid

```
Queries:  (Terms*, concatenation, empty)  -- Free monoid
Scores:   (ℝ, +, 0)                        -- Additive monoid
```

BM25 preserves this structure:

```
score(q₁ ⊕ q₂, doc) = score(q₁, doc) + score(q₂, doc)
```

### Practical Implications

This algebraic property enables:

1. **Compositional Scoring**
   ```python
   # Score terms independently, then combine
   total = sum(term_scores[t] for t in query_terms)
   ```

2. **Upper Bound Computation**
   ```python
   # Max possible score = sum of max term scores
   upper_bound = sum(max_term_scores[t] for t in query_terms)
   ```

3. **Parallel Scoring**
   ```python
   # Terms can be scored independently
   term_scores = parallel_map(score_term, query_terms)
   total = sum(term_scores)
   ```

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    Search Category                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Query ──tokenize──▶ Tokens ──score──▶ Scores ──rank──▶ Results │
│                                                              │
│  Functors: Lift operations to work on lists                 │
│  Coalgebra: Unfold search step by step, enable early exit   │
│  Comonad: Cache with context awareness                      │
│  Monoid: Enable compositional scoring and bounds            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Performance Impact

These abstractions aren't just theoretical—they enable real optimizations:

| Abstraction | Optimization | Impact |
|-------------|--------------|--------|
| Morphism composition | Fused pipelines | Reduced allocations |
| Functor | Vectorized scoring | 10-50x speedup |
| Coalgebra | MaxScore early exit | 20-40% faster queries |
| Comonad | LRU caching | Near-zero repeated query time |
| Monoid homomorphism | Upper bounds | Early termination |

## Further Reading

- [API Reference: Categorical](api/categorical.md) - Implementation details
- [Performance Tips](guide/performance.md) - Practical optimizations
- [Benchmarks](benchmarks.md) - Measured impact

### External Resources

- *Category Theory for Programmers* by Bartosz Milewski
- *Categories for the Working Mathematician* by Saunders Mac Lane
- Coalgebraic semantics for streaming/incremental computation
