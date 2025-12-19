# Categorical Search Improvement Plan

## The Problem

Vajra BM25 is currently ~2x slower than BM25S for query latency on 200K documents:
- Vajra: ~2.2ms per query
- BM25S: ~1.7ms per query

Brute-force engineering (Numba, GPU) can close this gap, but doesn't leverage Vajra's unique strength: its categorical foundations.

## The Insight

BM25S and rank-bm25 treat search as imperative computation. Vajra treats search as mathematical structure. This difference should manifest as performance advantage, not just correctness.

**Key observation**: The BM25 scoring function has rich algebraic structure we're not exploiting.

## Categorical Structure of BM25

### 1. Monoid Homomorphism

BM25 scoring is a **monoid homomorphism**:

```
Queries: (Terms*, ⊕, ε)     -- Free monoid over terms
Scores:  (ℝ, +, 0)          -- Additive monoid

score: Queries → Scores
score(q₁ ⊕ q₂) = score(q₁) + score(q₂)   -- Homomorphism
score(ε) = 0                               -- Preserves identity
```

**Implication**: Query results can be COMPOSED rather than recomputed.

### 2. Comonadic Caching

The current LRU cache treats queries as opaque strings. But queries have internal structure.

```
Query "machine learning algorithms" contains sub-queries:
  - "machine"
  - "learning"
  - "algorithms"
  - "machine learning"
  - ...

Comonadic structure: extract sub-results, extend to full results
```

**Implication**: Cache hits on sub-queries accelerate super-queries.

### 3. Coalgebraic Early Termination

Search is a coalgebra: `α: State → F(State)` where we unfold results lazily.

The **anamorphism** (unfold) can include **guards**:
```
unfold(state) =
  if guard(state) then Stop
  else Continue(next_state, result)
```

For BM25, the guard is: `current_score + upper_bound < threshold`

**Implication**: Most documents are never scored. Only those that CAN appear in top-k are evaluated.

### 4. Functorial Index Structure

The index is a functor: `Index: Term → PostingList`

Functors preserve structure. If we enrich the PostingList with pre-computed bounds:
```
PostingList = [(DocID, TF, MaxPossibleScore)]
```

The functor now carries optimization data through all transformations.

## Proposed Architecture

### Phase 1: Enriched Index (Functorial)

Modify `VectorizedIndexSparse` to compute and store:

1. **Max term contribution**: For each term, the maximum BM25 score it can contribute to any document
   ```python
   max_term_score[t] = max over all docs d of: IDF[t] * BM25_tf_component(t, d)
   ```

2. **Document upper bounds**: For each document, sum of max contributions from all its terms
   ```python
   doc_upper_bound[d] = sum of max_term_score[t] for t in doc[d]
   ```

This is functorial: we're lifting score computation from query-time to index-time.

### Phase 2: Compositional Query Cache (Comonadic)

Replace simple LRU cache with a **compositional cache**:

```python
class CompositionalCache:
    """
    Comonadic cache that understands query structure.

    Key insight: Queries are elements of a free monoid.
    Cache stores morphisms, not just results.
    """

    def __init__(self):
        self.term_scores = {}      # term → doc_scores (sparse)
        self.bigram_scores = {}    # (term1, term2) → doc_scores

    def get(self, query_terms: Tuple[str]) -> Optional[Scores]:
        # Try exact match
        if query_terms in self.cache:
            return self.cache[query_terms]

        # Try composition (monoid homomorphism!)
        if len(query_terms) > 1:
            # Split and compose
            mid = len(query_terms) // 2
            left = self.get(query_terms[:mid])
            right = self.get(query_terms[mid:])
            if left is not None and right is not None:
                return left + right  # Monoid operation

        return None
```

### Phase 3: Guarded Coalgebraic Scoring

Implement **MaxScore algorithm** using coalgebraic structure:

```python
class GuardedScorer:
    """
    Coalgebraic BM25 scorer with early termination.

    The guard checks if a document CAN appear in top-k.
    If not, we skip it (coalgebraic "stop" branch).
    """

    def score_with_guard(self, query_terms, top_k):
        # Sort terms by max contribution (essential → non-essential)
        sorted_terms = sorted(query_terms,
                              key=lambda t: self.max_term_score[t],
                              reverse=True)

        # Cumulative upper bounds
        upper_bounds = cumsum_reverse([self.max_term_score[t] for t in sorted_terms])

        threshold = 0.0
        scores = {}

        for i, term in enumerate(sorted_terms):
            remaining_potential = upper_bounds[i+1]

            for doc_id, tf in self.postings[term]:
                # COALGEBRAIC GUARD
                current = scores.get(doc_id, 0.0)
                if current + remaining_potential + self.max_term_score[term] < threshold:
                    continue  # Skip - can't make top-k

                # Score this term-document pair
                scores[doc_id] = current + self.bm25_score(term, doc_id, tf)

            # Update threshold periodically
            threshold = self.kth_largest(scores, top_k)

        return top_k_results(scores, top_k)
```

### Phase 4: Query Lattice Optimization (Functorial Lifting)

Queries form a **lattice** under term inclusion:
```
"machine" ≤ "machine learning" ≤ "machine learning algorithms"
```

If we've computed results for a sub-query, we can **lift** to super-queries:

```python
class QueryLattice:
    """
    Exploits lattice structure of queries.

    If query Q₁ ⊆ Q₂, then:
    - Documents matching Q₂ ⊆ Documents matching Q₁
    - We can filter Q₁ results instead of re-searching
    """

    def search(self, query_terms):
        # Find cached super-set query
        for cached_query, cached_results in self.cache.items():
            if set(cached_query) <= set(query_terms):
                # Lift cached results
                additional_terms = set(query_terms) - set(cached_query)
                return self.extend_results(cached_results, additional_terms)

        # No useful cache hit - compute fresh
        return self.compute_fresh(query_terms)
```

## Implementation Order

1. **Phase 1: Enriched Index** (1-2 days)
   - Add max_term_score computation to index build
   - No API changes, just richer internal structure

2. **Phase 2: Guarded Scorer** (2-3 days)
   - Implement MaxScore with coalgebraic guards
   - Expected: 2-5x speedup for top-k queries

3. **Phase 3: Compositional Cache** (2-3 days)
   - Replace LRU with monoid-aware cache
   - Expected: 10-100x speedup for repeated/similar queries

4. **Phase 4: Query Lattice** (3-5 days)
   - Implement lattice-based query optimization
   - Expected: Significant speedup for query workloads with overlap

## Expected Outcomes

| Scenario | Current | After Phase 2 | After Phase 4 |
|----------|---------|---------------|---------------|
| Cold query | 2.2ms | 0.8ms | 0.8ms |
| Warm query (exact) | 0.01ms | 0.01ms | 0.01ms |
| Warm query (subset) | 2.2ms | 0.8ms | 0.1ms |
| Query burst (similar) | 2.2ms × N | 0.8ms × N | 0.8ms + 0.1ms × (N-1) |

## Why This Is Different From BM25S

BM25S optimizes through:
- Numba JIT compilation
- Optimized C/Cython code
- Better memory layout

Vajra optimizes through:
- **Mathematical structure** (monoid homomorphism)
- **Compositional caching** (comonadic)
- **Guarded computation** (coalgebraic)
- **Lattice lifting** (functorial)

These are orthogonal. We can use BOTH:
- Numba for the inner loop
- Categorical structure for algorithmic speedup

The categorical approach gives **asymptotic** improvements (skip entire documents).
The engineering approach gives **constant factor** improvements (faster per-document scoring).

## Validation Criteria

1. **Correctness**: Results must be identical to current implementation (BM25 is exact)
2. **Cold performance**: Single query latency < 1ms on 200K docs
3. **Warm performance**: Similar queries should be 10x+ faster
4. **Memory**: Index size increase < 50%

## Open Questions

1. Should compositional cache be term-level or bigram-level?
2. How often should we update the top-k threshold in guarded scoring?
3. Can we use the lattice structure for approximate (but bounded-error) results?
4. How does this interact with document updates (incremental index)?
