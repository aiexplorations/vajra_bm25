# Mathematical Enhancements for Vajra BM25

## Philosophy

Search is not "scoring then selecting." Search is a mathematical structure with rich algebraic properties. This document explores the deep structure of BM25 search and how to exploit it for extreme performance.

The goal: **follow the structure**, not "optimize the loop."

---

## 1. The Free Monoid Structure of Queries

### Mathematical Foundation

A query is an element of the **free monoid** on terms: `(Term*, ⊕, ε)`

```
"machine learning" = "machine" ⊕ "learning"
```

The free monoid is the **universal** way to combine terms. This isn't notation—it's algebraic structure that we can exploit.

**Key Property**: BM25 scoring is a **monoid homomorphism**:
```
score: (Term*, ⊕, ε) → (ℝ, +, 0)
score(q₁ ⊕ q₂) = score(q₁) + score(q₂)
```

### Implementation

```python
class CompositionalScorer:
    """
    Exploit monoid homomorphism for compositional scoring.

    Since score(q₁ ⊕ q₂) = score(q₁) + score(q₂), we can:
    1. Cache term-level scores
    2. Compose cached results for multi-term queries
    3. Incrementally update scores as terms are added
    """

    def __init__(self, index):
        self.index = index
        # Cache: term → sparse score vector over all docs
        self.term_scores: Dict[str, SparseVector] = {}

    def get_term_scores(self, term: str) -> SparseVector:
        """Get or compute scores for a single term."""
        if term not in self.term_scores:
            # Compute IDF × TF for all docs containing term
            self.term_scores[term] = self._compute_term_scores(term)
        return self.term_scores[term]

    def score_query(self, query_terms: List[str]) -> SparseVector:
        """Compose term scores using monoid structure."""
        # Monoid homomorphism: just add the sparse vectors!
        result = SparseVector.zero()
        for term in query_terms:
            result = result + self.get_term_scores(term)
        return result
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Compositional caching** | Cache term-level scores, reuse across queries | 10-100x for repeated terms |
| **Incremental updates** | Add term scores one at a time | Enables early termination |
| **Parallel decomposition** | Score terms independently, combine | Linear speedup with cores |
| **Query algebra** | Combine cached sub-queries | Fast for query refinement |

---

## 2. The Coend Formula for Scoring

### Mathematical Foundation

The BM25 score is a **coend**:

```
score(q, d) = ∫^t q(t) ⊗ R(t, d)
```

Where:
- `q: Term → ℝ` is the query as a weighted term function (IDF weights)
- `R: Term × Doc → ℝ` is the relevance relation (TF component)
- `∫^t` is the coend—the universal "sum" that respects the indexing

**Interpretation**: Scoring is the **trace** of the query-document bimodule. The coend "contracts" the term index, leaving only the query-document relationship.

**Key Insight**: The coend only sums over **non-zero entries**. This is why sparse representations are natural—they represent the coend directly.

### Implementation

```python
class CoendScorer:
    """
    BM25 scoring as coend computation.

    The coend formula: score(q, d) = ∫^t q(t) ⊗ R(t, d)

    Implementation insight: the coend is computed by:
    1. Iterating over the "support" (non-zero entries)
    2. Accumulating contributions
    3. The iteration order doesn't affect the result (coend is universal)

    This means we can choose ANY iteration order for efficiency!
    """

    def __init__(self, index):
        self.index = index
        # R(t, d) factored as IDF[t] × TF_component[t, d]
        self.idf = index.idf_cache
        self.tf_matrix = index.term_doc_matrix  # Sparse CSR

    def compute_coend(self, query_terms: List[str],
                      candidates: np.ndarray) -> np.ndarray:
        """
        Compute coend over candidate documents.

        The coend contracts over terms:
        score[d] = Σ_t IDF[t] × TF_component[t, d]

        We iterate over terms (rows of sparse matrix) and accumulate.
        """
        scores = np.zeros(len(candidates), dtype=np.float32)

        for term in query_terms:
            term_id = self.index.term_to_id.get(term)
            if term_id is None:
                continue

            # Get sparse row for this term
            row = self.tf_matrix.getrow(term_id)

            # Coend contribution: IDF[t] × TF_component[t, d]
            # Only non-zero entries contribute (coend property)
            idf = self.idf[term_id]

            for doc_idx, tf in zip(row.indices, row.data):
                if candidates[doc_idx]:
                    tf_component = self._tf_component(tf, doc_idx)
                    scores[doc_idx] += idf * tf_component

        return scores
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Principled sparsity** | Coend naturally handles sparse data | No wasted computation |
| **Order independence** | Any term order gives same result | Enables optimal ordering |
| **Factorization** | R = IDF × TF factors the computation | Precompute IDF once |
| **Incremental coend** | Add term contributions one at a time | Enables early stopping |

---

## 3. The Profunctor Structure of the Index

### Mathematical Foundation

A **profunctor** `P: A ↛ B` is a functor `A^op × B → Set`.

The term-document relationship is naturally a profunctor:
```
P: Term ↛ Doc
P(t, d) = {occurrences of t in d}
```

The **inverted index** is the **collage** (or cograph) of this profunctor—a category whose:
- Objects are terms and documents
- Morphisms from term t to doc d are the occurrences

**Key Insight**: Profunctors compose! Query processing is **profunctor composition**:
```
Query ↛ Term ↛ Doc
```

The composition gives us the query-document relationship directly.

### Implementation

```python
class ProfunctorIndex:
    """
    Inverted index as profunctor representation.

    The profunctor P: Term ↛ Doc is represented as:
    - For each term t: the set of (doc, tf) pairs
    - This is P(t, -): Doc → Set

    Profunctor composition corresponds to:
    - Query terms select relevant postings
    - Postings are "multiplied" (scored) and "summed" (aggregated)

    This is exactly matrix multiplication in the profunctor category!
    """

    def __init__(self):
        # Profunctor representation: term → List[(doc, weight)]
        self.postings: Dict[int, List[Tuple[int, float]]] = {}

        # Enrichment: store additional structure per posting
        self.max_weight: Dict[int, float] = {}  # Max weight per term

    def compose_with_query(self, query_weights: Dict[str, float]) -> Dict[int, float]:
        """
        Compose query profunctor with index profunctor.

        Query: 1 ↛ Term (query as weighted terms)
        Index: Term ↛ Doc (index as postings)

        Composition: 1 ↛ Doc (documents with scores)

        In matrix terms: score = query × index
        """
        scores = defaultdict(float)

        for term, query_weight in query_weights.items():
            term_id = self.term_to_id.get(term)
            if term_id is None:
                continue

            # Compose: query_weight × posting_weight
            for doc_id, posting_weight in self.postings[term_id]:
                scores[doc_id] += query_weight * posting_weight

        return scores
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Compositional queries** | Queries compose with index naturally | Clean abstractions |
| **Enriched postings** | Store max weights, bounds per posting | Enable pruning |
| **Profunctor transformations** | Map, filter, transform postings | Flexible scoring |
| **Weighted composition** | Query weights flow through naturally | Boosting, field weights |

---

## 4. The Galois Connection for Candidate Refinement

### Mathematical Foundation

A **Galois connection** between posets (A, ≤) and (B, ≤) is a pair of monotone functions:
```
α: A → B    (left adjoint)
β: B → A    (right adjoint)

such that: α(a) ≤ b  ⟺  a ≤ β(b)
```

For search, we have a Galois connection between term sets and document sets:
```
α: P(Terms) → P(Docs)    -- documents containing ALL terms
β: P(Docs) → P(Terms)    -- terms appearing in ALL documents

Key property: α is ANTITONE in the subset ordering
α(T₁) ⊇ α(T₁ ∪ {t})     -- more terms = fewer documents
```

**Key Insight**: The Galois connection tells us **which terms to process first**. The most selective term (smallest `α({t})`) shrinks the candidate set most.

### Implementation

```python
class GaloisCandidateRefinement:
    """
    Use Galois connection structure for candidate set refinement.

    The Galois connection α ⊣ β gives us:
    1. α(T) = documents containing all terms in T
    2. Adding terms to T shrinks α(T)
    3. Process most selective terms first

    This is not an optimization—it's following the structure.
    """

    def __init__(self, index):
        self.index = index
        # Precompute selectivity (document frequency) for each term
        self.selectivity = {
            term: len(postings)
            for term, postings in index.postings.items()
        }

    def refine_candidates(self, query_terms: List[str]) -> Iterator[Set[int]]:
        """
        Iteratively refine candidate set using Galois connection.

        Yields progressively smaller candidate sets.

        The refinement follows the lattice structure:
        α({}) ⊇ α({t₁}) ⊇ α({t₁, t₂}) ⊇ ...
        """
        # Order by selectivity (most selective = smallest posting list)
        ordered_terms = sorted(query_terms,
                               key=lambda t: self.selectivity.get(t, float('inf')))

        candidates = None  # Start with all docs (top of lattice)

        for term in ordered_terms:
            term_docs = self.index.get_doc_set(term)

            if candidates is None:
                candidates = term_docs
            else:
                # Galois: intersection refines the set
                candidates = candidates & term_docs

            yield candidates

            # Early termination: if candidates too small, stop
            if len(candidates) < self.min_candidates:
                break

    def search_with_galois(self, query_terms: List[str], top_k: int):
        """
        Search using Galois-guided candidate refinement.
        """
        for candidates in self.refine_candidates(query_terms):
            # Score only current candidates
            scores = self.score_candidates(candidates, query_terms)

            # Check if we have enough high-quality results
            if self.have_enough_results(scores, top_k):
                return self.extract_top_k(scores, top_k)

        # Final scoring on remaining candidates
        return self.extract_top_k(scores, top_k)
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Optimal term ordering** | Most selective terms first | 2-10x fewer candidates |
| **Progressive refinement** | Candidate set shrinks monotonically | Early termination |
| **Lattice structure** | Well-defined refinement steps | Predictable behavior |
| **Intersection-first** | Cheap set intersection before expensive scoring | Major speedup |

---

## 5. The Tropical Semiring for Bound Propagation

### Mathematical Foundation

The **tropical semiring** is `(ℝ ∪ {-∞}, ⊕, ⊗)` where:
- `a ⊕ b = max(a, b)` (tropical addition)
- `a ⊗ b = a + b` (tropical multiplication)

BM25 upper bound computation naturally lives in this semiring:
```
upper_bound[d] = current_score[d] ⊗ remaining_potential
              = current_score[d] + Σ max_contribution[remaining_terms]
```

**Tropical Convexity**: The set of achievable (score, upper_bound) pairs forms a tropical polytope. Documents outside this polytope can be pruned.

**Key Insight**: Tropical algebra gives us **geometric** intuition for pruning. We're not just comparing numbers—we're doing geometry in log-space.

### Implementation

```python
class TropicalBoundTracker:
    """
    Track score bounds using tropical semiring structure.

    The tropical semiring (max, +) naturally models:
    - Upper bounds: current + remaining potential
    - Threshold updates: max of k-th best scores
    - Pruning: tropical comparison

    Tropical convexity gives geometric pruning.
    """

    def __init__(self, index, k: int):
        self.index = index
        self.k = k

        # Tropical "negative infinity"
        self.NEG_INF = float('-inf')

        # Current state
        self.scores = np.full(index.num_docs, self.NEG_INF, dtype=np.float32)
        self.threshold = self.NEG_INF

    def tropical_add(self, a: float, b: float) -> float:
        """Tropical addition: max(a, b)"""
        return max(a, b)

    def tropical_mult(self, a: float, b: float) -> float:
        """Tropical multiplication: a + b"""
        if a == self.NEG_INF or b == self.NEG_INF:
            return self.NEG_INF
        return a + b

    def compute_upper_bound(self, doc_idx: int,
                            remaining_potential: float) -> float:
        """
        Upper bound in tropical algebra.

        upper = score ⊗ remaining = score + remaining
        """
        return self.tropical_mult(self.scores[doc_idx], remaining_potential)

    def update_threshold(self, new_scores: np.ndarray):
        """
        Update threshold using tropical max over top-k.

        threshold = ⊕_{i=1}^{k} score[i] = max of top-k scores

        Actually we want the k-th largest, which is a tropical quantile.
        """
        # k-th largest score becomes new threshold
        if len(new_scores) >= self.k:
            # Partial sort to find k-th largest
            kth_largest = np.partition(new_scores, -self.k)[-self.k]
            self.threshold = self.tropical_add(self.threshold, kth_largest)

    def can_prune(self, doc_idx: int, remaining_potential: float) -> bool:
        """
        Tropical pruning condition.

        Prune if: upper_bound < threshold
        In tropical terms: score ⊗ remaining < threshold
        """
        upper = self.compute_upper_bound(doc_idx, remaining_potential)
        return upper < self.threshold
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Geometric intuition** | Pruning as tropical convexity | Better algorithms |
| **Natural bounds** | Upper bounds are tropical products | Clean formulation |
| **Composable** | Tropical operations compose | Multi-level bounds |
| **Hardware-friendly** | max and + are fast operations | SIMD-compatible |

---

## 6. The Span Structure of Search

### Mathematical Foundation

A **span** from A to B is a diagram: `A ← S → B`

The term-document relationship is a span:
```
        Occurrences
        /         \
       ↓           ↓
    Terms       Documents
```

**Key Property**: Spans compose! Given spans A ← S → B and B ← T → C, we get A ← S ×_B T → C.

Query processing is **span composition**:
```
Query ← QueryTermOccurrences → Terms ← TermDocOccurrences → Documents
```

The composition "traces through" the term space, connecting queries to documents.

### Implementation

```python
class SpanBasedSearch:
    """
    Search as span composition.

    A span A ← S → B represents a "many-to-many" relationship.
    Composition of spans corresponds to:
    1. Finding common elements in the middle
    2. Pairing up related elements
    3. Projecting to the endpoints

    For search:
    - Query span: Query ← (query, term, weight) → Terms
    - Index span: Terms ← (term, doc, tf) → Docs
    - Composition: Query ← ??? → Docs (scored results)
    """

    def __init__(self, index):
        self.index = index

        # Index span: term → List[(doc, weight)]
        self.term_doc_span = index.postings

    def compose_spans(self,
                      query_span: Dict[str, float],  # term → weight
                      ) -> Dict[int, float]:  # doc → score
        """
        Compose query span with index span.

        The composition is computed as:
        score[d] = Σ_{t: (t,d) in index} query_weight[t] × index_weight[t,d]

        This is the "pullback" along the term dimension.
        """
        doc_scores = defaultdict(float)

        for term, query_weight in query_span.items():
            if term not in self.term_doc_span:
                continue

            # Span composition: match on term, multiply weights
            for doc_id, index_weight in self.term_doc_span[term]:
                doc_scores[doc_id] += query_weight * index_weight

        return doc_scores

    def compose_with_filter(self,
                           query_span: Dict[str, float],
                           doc_filter: Set[int]) -> Dict[int, float]:
        """
        Compose spans with a document filter.

        The filter is a sub-span: Docs ← FilteredDocs → Docs
        Composition with filter restricts to filtered docs.
        """
        doc_scores = defaultdict(float)

        for term, query_weight in query_span.items():
            if term not in self.term_doc_span:
                continue

            for doc_id, index_weight in self.term_doc_span[term]:
                if doc_id in doc_filter:  # Filter composition
                    doc_scores[doc_id] += query_weight * index_weight

        return doc_scores
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Compositional** | Complex queries as span compositions | Modular design |
| **Filter integration** | Filters are just more spans | Clean filtering |
| **Parallelizable** | Span composition is embarrassingly parallel | Linear speedup |
| **Generalizable** | Works for any weighted relationship | Multi-field, joins |

---

## 7. Weighted Colimits for Top-k Selection

### Mathematical Foundation

A **colimit** is a universal "gluing" construction. A **weighted colimit** generalizes this with weights.

Top-k selection is a weighted colimit where:
- Objects are documents
- Weights are scores
- The colimit extracts the "heaviest" elements

Formally, we have a diagram D: I → Set (documents indexed by I) and weights W: I → ℝ. The weighted colimit is:
```
colim_W D = {elements of ∐_i D(i) with weight ≥ threshold}
```

**Key Insight**: The colimit perspective shows that top-k is about **thresholding a weighted sum**, not sorting.

### Implementation

```python
class WeightedColimitSelector:
    """
    Top-k selection as weighted colimit.

    The weighted colimit perspective:
    1. We have a diagram of documents with weights (scores)
    2. The colimit "glues" them by weight
    3. Top-k extracts elements above the k-th weight threshold

    Key insight: we don't need to sort all elements.
    We need to find the threshold and filter.
    """

    def __init__(self, k: int):
        self.k = k

    def select_top_k(self,
                     scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Compute weighted colimit (top-k selection).

        Algorithm:
        1. Find k-th largest score (the threshold)
        2. Filter to elements above threshold
        3. Sort only the filtered elements

        This is O(n) for finding threshold + O(k log k) for sorting k elements.
        """
        if len(scores) <= self.k:
            # Colimit is everything
            return sorted(scores.items(), key=lambda x: -x[1])

        # Find threshold (k-th largest) in O(n)
        score_values = np.array(list(scores.values()))
        threshold = np.partition(score_values, -self.k)[-self.k]

        # Filter to elements above threshold
        above_threshold = [
            (doc_id, score)
            for doc_id, score in scores.items()
            if score >= threshold
        ]

        # Sort only the filtered elements: O(k log k)
        return sorted(above_threshold, key=lambda x: -x[1])[:self.k]

    def incremental_colimit(self,
                           current_top: List[Tuple[int, float]],
                           new_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Incrementally update weighted colimit.

        When new scores arrive, we don't recompute from scratch.
        We merge the new scores with current top-k.

        This is the colimit of a filtered diagram.
        """
        # Current threshold
        if len(current_top) >= self.k:
            threshold = current_top[-1][1]
        else:
            threshold = float('-inf')

        # Filter new scores
        candidates = [
            (doc_id, score)
            for doc_id, score in new_scores.items()
            if score > threshold
        ]

        # Merge and re-select
        merged = current_top + candidates
        return sorted(merged, key=lambda x: -x[1])[:self.k]
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **O(n + k log k)** | Partial sort instead of full sort | 10x for large n, small k |
| **Incremental** | Update top-k without recomputing | Streaming queries |
| **Threshold-based** | Filter before sort | Fewer comparisons |
| **Parallel-friendly** | Find local top-k, merge | Distributed search |

---

## 8. Kan Extensions for Query Transformation

### Mathematical Foundation

Given functors F: C → D and G: C → E, the **left Kan extension** Lan_F G: D → E is the "best approximation" to G along F.

```
C --G--> E
|        ↗
F    Lan_F G
↓   /
D
```

For search:
- **Left Kan extension**: Query expansion (add related terms)
- **Right Kan extension**: Query contraction (find common core)

**Query Expansion**: Given a query Q and a term-similarity relation S, the left Kan extension adds similar terms:
```
expanded_query = Lan_S(Q)
```

### Implementation

```python
class KanQueryTransformer:
    """
    Query transformation via Kan extensions.

    Left Kan extension: expand query with related terms
    Right Kan extension: contract query to essential core

    The Kan extension is the "optimal" way to transform queries
    while preserving as much structure as possible.
    """

    def __init__(self, index, similarity_matrix: np.ndarray):
        self.index = index
        # Term similarity matrix: S[i,j] = similarity of term i to term j
        self.similarity = similarity_matrix

    def left_kan_expand(self,
                        query_terms: List[str],
                        expansion_threshold: float = 0.5) -> Dict[str, float]:
        """
        Left Kan extension: query expansion.

        For each query term t, find similar terms t' with S(t, t') > threshold.
        Weight expanded terms by their similarity.

        Lan_S(Q)(t') = sup_{t: S(t,t')>0} Q(t) × S(t, t')

        This is the "best approximation" to the query along similarity.
        """
        expanded = defaultdict(float)

        for term in query_terms:
            term_id = self.index.term_to_id.get(term)
            if term_id is None:
                continue

            # Original term with weight 1.0
            expanded[term] = max(expanded[term], 1.0)

            # Find similar terms (Kan extension formula)
            similarities = self.similarity[term_id, :]
            for other_id, sim in enumerate(similarities):
                if sim > expansion_threshold:
                    other_term = self.index.id_to_term[other_id]
                    # Kan extension: take supremum of weighted similarities
                    expanded[other_term] = max(expanded[other_term], sim)

        return dict(expanded)

    def right_kan_contract(self,
                          query_terms: List[str],
                          min_coverage: float = 0.8) -> List[str]:
        """
        Right Kan extension: query contraction.

        Find the minimal set of terms that "covers" the query.

        Ran_S(Q)(t') = inf_{t: S(t',t)>0} Q(t) / S(t', t)

        This finds terms that are "necessary" for the query.
        """
        # Compute term importance (how much of query does each term cover)
        term_coverage = {}

        for term in query_terms:
            term_id = self.index.term_to_id.get(term)
            if term_id is None:
                continue

            # How well does this term represent other query terms?
            similarities = self.similarity[term_id, :]
            coverage = sum(
                similarities[self.index.term_to_id[t]]
                for t in query_terms
                if t in self.index.term_to_id
            ) / len(query_terms)

            term_coverage[term] = coverage

        # Greedy selection of covering terms
        contracted = []
        remaining_coverage = 1.0

        for term, coverage in sorted(term_coverage.items(),
                                      key=lambda x: -x[1]):
            contracted.append(term)
            remaining_coverage -= coverage / len(query_terms)

            if remaining_coverage <= 1.0 - min_coverage:
                break

        return contracted
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Principled expansion** | Optimal way to add related terms | Better recall |
| **Weighted expansion** | Similar terms get proportional weight | Precision preserved |
| **Query compression** | Find essential query core | Faster search |
| **Universality** | Kan extensions are universal constructions | Theoretically optimal |

---

## 9. Simplicial Structure for Document Representation

### Mathematical Foundation

Documents can be viewed as **simplices**:
- 0-simplices (vertices): individual terms
- 1-simplices (edges): term co-occurrences
- 2-simplices (triangles): three-term co-occurrences
- etc.

A query is a **chain** in this simplicial complex. Scoring is a **chain map** from query chains to score chains.

**Key Insight**: Higher-order co-occurrence (phrases, n-grams) naturally lives in this simplicial structure.

### Implementation

```python
class SimplicialDocumentIndex:
    """
    Documents as simplices, queries as chains.

    The simplicial structure captures:
    - Individual terms (0-simplices)
    - Term pairs / bigrams (1-simplices)
    - Higher-order co-occurrence (k-simplices)

    Scoring is a chain map: it preserves the simplicial structure.
    """

    def __init__(self, index, max_simplex_dim: int = 2):
        self.index = index
        self.max_dim = max_simplex_dim

        # Simplices at each dimension
        # 0-simplices: terms
        self.simplices_0 = index.term_to_id

        # 1-simplices: term pairs with co-occurrence count
        self.simplices_1: Dict[Tuple[int, int], int] = {}

        # Build higher simplices from corpus
        self._build_simplices()

    def _build_simplices(self):
        """Build simplicial complex from document co-occurrences."""
        from itertools import combinations

        for doc_id, terms in self.index.doc_terms.items():
            term_ids = [self.index.term_to_id[t] for t in terms
                       if t in self.index.term_to_id]

            # Build 1-simplices (pairs)
            for t1, t2 in combinations(sorted(term_ids), 2):
                key = (t1, t2)
                self.simplices_1[key] = self.simplices_1.get(key, 0) + 1

    def query_as_chain(self, query_terms: List[str]) -> Dict[int, List]:
        """
        Convert query to simplicial chain.

        A chain is a formal sum of simplices.
        The query "machine learning" becomes:
        - 0-chain: [machine] + [learning]
        - 1-chain: [machine, learning] (if they co-occur)
        """
        chain = {0: [], 1: []}

        term_ids = [self.index.term_to_id[t] for t in query_terms
                   if t in self.index.term_to_id]

        # 0-simplices
        chain[0] = term_ids

        # 1-simplices (pairs that exist in complex)
        from itertools import combinations
        for t1, t2 in combinations(sorted(term_ids), 2):
            if (t1, t2) in self.simplices_1:
                chain[1].append((t1, t2))

        return chain

    def score_chain(self, chain: Dict[int, List], doc_id: int) -> float:
        """
        Score document using chain map.

        The chain map preserves simplicial structure:
        - 0-simplices contribute individual term scores
        - 1-simplices contribute co-occurrence bonuses
        """
        score = 0.0

        # 0-simplex contributions (standard BM25)
        for term_id in chain[0]:
            score += self.index.get_term_score(term_id, doc_id)

        # 1-simplex contributions (co-occurrence bonus)
        for t1, t2 in chain[1]:
            if self._terms_adjacent_in_doc(t1, t2, doc_id):
                # Bonus for phrase match
                score += self.phrase_bonus(t1, t2, doc_id)

        return score
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Phrase detection** | 1-simplices capture term pairs | Better relevance |
| **Structure preservation** | Chain maps preserve relationships | Consistent scoring |
| **Hierarchical features** | Higher simplices = higher-order patterns | Rich representations |
| **Topological features** | Homology, Betti numbers | Novel ranking signals |

---

## 10. Topos Structure for Fuzzy/Probabilistic Scoring

### Mathematical Foundation

A **topos** is a category that behaves like the category of sets, but with a generalized notion of "truth."

For search, the topos of **sheaves over term space** provides:
- Fuzzy term matching (not just exact match)
- Probabilistic scoring (uncertainty in relevance)
- Local-to-global reasoning (aggregate local scores)

The **subobject classifier** Ω in this topos is not just {true, false} but a richer truth-value space (e.g., [0, 1] for probabilities).

### Implementation

```python
class ToposScorer:
    """
    Scoring in the topos of sheaves over term space.

    In this topos:
    - Truth values are probabilities/fuzzy values
    - "Term matches document" can be partial
    - Scoring aggregates local (per-term) truth values

    This enables:
    - Fuzzy matching (synonyms, typos)
    - Probabilistic relevance
    - Uncertainty quantification
    """

    def __init__(self, index, fuzzy_threshold: float = 0.8):
        self.index = index
        self.fuzzy_threshold = fuzzy_threshold

        # Subobject classifier: [0, 1] instead of {0, 1}
        self.truth_values = np.linspace(0, 1, 100)

    def fuzzy_match(self, query_term: str, doc_term: str) -> float:
        """
        Fuzzy term matching (generalized truth value).

        Instead of exact match (0 or 1), return match degree.
        """
        if query_term == doc_term:
            return 1.0

        # Edit distance-based fuzzy match
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, query_term, doc_term).ratio()

        return ratio if ratio >= self.fuzzy_threshold else 0.0

    def sheaf_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Score using sheaf structure.

        A sheaf assigns data to each "open set" (subset of terms).
        The score is the global section (consistent assignment).

        Local scores (per-term) are aggregated to global score.
        """
        local_scores = []

        for query_term in query_terms:
            # Local score: best fuzzy match in document
            best_match = 0.0

            for doc_term in self.index.get_doc_terms(doc_id):
                match = self.fuzzy_match(query_term, doc_term)
                best_match = max(best_match, match)

            local_scores.append(best_match)

        # Global section: aggregate local scores
        # Use product for "all terms must match" semantics
        # Use sum for "any term matches" semantics
        return sum(local_scores)  # BM25-like aggregation

    def probabilistic_score(self,
                           query_terms: List[str],
                           doc_id: int) -> Tuple[float, float]:
        """
        Probabilistic scoring with uncertainty.

        Returns (expected_score, uncertainty).

        Uncertainty comes from:
        - Fuzzy matches (partial evidence)
        - Missing terms (incomplete information)
        """
        scores = []
        uncertainties = []

        for term in query_terms:
            term_score = self.index.get_term_score(term, doc_id)
            term_uncertainty = 1.0 - self.index.get_term_confidence(term, doc_id)

            scores.append(term_score)
            uncertainties.append(term_uncertainty)

        expected = sum(scores)
        # Uncertainty propagation (assuming independence)
        uncertainty = np.sqrt(sum(u**2 for u in uncertainties))

        return expected, uncertainty
```

### Benefits

| Benefit | Description | Expected Impact |
|---------|-------------|-----------------|
| **Fuzzy matching** | Partial matches contribute | Better recall |
| **Uncertainty quantification** | Know how confident scores are | Better ranking |
| **Local-to-global** | Aggregate local evidence | Principled scoring |
| **Generalized truth** | Richer than binary matching | Nuanced relevance |

---

## Implementation Priority

Based on expected impact and implementation complexity:

| Priority | Enhancement | Complexity | Expected Speedup | Accuracy Impact |
|----------|-------------|------------|------------------|-----------------|
| 1 | Galois Connection (candidate refinement) | Low | 2-5x | Neutral |
| 2 | Coend-aware iteration order | Low | 1.5-2x | Neutral |
| 3 | Tropical bound tracking | Medium | 2-3x | Neutral |
| 4 | Compositional term caching | Medium | 10-100x (warm) | Neutral |
| 5 | Span-based query composition | Medium | 1.5x | Neutral |
| 6 | Weighted colimit top-k | Low | 2x | Neutral |
| 7 | Simplicial phrase detection | High | - | +5-10% relevance |
| 8 | Kan extension query expansion | High | - | +10-20% recall |
| 9 | Topos fuzzy matching | High | - | +5-15% recall |

## Conclusion

These are not optimizations—they are **structural insights**. Each enhancement follows from understanding what BM25 search **is**, mathematically:

1. **Queries are monoids** → compositional caching
2. **Scoring is a coend** → sparse, order-independent computation
3. **Index is a profunctor** → compositional query processing
4. **Candidate sets form a Galois connection** → optimal refinement order
5. **Bounds live in tropical semiring** → geometric pruning
6. **Search is span composition** → modular, parallelizable
7. **Top-k is weighted colimit** → threshold-based selection
8. **Query expansion is Kan extension** → optimal transformation
9. **Documents are simplices** → higher-order features
10. **Fuzzy matching is topos-theoretic** → generalized truth values

The fastest BM25 implementation will be the one that most faithfully follows this structure.
