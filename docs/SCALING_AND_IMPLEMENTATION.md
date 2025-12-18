# Vajra BM25: Scaling and Implementation Deep Dive

This document provides a technical breakdown of how Vajra achieves its performance and how it scales with corpus size.

## 1. Sub-Linear Scaling Theory

Traditional BM25 implementations typically scale linearly ((N)$) with the number of documents. As you add more documents, the search time increases proportionally.

Vajra exhibits **sub-linear scaling** at larger corpus sizes (10K to 100K documents).

### Performance Growth Analysis

| Corpus Size | rank-bm25 | Vajra (Optimized) | Latency Growth (Vajra) |
| :---------- | :-------- | :---------------- | :--------------------- |
| 10,000      | 9.11 ms   | 0.13 ms           | -                      |
| 100,000     | 102.19 ms | 0.44 ms           | **3.4x** (for 10x size) |

**Why is it sub-linear?**
1. **Dynamic Sparsity**: As the vocabulary grows, the term-document matrix becomes increasingly sparse. Vajra's use of CSR (Compressed Sparse Row) matrices allows it to skip empty intersections almost entirely.
2. **Matrix Slicing Efficiency**: Slicing a sparse matrix for a query with few terms is extremely fast, regardless of the total number of documents.
3. **Vectorized Aggregation**: Once the relevant rows are sliced, NumPy's C-intensive aggregation handles the rest at speeds approaching hardware limits.

## 2. Memory Efficiency (Sparse vs Dense)

For small corpora, a dense matrix is acceptable. However, search engines for larger datasets must manage memory carefully.

| Corpus Size | Dense Matrix | Sparse Matrix | **Savings** |
| :---------- | :----------- | :------------ | :---------- |
| 10,000      | ~150 MB      | ~47 MB        | 3.2x        |
| 50,000      | ~750 MB      | ~89 MB        | **8.4x**    |
| 100,000     | ~1.5 GB      | ~240 MB       | **6.3x**    |

*Note: Memory reduction is non-linear due to the distribution of terms in natural language (Zipf's Law).*

## 3. The Engineering Flow

The following diagram illustrates how Vajra treats search as a **coalgebraic unfolding** ($\alpha: S \to F(S)$):

![Architectural Flow](../docs/bm25_flow.png)

1. **Carrier (State)**: The immutable `QueryState` containing tokenized terms and expansion depth.
2. **Unfolding ($\alpha$)**: The process of using the Inverted Index to identify candidates (Term $\to$ Postings).
3. **Functor Application**: Lifting the scoring **Morphism**  \to \mathbb{R}$ over the list of candidates using the **List Functor**.
4. **Observation**: The final ranked list of documents.

## 4. Optimization Breakdown

Vajra's 291x speedup is the result of a "layered" optimization strategy:

| Technique | Implementation | Impact |
| :--- | :--- | :--- |
| **Vectorization** | NumPy matrix operations | **10-50x** over Python loops |
| **Sparse Storage** | SciPy CSR Matrices | Significant memory reduction & slice speed |
| **Top-K Sorting** | O(N) Partial Sort (`np.argpartition`) | Prevents O(N log N) overhead on large lists |
| **Caching** | Comonadic LRU Cache | Near-zero latency for repeated queries |
| **Parallelism** | Thread Pool (8 workers) | **1.3x - 3x** speedup on multi-core systems |

## 5. Implementation Summary

Vajra proves that **Category Theory** is not just an academic exercise but a rigorous design vocabulary for high-performance systems. By modeling search as a coalgebra, we decoupled the algorithm's structure from its execution context, allowing us to drop in highly optimized NumPy/SciPy backends without changing the core architectural "arrow".

