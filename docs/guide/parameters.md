# BM25 Parameters

Vajra implements the standard BM25 ranking algorithm with configurable parameters.

## The BM25 Formula

$$
\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

Where:

- $f(q_i, D)$ = frequency of term $q_i$ in document $D$
- $|D|$ = document length
- $\text{avgdl}$ = average document length
- $k_1$ = term frequency saturation parameter
- $b$ = length normalization parameter

## Parameters

### k1 (Term Frequency Saturation)

Controls how quickly term frequency saturates.

- **Default**: 1.5
- **Range**: 0.0 - 3.0 typical
- **Effect**: Higher values give more weight to term frequency

```python
# Lower k1: Term frequency saturates quickly
engine = VajraSearchOptimized(corpus, k1=0.5)

# Higher k1: More weight to repeated terms
engine = VajraSearchOptimized(corpus, k1=2.5)
```

| k1 Value | Behavior |
|----------|----------|
| 0.0 | Binary term presence (ignore frequency) |
| 1.2 | Standard (works well for most corpora) |
| 1.5 | Default (balanced) |
| 2.0+ | Favor documents with many term occurrences |

### b (Length Normalization)

Controls document length normalization.

- **Default**: 0.75
- **Range**: 0.0 - 1.0
- **Effect**: Higher values penalize longer documents more

```python
# No length normalization
engine = VajraSearchOptimized(corpus, b=0.0)

# Strong length normalization
engine = VajraSearchOptimized(corpus, b=1.0)
```

| b Value | Behavior |
|---------|----------|
| 0.0 | No length normalization (favor long docs) |
| 0.5 | Mild normalization |
| 0.75 | Default (balanced) |
| 1.0 | Strong normalization (favor short docs) |

## Using BM25Parameters

For the base `VajraSearch` engine:

```python
from vajra_bm25 import VajraSearch, BM25Parameters

params = BM25Parameters(k1=1.2, b=0.75)
engine = VajraSearch(corpus, params=params)
```

For optimized engines, pass directly:

```python
engine = VajraSearchOptimized(corpus, k1=1.2, b=0.75)
```

## Tuning Guidelines

### Short Documents (tweets, titles)

```python
# Less length normalization needed
engine = VajraSearchOptimized(corpus, k1=1.2, b=0.3)
```

### Long Documents (articles, papers)

```python
# More length normalization
engine = VajraSearchOptimized(corpus, k1=1.5, b=0.75)
```

### Technical/Scientific Corpus

```python
# Standard parameters usually work well
engine = VajraSearchOptimized(corpus, k1=1.5, b=0.75)
```

### Web Search (mixed lengths)

```python
# Balanced approach
engine = VajraSearchOptimized(corpus, k1=1.2, b=0.75)
```

## Experimentation

The best parameters depend on your corpus. Use evaluation metrics to tune:

```python
def evaluate_parameters(corpus, queries, relevant_docs, k1, b):
    engine = VajraSearchOptimized(corpus, k1=k1, b=b)

    total_recall = 0
    for query, relevant in zip(queries, relevant_docs):
        results = engine.search(query, top_k=10)
        retrieved = {r.document.id for r in results}
        recall = len(retrieved & relevant) / len(relevant)
        total_recall += recall

    return total_recall / len(queries)

# Grid search
for k1 in [1.0, 1.2, 1.5, 2.0]:
    for b in [0.5, 0.75, 1.0]:
        score = evaluate_parameters(corpus, queries, relevant, k1, b)
        print(f"k1={k1}, b={b}: Recall@10 = {score:.3f}")
```
