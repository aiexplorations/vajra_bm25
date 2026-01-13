# Installation

## Requirements

- Python 3.8 or higher
- pip package manager

## Basic Installation

Install Vajra BM25 with zero dependencies:

```bash
pip install vajra-bm25
```

## Optional Dependencies

Vajra offers several optional feature sets:

### Optimized Performance

For production workloads with NumPy and SciPy optimizations:

```bash
pip install vajra-bm25[optimized]
```

This enables:

- Sparse matrix operations (6-8x memory reduction)
- Vectorized scoring (10-50x speedup)
- Numba JIT compilation (if installed separately)

### PDF Support

To index and search PDF documents:

```bash
pip install vajra-bm25[pdf]
```

### Index Persistence

To save and load indexes:

```bash
pip install vajra-bm25[persistence]
```

### Interactive CLI

For the rich command-line interface:

```bash
pip install vajra-bm25[cli]
```

### Everything

Install all optional dependencies:

```bash
pip install vajra-bm25[all]
```

## Development Installation

To contribute to Vajra:

```bash
git clone https://github.com/aiexplorations/vajra_bm25.git
cd vajra_bm25
pip install -e ".[dev]"
```

## Verifying Installation

```python
import vajra_bm25
print(vajra_bm25.__version__)
```

## Optional: Numba

For additional JIT compilation speedup, install Numba separately:

```bash
pip install numba
```

Vajra will automatically detect and use Numba if available.
