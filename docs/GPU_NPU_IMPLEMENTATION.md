# GPU/NPU Acceleration for Vajra BM25

## Objective

Build the fastest BM25 implementation by leveraging GPU and NPU acceleration, targeting:
- **Apple Silicon** (M1/M2/M3/M4) via Metal Performance Shaders (MPS)
- **NVIDIA GPUs** via CUDA/PyTorch
- Future: Intel NPU, AMD XDNA (low priority)

## Current Performance Baseline

At 100K documents:
- Vajra Parallel (CPU, 8 workers): **0.35ms** (291x vs rank-bm25)
- BM25S (CPU): 0.85ms (120x vs rank-bm25)

**Goal**: Achieve **sub-0.1ms** query latency at 100K+ documents.

## Architecture Design

### Backend Abstraction

Create a device-agnostic backend system:

```python
class AcceleratorBackend(ABC):
    """Abstract base for CPU/GPU/NPU backends."""

    @abstractmethod
    def to_device(self, tensor): pass

    @abstractmethod
    def sparse_matmul(self, sparse_index, query_vector): pass

    @abstractmethod
    def score_candidates(self, candidates, query_terms, idf, doc_lengths): pass

    @abstractmethod
    def topk(self, scores, k): pass

class CPUBackend(AcceleratorBackend): ...
class MPSBackend(AcceleratorBackend): ...  # Apple Silicon
class CUDABackend(AcceleratorBackend): ...  # NVIDIA GPU
```

### Auto-Detection

```python
def get_best_backend():
    if torch.backends.mps.is_available():
        return MPSBackend()  # Apple NPU/GPU
    elif torch.cuda.is_available():
        return CUDABackend()  # NVIDIA GPU
    else:
        return CPUBackend()
```

## Implementation Phases

### Phase 1: PyTorch MPS (Apple Silicon) ✓ Priority 1

**Why MPS First:**
- PyTorch has native MPS support (`torch.device('mps')`)
- Utilizes both Apple GPU and Neural Engine
- Same API as CUDA (easy to port)
- Large install base (M1/M2/M3 Macs)

**Operations to Accelerate:**

1. **Index Storage**: Convert inverted index to sparse COO tensor on MPS device
   ```python
   # CPU: dict[term, list[(doc_id, freq)]]
   # MPS: torch.sparse_coo_tensor on mps device
   ```

2. **Candidate Retrieval**: Sparse matrix slicing
   ```python
   # For each query term, slice posting list from sparse tensor
   candidates = sparse_index[query_term_ids].coalesce()
   ```

3. **BM25 Scoring**: Vectorized on GPU
   ```python
   # Batch compute for all candidates simultaneously
   scores = idf * (tf * (k1 + 1)) / (tf + k1 * norm)  # All on MPS
   ```

4. **Top-K Selection**: Use torch.topk on MPS
   ```python
   topk_scores, topk_indices = torch.topk(scores, k)
   ```

**Expected Speedup**: 5-10x over CPU (targeting **0.05-0.07ms** at 100K docs)

### Phase 2: CUDA (NVIDIA GPUs) ✓ Priority 1

Same architecture as MPS, but using `torch.device('cuda')`:
- Mature CUDA sparse operations
- Larger memory bandwidth
- Better for batch queries

**Expected Speedup**: 10-20x over CPU (targeting **0.02-0.03ms** at 100K docs)

### Phase 3: Optimizations

1. **Persistent GPU Index**: Keep index on GPU between queries
2. **Batch Query Processing**: Process multiple queries simultaneously
3. **Mixed Precision**: Use FP16 for scoring (2x memory, 2x speed)
4. **Custom CUDA Kernels**: For critical paths if PyTorch overhead is significant

### Phase 4: Intel/AMD NPU (Future, Low Priority)

- Intel NPU via OpenVINO
- AMD XDNA via ROCm
- Only if there's demand

## Key Operations Analysis

### 1. Index Building (One-Time Cost)

**Current (CPU)**:
```python
for doc in corpus:
    tokens = preprocess(doc)
    for term, freq in Counter(tokens).items():
        inverted_index[term].append((doc.id, freq))
```

**GPU-Accelerated**:
```python
# Tokenize entire corpus in parallel
all_tokens = parallel_tokenize(corpus)  # On GPU

# Build COO sparse tensor
indices = []  # (term_id, doc_id) pairs
values = []   # frequencies
for doc_id, tokens in enumerate(all_tokens):
    for term_id, freq in token_counts:
        indices.append([term_id, doc_id])
        values.append(freq)

sparse_index = torch.sparse_coo_tensor(
    indices=torch.tensor(indices).T,
    values=torch.tensor(values),
    device='mps'  # or 'cuda'
)
```

### 2. Query Processing (Hot Path)

**Current (CPU, NumPy)**:
```python
# Get candidates
candidates = set()
for term in query_terms:
    candidates.update(inverted_index[term])

# Score candidates
scores = np.zeros(len(candidates))
for term in query_terms:
    idf = idf_cache[term]
    tf = get_term_freq(term, candidates)
    scores += idf * (tf * (k1+1)) / (tf + k1 * norm)
```

**GPU-Accelerated (PyTorch)**:
```python
# Get candidates (sparse tensor slicing)
query_term_indices = torch.tensor(query_term_ids, device=device)
candidate_mask = sparse_index[query_term_indices].coalesce()

# Score candidates (all vectorized on GPU)
idf_tensor = idf_cache[query_term_indices]  # Pre-loaded on GPU
tf_tensor = candidate_mask.values()
norm_tensor = doc_lengths[candidate_mask.indices()[1]]  # Doc lengths on GPU

scores = idf_tensor * (tf_tensor * (k1+1)) / (tf_tensor + k1 * norm_tensor)

# Top-K (on GPU)
topk_scores, topk_indices = torch.topk(scores, k)
```

## Implementation Checklist

### MPS Backend (Apple Silicon)
- [ ] Create `MPSBackend` class
- [ ] Implement sparse tensor conversion from inverted index
- [ ] Implement GPU-accelerated candidate retrieval
- [ ] Implement GPU-accelerated BM25 scoring
- [ ] Implement GPU top-k selection
- [ ] Add MPS detection and auto-switching
- [ ] Benchmark vs CPU on M1/M2/M3

### CUDA Backend (NVIDIA)
- [ ] Create `CUDABackend` class
- [ ] Port MPS implementation to CUDA device
- [ ] Optimize for CUDA-specific features
- [ ] Benchmark vs CPU and MPS

### Integration
- [ ] Create `VajraSearchGPU` class
- [ ] Update `VajraSearchParallel` to optionally use GPU backend
- [ ] Add device selection parameter (`device='auto'|'cpu'|'mps'|'cuda'`)
- [ ] Comprehensive benchmarks (CPU vs MPS vs CUDA)
- [ ] Update documentation

### Testing
- [ ] Unit tests for each backend
- [ ] Numerical correctness tests (GPU results == CPU results within epsilon)
- [ ] Performance regression tests
- [ ] Memory usage profiling

## Expected Results

### Query Latency (100K documents)

| Backend | Expected Latency | Speedup vs CPU | Speedup vs rank-bm25 |
|---------|-----------------|----------------|----------------------|
| CPU (current) | 0.35ms | 1x | 291x |
| MPS (Apple) | **0.05-0.07ms** | 5-7x | **1,500-2,000x** |
| CUDA (NVIDIA) | **0.02-0.03ms** | 12-17x | **3,400-5,100x** |

### Scaling (1M documents)

| Backend | Expected Latency | Notes |
|---------|-----------------|-------|
| CPU | ~3-5ms | Linear scaling bottleneck |
| MPS | **0.3-0.5ms** | GPU memory bandwidth advantage |
| CUDA | **0.1-0.2ms** | Best for large-scale |

## Dependencies

```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.0.0",  # MPS support requires 2.0+
]
cuda = [
    "torch[cuda]>=2.0.0",
]
```

## Usage Example

```python
from vajra_bm25 import VajraSearchGPU

# Auto-detect best device
engine = VajraSearchGPU(corpus, device='auto')  # Uses MPS on Mac, CUDA on NVIDIA

# Or explicitly choose
engine = VajraSearchGPU(corpus, device='mps')   # Force Apple Silicon
engine = VajraSearchGPU(corpus, device='cuda')  # Force NVIDIA GPU
engine = VajraSearchGPU(corpus, device='cpu')   # Force CPU

results = engine.search("category theory functors", top_k=10)
# Sub-0.1ms query latency!
```

## References

- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
- [PyTorch Sparse Tensors](https://pytorch.org/docs/stable/sparse.html)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [CUDA Sparse Operations](https://docs.nvidia.com/cuda/cusparse/)
