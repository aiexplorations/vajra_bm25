# Vajra BM25 0.5.1: results and model matrix

Measured on 2026-11-19 on one Apple-silicon machine (18 logical CPUs, MPS for embeddings), Python 3.11,
branch `feat/v0.5.1`. The 0.5.0 numbers come from `results/baseline-0.5.0/` and from a git-archive
extract of the 0.5.0 commit run in the same session. Every figure is a single run: differences under
about 3% (search) or 1 s (builds) are noise.

## 1. What 0.5.1 changes (Verity findings)

| Finding | Change | Status |
|---|---|---|
| Stale scorer after `add_document`/`remove_document` | One `_build_scorers()` serves `__init__`, `load_index`, add and remove; the vector/hybrid engine also receives added documents | Fixed, reproduced on 0.5.0 first |
| Pool hang in scripts and threaded servers | In-process tokenisation by default; `spawn` pool only for `n_jobs>1` and 20,000+ docs | Fixed; a pool without a `__main__` guard can still hang |
| No custom tokenisation (equipment tags V-110, P-400A/B) | `analyzer=` hook, saved with the index; `tag_aware_analyzer` | Added; tested on a synthetic fixture only |
| Logging configured at import | `NullHandler` only; `configure_logging()` is opt-in | Fixed |
| Wrong `__version__` | Read from package metadata | Fixed |

Not in 0.5.1 (0.6 inputs): stemming, synonyms, metadata filters, incremental updates (add still
rebuilds the whole index), hybrid defaults and fusion depth, dropping SciPy.

Also changed: `preprocess_text` no longer allocates a `Token` per word (identical output, checked on
3,000 Wikipedia docs, with and without stop words), and `requests` is declared in the benchmark/CLI extras
because `beir` imports it without declaring it. Tests: 360 pass (48 new, in `tests/test_v051.py`).

## 2. Lexical quality and search latency versus 0.5.0

Quality is unchanged: SciFact recall@10 78.89, nDCG@10 66.97, MRR 63.96 for Vajra, and all lexical and
hybrid systems match the baseline. MiniLM hybrid runs in section 4 also match the 0.5.0 baseline, except
Tantivy on NFCorpus and ArguAna, which moves by under 0.05 nDCG (run-to-run variation).

Single-query latency (1,000 Wikipedia title queries, top_k=10, warmed up; `benchmarks/bench_search.py`):

| Docs | Version | Mean | p50 | p95 | Queries/s |
|---|---|---:|---:|---:|---:|
| 20,000 | 0.5.0 | 0.071 ms | 0.070 | 0.119 | 14,012 |
| 20,000 | 0.5.1 | 0.069 ms | 0.068 | 0.118 | 14,421 |
| 200,000 | 0.5.0 | 0.303 ms | 0.251 | 0.725 | 3,302 |
| 200,000 | 0.5.1 | 0.293 ms | 0.241 | 0.689 | 3,410 |

SciFact (BEIR) mean latency: 0.121 ms (0.5.0) and 0.119 ms (0.5.1). Search is unchanged: the scoring path was
not touched. Batch and thread-pool search were not re-measured. Wikipedia 200k to 5M comparisons against
bm25s, Tantivy and Pyserini are pending (run by the owner with `benchmarks/run_suite.sh`).

## 3. Index build time (sparse index, `benchmarks/bench_build.py`)

| Docs | 0.5.0 (always a pool) | 0.5.1 default (`n_jobs=1`) | `n_jobs=8` | `n_jobs=12` | `n_jobs=24` | `n_jobs=-1` (18) |
|---|---:|---:|---:|---:|---:|---:|
| 629 | 0.27 s | 0.07 s | - | - | - | - |
| 5,183 | 0.50 s | 0.47 s | - | - | - | - |
| 20,000 | 3.05 s | 4.3 s (10.0 s before the tokenizer change) | 3.4 s | 3.3 s | 3.7 s | 4.6 s |
| 200,000 | 17.3 s | 28.2 s (73.6 s before) | 17.8-20.6 s | 18.9 s | 19.7 s | 18.9 s |

- Small corpora build about 4x faster than 0.5.0 because no pool is started.
- The default is still about 1.6x slower than 0.5.0 above 20,000 docs. Passing `n_jobs=8` restores 0.5.0 speed.
- More than 8 workers does not help: the rest of the build (term dictionary, COO/CSR matrices, eager scores)
  is single-threaded in the parent. 24 workers on 18 CPUs is oversubscribed and no faster than 12.
- Profile of a 100k-doc in-process build: about 60% tokenisation (4.8 s of 16.9 s is regex matching).
- Open decision: an automatic default (pool of up to 8 above about 50,000 docs) would match 0.5.0 on large
  corpora with no flags, at the cost of the hang risk in scripts without a `__main__` guard.

## 4. Embedding model matrix (hybrid benchmark)

Models: `all-MiniLM-L6-v2` (current default), `BAAI/bge-small-en-v1.5` and `all-mpnet-base-v2`. Datasets:
BEIR SciFact, NFCorpus, ArguAna, FiQA. Same corpus, queries and code for every model
(`benchmarks/run_model_matrix.sh`, summarised by `benchmarks/summarize_models.py`). Notes:

- BGE queries get the prefix "Represent this sentence for searching relevant passages: "; documents do not.
- nDCG is binary-relevance, so it does not match published graded nDCG. ArguAna excludes the query's own document.
- Hybrid `alpha` is the BM25 weight: `a=0.3` is 30% BM25 and 70% vector.
- Tantivy scores 10.7 on ArguAna because its wrapper silently drops queries that fail to parse; this is a
  benchmark-wrapper issue, not a Vajra one.
- Query-embedding times vary between runs of the same model (mpnet: 18.3 ms on SciFact, 6.7 ms on FiQA), so read
  them as order of magnitude. Pyserini was not part of this matrix (the 0.5.0 baseline has it).

### Lexical reference (same for every model)

| System | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| vajra-bm25 | 67.0 | 30.9 | 47.7 | 23.2 |
| bm25s | 66.2 | 30.7 | 48.2 | 23.3 |
| tantivy | 60.4 | 28.6 | 10.7 | 20.7 |

### Vector only (flat, exact) nDCG@10

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 64.8 | 32.0 | 49.1 | 36.9 |
| BAAI_bge-small-en-v1.5 | 70.6 | 34.3 | 59.4 | 40.3 |
| all-mpnet-base-v2 | 65.3 | 33.4 | 45.9 | 50.0 |

### Default hybrid (rrf, alpha 0.5) nDCG@10

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 70.4 | 34.1 | 52.6 | 35.0 |
| BAAI_bge-small-en-v1.5 | 72.1 | 35.7 | 57.4 | 36.1 |
| all-mpnet-base-v2 | 69.7 | 35.9 | 51.5 | 43.1 |

### Best hybrid fusion (vajra + flat) nDCG@10

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 71.7 (linear a=0.5) | 34.7 (rsf a=0.5) | 53.5 (linear a=0.7) | 38.4 (linear a=0.3) |
| BAAI_bge-small-en-v1.5 | 72.9 (linear a=0.5) | 35.8 (rsf a=0.5) | 60.6 (linear a=0.5) | 40.4 (linear a=0.3) |
| all-mpnet-base-v2 | 72.4 (linear a=0.5) | 36.1 (rsf a=0.5) | 52.1 (rsf a=0.5) | 50.2 (linear a=0.3) |

### Vector recall@10 (flat)

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 78.8 | 15.9 | 76.2 | 44.1 |
| BAAI_bge-small-en-v1.5 | 83.0 | 16.2 | 84.9 | 46.4 |
| all-mpnet-base-v2 | 79.0 | 16.6 | 73.6 | 58.1 |

### Corpus embedding time (s)

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 4 | 3 | 6 | 32 |
| BAAI_bge-small-en-v1.5 | 11 | 9 | 12 | 73 |
| all-mpnet-base-v2 | 35 | 25 | 39 | 229 |

### Query embedding, mean (ms)

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 2.9 | 2.9 | 3.4 | 2.9 |
| BAAI_bge-small-en-v1.5 | 4.6 | 4.1 | 6.6 | 4.5 |
| all-mpnet-base-v2 | 18.3 | 13.5 | 15.1 | 6.7 |

### Default hybrid p50 latency (ms, includes query embedding)

| Model | SciFact | NFCorpus | ArguAna | FiQA |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 3.8 | 3.4 | 6.3 | 12.5 |
| BAAI_bge-small-en-v1.5 | 5.4 | 4.5 | 8.8 | 14.6 |
| all-mpnet-base-v2 | 9.4 | 5.9 | 16.4 | 25.6 |

Embedding dimension: all-MiniLM-L6-v2 384, BAAI_bge-small-en-v1.5 384, all-mpnet-base-v2 768

### Reading the model results

- **No model wins everywhere.** BGE-small is best on SciFact (70.6 vector-only), NFCorpus and ArguAna (59.4), and
  ahead of MiniLM on all four datasets. mpnet wins FiQA by a wide margin (50.0 against 40.3 for BGE and 36.9 for
  MiniLM) but is below MiniLM on ArguAna (45.9 against 49.1).
- **BGE-small is a cheap upgrade.** Same 384 dimensions as MiniLM, about 2.5x the corpus-embedding time and 1.5x
  the query-embedding time, and better nDCG on every dataset here.
- **mpnet is expensive.** 768 dimensions, about 7x MiniLM's corpus-embedding time (229 s against 32 s on FiQA), and
  double the hybrid p50 latency. It pays off on FiQA and nowhere else in this matrix.
- **Hybrid helps weaker embedders most.** With MiniLM on SciFact, vector-only is 64.8 and the best hybrid is 71.7.
  With the stronger model on its best dataset, hybrid adds little: mpnet on FiQA is 50.0 vector-only and 50.2 best
  hybrid; BGE on ArguAna is 59.4 and 60.6.
- **The default fusion (rrf, alpha 0.5) can hurt.** mpnet on FiQA scores 43.1 with the default against 50.0 for
  vector-only; BGE on FiQA scores 36.1 against 40.3; MiniLM on FiQA 35.0 against 36.9. The best fusion on FiQA is
  linear with alpha 0.3, on SciFact linear 0.5, on NFCorpus rsf 0.5. This supports revisiting the hybrid default in 0.6.
- **Caveats.** Four datasets, one run each, binary nDCG, FiQA lexical is weak (23), and no domain data from Verity.
  Do not change the default model on this evidence alone; check the candidate on Verity's own queries first.

## 5. Reproduce

```bash
TAG=after-0.5.1 caffeinate -i benchmarks/run_suite.sh beir-scifact hybrid-scifact
TAG=models-0.5.1 benchmarks/run_model_matrix.sh && python benchmarks/summarize_models.py models-0.5.1
python benchmarks/bench_build.py v0.5.1 wiki20000 wiki200000 wiki200000@8   # needs the Wikipedia corpora
python benchmarks/bench_search.py v0.5.1 200000
```
