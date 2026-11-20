# Vajra BM25 0.5.1: results and model matrix

Measured on 2026-11-19 on one Apple-silicon machine (18 logical CPUs, MPS for embeddings), Python 3.11,
branch `feat/v0.5.1`. The 0.5.0 numbers come from `results/baseline-0.5.0/` and from a git-archive
extract of the 0.5.0 commit run in the same session. Every figure is a single run: differences under
about 3% (search) or 1 s (builds) are noise.

## 1. What 0.5.1 changes (downstream-reported findings)

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
- **Caveats.** Four datasets, one run each, binary nDCG, FiQA lexical is weak (23), and no domain-specific data.
  Do not change the default model on this evidence alone; check the candidate on your own queries first.

## 5. Comparison with other engines (BM25S, Tantivy, Pyserini/Lucene)

Same corpora, same queries, same metrics. Wikipedia 200k to 5M comparisons are not in this file yet
(pending the owner's runs of `benchmarks/run_suite.sh`).

### 5.1 SciFact, all engines, single run (`beir-scifact-all-engines.json`)

Vajra rows are 0.5.1. "Single" is mean per-query latency; "Batch QPS" runs the 300 queries as a batch
(16 workers for the parallel variants).

| Engine | Build (s) | Single (ms) | Single QPS | Batch QPS | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| vajra | 0.37 | 0.115 | 8,664 | 8,305 | 78.9 | 67.0 |
| vajra-parallel | 0.34 | 0.116 | 8,637 | 6,959 | 78.9 | 67.0 |
| bm25s | 0.26 | 0.125 | 7,987 | 15,839 | 77.4 | 66.2 |
| bm25s-parallel | 0.25 | 0.125 | 7,981 | 11,398 | 77.4 | 66.2 |
| tantivy | 0.03 | 0.178 | 5,610 | 5,441 | 73.1 | 60.4 |
| pyserini (Lucene) | 1.28 | 0.534 | 1,874 | 8,568 | 81.7 | 68.8 |

### 5.2 Four BEIR datasets (nDCG@10, recall@10, p50 single-query latency, index build)

Vajra, BM25S and Tantivy come from the 0.5.1 model-matrix runs; Pyserini comes from the 0.5.0 baseline
(it does not depend on Vajra code). Vajra and BM25S scores are identical across 0.5.0 and 0.5.1.

| Dataset (docs) | Engine | nDCG@10 | Recall@10 | MRR | p50 (ms) | Build (s) |
|---|---|---:|---:|---:|---:|---:|
| SciFact (5,183) | vajra-bm25 | 67.0 | 78.9 | 64.0 | 0.113 | 0.36 |
| | bm25s | 66.2 | 77.4 | 63.1 | 0.129 | 0.25 |
| | tantivy | 60.4 | 73.1 | 57.0 | 0.178 | 0.02 |
| | pyserini | 68.8 | 81.7 | 65.3 | 0.380 | 3.21 |
| NFCorpus (3,633) | vajra-bm25 | 30.9 | 15.2 | 51.6 | 0.042 | 0.26 |
| | bm25s | 30.7 | 14.5 | 51.9 | 0.100 | 0.20 |
| | tantivy | 28.5 | 13.7 | 48.2 | 0.033 | 0.02 |
| | pyserini | 32.6 | 15.6 | 52.9 | 0.272 | 3.09 |
| ArguAna (8,674) | vajra-bm25 | 47.7 | 74.5 | 39.4 | 1.474 | 0.51 |
| | bm25s | 48.2 | 75.4 | 39.7 | 0.330 | 0.33 |
| | tantivy | 10.7 | 17.3 | 8.6 | 0.026 | 0.02 |
| | pyserini | 48.5 | 76.0 | 40.0 | 1.024 | 3.34 |
| FiQA (57,638) | vajra-bm25 | 23.2 | 29.3 | 28.8 | 0.255 | 1.73 |
| | bm25s | 23.3 | 29.6 | 28.7 | 0.417 | 1.81 |
| | tantivy | 20.7 | 26.1 | 25.9 | 0.553 | 0.10 |
| | pyserini | 25.3 | 32.3 | 30.7 | 0.546 | 5.10 |

Vajra index build on these datasets, 0.5.0 against 0.5.1 (the default in-process build): SciFact 0.56 s to
0.36 s, NFCorpus 0.40 s to 0.26 s, ArguAna 0.75 s to 0.51 s, FiQA 3.01 s to 1.73 s.

### 5.3 Reading the comparison

- **Quality.** Vajra is level with BM25S on all four datasets (within 0.8 nDCG; Vajra ahead on SciFact and NFCorpus). Pyserini/Lucene is higher on every
  dataset, by 0.8 to 2.1 nDCG over Vajra, probably because it applies stemming, which Vajra and BM25S here do not; I have
  not isolated the cause. Tantivy is 2 to 7 nDCG behind. Its ArguAna score (10.7) is a wrapper defect: queries that
  fail to parse are silently dropped, so ignore it. Treat the Tantivy numbers as a lower bound.
- **Single-query latency.** Vajra is the fastest on SciFact and FiQA among BM25S, Tantivy and Pyserini, and
  about 2.4x faster than BM25S on NFCorpus. Tantivy is faster on NFCorpus (0.033 ms against 0.042 ms) and on ArguAna,
  where its number is meaningless. **ArguAna is a weakness:** Vajra's p50 is 1.47 ms against 0.33 ms for BM25S,
  because ArguAna queries are whole documents (long queries). This needs a look in 0.6.
- **Batch throughput.** BM25S is about 1.9x Vajra's batch QPS on SciFact (15,839 against 8,305), and Vajra's
  thread-pool variant is slower than the plain engine on this small corpus (6,959 against 8,305).
- **Build time.** Tantivy is the fastest builder by a wide margin (0.03 s on SciFact, 0.1 s on FiQA). Vajra builds
  slower than BM25S on the small corpora and about level with it on FiQA (1.73 s against 1.81 s).
- **Not covered here.** Wikipedia 200k to 5M (scaling and build cost at size), memory use, and rank-bm25.
  Single runs, one machine, binary-relevance nDCG.

## 6. Reproduce

```bash
TAG=after-0.5.1 caffeinate -i benchmarks/run_suite.sh beir-scifact hybrid-scifact
# all six engines on SciFact (pyserini needs JAVA_HOME set to a JDK 11+):
python benchmarks/benchmark.py --datasets beir-scifact --engines vajra vajra-parallel bm25s bm25s-parallel tantivy pyserini
TAG=models-0.5.1 benchmarks/run_model_matrix.sh && python benchmarks/summarize_models.py models-0.5.1
python benchmarks/bench_build.py v0.5.1 wiki20000 wiki200000 wiki200000@8   # needs the Wikipedia corpora
python benchmarks/bench_search.py v0.5.1 200000
```
