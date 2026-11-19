# Vajra BM25 0.5.0: performance baseline

Reference measurements taken **before** the 0.5.1 changes, so the patch release can be compared
against them. Raw results live next to this file in `results/baseline-0.5.0/`.

## Status

| Part | State |
|---|---|
| Test suite | 312 tests pass |
| BEIR SciFact, lexical (Vajra, bm25s) | Done (below) |
| Hybrid and vector on SciFact, NFCorpus, ArguAna, FiQA | Done (below) |
| Lexical on Wikipedia 200k, 500k, 1M, 2M, 5M (adds Tantivy; Pyserini up to 1M) | **Pending**: run on the Mac Studio, then add here |
| Tantivy and Pyserini on SciFact through `run_suite.sh` | Pending (both appear in the hybrid tables below) |

## Environment

- Code: commit `b3f991b`, run from branch `feat/v0.5.1` with the library code identical to `main` (only `benchmarks/` changed) (package version metadata `0.5.0`; `__version__` in the wheel still says 0.4.1, which is bug 5 in the 0.5.1 plan)
- Machine: Apple M5 Max, 18 cores, 128 GB RAM, Python 3.11.18, arm64
- Packages: numpy 2.4.6, scipy 1.17.1, numba 0.68.1, bm25s 0.3.16, tantivy 0.27.0, pyserini 1.6.0, sentence-transformers 6.3.0, torch 2.15.0
- Java: OpenJDK 21 (Pyserini)
- Embedding model for all dense and hybrid rows: `all-MiniLM-L6-v2` (384 dimensions) on Apple GPU (MPS)

## Lexical: BEIR SciFact (5,183 docs, 300 queries)

| Engine | Build (s) | Avg query (ms) | Recall@10 | nDCG@10 | QPS | Batch QPS |
|---|---:|---:|---:|---:|---:|---:|
| vajra | 0.57 | 0.121 | 78.9 | 67.0 | 8241 | 8099 |
| vajra-parallel | shared index | 0.118 | 78.9 | 67.0 | 8453 | 8069 |
| bm25s | 0.25 | 0.129 | 77.4 | 66.2 | 7732 | 15805 |
| bm25s-parallel | shared index | 0.123 | 77.4 | 66.2 | 8121 | 11558 |

SciFact uses Vajra's dense-matrix scorer (the log says "Using dense matrices for corpus of 5183
documents"), not the sparse eager scorer used for large corpora. The parallel variants share the
single-threaded index, so their build time is not separate.

## Hybrid and vector (BEIR, human relevance judgments)

Vajra code is used for: BM25, the flat and HNSW vector indices, and the hybrid fusion. Not Vajra:
bm25s, Tantivy, Pyserini (Lucene) and the embedding model. In "Hybrid: Lucene + Vajra dense" Lucene
does the keyword half while Vajra does the dense half and the fusion.

### Quality: nDCG@10 (0-100, higher is better)

| Dataset | Vajra BM25 | bm25s | tantivy | Pyserini (Lucene) | Vajra dense, exact | Vajra dense, HNSW ef=50 | Vajra hybrid, default (rrf, a=0.5) | Hybrid: Lucene + Vajra dense | Vajra hybrid, best of 9 configs* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact (5,183 docs) | 67.0 | 66.2 | 60.4 | 68.8 | 64.8 | 63.0 | 70.4 | 71.9 | 71.7 (linear a=0.5) |
| NFCorpus (3,633 docs) | 30.9 | 30.7 | 28.5 | 32.6 | 32.0 | 30.3 | 34.1 | 34.9 | 34.7 (rsf a=0.5) |
| ArguAna (8,674 docs) | 47.7 | 48.2 | 10.8 | 48.5 | 49.1 | 48.6 | 52.6 | 52.9 | 53.5 (linear a=0.7) |
| FiQA (57,638 docs) | 23.2 | 23.3 | 20.7 | 25.3 | 36.9 | 34.5 | 35.0 | 36.6 | 38.4 (linear a=0.3) |

*Best config chosen on the same test queries it is scored on: an upper bound, not a fair expectation. The default column is the fair comparison.

### Median query latency (ms), including query embedding for dense and hybrid

| Dataset | Vajra BM25 | bm25s | tantivy | Pyserini (Lucene) | Vajra dense, exact | Vajra dense, HNSW ef=50 | Vajra hybrid, default (rrf, a=0.5) | Hybrid: Lucene + Vajra dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 0.12 | 0.13 | 0.18 | 0.38 | 3.56 | 3.15 | 3.82 | 4.35 |
| NFCorpus | 0.05 | 0.10 | 0.03 | 0.27 | 3.37 | 3.14 | 3.49 | 3.98 |
| ArguAna | 1.50 | 0.33 | 0.03 | 1.02 | 4.56 | 3.55 | 6.50 | 5.97 |
| FiQA | 0.26 | 0.41 | 0.61 | 0.55 | 11.98 | 3.21 | 12.53 | 13.41 |

### Build cost (seconds)

| Dataset | Vajra BM25 build | Pyserini build | Embed corpus (MiniLM) | HNSW index add | Query embedding (avg) |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.6 | 3.2 | 4.6 | 4.5 | 11.4 ms |
| NFCorpus | 0.4 | 3.1 | 2.9 | 2.9 | 10.9 ms |
| ArguAna | 0.8 | 3.3 | 5.8 | 7.5 | 4.9 ms |
| FiQA | 3.0 | 5.1 | 31.8 | 83.6 | 2.9 ms |


## How to read the numbers

- nDCG@10 and recall@10 use binary relevance (as in `benchmarks/benchmark.py`). NFCorpus has graded
  judgments, so its nDCG is not comparable with published graded numbers.
- Latency is per query, single-threaded, 5 warm-up queries, and includes query embedding for dense
  and hybrid rows (the embedding cache is off).
- Dense "exact" is brute-force cosine search; HNSW is approximate (M=16, ef_construction=200).
- ArguAna excludes a query's own document from its results, as BEIR does.
- The seed is 42 (HNSW level sampling). Each configuration was run once, and run-to-run noise
  was not measured, so treat small differences (under about 1 nDCG point) with caution.

## Observations that matter for 0.5.1 and 0.6

1. **Stemming helps the keyword side.** With the same dense half and the same fusion, Lucene as the
   keyword engine beats Vajra BM25 on SciFact (71.9 vs 70.4), NFCorpus (34.9 vs 34.1) and FiQA
   (36.6 vs 35.0). Lucene's default analyzer stems and Vajra does not, which is consistent with
   stemming being the cause, but this baseline did not isolate it (stop words and BM25 details also
   differ). Stemming is already scheduled for 0.6.
2. **The hybrid default is not safe everywhere.** The default (rrf, alpha 0.5) scores 35.0 on FiQA,
   below dense alone (36.9). The best config differs per dataset, and the spread on FiQA is 9 points
   (best 38.4, linear alpha 0.3; worst 29.1, rrf alpha 0.7).
3. **Fusion depth is fixed.** `HybridSearchEngine` fuses only the top `2 * top_k` (20) results from
   each side.
4. **Hybrid latency is query embedding.** Dense and hybrid queries cost 3 to 13 ms against 0.05 to
   1.5 ms for BM25.
5. **HNSW costs quality and build time.** ef=50 loses 0.5 to 2.4 nDCG points against exact search on
   these datasets; on FiQA the HNSW build took 84 s against 32 s to embed the corpus.

## Known problems in this baseline

- **Tantivy on ArguAna (10.8 nDCG) is not valid.** The wrapper returns an empty list when query
  parsing fails, and ArguAna queries are long passages with punctuation. Do not use Tantivy's
  numbers until the wrapper escapes special characters.
- **The Wikipedia benchmark queries are synthetic**: a document's title (or a 3 to 5 word fragment)
  with the source document as the only relevant result. They measure self-retrieval, not relevance,
  and BM25 wins on them by construction.
- The Wikipedia corpora (`~/Github/ir_benchmark_data/wikipedia/`) were rebuilt on 2026-11-19 from the
  HuggingFace `wikimedia/wikipedia` `20231101.en` dump (6.08M articles of 200+ characters, first N in
  shard order). They differ from the January WikIR corpora, so Wikipedia numbers are not comparable
  with the committed January results in `results/benchmark_results.json`.

## Reproduce

```bash
source .venv/bin/activate
TAG=baseline-0.5.0 benchmarks/run_suite.sh beir-scifact hybrid-scifact hybrid-nfcorpus hybrid-arguana hybrid-fiqa
TAG=baseline-0.5.0 caffeinate -i benchmarks/run_suite.sh wiki-200k wiki-500k wiki-1m wiki-2m wiki-5m
python benchmarks/summarize_hybrid.py baseline-0.5.0
```

Use a different `TAG` for later runs: the index cache is keyed on the corpus only, so reusing a
cache across code changes would hide build-time differences.
