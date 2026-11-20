# vajra-bm25 0.5.1 release candidate: test report

Date: 2026-11-20. Platform: macOS arm64 (Apple silicon, MPS), Python 3.11.18.
Tree: `release/v0.5.1`, built from the `main` 68f48b8 file state, plus the deprecation warning (`ba8b140`, with a
dynamic version), plus numpy as a base dependency. Not yet committed.

## Verdict

**Go for a release candidate**, subject to the two open decisions at the end. No defect was found in the search code.

## 1. Changes in this tree relative to `release/v0.5.1` as pushed

| Change | Why |
|---|---|
| 0.5.1 fixes from `main` brought in (stale scorers, pool hang, analyzer hook, logging, `__version__`) | The pushed branch was based on 0.5.0 and had none of them |
| `pyproject.toml` version 0.5.1 | Pushed branch said 0.5.0 |
| `__version__` read from package metadata; `tag_aware_analyzer` export kept | Pushed branch hardcoded 0.4.1 and dropped the export |
| Deprecation warning text uses `__version__` | It quoted v0.4.1 |
| `dependencies = ["numpy>=1.20.0"]` | A bare `pip install` could not be imported (numpy is imported at load time) |
| Downstream-reporter name removed from docs, tests and results; internal plan file left out | Requested |

Search, index and scoring code in `vajra_bm25/` is identical to `main`. Only `__init__.py` (+10 lines) and
`pyproject.toml` (+1 line) differ from it.

## 2. Unit and integration tests

| Run | Result |
|---|---|
| `pytest tests/` on final tree | **372 passed, 0 failed** (360 existing + 12 release checks) |
| Coverage | 76% overall; `optimized.py` 82%, `parallel.py` 100%, `text_processing.py` 100%, `search.py` 65% |
| `main` alone | 360 passed |
| `release/v0.5.1` as pushed | 3 failed (version checks, build), 321 passed; lacked the 48 fix tests |

The release checks include the sdist and wheel build with `twine check`.

## 3. Packaging

| Check | Result |
|---|---|
| sdist and wheel build | `vajra_bm25-0.5.1` both build |
| `twine check` | Both pass |
| Name scan of wheel and sdist | No matches; no `references/` or `plans/` in the sdist |
| Bare wheel install (numpy only) | Imports; `VajraSearch` and `VajraSearchOptimized` work without scipy; HNSW works |
| Wheel with `optimized,persistence,pdf,cli` extras | Imports, searches, `vajra-search --help` works |

## 4. Benchmarks (tag `rc1`; raw data in this directory)

### BM25, BEIR SciFact (5,183 docs, 300 queries) vs `results/after-0.5.1`

| Engine | nDCG@10 | Recall@10 | MRR | p50 ms (rc1 / 0.5.1) |
|---|---:|---:|---:|---|
| vajra | 66.97 | 78.89 | 63.96 | 0.111 / 0.114 |
| vajra-parallel | 66.97 | 78.89 | 63.96 | 0.111 / 0.113 |
| bm25s | 66.17 | 77.39 | 63.12 | 0.130 / 0.135 |
| tantivy | 60.44 | 73.08 | 57.02 | 0.180 / 0.172 |

Quality identical to the baseline for every engine. Vajra QPS 8,672 (baseline 8,393); build 0.40 s (0.56 s).

### Hybrid and vector (MiniLM, 18 systems per dataset) vs `results/baseline-0.5.0`

| Dataset | Max nDCG@10 change | Median p50 latency ratio |
|---|---:|---:|
| SciFact | 0.000 | 1.00 |
| NFCorpus | 0.054 (Tantivy rows) | 1.00 |
| FiQA | 0.000 | 1.00 |
| ArguAna | 0.017 | 0.99 |

All 72 rows matched a baseline row. Changes under 0.06 nDCG are run-to-run variation on Tantivy.

### Wikipedia 200k (200,000 docs, 464 queries; first run, no baseline for this harness)

| Engine | Build s | p50 ms | QPS |
|---|---:|---:|---:|
| vajra | 29.1 | 0.393 | 2,406 |
| vajra-parallel | 0.0 (shares index) | 0.394 | 2,397 |
| bm25s | 33.5 | 0.598 | 1,584 |
| tantivy | 2.1 | 1.839 | 470 |

`results/after-0.5.1/RESULTS.md` reports 0.241 ms p50 at 200k, but from `bench_search.py` (1,000 queries), so it is
not like-for-like. The search code is identical to `main`, so this is a harness difference, not a regression. A
like-for-like run of `bench_search.py` has not been done on this tree.

## 5. Known issues and open decisions

1. **Deprecation warning wording.** It says the package is "no longer actively developed", which conflicts with the
   v0.6 plan. It fires on every import and on every `vajra-search` run, and `-W error::DeprecationWarning` makes the
   package unimportable. Decide the final text before shipping.
2. **Not covered:** Wikipedia 500k to 5M, memory use, Python versions other than 3.11, Linux, Pyserini.
3. **`requires-python = ">=3.8"`** is declared, but only 3.11 was tested.
4. The `rc1` results are tied to this machine; run-to-run noise is about 3%.
