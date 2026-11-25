# Changelog

## 0.5.1

Fixes reported by a downstream user (benchmark of 17 November 2026, run against 0.5.0). No API breaks.

- **A document added later was never found.** `add_document` and `remove_document` rebuilt only
  `self.scorer`; the eager, Numba and MaxScore scorers kept answering from the old index, so on sparse
  indices a new document returned no results. All scorers now come from one `_build_scorers()` that every
  build, load, add and remove path calls. New `add_documents(list)` rebuilds once for a batch
  (`batch_add_documents` stays as an alias), and the duplicate-ID check no longer scans the whole corpus.
  The lazy vector engine used by `mode="vector"` and `mode="hybrid"` now receives added documents too.
- **Every index build used a `multiprocessing.Pool`.** Starting workers cost more than tokenising small
  corpora (629 documents: 0.27 s to 0.07 s here), and forking a threaded server is unsafe. Builds now
  tokenise in-process by default. New `n_jobs` argument (`-1` = all cores) uses a `spawn` pool from
  20,000 documents; large corpora that relied on the old default should pass `n_jobs=-1`
  (200,000 documents: about 28 s in-process against 18 s with `n_jobs=8`; 0.5.0 took 17 s).
  `preprocess_text` no longer allocates a `Token` object per word (identical output, about 2.6x
  faster), which is what makes the in-process default competitive.
- **Tags could not be indexed as one token.** New `analyzer` argument for `VajraSearchOptimized` and
  `VajraSearchParallel` (applied to documents and queries, saved with the index) and an optional
  `tag_aware_analyzer` that also emits `v110` for `V-110` and `p400a`/`p400b` for `P-400A/B`. The default
  analyzer is unchanged: SciFact results are identical to 0.5.0.
- **The library installed its own log handler.** Vajra now attaches only a `NullHandler`, so records reach
  the application's logging once. `configure_logging()` remains as an opt-in for scripts.
- **`__version__` said 0.4.1 in the 0.5.0 wheel.** It now reads the installed package metadata.
- **A bare `pip install vajra-bm25` could not be imported.** The package imports NumPy at load time but did not
  declare it. NumPy is now a base dependency; SciPy stays in the `optimized` extra.
- Benchmarks: tagged suite runner, hybrid/vector benchmark on BEIR, `--vajra-n-jobs`; the `benchmark` and
  `cli` extras now include `requests`, which `beir` imports without declaring.
