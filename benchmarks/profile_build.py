#!/usr/bin/env python3
"""Profile index building phases at different scales."""

import time
import sys
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from scipy.sparse import coo_matrix, csr_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from vajra_bm25 import DocumentCorpus
from vajra_bm25.text_processing import preprocess_text


def _tokenize_document(doc):
    full_text = doc.title + ' ' + doc.content
    terms = preprocess_text(full_text)
    term_counts = Counter(terms)
    return doc.id, term_counts, len(terms)


def profile_build(corpus_size):
    wiki_file = Path.home() / 'Github' / 'ir_benchmark_data' / 'wikipedia' / f'wikipedia_{corpus_size}.jsonl'

    print(f'\n{"="*70}')
    print(f'PROFILING {corpus_size:,} DOCUMENTS')
    print(f'{"="*70}')

    timings = {}

    # 1. Load corpus
    t0 = time.time()
    corpus = DocumentCorpus.load_jsonl(wiki_file)
    timings['load'] = time.time() - t0
    print(f'1. Load corpus: {timings["load"]:.1f}s')

    # 2. Parallel tokenization
    t0 = time.time()
    n_jobs = min(cpu_count(), len(corpus) // 100 + 1)
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_tokenize_document, corpus.documents)
    timings['tokenize'] = time.time() - t0
    print(f'2. Tokenization ({n_jobs} workers): {timings["tokenize"]:.1f}s')

    # 3. Extract results
    t0 = time.time()
    doc_ids, doc_term_counts, doc_lengths_list = zip(*results)
    doc_ids = list(doc_ids)
    doc_term_counts = list(doc_term_counts)
    timings['extract'] = time.time() - t0
    print(f'3. Extract results: {timings["extract"]:.2f}s')

    # 4. Build vocabulary - compare methods
    print(f'\n--- Vocabulary Building ---')

    # Method A: set.union (current)
    t0 = time.time()
    term_set_A = set().union(*[tc.keys() for tc in doc_term_counts])
    time_union = time.time() - t0
    print(f'   A. set.union: {time_union:.2f}s')

    # Method B: repeated update (original)
    t0 = time.time()
    term_set_B = set()
    for tc in doc_term_counts:
        term_set_B.update(tc.keys())
    time_update = time.time() - t0
    print(f'   B. set.update loop: {time_update:.2f}s')

    winner = "set.union" if time_union < time_update else "set.update"
    print(f'   Winner: {winner} (diff: {abs(time_union-time_update):.2f}s)')

    term_set = term_set_A
    timings['vocab'] = min(time_union, time_update)
    print(f'   Terms: {len(term_set):,}')

    # 5. Create term_to_id
    t0 = time.time()
    term_to_id = {term: idx for idx, term in enumerate(term_set)}
    timings['term_to_id'] = time.time() - t0
    print(f'\n4. Create term_to_id: {timings["term_to_id"]:.2f}s')

    # 6. Count total entries
    t0 = time.time()
    total_entries = sum(len(tc) for tc in doc_term_counts)
    timings['count'] = time.time() - t0
    print(f'5. Count entries: {timings["count"]:.2f}s ({total_entries:,} entries)')

    # 7. COO construction - compare methods
    print(f'\n--- COO Matrix Construction ---')

    # Method A: NumPy pre-alloc (current)
    t0 = time.time()
    rows_A = np.empty(total_entries, dtype=np.int32)
    cols_A = np.empty(total_entries, dtype=np.int32)
    data_A = np.empty(total_entries, dtype=np.float32)
    idx = 0
    for doc_idx, term_counts in enumerate(doc_term_counts):
        for term, count in term_counts.items():
            rows_A[idx] = term_to_id[term]
            cols_A[idx] = doc_idx
            data_A[idx] = count
            idx += 1
    time_numpy = time.time() - t0
    print(f'   A. NumPy pre-alloc: {time_numpy:.1f}s')

    # Method B: Python lists (original)
    t0 = time.time()
    rows_B = []
    cols_B = []
    data_B = []
    for doc_idx, term_counts in enumerate(doc_term_counts):
        for term, count in term_counts.items():
            term_id = term_to_id[term]
            rows_B.append(term_id)
            cols_B.append(doc_idx)
            data_B.append(count)
    time_lists = time.time() - t0
    print(f'   B. Python lists: {time_lists:.1f}s')

    winner = "NumPy" if time_numpy < time_lists else "Python lists"
    speedup = time_lists / time_numpy if time_numpy < time_lists else time_numpy / time_lists
    print(f'   Winner: {winner} ({speedup:.2f}x faster)')

    timings['coo'] = min(time_numpy, time_lists)

    # 8. Build sparse matrix
    t0 = time.time()
    num_terms = len(term_set)
    num_docs = len(corpus)
    coo = coo_matrix((data_A, (rows_A, cols_A)), shape=(num_terms, num_docs), dtype=np.float32)
    term_doc_matrix = coo.tocsr()
    timings['csr'] = time.time() - t0
    print(f'\n6. COO->CSR: {timings["csr"]:.1f}s')

    # 9. IDF computation
    t0 = time.time()
    doc_freqs = np.asarray((term_doc_matrix > 0).sum(axis=1)).flatten()
    idf_cache = np.log((num_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0).astype(np.float32)
    timings['idf'] = time.time() - t0
    print(f'7. IDF: {timings["idf"]:.1f}s')

    # 10. Term bounds
    doc_lengths = np.array(list(doc_lengths_list), dtype=np.int32)
    avg_doc_length = float(doc_lengths.mean())
    norm_factors = (1.0 - 0.75 + 0.75 * (doc_lengths / avg_doc_length)).astype(np.float32)

    # 11. Eager score matrix - compare methods
    print(f'\n--- Eager Score Matrix (term_id expansion) ---')

    indptr = term_doc_matrix.indptr
    indices = term_doc_matrix.indices
    csr_data = term_doc_matrix.data

    # Method A: np.repeat (current)
    t0 = time.time()
    term_ids_A = np.repeat(np.arange(num_terms, dtype=np.int32), np.diff(indptr))
    time_repeat = time.time() - t0
    print(f'   A. np.repeat: {time_repeat:.3f}s')

    # Method B: Python loop (original)
    t0 = time.time()
    term_ids_B = np.zeros(len(csr_data), dtype=np.int32)
    for term_id in range(num_terms):
        row_start = indptr[term_id]
        row_end = indptr[term_id + 1]
        term_ids_B[row_start:row_end] = term_id
    time_loop = time.time() - t0
    print(f'   B. Python loop: {time_loop:.3f}s')

    # Method C: Numba (if available)
    try:
        from numba import njit, prange

        @njit(parallel=True)
        def fill_term_ids_numba(indptr, out):
            for term_id in prange(len(indptr) - 1):
                for i in range(indptr[term_id], indptr[term_id + 1]):
                    out[i] = term_id

        term_ids_C = np.zeros(len(csr_data), dtype=np.int32)
        # Warm up
        fill_term_ids_numba(indptr[:100], term_ids_C[:indptr[99]])

        t0 = time.time()
        term_ids_C = np.zeros(len(csr_data), dtype=np.int32)
        fill_term_ids_numba(indptr, term_ids_C)
        time_numba = time.time() - t0
        print(f'   C. Numba parallel: {time_numba:.3f}s')
    except ImportError:
        time_numba = float('inf')
        print(f'   C. Numba: not available')

    best_time = min(time_repeat, time_loop, time_numba)
    if best_time == time_repeat:
        winner = "np.repeat"
    elif best_time == time_loop:
        winner = "Python loop"
    else:
        winner = "Numba"
    print(f'   Winner: {winner}')

    timings['term_ids'] = best_time

    # Full score matrix build
    t0 = time.time()
    tf = csr_data.astype(np.float32)
    doc_norms = norm_factors[indices]
    k1 = np.float32(1.5)
    tf_component = (tf * (k1 + 1)) / (tf + k1 * doc_norms)
    idf_for_entries = idf_cache[term_ids_A]
    score_data = (tf_component * idf_for_entries).astype(np.float32)
    score_matrix = csr_matrix((score_data, indices.copy(), indptr.copy()),
                              shape=(num_terms, num_docs), dtype=np.float32)
    timings['score_matrix'] = time.time() - t0
    print(f'\n8. Score matrix (total): {timings["score_matrix"]:.1f}s')

    # Summary
    total = sum(timings.values())
    print(f'\n--- SUMMARY ---')
    print(f'Total time: {total:.1f}s')
    print(f'\nBreakdown:')
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = t / total * 100
        print(f'  {name:15s}: {t:7.1f}s ({pct:5.1f}%)')

    return timings


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=500000)
    args = parser.parse_args()

    profile_build(args.size)
