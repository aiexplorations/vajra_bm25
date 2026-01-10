#!/usr/bin/env python3
"""Test different optimization approaches for index building."""

import time
import numpy as np
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from scipy.sparse import coo_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from vajra_bm25 import DocumentCorpus
from vajra_bm25.text_processing import preprocess_text


def _tokenize_document(doc):
    """Worker function for parallel tokenization."""
    full_text = doc.title + ' ' + doc.content
    terms = preprocess_text(full_text)
    term_counts = Counter(terms)
    return doc.id, term_counts, len(terms)


def test_vocab_approaches(doc_term_counts):
    """Compare vocabulary building approaches."""
    print("\n--- Vocabulary Building Approaches ---")

    # Approach A: set.update loop (current)
    t0 = time.time()
    term_set_A = set()
    for tc in doc_term_counts:
        term_set_A.update(tc.keys())
    time_A = time.time() - t0
    print(f"  A. set.update loop: {time_A:.2f}s")

    # Approach B: Combined vocab + term_to_id in ONE pass
    t0 = time.time()
    term_to_id_B = {}
    next_id = 0
    for tc in doc_term_counts:
        for term in tc:
            if term not in term_to_id_B:
                term_to_id_B[term] = next_id
                next_id += 1
    time_B = time.time() - t0
    print(f"  B. Combined dict build: {time_B:.2f}s")

    # Now compare term_to_id creation from set
    t0 = time.time()
    term_to_id_A = {term: idx for idx, term in enumerate(term_set_A)}
    time_dict = time.time() - t0
    print(f"  + dict from set: {time_dict:.2f}s")
    print(f"  Total A (set.update + dict): {time_A + time_dict:.2f}s")
    print(f"  Total B (combined): {time_B:.2f}s")

    savings = (time_A + time_dict) - time_B
    print(f"  Savings: {savings:.2f}s ({savings/(time_A+time_dict)*100:.1f}%)")

    assert len(term_to_id_A) == len(term_to_id_B)
    return term_to_id_B


def test_coo_approaches(doc_term_counts, term_to_id):
    """Compare COO construction approaches."""
    print("\n--- COO Construction Approaches ---")

    total_entries = sum(len(tc) for tc in doc_term_counts)
    print(f"  Total entries: {total_entries:,}")

    # Approach A: Current NumPy pre-alloc with nested loops
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
    time_A = time.time() - t0
    print(f"  A. NumPy pre-alloc (nested loops): {time_A:.2f}s")

    # Approach B: Build per-document arrays, then concatenate
    t0 = time.time()
    all_rows = []
    all_cols = []
    all_data = []
    for doc_idx, term_counts in enumerate(doc_term_counts):
        n = len(term_counts)
        if n == 0:
            continue
        terms_list = list(term_counts.keys())
        counts_list = list(term_counts.values())

        doc_rows = np.array([term_to_id[t] for t in terms_list], dtype=np.int32)
        doc_cols = np.full(n, doc_idx, dtype=np.int32)
        doc_data = np.array(counts_list, dtype=np.float32)

        all_rows.append(doc_rows)
        all_cols.append(doc_cols)
        all_data.append(doc_data)

    rows_B = np.concatenate(all_rows)
    cols_B = np.concatenate(all_cols)
    data_B = np.concatenate(all_data)
    time_B = time.time() - t0
    print(f"  B. Per-doc arrays + concat: {time_B:.2f}s")

    # Approach C: Use list comprehensions (baseline comparison)
    t0 = time.time()
    rows_C = []
    cols_C = []
    data_C = []
    for doc_idx, term_counts in enumerate(doc_term_counts):
        for term, count in term_counts.items():
            rows_C.append(term_to_id[term])
            cols_C.append(doc_idx)
            data_C.append(count)
    rows_C = np.array(rows_C, dtype=np.int32)
    cols_C = np.array(cols_C, dtype=np.int32)
    data_C = np.array(data_C, dtype=np.float32)
    time_C = time.time() - t0
    print(f"  C. List appends + array: {time_C:.2f}s")

    # Find winner
    times = {'NumPy pre-alloc': time_A, 'Per-doc concat': time_B, 'List appends': time_C}
    winner = min(times, key=times.get)
    print(f"  Winner: {winner}")

    return rows_A, cols_A, data_A


def test_combined_vocab_and_coo(doc_term_counts):
    """Test building vocab and COO arrays in fewer passes."""
    print("\n--- Combined Vocab + COO (Single Iteration) ---")

    # Current approach: 3 iterations
    # 1. Build vocab set
    # 2. Count total entries
    # 3. Fill COO arrays

    # New approach: 2 iterations
    # 1. Build vocab dict + count entries in one pass
    # 2. Fill COO arrays

    t0 = time.time()

    # Pass 1: Build term_to_id and count entries
    term_to_id = {}
    next_id = 0
    total_entries = 0
    for tc in doc_term_counts:
        total_entries += len(tc)
        for term in tc:
            if term not in term_to_id:
                term_to_id[term] = next_id
                next_id += 1

    time_pass1 = time.time() - t0
    print(f"  Pass 1 (vocab + count): {time_pass1:.2f}s")
    print(f"    Terms: {len(term_to_id):,}, Entries: {total_entries:,}")

    # Pass 2: Fill COO arrays
    t0 = time.time()
    rows = np.empty(total_entries, dtype=np.int32)
    cols = np.empty(total_entries, dtype=np.int32)
    data = np.empty(total_entries, dtype=np.float32)
    idx = 0
    for doc_idx, term_counts in enumerate(doc_term_counts):
        for term, count in term_counts.items():
            rows[idx] = term_to_id[term]
            cols[idx] = doc_idx
            data[idx] = count
            idx += 1

    time_pass2 = time.time() - t0
    print(f"  Pass 2 (fill COO): {time_pass2:.2f}s")
    print(f"  Total (2-pass): {time_pass1 + time_pass2:.2f}s")

    return term_to_id, rows, cols, data


def main(corpus_size=100000):
    wiki_file = Path.home() / 'Github' / 'ir_benchmark_data' / 'wikipedia' / f'wikipedia_{corpus_size}.jsonl'

    if not wiki_file.exists():
        print(f"Wikipedia {corpus_size} not found at {wiki_file}")
        # Try smaller sizes
        for size in [500000, 200000, 100000]:
            alt_file = Path.home() / 'Github' / 'ir_benchmark_data' / 'wikipedia' / f'wikipedia_{size}.jsonl'
            if alt_file.exists():
                wiki_file = alt_file
                corpus_size = size
                break
        else:
            print("No Wikipedia data found!")
            return

    print(f"\n{'='*70}")
    print(f"TESTING OPTIMIZATIONS ON {corpus_size:,} DOCUMENTS")
    print(f"{'='*70}")

    # Load and tokenize
    print(f"\n1. Loading corpus...")
    t0 = time.time()
    corpus = DocumentCorpus.load_jsonl(wiki_file)
    print(f"   Loaded in {time.time()-t0:.1f}s")

    print(f"\n2. Tokenizing...")
    t0 = time.time()
    n_jobs = min(cpu_count(), len(corpus) // 100 + 1)
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_tokenize_document, corpus.documents)
    print(f"   Tokenized in {time.time()-t0:.1f}s ({n_jobs} workers)")

    doc_ids, doc_term_counts, doc_lengths = zip(*results)
    doc_term_counts = list(doc_term_counts)

    # Test vocabulary approaches
    term_to_id = test_vocab_approaches(doc_term_counts)

    # Test COO approaches
    test_coo_approaches(doc_term_counts, term_to_id)

    # Test combined approach
    test_combined_vocab_and_coo(doc_term_counts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=100000)
    args = parser.parse_args()
    main(args.size)
