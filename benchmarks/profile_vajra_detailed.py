#!/usr/bin/env python3
"""
Detailed profiling of Vajra index building.

Breaks down time spent in each phase:
- Document loading
- Tokenization
- Vocabulary building
- Sparse matrix construction
- IDF computation
"""

import time
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from vajra_bm25 import DocumentCorpus
from vajra_bm25.text_processing import preprocess_text

try:
    from scipy.sparse import lil_matrix
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    lil_matrix = None
    np = None


def profile_vajra_detailed(corpus):
    """Profile each step of Vajra index building."""

    print(f"\n{'='*70}")
    print(f"DETAILED VAJRA INDEX BUILD PROFILING")
    print(f"{'='*70}")
    print(f"Corpus: {len(corpus):,} documents")
    print(f"Scipy available: {SCIPY_AVAILABLE}")
    print(f"{'='*70}\n")

    timings = {}

    # ========================================================================
    # PHASE 1: First pass - tokenization and vocabulary building
    # ========================================================================
    print("Phase 1: Tokenization and vocabulary building...")
    phase1_start = time.time()

    term_set = set()
    doc_term_counts = []
    doc_lengths = []

    for doc_idx, doc in enumerate(corpus):
        full_text = doc.title + " " + doc.content
        terms = preprocess_text(full_text)

        term_counts = Counter(terms)
        term_set.update(term_counts.keys())
        doc_term_counts.append(term_counts)
        doc_lengths.append(len(terms))

        # Progress indicator
        if (doc_idx + 1) % 10000 == 0:
            elapsed = time.time() - phase1_start
            rate = (doc_idx + 1) / elapsed
            print(f"  Processed {doc_idx + 1:,} docs ({rate:.0f} docs/sec)")

    phase1_time = time.time() - phase1_start
    timings['tokenization'] = phase1_time

    num_docs = len(corpus)
    num_terms = len(term_set)

    print(f"  ✓ Phase 1 complete: {phase1_time:.2f}s")
    print(f"  Unique terms: {num_terms:,}")
    print(f"  Throughput: {num_docs/phase1_time:.0f} docs/sec\n")

    # ========================================================================
    # PHASE 2: Term ID assignment
    # ========================================================================
    print("Phase 2: Assigning term IDs...")
    phase2_start = time.time()

    term_to_id = {}
    for term_id, term in enumerate(sorted(term_set)):
        term_to_id[term] = term_id

    phase2_time = time.time() - phase2_start
    timings['term_id_assignment'] = phase2_time

    print(f"  ✓ Phase 2 complete: {phase2_time:.2f}s\n")

    # ========================================================================
    # PHASE 3: Sparse matrix construction
    # ========================================================================
    print("Phase 3: Building sparse term-document matrix...")
    phase3_start = time.time()

    if SCIPY_AVAILABLE:
        # Build LIL matrix (efficient for construction)
        lil = lil_matrix((num_terms, num_docs), dtype=np.float32)

        for doc_idx, term_counts in enumerate(doc_term_counts):
            for term, count in term_counts.items():
                term_id = term_to_id[term]
                lil[term_id, doc_idx] = count

            # Progress indicator
            if (doc_idx + 1) % 10000 == 0:
                elapsed = time.time() - phase3_start
                rate = (doc_idx + 1) / elapsed
                print(f"  Built matrix for {doc_idx + 1:,} docs ({rate:.0f} docs/sec)")

        # Convert to CSR
        print("  Converting LIL to CSR format...")
        csr_start = time.time()
        term_doc_matrix = lil.tocsr()
        csr_time = time.time() - csr_start
        timings['lil_to_csr_conversion'] = csr_time
        print(f"  ✓ Conversion complete: {csr_time:.2f}s")

        # Matrix stats
        nnz = term_doc_matrix.nnz
        sparsity = 1.0 - (nnz / (num_terms * num_docs))
        memory_mb = (term_doc_matrix.data.nbytes +
                     term_doc_matrix.indices.nbytes +
                     term_doc_matrix.indptr.nbytes) / (1024**2)

        print(f"  Matrix shape: {num_terms:,} × {num_docs:,}")
        print(f"  Non-zero entries: {nnz:,}")
        print(f"  Sparsity: {sparsity*100:.2f}%")
        print(f"  Memory: {memory_mb:.1f} MB")
    else:
        print("  ✗ Scipy not available - would use dense matrix (not recommended)")
        term_doc_matrix = None

    phase3_time = time.time() - phase3_start
    timings['sparse_matrix_build'] = phase3_time

    print(f"  ✓ Phase 3 complete: {phase3_time:.2f}s\n")

    # ========================================================================
    # PHASE 4: IDF computation
    # ========================================================================
    print("Phase 4: Computing IDF values...")
    phase4_start = time.time()

    if SCIPY_AVAILABLE and term_doc_matrix is not None:
        # Document frequencies
        doc_freqs = np.asarray((term_doc_matrix > 0).sum(axis=1)).flatten()

        # IDF computation
        idf_cache = np.log((num_docs - doc_freqs + 0.5) / (doc_freqs + 0.5) + 1.0)

        print(f"  IDF range: [{idf_cache.min():.3f}, {idf_cache.max():.3f}]")

    phase4_time = time.time() - phase4_start
    timings['idf_computation'] = phase4_time

    print(f"  ✓ Phase 4 complete: {phase4_time:.2f}s\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    total_time = sum(timings.values())

    print(f"{'='*70}")
    print("TIMING BREAKDOWN")
    print(f"{'='*70}")

    for phase, duration in timings.items():
        percentage = (duration / total_time) * 100
        print(f"  {phase:.<40} {duration:>8.2f}s ({percentage:>5.1f}%)")

    print(f"  {'─'*68}")
    print(f"  {'TOTAL':.<40} {total_time:>8.2f}s (100.0%)")
    print(f"{'='*70}\n")

    print(f"Overall throughput: {num_docs/total_time:.0f} docs/sec")
    print(f"Memory per document: {(memory_mb * 1024) / num_docs:.2f} KB/doc\n")

    # Bottleneck identification
    max_phase = max(timings.items(), key=lambda x: x[1])
    print(f"⚠️  BOTTLENECK: {max_phase[0]} ({max_phase[1]:.2f}s, {(max_phase[1]/total_time)*100:.1f}%)")

    return timings, total_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detailed Vajra index profiling")
    parser.add_argument('--corpus', type=Path, required=True, help="Path to JSONL corpus")
    parser.add_argument('--max-docs', type=int, help="Limit number of documents")

    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus}...")
    load_start = time.time()
    corpus = DocumentCorpus.load_jsonl(args.corpus)
    load_time = time.time() - load_start

    if args.max_docs:
        corpus = DocumentCorpus(corpus.documents[:args.max_docs])

    print(f"✓ Loaded {len(corpus):,} documents in {load_time:.2f}s\n")

    # Run detailed profiling
    timings, total_time = profile_vajra_detailed(corpus)

    # Recommendations
    print(f"{'='*70}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*70}")

    if timings.get('tokenization', 0) / total_time > 0.5:
        print("• Tokenization is >50% of time")
        print("  → Consider parallel tokenization with multiprocessing")
        print("  → Or use faster tokenizer (e.g., pre-compiled regex)")

    if timings.get('sparse_matrix_build', 0) / total_time > 0.3:
        print("• Sparse matrix building is >30% of time")
        print("  → Use COO format construction (faster than LIL for this pattern)")
        print("  → Batch inserts instead of per-element")

    if timings.get('term_id_assignment', 0) / total_time > 0.1:
        print("• Term ID assignment is >10% of time")
        print("  → Skip sorting - use insertion order (Python 3.7+ dicts)")
        print("  → Or assign IDs during first pass")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
