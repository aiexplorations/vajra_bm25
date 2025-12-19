#!/usr/bin/env python3
"""
Profile index building performance to identify bottlenecks.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vajra_bm25 import DocumentCorpus, VajraSearchOptimized
from rank_bm25 import BM25Okapi
from vajra_bm25.text_processing import preprocess_text


def profile_vajra_index(corpus, label):
    """Profile Vajra index building."""
    print(f"\n{'='*60}")
    print(f"Profiling Vajra: {label}")
    print(f"{'='*60}")

    start = time.time()
    engine = VajraSearchOptimized(corpus)
    total_time = time.time() - start

    print(f"Total index build time: {total_time:.2f}s")
    print(f"Throughput: {len(corpus)/total_time:.0f} docs/sec")

    return total_time


def profile_rank_bm25_index(corpus, label):
    """Profile rank-bm25 index building."""
    print(f"\n{'='*60}")
    print(f"Profiling rank-bm25: {label}")
    print(f"{'='*60}")

    # Tokenize corpus
    start = time.time()
    tokenized = []
    for doc in corpus:
        full_text = doc.title + " " + doc.content
        tokens = preprocess_text(full_text)
        tokenized.append(tokens)
    tokenize_time = time.time() - start
    print(f"Tokenization: {tokenize_time:.2f}s ({len(corpus)/tokenize_time:.0f} docs/sec)")

    # Build index
    start = time.time()
    bm25 = BM25Okapi(tokenized)
    build_time = time.time() - start
    print(f"Index building: {build_time:.2f}s")

    total_time = tokenize_time + build_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(corpus)/total_time:.0f} docs/sec")

    return total_time


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile index building")
    parser.add_argument('--corpus', type=Path, required=True, help="Path to JSONL corpus")
    parser.add_argument('--max-docs', type=int, help="Limit number of documents")

    args = parser.parse_args()

    print(f"Loading corpus from {args.corpus}...")
    corpus = DocumentCorpus.load_jsonl(args.corpus)

    if args.max_docs:
        corpus = DocumentCorpus(corpus.documents[:args.max_docs])

    print(f"Corpus: {len(corpus):,} documents")

    # Profile both
    vajra_time = profile_vajra_index(corpus, f"{len(corpus):,} docs")
    rank_time = profile_rank_bm25_index(corpus, f"{len(corpus):,} docs")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Vajra: {vajra_time:.2f}s ({len(corpus)/vajra_time:.0f} docs/sec)")
    print(f"rank-bm25: {rank_time:.2f}s ({len(corpus)/rank_time:.0f} docs/sec)")

    if vajra_time < rank_time:
        speedup = rank_time / vajra_time
        print(f"\nVajra is {speedup:.1f}x FASTER at index building")
    else:
        slowdown = vajra_time / rank_time
        print(f"\nVajra is {slowdown:.1f}x SLOWER at index building")


if __name__ == "__main__":
    main()
