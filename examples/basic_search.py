#!/usr/bin/env python3
"""
Basic Vajra BM25 Search Example

Demonstrates:
- Creating documents and corpus
- Building search engine
- Executing queries
- Explaining results
"""

from vajra_bm25 import (
    Document,
    DocumentCorpus,
    VajraSearch,
    create_sample_corpus,
)


def main():
    print("=" * 70)
    print("VAJRA BM25 - Basic Search Example")
    print("=" * 70)

    # Create sample corpus (10 documents about category theory, programming, algorithms)
    print("\n1. Creating sample corpus...")
    corpus = create_sample_corpus()
    print(f"   Created corpus with {len(corpus)} documents")

    # Display documents
    print("\n   Documents:")
    for doc in corpus:
        print(f"   - [{doc.id}] {doc.title}")

    # Build search engine
    print("\n2. Building search engine...")
    engine = VajraSearch(corpus)
    print(f"   Index stats: {engine.index}")

    # Execute searches
    test_queries = [
        "category theory functors",
        "search algorithms BFS DFS",
        "functional programming monads",
        "lambda calculus",
    ]

    print("\n3. Executing searches...")
    for query in test_queries:
        print("\n" + "-" * 70)
        print(f"Query: '{query}'")
        print("-" * 70)

        results = engine.search(query, top_k=3)

        if not results:
            print("   No results found.")
            continue

        for r in results:
            print(f"\n   {r.rank}. [{r.document.id}] {r.document.title}")
            print(f"      Score: {r.score:.4f}")
            print(f"      Content: {r.document.content[:80]}...")

    # Explain a result
    print("\n4. Explaining a result...")
    query = "category theory morphisms"
    results = engine.search(query, top_k=1)

    if results:
        top_result = results[0]
        print(f"\n   Query: '{query}'")
        print(f"   Top result: {top_result.document.title}")
        print(f"   Total score: {top_result.score:.4f}")

        explanation = engine.explain_result(query, top_result.document.id)
        print("\n   Score breakdown by term:")
        for term, score in sorted(explanation['term_scores'].items(),
                                 key=lambda x: x[1], reverse=True):
            if score > 0:
                print(f"      '{term}': {score:.4f}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
