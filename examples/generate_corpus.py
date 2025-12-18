#!/usr/bin/env python3
"""
Corpus Generation Example

Demonstrates:
- Creating custom documents
- Saving corpus to JSONL
- Loading corpus from JSONL
"""

import random
from pathlib import Path
from vajra_bm25 import Document, DocumentCorpus


# Sample topics and vocabulary for generating synthetic documents
TOPICS = {
    "machine_learning": [
        "neural network", "deep learning", "gradient descent", "backpropagation",
        "activation function", "loss function", "optimizer", "batch size",
        "learning rate", "regularization", "dropout", "overfitting",
        "training data", "validation set", "test accuracy", "model weights",
    ],
    "statistics": [
        "hypothesis testing", "p-value", "confidence interval", "sample size",
        "standard deviation", "variance", "mean", "median", "regression",
        "correlation", "statistical significance", "null hypothesis",
        "probability distribution", "normal distribution", "chi-square",
    ],
    "algorithms": [
        "sorting algorithm", "binary search", "dynamic programming", "recursion",
        "time complexity", "space complexity", "hash table", "tree traversal",
        "graph algorithm", "shortest path", "divide and conquer", "greedy",
        "breadth first search", "depth first search", "big O notation",
    ],
    "databases": [
        "SQL query", "database index", "transaction", "ACID properties",
        "normalization", "foreign key", "primary key", "join operation",
        "query optimization", "relational database", "NoSQL", "data modeling",
        "indexing strategy", "query plan", "table scan", "B-tree",
    ],
}


def generate_document(doc_id: int, topic: str) -> Document:
    """Generate a synthetic document for a topic."""
    terms = TOPICS[topic]
    num_terms = random.randint(5, 12)
    selected_terms = random.sample(terms, min(num_terms, len(terms)))

    # Create title
    title_terms = random.sample(selected_terms, min(3, len(selected_terms)))
    title = f"Understanding {' and '.join(title_terms).title()}"

    # Create content
    content_parts = []
    for term in selected_terms:
        templates = [
            f"The concept of {term} is fundamental to understanding this field.",
            f"When working with {term}, it's important to consider several factors.",
            f"Modern approaches to {term} have evolved significantly.",
            f"Applications of {term} can be found in many domains.",
            f"Research on {term} continues to advance rapidly.",
        ]
        content_parts.append(random.choice(templates))

    content = " ".join(content_parts)

    return Document(
        id=f"doc_{doc_id:05d}",
        title=title,
        content=content,
        metadata={"topic": topic, "generated": True}
    )


def generate_corpus(num_docs: int = 100) -> DocumentCorpus:
    """Generate a corpus with documents from various topics."""
    documents = []
    topics = list(TOPICS.keys())

    for i in range(num_docs):
        topic = random.choice(topics)
        doc = generate_document(i + 1, topic)
        documents.append(doc)

    return DocumentCorpus(documents)


def main():
    print("=" * 70)
    print("VAJRA BM25 - Corpus Generation Example")
    print("=" * 70)

    # Generate corpus
    num_docs = 100
    print(f"\n1. Generating {num_docs} synthetic documents...")
    corpus = generate_corpus(num_docs)
    print(f"   Created corpus with {len(corpus)} documents")

    # Show topic distribution
    topic_counts = {}
    for doc in corpus:
        topic = doc.metadata.get("topic", "unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("\n   Topic distribution:")
    for topic, count in sorted(topic_counts.items()):
        print(f"   - {topic}: {count} documents")

    # Save to JSONL
    output_path = Path("generated_corpus.jsonl")
    print(f"\n2. Saving corpus to {output_path}...")
    corpus.save_jsonl(output_path)
    print(f"   Saved {len(corpus)} documents")

    # Reload to verify
    print(f"\n3. Reloading corpus from {output_path}...")
    loaded_corpus = DocumentCorpus.load_jsonl(output_path)
    print(f"   Loaded {len(loaded_corpus)} documents")

    # Show sample documents
    print("\n4. Sample documents:")
    for doc in list(loaded_corpus)[:3]:
        print(f"\n   [{doc.id}] {doc.title}")
        print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"   Content: {doc.content[:100]}...")

    # Test search on generated corpus
    print("\n5. Testing search on generated corpus...")
    from vajra_bm25 import VajraSearch

    engine = VajraSearch(loaded_corpus)

    test_queries = [
        "neural network deep learning",
        "database indexing optimization",
        "sorting algorithm complexity",
    ]

    for query in test_queries:
        results = engine.search(query, top_k=2)
        print(f"\n   Query: '{query}'")
        for r in results:
            print(f"   - {r.document.title} (score: {r.score:.3f})")

    # Clean up
    print(f"\n6. Cleaning up...")
    output_path.unlink()
    print(f"   Removed {output_path}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
