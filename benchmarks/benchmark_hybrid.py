#!/usr/bin/env python3
"""
Hybrid / vector benchmark for Vajra BM25 on BEIR datasets with human relevance judgments.

Systems measured on one dataset (same documents, same queries, same embedding model):

  lexical   vajra-bm25, bm25s, tantivy, pyserini
  vector    vajra-flat (exact), vajra-hnsw (ef_search 50 and 100)
  hybrid    vajra BM25 + flat vector, fusion in {rrf, linear, rsf} x alpha in {0.3, 0.5, 0.7}
            vajra BM25 + HNSW vector (rrf, alpha 0.5)
            {bm25s, tantivy, pyserini} + flat vector (rrf, alpha 0.5): same fusion code and
            embeddings, only the lexical engine changes

Metrics match benchmarks/benchmark.py (binary relevance, recall@10, nDCG@10, MRR), so BM25 rows
are comparable with the lexical suite. Note nDCG here is binary: graded datasets such as NFCorpus
will not match published graded nDCG.

Latency is per query, single-threaded, and includes query embedding for vector and hybrid
systems (the embedding cache is disabled). Index build is reported in separate parts:
corpus embedding, flat index add, HNSW index add, BM25 build.

Usage:
    python benchmarks/benchmark_hybrid.py --dataset scifact --output results/TAG/hybrid-scifact.json
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import benchmark as b  # noqa: E402  (loaders, metrics, lexical engine wrappers)
from vajra_bm25 import Document  # noqa: E402
from vajra_bm25.search import SearchResult  # noqa: E402
from vajra_bm25.vector import (  # noqa: E402
    FlatVectorIndex,
    HybridSearchEngine,
    NativeHNSWIndex,
    TextEmbeddingMorphism,
    VajraVectorSearch,
)

# BEIR names. ArguAna queries are documents in the corpus, so BEIR's identical-id exclusion
# is structural there (the query's own document must not count as a hit).
DATASETS = {
    "scifact": {"ignore_identical_ids": False},
    "nfcorpus": {"ignore_identical_ids": False},
    "fiqa": {"ignore_identical_ids": False},
    "arguana": {"ignore_identical_ids": True},
}

LEXICAL_ENGINES = ["vajra", "bm25s", "tantivy", "pyserini"]
TOP_K = 10


@dataclass
class SystemResult:
    system: str
    kind: str  # lexical | vector | hybrid
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    params: Dict = field(default_factory=dict)


class TimedEmbedder(TextEmbeddingMorphism):
    """Text embedder that accumulates time spent embedding documents and keeps the vectors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_seconds = 0.0
        self.captured: List[np.ndarray] = []

    def embed_batch(self, texts):
        t = time.perf_counter()
        out = super().embed_batch(texts)
        self.batch_seconds += time.perf_counter() - t
        self.captured.append(out)
        return out


class LexicalAdapter:
    """Presents a benchmark.py lexical engine (returns doc ids) as a HybridSearchEngine BM25 side."""

    def __init__(self, engine, docs_by_id: Dict[str, Document]):
        self.engine = engine
        self.docs_by_id = docs_by_id

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        ids = self.engine.search(query, top_k)
        return [
            SearchResult(document=self.docs_by_id[i], score=1.0 / (r + 1), rank=r + 1)
            for r, i in enumerate(ids) if i in self.docs_by_id
        ]


def evaluate(name: str, kind: str, search: Callable[[str], List[str]], queries, ignore_identical: bool,
             params: Optional[Dict] = None) -> SystemResult:
    """Run every query once, timing each, and compute mean recall/nDCG/MRR."""
    for q in queries[:5]:  # warm-up
        search(q.text)

    latencies, recalls, ndcgs, mrrs = [], [], [], []
    for q in queries:
        t = time.perf_counter()
        ids = search(q.text)
        latencies.append((time.perf_counter() - t) * 1000)
        if ignore_identical:
            ids = [i for i in ids if i != q.query_id]
        ids = ids[:TOP_K]
        recalls.append(b.recall_at_k(ids, q.relevant_docs, TOP_K))
        ndcgs.append(b.ndcg_at_k(ids, q.relevant_docs, TOP_K))
        mrrs.append(b.mrr(ids, q.relevant_docs))

    latencies.sort()
    res = SystemResult(
        system=name, kind=kind,
        recall_at_10=100 * statistics.fmean(recalls),
        ndcg_at_10=100 * statistics.fmean(ndcgs),
        mrr=100 * statistics.fmean(mrrs),
        avg_latency_ms=statistics.fmean(latencies),
        p50_latency_ms=latencies[len(latencies) // 2],
        p95_latency_ms=latencies[int(len(latencies) * 0.95)],
        params=params or {},
    )
    print(f"  {name:<34} nDCG@10 {res.ndcg_at_10:5.1f}  R@10 {res.recall_at_10:5.1f}  "
          f"p50 {res.p50_latency_ms:8.2f} ms", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--lexical", nargs="+", choices=LEXICAL_ENGINES, default=LEXICAL_ENGINES)
    ap.add_argument("--embed-batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ignore_identical = DATASETS[args.dataset]["ignore_identical_ids"]
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== hybrid benchmark: {args.dataset} ===", flush=True)
    corpus, queries = b.load_beir_dataset(args.dataset)
    print(f"  {len(corpus):,} docs, {len(queries)} queries, model {args.model}", flush=True)
    docs_by_id = {d.id: d for d in corpus}

    results: List[SystemResult] = []
    build: Dict[str, float] = {}

    # ---- lexical engines ---------------------------------------------------------------
    lexical: Dict[str, object] = {}
    for name in args.lexical:
        if name == "vajra":
            eng = b.VajraEngine(corpus)
        elif name == "bm25s":
            if not b.BM25S_AVAILABLE:
                continue
            eng = b.BM25SEngine(corpus)
        elif name == "tantivy":
            if not b.TANTIVY_AVAILABLE:
                continue
            eng = b.TantivyEngine(corpus)
        else:
            # Pyserini last: the JVM does not coexist with a later fork()
            eng = b.PyseriniEngine(corpus)
        t = time.perf_counter()
        try:
            eng.build()
        except Exception as e:  # e.g. no Java for pyserini
            print(f"  skipping {name}: {e}")
            continue
        build[f"{name}_build_s"] = time.perf_counter() - t
        lexical[name] = eng
        label = "vajra-bm25" if name == "vajra" else name
        results.append(evaluate(label, "lexical", lambda q, e=eng: e.search(q, TOP_K), queries, ignore_identical))

    # ---- vector: embed once, then build flat and HNSW indices -----------------------------
    # Title is folded into the embedded text so vector and lexical sides see the same text.
    vec_docs = [
        Document(id=d.id, title=d.title,
                 content=f"{d.title}. {d.content}" if d.title else d.content)
        for d in corpus
    ]
    embedder = TimedEmbedder(args.model)
    embedder.dimension  # load model outside the timed region
    dim = embedder.dimension

    flat = VajraVectorSearch(embedder, FlatVectorIndex(dimension=dim), cache_size=0)
    t = time.perf_counter()
    flat.index_documents(vec_docs, batch_size=args.embed_batch_size, show_progress=False)
    flat_total = time.perf_counter() - t
    build["embed_corpus_s"] = embedder.batch_seconds
    build["flat_add_s"] = flat_total - embedder.batch_seconds
    print(f"  embedded {len(vec_docs):,} docs in {embedder.batch_seconds:.1f}s", flush=True)

    hnsw_index = NativeHNSWIndex(dimension=dim, metric="cosine", M=16, ef_construction=200, ef_search=50)
    hnsw = VajraVectorSearch(embedder, hnsw_index, cache_size=0)
    t = time.perf_counter()
    hnsw_index.add([d.id for d in vec_docs], np.vstack(embedder.captured))
    build["hnsw_add_s"] = time.perf_counter() - t
    hnsw._documents = dict(flat._documents)
    print(f"  HNSW index add: {build['hnsw_add_s']:.1f}s", flush=True)

    q_embed = []
    for q in queries[:200]:
        t = time.perf_counter()
        embedder.embed(q.text)
        q_embed.append((time.perf_counter() - t) * 1000)
    build["query_embed_ms_avg"] = statistics.fmean(q_embed)

    def vec_search(engine):
        return lambda q: [r.document.id for r in engine.search(q, top_k=TOP_K)]

    results.append(evaluate("vajra-flat", "vector", vec_search(flat), queries, ignore_identical))
    for ef in (50, 100):
        hnsw_index.ef_search = ef
        results.append(evaluate(f"vajra-hnsw ef={ef}", "vector", vec_search(hnsw), queries, ignore_identical,
                                {"ef_search": ef, "M": 16, "ef_construction": 200}))
    hnsw_index.ef_search = 50

    # ---- hybrid: Vajra BM25 + flat, fusion sweep ----------------------------------------
    bm25_engine = lexical["vajra"].engine if "vajra" in lexical else None
    if bm25_engine is not None:
        for method in ("rrf", "linear", "rsf"):
            for alpha in (0.3, 0.5, 0.7):
                hybrid = HybridSearchEngine(bm25_engine, flat, alpha=alpha, method=method)
                results.append(evaluate(
                    f"hybrid vajra+flat {method} a={alpha}", "hybrid",
                    lambda q, h=hybrid: [r.document.id for r in h.search(q, top_k=TOP_K)],
                    queries, ignore_identical, {"fusion": method, "alpha": alpha, "vector": "flat"}))
        hybrid = HybridSearchEngine(bm25_engine, hnsw, alpha=0.5, method="rrf")
        results.append(evaluate(
            "hybrid vajra+hnsw rrf a=0.5", "hybrid",
            lambda q: [r.document.id for r in hybrid.search(q, top_k=TOP_K)],
            queries, ignore_identical, {"fusion": "rrf", "alpha": 0.5, "vector": "hnsw ef=50"}))

    # ---- hybrid with other lexical engines (same fusion, same vectors) ----------------------
    for name in ("bm25s", "tantivy", "pyserini"):
        if name not in lexical:
            continue
        hybrid = HybridSearchEngine(LexicalAdapter(lexical[name], docs_by_id), flat, alpha=0.5, method="rrf")
        results.append(evaluate(
            f"hybrid {name}+flat rrf a=0.5", "hybrid",
            lambda q, h=hybrid: [r.document.id for r in h.search(q, top_k=TOP_K)],
            queries, ignore_identical, {"fusion": "rrf", "alpha": 0.5, "vector": "flat"}))

    # ---- save ---------------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "benchmark": "hybrid",
        "dataset": f"BEIR/{args.dataset}",
        "timestamp": datetime.now().isoformat(),
        "corpus_size": len(corpus),
        "num_queries": len(queries),
        "model": args.model,
        "embedding_dim": dim,
        "ignore_identical_ids": ignore_identical,
        "seed": args.seed,
        "device": getattr(getattr(embedder, "_model", None), "device", None) and str(embedder._model.device),
        "platform": platform.platform(),
        "build": build,
        "results": [asdict(r) for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")
    print("Build:", {k: round(v, 2) for k, v in build.items()})


if __name__ == "__main__":
    main()
