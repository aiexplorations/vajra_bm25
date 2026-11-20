#!/usr/bin/env python3
"""
Print markdown tables from results/<tag>/hybrid-*.json (written by benchmark_hybrid.py).

    python benchmarks/summarize_hybrid.py baseline-0.5.0
"""

import json
import sys
from pathlib import Path

DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA"}


def load(tag):
    out = {}
    for ds in DATASETS:
        path = Path("results") / tag / f"hybrid-{ds}.json"
        if path.exists():
            out[ds] = json.loads(path.read_text())
    return out


def get(data, system):
    return next((r for r in data["results"] if r["system"] == system), None)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    tag = sys.argv[1]
    data = load(tag)
    if not data:
        sys.exit(f"No hybrid-*.json found in results/{tag}")

    cols = [  # (column label, result system name)
        ("Vajra BM25", "vajra-bm25"), ("bm25s", "bm25s"), ("tantivy", "tantivy"), ("Pyserini (Lucene)", "pyserini"),
        ("Vajra dense, exact", "vajra-flat"), ("Vajra dense, HNSW ef=50", "vajra-hnsw ef=50"),
        ("Vajra hybrid, default (rrf, a=0.5)", "hybrid vajra+flat rrf a=0.5"),
        ("Hybrid: Lucene + Vajra dense", "hybrid pyserini+flat rrf a=0.5"),
    ]

    def fmt(ds, system, field="ndcg_at_10", digits=1):
        r = get(data[ds], system)
        return f"{r[field]:.{digits}f}" if r else "–"

    print(f"## {tag}: nDCG@10 (0-100, higher is better)\n")
    rows = []
    for ds in data:
        hy = [r for r in data[ds]["results"] if r["system"].startswith("hybrid vajra+flat")]
        best = max(hy, key=lambda r: r["ndcg_at_10"])
        best_label = best["system"].replace("hybrid vajra+flat ", "")
        rows.append([f"{NAMES[ds]} ({data[ds]['corpus_size']:,} docs)"]
                    + [fmt(ds, s) for _, s in cols]
                    + [f"{best['ndcg_at_10']:.1f} ({best_label})"])
    print(table(["Dataset"] + [c for c, _ in cols] + ["Vajra hybrid, best of 9 configs*"], rows))
    print("\n*Best config chosen on the same test queries it is scored on: an upper bound, not a "
          "fair expectation. The default column is the fair comparison.")

    print("\n## Median query latency (ms), including query embedding for dense and hybrid\n")
    rows = [[NAMES[ds]] + [fmt(ds, s, "p50_latency_ms", 2) for _, s in cols] for ds in data]
    print(table(["Dataset"] + [c for c, _ in cols], rows))

    print("\n## Build cost (seconds)\n")
    rows = []
    for ds in data:
        b = data[ds]["build"]
        rows.append([NAMES[ds], f"{b['vajra_build_s']:.1f}", f"{b.get('pyserini_build_s', float('nan')):.1f}",
                     f"{b['embed_corpus_s']:.1f}", f"{b['hnsw_add_s']:.1f}", f"{b['query_embed_ms_avg']:.1f} ms"])
    print(table(["Dataset", "Vajra BM25 build", "Pyserini build", "Embed corpus (MiniLM)",
                 "HNSW index add", "Query embedding (avg)"], rows))


if __name__ == "__main__":
    main()
