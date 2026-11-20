#!/usr/bin/env python3
"""
Markdown tables comparing embedding models from results/<tag>/<model>/hybrid-<dataset>.json
(written by benchmarks/run_model_matrix.sh).

    python benchmarks/summarize_models.py models-0.5.1
"""

import json
import sys
from pathlib import Path

DATASETS = ["scifact", "nfcorpus", "arguana", "fiqa"]
NAMES = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "arguana": "ArguAna", "fiqa": "FiQA"}


def sysrow(d, name):
    return next(r for r in d["results"] if r["system"] == name)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    root = Path("results") / sys.argv[1]
    models = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        models[mdir.name] = {ds: json.loads((mdir / f"hybrid-{ds}.json").read_text())
                             for ds in DATASETS if (mdir / f"hybrid-{ds}.json").exists()}
    order = sorted(models, key=lambda m: (m != "all-MiniLM-L6-v2", m))

    first = next(iter(models.values()))
    print("### Lexical reference (same for every model)\n")
    print(table(["System"] + [NAMES[d] for d in DATASETS],
                [[s] + [f"{sysrow(first[d], s)['ndcg_at_10']:.1f}" for d in DATASETS]
                 for s in ("vajra-bm25", "bm25s", "tantivy")]))

    def grid(title, fn):
        print(f"\n### {title}\n")
        print(table(["Model"] + [NAMES[d] for d in DATASETS],
                    [[m] + [fn(models[m][d]) if d in models[m] else "-" for d in DATASETS] for m in order]))

    def best(d):
        hyb = [r for r in d["results"] if r["system"].startswith("hybrid vajra+flat")]
        b = max(hyb, key=lambda r: r["ndcg_at_10"])
        return f"{b['ndcg_at_10']:.1f} ({b['system'].split('flat ')[1]})"

    grid("Vector only (flat, exact) nDCG@10", lambda d: f"{sysrow(d, 'vajra-flat')['ndcg_at_10']:.1f}")
    grid("Default hybrid (rrf, alpha 0.5) nDCG@10", lambda d: f"{sysrow(d, 'hybrid vajra+flat rrf a=0.5')['ndcg_at_10']:.1f}")
    grid("Best hybrid fusion (vajra + flat) nDCG@10", best)
    grid("Vector recall@10 (flat)", lambda d: f"{sysrow(d, 'vajra-flat')['recall_at_10']:.1f}")
    grid("Corpus embedding time (s)", lambda d: f"{d['build']['embed_corpus_s']:.0f}")
    grid("Query embedding, mean (ms)", lambda d: f"{d['build']['query_embed_ms_avg']:.1f}")
    grid("Default hybrid p50 latency (ms, includes query embedding)",
         lambda d: f"{sysrow(d, 'hybrid vajra+flat rrf a=0.5')['p50_latency_ms']:.1f}")
    print("\nEmbedding dimension: " + ", ".join(
        f"{m} {next(iter(models[m].values()))['embedding_dim']}" for m in order))


if __name__ == "__main__":
    main()
