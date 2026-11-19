import inspect, json, sys, time
from pathlib import Path

def load_scifact(n=None):
    from vajra_bm25 import Document, DocumentCorpus
    docs = []
    with open("datasets/scifact/corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs.append(Document(id=d["_id"], title=d.get("title", ""), content=d["text"]))
    return DocumentCorpus(docs[:n] if n else docs)

def load_wiki(n):
    from vajra_bm25 import DocumentCorpus
    c = DocumentCorpus.load_jsonl(Path.home() / "Github/ir_benchmark_data/wikipedia/wikipedia_200000.jsonl")
    return DocumentCorpus(c.documents[:n])

if __name__ == "__main__":
    import vajra_bm25
    from vajra_bm25 import VajraSearchOptimized
    from vajra_bm25.optimized import VectorizedIndexSparse
    label = sys.argv[1]
    cases = sys.argv[2:]  # e.g. sci629 sci5183 wiki20000 wiki200000
    supports_n_jobs = "n_jobs" in inspect.signature(VectorizedIndexSparse.build).parameters
    print(f"{label}: vajra_bm25 from {Path(vajra_bm25.__file__).parent.parent}, n_jobs supported: {supports_n_jobs}")
    for case in cases:
        name, _, njobs = case.partition("@")
        corpus = load_scifact(int(name[3:])) if name.startswith("sci") else load_wiki(int(name[4:]))
        kwargs = {}
        if njobs:
            if not supports_n_jobs:
                continue
            kwargs["n_jobs"] = int(njobs)
        t = time.perf_counter()
        idx = VectorizedIndexSparse()
        idx.build(corpus, **kwargs)
        dt = time.perf_counter() - t
        print(f"  {case:<18} {len(corpus):>8,} docs  sparse index build {dt:7.2f}s  terms {idx.num_terms:,}", flush=True)
