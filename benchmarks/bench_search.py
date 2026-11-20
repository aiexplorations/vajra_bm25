import sys, time, statistics as st
from pathlib import Path
import vajra_bm25
from vajra_bm25 import DocumentCorpus, VajraSearchOptimized
if __name__ == "__main__":
    n = int(sys.argv[2])
    c = DocumentCorpus.load_jsonl(Path.home()/"Github/ir_benchmark_data/wikipedia/wikipedia_200000.jsonl")
    c = DocumentCorpus(c.documents[:n])
    qs = [d.title for d in c.documents[:: max(1, n // 1000)]][:1000]
    e = VajraSearchOptimized(c)
    for q in qs[:50]: e.search(q, top_k=10)
    e.cache.clear() if hasattr(e, "cache") and hasattr(e.cache, "clear") else None
    lat = []
    for q in qs:
        t = time.perf_counter(); e.search(q, top_k=10); lat.append((time.perf_counter()-t)*1000)
    lat.sort()
    print(f"{sys.argv[1]} {n:>7} docs {len(qs)} q  mean {st.mean(lat):.3f} ms  p50 {lat[len(lat)//2]:.3f}  p95 {lat[int(len(lat)*.95)]:.3f}  qps {1000/st.mean(lat):,.0f}  ({Path(vajra_bm25.__file__).parent.parent.name})")
