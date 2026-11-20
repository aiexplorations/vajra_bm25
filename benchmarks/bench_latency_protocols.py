"""
Single-query latency of Vajra under each query protocol used in this repo's benchmarks.

`benchmark.py` and `bench_search.py` report different latencies for the same corpus because they differ in
query set, warm-up and cache setting. This script runs all of those combinations on one index so the
differences can be attributed, and writes the setup next to the numbers so a run can be reproduced.

    python benchmarks/bench_latency_protocols.py --n-docs 200000 --output results/<tag>/latency-wiki-200k.json
    PYTHONPATH=/path/to/older/checkout python benchmarks/bench_latency_protocols.py ...   # compare versions

Query sets
    bench_search : every document title, evenly spaced, up to 1,000 (the `bench_search.py` protocol)
    benchmark    : `load_wikipedia` from benchmark.py (seed 42; titles of 3+ words) so it cannot drift

Each query set runs with and without a 50-query warm-up, on an index built with the default cache and
with `cache_size=0` (what benchmark.py uses).
"""

import argparse
import json
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vajra_bm25  # noqa: E402
from vajra_bm25 import DocumentCorpus, VajraSearchOptimized  # noqa: E402
from vajra_bm25.text_processing import preprocess_text  # noqa: E402


def git_sha(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def bench_search_queries(corpus: DocumentCorpus, n: int, limit: int = 1000) -> list:
    """Same selection as benchmarks/bench_search.py."""
    return [d.title for d in corpus.documents[:: max(1, n // 1000)]][:limit]


def time_queries(engine, queries: list, warmup: bool) -> dict:
    if warmup:
        for q in queries[:50]:
            engine.search(q, top_k=10)
    lat = []
    for q in queries:
        t = time.perf_counter()
        engine.search(q, top_k=10)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    return {
        "mean_ms": st.mean(lat),
        "p50_ms": lat[len(lat) // 2],
        "p95_ms": lat[int(len(lat) * 0.95)],
        "qps": 1000 / st.mean(lat),
    }


def main() -> None:
    from benchmark import load_wikipedia  # imported late: pulls in the benchmark dependencies

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-docs", type=int, default=200000, help="Wikipedia corpus size (wikipedia_<n>.jsonl)")
    p.add_argument("--data-dir", type=Path, default=Path.home() / "Github" / "ir_benchmark_data")
    p.add_argument("--repeats", type=int, default=3, help="Repeat each measurement; the median repeat is reported")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    corpus, benchmark_eval = load_wikipedia(args.n_docs, args.data_dir)
    query_sets = {
        "bench_search": bench_search_queries(corpus, args.n_docs),
        "benchmark": [q.text for q in benchmark_eval],
    }
    code_root = Path(vajra_bm25.__file__).resolve().parent.parent
    meta = {
        "script": "benchmarks/bench_latency_protocols.py",
        "code_root": str(code_root),
        "code_git_sha": git_sha(code_root),
        "vajra_bm25_version": vajra_bm25.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpus": f"wikipedia_{args.n_docs}.jsonl",
        "corpus_size": len(corpus),
        "repeats": args.repeats,
        "query_sets": {
            name: {
                "n_queries": len(qs),
                "avg_terms_after_preprocessing": st.mean(len(preprocess_text(q)) for q in qs),
                "avg_words_raw": st.mean(len(q.split()) for q in qs),
            }
            for name, qs in query_sets.items()
        },
    }
    print(json.dumps(meta, indent=2), flush=True)

    rows = []
    for cache_label, kwargs in (("default", {}), ("0", {"cache_size": 0})):
        engine = VajraSearchOptimized(corpus, **kwargs)
        for qs_name, qs in query_sets.items():
            for warm in (True, False):
                runs = []
                for _ in range(args.repeats):
                    if hasattr(engine, "clear_cache"):
                        engine.clear_cache()
                    runs.append(time_queries(engine, qs, warm))
                runs.sort(key=lambda r: r["p50_ms"])
                row = {"cache": cache_label, "query_set": qs_name, "warmup": warm,
                       **runs[len(runs) // 2], "all_p50_ms": [r["p50_ms"] for r in runs]}
                rows.append(row)
                print(f"cache={cache_label:7} {qs_name:12} warmup={str(warm):5} "
                      f"p50 {row['p50_ms']:.3f}  mean {row['mean_ms']:.3f}  p95 {row['p95_ms']:.3f} ms", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"meta": meta, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
