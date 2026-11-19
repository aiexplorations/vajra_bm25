#!/usr/bin/env bash
# Run the benchmark suite and keep results, logs and index caches under a tag.
#
#   TAG=baseline-0.5.0 benchmarks/run_suite.sh            # all datasets
#   TAG=baseline-0.5.0 benchmarks/run_suite.sh wiki-200k  # selected datasets
#
# Each dataset is a separate benchmark.py run so one failure or crash keeps the
# other results. The index cache is per tag: the cache key ignores code version,
# so reusing a cache across code changes would hide build-time differences.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${TAG:?set TAG, e.g. TAG=baseline-0.5.0}"
WORKERS="${WORKERS:-16}"
# Tokenizer processes for Vajra index builds (-1 = all cores); empty = library default
VAJRA_N_JOBS="${VAJRA_N_JOBS:-}"
OUT="results/$TAG"
CACHE=".index_cache/$TAG"
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(beir-scifact hybrid-scifact hybrid-nfcorpus hybrid-fiqa hybrid-arguana
            wiki-200k wiki-500k wiki-1m wiki-2m wiki-5m)
fi

mkdir -p "$OUT"

python - "$OUT/environment.json" "$WORKERS" "$VAJRA_N_JOBS" <<'EOF'
import json, os, platform, subprocess, sys
from importlib import metadata

def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

def ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None

env = {
    "git_sha": git("rev-parse", "HEAD"),
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_dirty_files": git("status", "--porcelain", "--untracked-files=no").splitlines(),
    "python": platform.python_version(),
    "machine": platform.machine(),
    "cpu_count": os.cpu_count(),
    "workers": int(sys.argv[2]),
    "vajra_n_jobs": sys.argv[3] or None,
    "packages": {p: ver(p) for p in
                 ["vajra-bm25", "numpy", "scipy", "numba", "bm25s", "joblib", "tantivy",
                  "pyserini", "beir", "sentence-transformers", "torch"]},
}
json.dump(env, open(sys.argv[1], "w"), indent=2)
print(json.dumps(env, indent=2))
EOF

# Pyserini needs a JVM. Brew's openjdk is not registered with /usr/libexec/java_home.
if [ -z "${JAVA_HOME:-}" ] && [ -d /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home ]; then
  export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
fi

for ds in "${DATASETS[@]}"; do
  # hybrid-<beir name>: lexical + vector + hybrid on a BEIR dataset (benchmark_hybrid.py)
  if [[ "$ds" == hybrid-* ]]; then
    echo "=== $ds  [$TAG] ==="
    python benchmarks/benchmark_hybrid.py --dataset "${ds#hybrid-}" --output "$OUT/$ds.json"
    continue
  fi

  # Pyserini goes last: the JVM does not survive a later fork() from vajra's index build.
  # Pyserini indexes single-threaded, so it only runs up to 1M docs; 5M also drops the
  # parallel variants to keep memory and time in check.
  case "$ds" in
    wiki-2m) ENGINES=(vajra vajra-parallel bm25s bm25s-parallel tantivy) ;;
    wiki-5m) ENGINES=(vajra bm25s tantivy) ;;
    *)       ENGINES=(vajra vajra-parallel bm25s bm25s-parallel tantivy pyserini) ;;
  esac
  echo "=== $ds  [$TAG]  engines: ${ENGINES[*]} ==="
  python benchmarks/benchmark.py \
    --datasets "$ds" \
    --engines "${ENGINES[@]}" \
    --workers "$WORKERS" \
    ${VAJRA_N_JOBS:+--vajra-n-jobs "$VAJRA_N_JOBS"} \
    --output "$OUT/$ds.json" \
    --log "$OUT/benchmark.log" \
    --index-cache-dir "$CACHE"
done
