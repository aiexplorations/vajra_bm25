#!/usr/bin/env bash
# Hybrid benchmark across embedding models and BEIR datasets (same code, same queries).
#
#   TAG=models-0.5.1 benchmarks/run_model_matrix.sh
#   TAG=models-0.5.1 MODELS="BAAI/bge-small-en-v1.5" DATASETS="scifact" benchmarks/run_model_matrix.sh
#
# Lexical engines are limited to vajra, bm25s and tantivy (lexical results do not depend on the
# model; the baseline already covers pyserini). Results: results/$TAG/<model>/hybrid-<dataset>.json
set -uo pipefail
cd "$(dirname "$0")/.."
TAG="${TAG:?set TAG, e.g. TAG=models-0.5.1}"
MODELS="${MODELS:-all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 all-mpnet-base-v2}"
DATASETS="${DATASETS:-scifact nfcorpus arguana fiqa}"
for m in $MODELS; do
  for ds in $DATASETS; do
    out="results/$TAG/${m//\//_}"
    mkdir -p "$out"
    echo "=== $m  $ds ==="
    python benchmarks/benchmark_hybrid.py --dataset "$ds" --model "$m" \
      --lexical vajra bm25s tantivy --output "$out/hybrid-$ds.json" || echo "FAILED: $m $ds"
  done
done
