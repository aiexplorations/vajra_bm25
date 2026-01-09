# Plan: Full Wikipedia (~7M Articles) Benchmark Feasibility

## Context

User wants to validate how Vajra and BM25S scale to full English Wikipedia (~7M articles) out of curiosity. This is not for production deployment.

## Key Findings from Research

### 1. Full English Wikipedia Scale

- **Articles**: ~7.1 million
- **Total text**: ~58GB
- **Current data available**: Up to 1M documents (~5.3GB JSONL)

### 2. Memory Requirements

| Corpus Size | Vajra (sparse) | Vajra (eager) | BM25S | Elasticsearch |
|-------------|---------------|---------------|-------|---------------|
| 1M docs | 220 MB | 330 MB | ~250 MB | 2-4 GB |
| 2.46M docs | 540 MB | 800 MB | ~600 MB | 4-8 GB |
| 7M docs | **1.5 GB** | **2.3 GB** | ~1.8 GB | 8-16 GB |

**Your machine**: 24GB RAM (M4 Pro) - **easily handles 7M documents**

### 3. Build Time Estimates

Based on 500K benchmark scaling:

| Corpus | Vajra Build | BM25S Build |
|--------|-------------|-------------|
| 500K | 7 min | 4 min |
| 1M | ~15 min | ~10 min |
| 2.46M | ~35 min | ~20 min |
| 7M | **~1.7 hours** | **~1 hour** |

### 4. Can They Handle 7M?

| Framework | 7M Feasibility | Notes |
|-----------|----------------|-------|
| **Vajra** | **Yes** | ~1.5-2.3GB RAM, 1.7hr build |
| **BM25S** | **Yes** | ~1.8GB RAM, 1hr build, has mmap for even less RAM |
| **Tantivy** | Yes | Faster build, disk-based |
| **Elasticsearch** | Yes | Overkill for validation, but production-ready |

### 5. Potential Bottlenecks

1. **Vocabulary size**: 7M docs → ~2-3M unique terms → ~100-150MB for dictionaries
2. **Peak memory during build**: COO→CSR conversion needs extra RAM temporarily (~2x final size)
3. **Build time**: ~1.7 hours is manageable for a one-time validation
4. **Disk I/O**: 58GB corpus needs to be read from disk during indexing

## Recommendations

### Option A: Test at 1M First (Recommended Start)
- We already have 1M docs (5.3GB)
- Feasible: ~15min build, ~2GB RAM
- Validates scaling behavior before committing to larger download

### Option B: Download 2.46M via ir-datasets
- Dataset: `wikir/en78k` provides 2.46M Wikipedia articles
- Download time: 30-60 minutes
- Build time: ~45min-1hr per engine
- RAM needed: ~1-1.5GB

### Option C: Full 7M Wikipedia
- Need to download from Wikipedia dumps directly
- Download: ~20GB compressed, ~58GB text
- Build time: ~2-3 hours per engine
- RAM needed: ~3-4GB (sparse mode)
- **Feasible on your 24GB M4 Pro**

## Proposed Next Steps

1. **Run 1M benchmark** to validate current scaling
2. **Measure actual memory usage** during 1M build
3. **If 1M works well**, download 2.46M via `ir-datasets`
4. **If still curious**, download full Wikipedia dump for 7M test

## Verification

After running 1M benchmark:
- Check actual memory usage with `psutil` or Activity Monitor
- Verify index size on disk
- Extrapolate to 44M feasibility

## Files to Modify

None for this evaluation - this is a research/planning task.
