#!/usr/bin/env python3
"""
Download and cache Wikipedia corpus for BM25 benchmarking.

Downloads Wikipedia data using ir-datasets and caches in JSONL format
for fast loading in future benchmarks.

Data stored in: ~/Github/ir_benchmark_data/wikipedia/

Requirements:
    pip install ir-datasets tqdm
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import asdict

try:
    import ir_datasets
    IR_DATASETS_AVAILABLE = True
except ImportError:
    IR_DATASETS_AVAILABLE = False
    print("Error: ir-datasets not installed. Run: pip install ir-datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Progress bars disabled.")
    print("Install with: pip install tqdm")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vajra_bm25 import Document, DocumentCorpus


# Wikipedia dataset configurations
WIKIPEDIA_DATASETS = {
    "wikir-en78k": {
        "name": "wikir/en78k",
        "description": "WikIR English Wikipedia corpus (2.46M articles)",
        "estimated_docs": 2456637,
    },
    "wikir-en59k": {
        "name": "wikir/en59k",
        "description": "WikIR English Wikipedia corpus (2.45M articles)",
        "estimated_docs": 2454785,
    },
    "wikir-en1k": {
        "name": "wikir/en1k",
        "description": "WikIR English Wikipedia corpus (369K articles)",
        "estimated_docs": 369721,
    },
}


class WikipediaDownloader:
    """Download and cache Wikipedia corpus."""

    def __init__(self, cache_dir: Path = Path.home() / "Github" / "ir_benchmark_data"):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloaded data (default: ~/Github/ir_benchmark_data)
        """
        self.cache_dir = Path(cache_dir)
        self.wiki_dir = self.cache_dir / "wikipedia"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        print(f"Cache directory: {self.cache_dir}")
        print(f"Wikipedia data will be stored in: {self.wiki_dir}")

    def get_cached_corpus_path(self, max_docs: Optional[int] = None) -> Path:
        """Get path to cached corpus file."""
        if max_docs:
            return self.wiki_dir / f"wikipedia_{max_docs}.jsonl"
        return self.wiki_dir / "wikipedia_full.jsonl"

    def is_cached(self, max_docs: Optional[int] = None) -> bool:
        """Check if corpus is already cached."""
        path = self.get_cached_corpus_path(max_docs)
        return path.exists()

    def download_and_cache(
        self,
        max_docs: Optional[int] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download Wikipedia and cache as JSONL.

        Args:
            max_docs: Maximum number of documents to download (None for all)
            force_download: Force re-download even if cached

        Returns:
            Path to cached JSONL file
        """
        corpus_path = self.get_cached_corpus_path(max_docs)

        # Check cache
        if corpus_path.exists() and not force_download:
            print(f"✓ Wikipedia corpus already cached: {corpus_path}")
            file_size = corpus_path.stat().st_size / (1024**3)  # GB
            print(f"  File size: {file_size:.2f} GB")
            return corpus_path

        print(f"\nDownloading Wikipedia corpus (WikIR dataset)...")
        if max_docs:
            print(f"Target: {max_docs:,} documents")
        else:
            print(f"Target: All documents (~2.46M max from WikIR)")

        # Try different Wikipedia sources
        dataset = self._get_wikipedia_dataset()

        if dataset is None:
            print("\nError: Could not load Wikipedia dataset.")
            print("\nThis usually means ir-datasets is not installed or has an issue.")
            print("Install with: pip install ir-datasets")
            print("\nAvailable Wikipedia datasets via ir-datasets:")
            print("  - wikir/en78k  (2.46M documents)")
            print("  - wikir/en59k  (2.45M documents)")
            print("  - wikir/en1k   (369K documents)")
            sys.exit(1)

        # Process and cache documents
        print(f"\nProcessing documents and writing to: {corpus_path}")

        doc_count = 0
        start_time = time.time()

        with open(corpus_path, 'w', encoding='utf-8') as f:
            iterator = dataset.docs_iter()

            if TQDM_AVAILABLE:
                total = max_docs if max_docs else 2500000
                iterator = tqdm(iterator, total=total, desc="Processing docs", unit="docs")

            for doc in iterator:
                if max_docs and doc_count >= max_docs:
                    break

                # Convert to Vajra Document format
                vajra_doc = Document(
                    id=doc.doc_id,
                    title=getattr(doc, 'title', ''),
                    content=doc.text if hasattr(doc, 'text') else str(doc)
                )

                # Write as JSONL
                json.dump(asdict(vajra_doc), f, ensure_ascii=False)
                f.write('\n')

                doc_count += 1

                # Progress updates (if no tqdm)
                if not TQDM_AVAILABLE and doc_count % 100000 == 0:
                    elapsed = time.time() - start_time
                    rate = doc_count / elapsed
                    print(f"  Processed {doc_count:,} documents ({rate:.0f} docs/sec)")

        elapsed = time.time() - start_time
        file_size = corpus_path.stat().st_size / (1024**3)  # GB

        print(f"\n✓ Download complete!")
        print(f"  Documents: {doc_count:,}")
        print(f"  File size: {file_size:.2f} GB")
        print(f"  Time: {elapsed:.1f}s ({doc_count/elapsed:.0f} docs/sec)")
        print(f"  Saved to: {corpus_path}")

        return corpus_path

    def _get_wikipedia_dataset(self):
        """Try to load Wikipedia dataset from available sources."""

        # WikIR datasets are available and work well
        # wikir/en78k has ~2.46M documents (good for 1M-2M benchmarks)
        # We'll use this as our primary source

        dataset_ids = [
            "wikir/en78k",      # 2.46M docs - best choice
            "wikir/en59k",      # 2.45M docs - alternative
            "wikir/en1k",       # 369K docs - fallback
        ]

        for dataset_id in dataset_ids:
            try:
                print(f"  Trying {dataset_id}...")
                dataset = ir_datasets.load(dataset_id)
                print(f"  ✓ Loaded {dataset_id} (Wikipedia corpus)")
                return dataset
            except Exception as e:
                print(f"  ✗ {dataset_id} not available: {e}")
                continue

        # If none work, inform user about the issue
        print("\n  Error: No Wikipedia datasets available via ir-datasets")
        return None

    def _try_huggingface_wikipedia(self):
        """Try to use HuggingFace datasets as fallback."""
        try:
            from datasets import load_dataset
            print("  Loading Wikipedia from HuggingFace...")

            # This is a wrapper to make HF dataset look like ir-datasets
            class HFWikipediaAdapter:
                def __init__(self):
                    print("  Downloading Wikipedia dataset (this may take a while)...")
                    self.dataset = load_dataset('wikipedia', '20220301.en', split='train')
                    print(f"  ✓ Loaded {len(self.dataset):,} articles")

                def docs_iter(self):
                    for idx, item in enumerate(self.dataset):
                        # Create a doc-like object
                        class WikiDoc:
                            def __init__(self, doc_id, title, text):
                                self.doc_id = doc_id
                                self.title = title
                                self.text = text

                        yield WikiDoc(
                            doc_id=str(idx),
                            title=item.get('title', ''),
                            text=item.get('text', '')
                        )

            return HFWikipediaAdapter()

        except ImportError:
            print("  ✗ HuggingFace datasets not installed")
            print("    Install with: pip install datasets")
            return None
        except Exception as e:
            print(f"  ✗ Error loading from HuggingFace: {e}")
            return None

    def load_corpus(self, max_docs: Optional[int] = None) -> DocumentCorpus:
        """
        Load cached Wikipedia corpus.

        Args:
            max_docs: Load only first N documents (None for all)

        Returns:
            DocumentCorpus with Wikipedia documents
        """
        corpus_path = self.get_cached_corpus_path(max_docs)

        if not corpus_path.exists():
            print(f"Corpus not cached. Downloading first...")
            self.download_and_cache(max_docs)

        print(f"Loading corpus from {corpus_path}...")
        start_time = time.time()

        corpus = DocumentCorpus.load_jsonl(corpus_path)

        elapsed = time.time() - start_time
        print(f"✓ Loaded {len(corpus):,} documents in {elapsed:.1f}s")

        return corpus


def main():
    """Main function to download and cache Wikipedia."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and cache Wikipedia corpus for BM25 benchmarking"
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path.home() / "Github" / "ir_benchmark_data",
        help="Cache directory (default: ~/Github/ir_benchmark_data)"
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        help="Maximum number of documents to download (default: all ~2.46M from WikIR)"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force re-download even if cached"
    )
    parser.add_argument(
        '--sizes',
        nargs='+',
        type=int,
        default=[100000, 1000000, 2000000],
        help="Create multiple corpus sizes (default: 100K, 1M, 2M). Max: 2.46M from WikIR"
    )

    args = parser.parse_args()

    print("="*80)
    print("WIKIPEDIA CORPUS DOWNLOAD FOR VAJRA BM25 BENCHMARKING")
    print("="*80)

    downloader = WikipediaDownloader(cache_dir=args.cache_dir)

    if args.max_docs:
        # Download single size
        downloader.download_and_cache(max_docs=args.max_docs, force_download=args.force)
    else:
        # Download multiple sizes
        print(f"\nDownloading corpus sizes: {[f'{s:,}' for s in args.sizes]}")

        for size in sorted(args.sizes):
            print(f"\n{'='*80}")
            print(f"DOWNLOADING {size:,} DOCUMENTS")
            print(f"{'='*80}")
            downloader.download_and_cache(max_docs=size, force_download=args.force)

    print(f"\n{'='*80}")
    print("DOWNLOAD COMPLETE!")
    print(f"{'='*80}")
    print(f"\nCached corpora location: {downloader.wiki_dir}")
    print(f"\nTo use in benchmarks:")
    print(f"  from benchmarks.download_wikipedia import WikipediaDownloader")
    print(f"  downloader = WikipediaDownloader()")
    print(f"  corpus = downloader.load_corpus(max_docs=1000000)  # 1M docs")


if __name__ == "__main__":
    main()
