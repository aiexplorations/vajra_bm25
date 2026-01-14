#!/usr/bin/env python3
"""
Vajra Search CLI

Interactive search engine with BM25, vector, and hybrid search modes.

Usage:
    vajra-search                           # Interactive BM25 mode with scifact
    vajra-search -q "machine learning"     # Single query mode
    vajra-search --corpus my_docs.jsonl    # Custom JSONL corpus
    vajra-search --corpus document.pdf     # Single PDF file
    vajra-search --corpus ./pdf_folder/    # Directory of PDFs
    vajra-search --dataset beir-nfcorpus   # Use nfcorpus

    # Vector search (requires: pip install vajra-bm25[vector])
    vajra-search --mode vector             # Semantic vector search
    vajra-search --mode vector --model all-MiniLM-L6-v2

    # Hybrid search (BM25 + Vector fusion)
    vajra-search --mode hybrid --alpha 0.5 # 50% BM25, 50% vector
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Graceful Rich import
RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    pass

# Vajra imports
from vajra_bm25 import (
    VajraSearchOptimized,
    Document,
    DocumentCorpus,
    SearchResult,
    preprocess_text,
    create_sample_corpus,
    __version__,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SearchConfig:
    """CLI configuration."""
    dataset: str = "beir-scifact"
    corpus_path: Optional[str] = None
    corpus_format: Optional[str] = None  # 'jsonl', 'pdf', 'pdf_dir', or None for auto
    top_k: int = 10
    snippet_length: int = 200
    use_rich: bool = True
    # Search mode
    mode: str = "bm25"  # 'bm25', 'vector', or 'hybrid'
    # Vector search options
    embedding_model: str = "all-MiniLM-L6-v2"
    # Hybrid search options
    alpha: float = 0.5  # BM25 weight (1-alpha = vector weight)


# ============================================================================
# Plain Console Fallback
# ============================================================================

class PlainConsole:
    """Fallback console when Rich is not available."""

    def print(self, msg: str = "", **kwargs) -> None:
        """Print with Rich markup stripped."""
        if hasattr(msg, 'plain'):
            print(msg.plain)
        else:
            # Strip basic Rich markup
            import re
            plain = re.sub(r'\[/?[^\]]+\]', '', str(msg))
            print(plain)

    def status(self, msg: str):
        """Context manager that just prints the message."""
        import contextlib
        print(msg)
        return contextlib.nullcontext()


# ============================================================================
# Dataset Loading
# ============================================================================

def load_beir_dataset(dataset_name: str, console) -> DocumentCorpus:
    """Load a BEIR dataset with progress display.

    Args:
        dataset_name: Name like "scifact" or "nfcorpus" (without beir- prefix)
        console: Console for output

    Returns:
        DocumentCorpus ready for indexing
    """
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError:
        console.print(
            "[red]BEIR not installed.[/red]\n"
            "Run: [bold]pip install beir[/bold]\n"
            "Or: [bold]pip install vajra-bm25[cli][/bold]"
        )
        sys.exit(1)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = Path("./datasets") / dataset_name

    if not data_path.exists():
        console.print(f"[dim]Downloading {dataset_name} dataset...[/dim]")
        try:
            data_path = Path(util.download_and_unzip(url, "./datasets"))
        except Exception as e:
            console.print(f"[red]Failed to download dataset: {e}[/red]")
            sys.exit(1)

    console.print(f"[dim]Loading {dataset_name}...[/dim]")
    try:
        corpus_data, _, _ = GenericDataLoader(data_folder=str(data_path)).load(split="test")
    except Exception as e:
        console.print(f"[red]Failed to load dataset: {e}[/red]")
        sys.exit(1)

    documents = [
        Document(
            id=doc_id,
            title=doc.get("title", ""),
            content=doc.get("text", "")
        )
        for doc_id, doc in corpus_data.items()
    ]

    return DocumentCorpus(documents)


def load_custom_corpus(
    corpus_path: str,
    console,
    format: Optional[str] = None
) -> DocumentCorpus:
    """Load a custom corpus (JSONL, PDF, or directory of PDFs).

    Args:
        corpus_path: Path to file or directory
        console: Console for output
        format: Optional format hint ('jsonl', 'pdf', 'pdf_dir')
               If None, auto-detects based on path

    Returns:
        DocumentCorpus
    """
    path = Path(corpus_path)
    if not path.exists():
        console.print(f"[red]Path not found: {corpus_path}[/red]")
        sys.exit(1)

    # Determine format description for logging
    if format:
        format_desc = format
    elif path.is_dir():
        format_desc = "PDF directory"
    elif path.suffix.lower() == ".pdf":
        format_desc = "PDF"
    else:
        format_desc = "JSONL"

    console.print(f"[dim]Loading {format_desc} corpus from {corpus_path}...[/dim]")

    try:
        corpus = DocumentCorpus.load(path, format=format)
        console.print(f"[dim]Loaded {len(corpus)} documents[/dim]")
        return corpus
    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Failed to load corpus: {e}[/red]")
        sys.exit(1)


# ============================================================================
# Result Formatting
# ============================================================================

def create_snippet(content: str, query_terms: List[str], max_length: int = 200) -> str:
    """Create snippet with context around query terms.

    Args:
        content: Full document content
        query_terms: Preprocessed query terms
        max_length: Maximum snippet length

    Returns:
        Snippet string with ellipsis if truncated
    """
    if not content:
        return ""

    content_lower = content.lower()
    query_terms_lower = [t.lower() for t in query_terms if t]

    # Find best starting position (first occurrence of any query term)
    best_pos = 0
    for term in query_terms_lower:
        pos = content_lower.find(term)
        if pos != -1:
            best_pos = max(0, pos - 40)  # Start 40 chars before match
            break

    # Extract snippet
    snippet = content[best_pos:best_pos + max_length]

    # Clean up - don't break in middle of word
    if best_pos > 0:
        # Find first space and trim
        first_space = snippet.find(' ')
        if first_space > 0 and first_space < 20:
            snippet = snippet[first_space + 1:]
        snippet = "..." + snippet

    if len(content) > best_pos + max_length:
        # Find last space and trim
        last_space = snippet.rfind(' ')
        if last_space > len(snippet) - 20:
            snippet = snippet[:last_space]
        snippet = snippet + "..."

    return snippet


def highlight_snippet(snippet: str, query_terms: List[str]) -> "Text":
    """Create Rich Text with highlighted query terms.

    Args:
        snippet: Text snippet
        query_terms: Terms to highlight

    Returns:
        Rich Text object with highlighting
    """
    text = Text(snippet)
    for term in query_terms:
        if term:
            text.highlight_words([term, term.lower(), term.upper(), term.capitalize()],
                                 style="bold yellow")
    return text


# ============================================================================
# CLI Application
# ============================================================================

class VajraSearchCLI:
    """Main CLI application.

    Uses VajraSearchOptimized unified API for all search modes:
    - BM25: Traditional keyword search (always available)
    - Vector: Semantic search (lazy-loaded on first use)
    - Hybrid: Combined BM25 + vector fusion
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.console = Console() if (RICH_AVAILABLE and config.use_rich) else PlainConsole()
        self.engine: Optional[VajraSearchOptimized] = None
        self.corpus: Optional[DocumentCorpus] = None
        self.last_query: Optional[str] = None
        self.last_query_terms: List[str] = []

    # Legacy properties for backward compatibility with tests
    @property
    def bm25_engine(self):
        return self.engine

    @bm25_engine.setter
    def bm25_engine(self, value):
        self.engine = value

    @property
    def vector_engine(self):
        return self.engine._vector_engine if self.engine else None

    @property
    def hybrid_engine(self):
        return self.engine._hybrid_engine if self.engine else None

    def load_dataset(self) -> None:
        """Load dataset and build unified search engine."""
        # Load corpus
        if self.config.corpus_path:
            self.corpus = load_custom_corpus(
                self.config.corpus_path,
                self.console,
                format=self.config.corpus_format
            )
            source = Path(self.config.corpus_path).name
        else:
            dataset_name = self.config.dataset.replace("beir-", "")
            self.corpus = load_beir_dataset(dataset_name, self.console)
            source = self.config.dataset

        # Build unified search engine with embedding model for vector/hybrid modes
        self.console.print(f"[dim]Building search index for {len(self.corpus):,} documents...[/dim]")
        start_time = time.perf_counter()

        self.engine = VajraSearchOptimized(
            self.corpus,
            use_sparse=True,
            use_eager=True,
            cache_size=1000,
            embedding_model=self.config.embedding_model,
        )
        bm25_time = (time.perf_counter() - start_time) * 1000
        self.console.print(f"[green]BM25 index built in {bm25_time:.2f} ms[/green]")

        mode = self.config.mode

        # For vector/hybrid modes, initialize vector search upfront
        if mode in ("vector", "hybrid"):
            self._init_vector_search()

    def _init_vector_search(self) -> bool:
        """Initialize vector search components upfront."""
        if not self.engine:
            return False

        if not self.engine.vector_available:
            self.console.print(
                "[yellow]Vector search not available.[/yellow] "
                "Install with: pip install vajra-bm25[vector]"
            )
            return False

        self.console.print(f"[dim]Loading embedding model: {self.config.embedding_model}...[/dim]")
        start_time = time.perf_counter()

        try:
            # Trigger lazy initialization
            success = self.engine._init_vector_search()
            if success:
                vector_time = (time.perf_counter() - start_time) * 1000
                self.console.print(f"[green]Vector index built in {vector_time:.2f} ms[/green]")
                if self.config.mode == "hybrid":
                    self.console.print(
                        f"[green]Hybrid engine ready[/green] "
                        f"[dim](alpha={self.config.alpha}: {self.config.alpha*100:.0f}% BM25, "
                        f"{(1-self.config.alpha)*100:.0f}% vector)[/dim]"
                    )
                return True
            else:
                self.console.print("[red]Failed to initialize vector search[/red]")
                return False
        except Exception as e:
            self.console.print(f"[red]Vector search error: {e}[/red]")
            return False

    def search(self, query: str) -> List[SearchResult]:
        """Execute search using unified API.

        Args:
            query: Search query string

        Returns:
            List of SearchResult objects
        """
        if not self.engine:
            self.console.print("[red]Search engine not loaded[/red]")
            return []

        if not query.strip():
            self.console.print("[dim]Empty query[/dim]")
            return []

        self.last_query = query
        self.last_query_terms = preprocess_text(query)

        # Execute search using unified API with mode parameter
        start_time = time.perf_counter()
        mode = self.config.mode

        try:
            results = self.engine.search(
                query,
                top_k=self.config.top_k,
                mode=mode,
                alpha=self.config.alpha
            )
        except ValueError as e:
            self.console.print(f"[red]{e}[/red]")
            return []

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Display results
        self._display_results(results, query, latency_ms)

        return results

    def _display_results(self, results: List[SearchResult], query: str, latency_ms: float) -> None:
        """Display search results in a formatted table."""
        if not RICH_AVAILABLE or not self.config.use_rich:
            self._display_results_plain(results, query, latency_ms)
            return

        if not results:
            self.console.print(
                Panel(
                    f"[dim]No results found for:[/dim] [bold]{query}[/bold]",
                    title="Search Results",
                    border_style="yellow"
                )
            )
            return

        # Create results table
        table = Table(
            title=f"Results for: [bold cyan]{query}[/bold cyan]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title_style="bold",
            expand=True,
        )

        table.add_column("Rank", style="dim", width=5, justify="center")
        table.add_column("Score", style="yellow", width=10, justify="right")
        table.add_column("Document", style="white", ratio=1)

        for result in results:
            # Create snippet with highlighting
            content = f"{result.document.title}\n{result.document.content}" if result.document.title else result.document.content
            snippet = create_snippet(content, self.last_query_terms, self.config.snippet_length)
            snippet_text = highlight_snippet(snippet, self.last_query_terms)

            # Title line
            title_text = Text()
            title_text.append(result.document.title or f"[{result.document.id}]", style="bold")
            title_text.append("\n")
            title_text.append(snippet_text)

            table.add_row(
                str(result.rank),
                f"{result.score:.4f}",
                title_text
            )

        self.console.print(table)
        self.console.print(
            f"[dim]Found {len(results)} results in {latency_ms:.5f} ms[/dim]"
        )

    def _display_results_plain(self, results: List[SearchResult], query: str, latency_ms: float) -> None:
        """Display results in plain text format."""
        if not results:
            print(f"No results found for: {query}")
            return

        print(f"\nResults for: {query}")
        print("-" * 60)

        for result in results:
            title = result.document.title or f"[{result.document.id}]"
            snippet = create_snippet(result.document.content, self.last_query_terms, 150)
            print(f"{result.rank}. [{result.score:.4f}] {title}")
            print(f"   {snippet}")
            print()

        print(f"Found {len(results)} results in {latency_ms:.5f} ms")

    def explain(self, doc_id: str) -> None:
        """Show BM25 score breakdown for a document.

        Args:
            doc_id: Document ID to explain
        """
        if not self.bm25_engine or not self.last_query:
            self.console.print("[red]No previous query to explain (requires BM25 mode)[/red]")
            return

        try:
            explanation = self.bm25_engine.explain_result(self.last_query, doc_id)
        except Exception as e:
            self.console.print(f"[red]Could not explain: {e}[/red]")
            return

        if not RICH_AVAILABLE or not self.config.use_rich:
            print(f"\nScore breakdown for document '{doc_id}':")
            print(f"Query: {self.last_query}")
            for term, score in explanation.get('term_scores', {}).items():
                if score > 0:
                    print(f"  {term}: {score:.4f}")
            print(f"Total: {explanation.get('total_score', 0):.4f}")
            return

        table = Table(
            title=f"Score Breakdown: [cyan]{doc_id}[/cyan]",
            box=box.ROUNDED,
        )
        table.add_column("Term", style="cyan")
        table.add_column("Score", style="yellow", justify="right")

        term_scores = explanation.get('term_scores', {})
        for term, score in sorted(term_scores.items(), key=lambda x: x[1], reverse=True):
            if score > 0:
                table.add_row(term, f"{score:.4f}")

        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{explanation.get('total_score', 0):.4f}[/bold]"
        )

        self.console.print(table)
        self.console.print(f"[dim]Query: {self.last_query}[/dim]")

    def show_stats(self) -> None:
        """Display index statistics."""
        if not self.corpus:
            self.console.print("[red]No corpus loaded[/red]")
            return

        stats = {
            "Dataset": self.config.corpus_path or self.config.dataset,
            "Documents": f"{len(self.corpus):,}",
            "Search Mode": self.config.mode.upper(),
            "Top-K": str(self.config.top_k),
        }

        # Add BM25-specific stats
        if self.bm25_engine:
            stats["BM25 Terms"] = f"{len(self.bm25_engine.index.term_to_id):,}"
            stats["Avg Doc Length"] = f"{self.bm25_engine.index.avg_doc_length:.1f} tokens"

        # Add vector-specific stats
        if self.vector_engine:
            stats["Vector Index"] = self.vector_engine.index.__class__.__name__
            stats["Vectors"] = f"{self.vector_engine.num_documents:,}"

        # Add hybrid-specific stats
        if self.config.mode == "hybrid":
            stats["Alpha (BM25 weight)"] = f"{self.config.alpha}"

        if not RICH_AVAILABLE or not self.config.use_rich:
            print("\nIndex Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return

        table = Table(title="Index Statistics", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key, value)

        self.console.print(table)

    def _show_welcome(self) -> None:
        """Display welcome message."""
        corpus_size = len(self.corpus) if self.corpus else 0
        source = self.config.corpus_path or self.config.dataset
        mode = self.config.mode.upper()

        if not RICH_AVAILABLE or not self.config.use_rich:
            print(f"\nVajra Search Engine v{__version__}")
            print(f"Mode: {mode}")
            print(f"Dataset: {source} ({corpus_size:,} documents)")
            print(f"Top-K: {self.config.top_k}")
            print("\nType a query to search, or :help for commands.\n")
            return

        mode_info = f"Mode: [yellow]{mode}[/yellow]"
        if self.config.mode == "hybrid":
            mode_info += f" [dim](alpha={self.config.alpha})[/dim]"

        welcome_text = (
            f"[bold]Vajra Search Engine[/bold] v{__version__}\n\n"
            f"{mode_info}\n"
            f"Dataset: [cyan]{source}[/cyan] ({corpus_size:,} documents)\n"
            f"Top-K: [cyan]{self.config.top_k}[/cyan]\n\n"
            "Type a query to search, or [bold]:help[/bold] for commands."
        )

        panel = Panel(
            welcome_text,
            title="[bold blue]Welcome to Vajra Search[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(panel)

    def _show_help(self) -> None:
        """Display help information."""
        help_text = """
[bold]Commands:[/bold]
  [cyan]<query>[/cyan]           Search for documents
  [cyan]:help[/cyan] or [cyan]:h[/cyan]      Show this help
  [cyan]:stats[/cyan] or [cyan]:s[/cyan]     Show index statistics
  [cyan]:top N[/cyan] or [cyan]:k N[/cyan]   Set number of results (current: {top_k})
  [cyan]:explain ID[/cyan]      Score breakdown for document ID (BM25 only)
  [cyan]:e ID[/cyan]            (shorthand for :explain)
  [cyan]:clear[/cyan] or [cyan]:c[/cyan]     Clear screen
  [cyan]:quit[/cyan] or [cyan]:q[/cyan]      Exit

[bold]Current Mode:[/bold] [yellow]{mode}[/yellow]

[bold]Examples:[/bold]
  machine learning algorithms
  :top 20
  :explain doc_123
""".format(top_k=self.config.top_k, mode=self.config.mode.upper())

        if not RICH_AVAILABLE or not self.config.use_rich:
            import re
            print(re.sub(r'\[/?[^\]]+\]', '', help_text))
            return

        self.console.print(Panel(help_text, title="Help", box=box.ROUNDED))

    def _handle_command(self, cmd: str) -> bool:
        """Handle REPL command.

        Args:
            cmd: Command string starting with ':'

        Returns:
            True to continue REPL, False to exit
        """
        parts = cmd[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ('quit', 'q', 'exit'):
            return False

        elif command in ('help', 'h', '?'):
            self._show_help()

        elif command in ('stats', 's'):
            self.show_stats()

        elif command in ('clear', 'c'):
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            self._show_welcome()

        elif command in ('top', 'k'):
            try:
                new_k = int(arg)
                if new_k < 1:
                    self.console.print("[red]Top-K must be at least 1[/red]")
                else:
                    self.config.top_k = new_k
                    self.console.print(f"[green]Top-K set to {new_k}[/green]")
            except ValueError:
                self.console.print(f"[red]Invalid number: {arg}[/red]")

        elif command in ('explain', 'e'):
            if not arg:
                self.console.print("[red]Usage: :explain <doc_id>[/red]")
            else:
                self.explain(arg)

        else:
            self.console.print(f"[red]Unknown command: :{command}[/red]")
            self.console.print("[dim]Type :help for available commands[/dim]")

        return True

    def run_repl(self) -> None:
        """Run interactive REPL mode."""
        self._show_welcome()

        while True:
            try:
                # Get input - use plain input() for reliable terminal handling
                # Rich Prompt.ask() has backspace issues in some terminals
                if RICH_AVAILABLE and self.config.use_rich:
                    self.console.print()  # Newline
                    self.console.print("[bold cyan]vajra>[/bold cyan] ", end="")
                try:
                    user_input = input() if (RICH_AVAILABLE and self.config.use_rich) else input("\nvajra> ")
                except (EOFError, KeyboardInterrupt):
                    break

                user_input = user_input.strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith(':'):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Execute search
                self.search(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[dim]Use :quit to exit[/dim]")

        self.console.print("\n[dim]Goodbye![/dim]")


# ============================================================================
# Entry Point
# ============================================================================

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="vajra-search",
        description="Vajra Search Engine - BM25, Vector, and Hybrid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vajra-search                           # Interactive BM25 mode with scifact
  vajra-search -q "machine learning"     # Single query mode
  vajra-search --corpus my_docs.jsonl    # Custom JSONL corpus
  vajra-search --corpus doc.pdf          # Single PDF file
  vajra-search --corpus ./pdfs/          # Directory of PDFs
  vajra-search --dataset beir-nfcorpus   # Use nfcorpus dataset
  vajra-search --top-k 20                # Return 20 results

  # Vector search (semantic similarity)
  vajra-search --mode vector             # Uses all-MiniLM-L6-v2 by default
  vajra-search --mode vector --model sentence-transformers/all-mpnet-base-v2

  # Hybrid search (BM25 + Vector fusion)
  vajra-search --mode hybrid             # Default alpha=0.5
  vajra-search --mode hybrid --alpha 0.7 # 70% BM25, 30% vector

Supported corpus formats:
  .jsonl      JSONL file with {id, title, content} per line
  .pdf        Single PDF file (requires: pip install vajra-bm25[pdf])
  directory   Directory of PDF files

Search modes:
  bm25        Traditional keyword search (fast, exact matching)
  vector      Semantic search (requires: pip install vajra-bm25[vector])
  hybrid      Combined BM25 + vector with score fusion
"""
    )

    parser.add_argument(
        "-q", "--query",
        help="Single query (non-interactive mode)"
    )
    parser.add_argument(
        "-d", "--dataset",
        default="beir-scifact",
        choices=["beir-scifact", "beir-nfcorpus"],
        help="BEIR dataset to load (default: beir-scifact)"
    )
    parser.add_argument(
        "-c", "--corpus",
        help="Path to corpus: JSONL file, PDF file, or directory of PDFs"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["jsonl", "pdf", "pdf_dir"],
        help="Corpus format (auto-detected if not specified)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "-m", "--mode",
        choices=["bm25", "vector", "hybrid"],
        default="bm25",
        help="Search mode: bm25, vector, or hybrid (default: bm25)"
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for vector/hybrid search (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="BM25 weight for hybrid search, 0-1 (default: 0.5)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics and exit"
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable Rich formatting (plain text output)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    return parser.parse_args(args)


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Check Rich availability
    if not RICH_AVAILABLE and not args.no_rich:
        print("Note: Install 'rich' for enhanced output: pip install rich")

    # Validate alpha
    if args.alpha < 0 or args.alpha > 1:
        print("Error: --alpha must be between 0 and 1")
        return 1

    config = SearchConfig(
        dataset=args.dataset,
        corpus_path=args.corpus,
        corpus_format=args.format,
        top_k=args.top_k,
        use_rich=RICH_AVAILABLE and not args.no_rich,
        mode=args.mode,
        embedding_model=args.model,
        alpha=args.alpha,
    )

    cli = VajraSearchCLI(config)

    try:
        cli.load_dataset()

        if args.stats:
            cli.show_stats()
            return 0

        if args.query:
            # Single query mode
            cli.search(args.query)
            return 0

        # Interactive mode
        cli.run_repl()
        return 0

    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
