"""
Tests for the Vajra BM25 CLI application.

Tests cover:
- Argument parsing
- Snippet generation and highlighting
- CLI initialization with sample corpus
- Search functionality
"""

import pytest
from unittest.mock import patch, MagicMock

from vajra_bm25.cli import (
    parse_args,
    create_snippet,
    SearchConfig,
    VajraSearchCLI,
    PlainConsole,
)
from vajra_bm25 import create_sample_corpus, DocumentCorpus, Document


# ============================================================================
# Argument Parsing Tests
# ============================================================================

class TestParseArgs:
    """Tests for command line argument parsing."""

    def test_defaults(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.query is None
        assert args.dataset == "beir-scifact"
        assert args.corpus is None
        assert args.top_k == 10
        assert args.stats is False
        assert args.no_rich is False
        assert args.mode == "bm25"
        assert args.model == "all-MiniLM-L6-v2"
        assert args.alpha == 0.5
        assert args.format is None

    def test_query_mode(self):
        """Test single query mode."""
        args = parse_args(["-q", "machine learning"])

        assert args.query == "machine learning"

    def test_query_long_form(self):
        """Test --query long form."""
        args = parse_args(["--query", "search terms"])

        assert args.query == "search terms"

    def test_dataset_selection(self):
        """Test dataset selection."""
        args = parse_args(["-d", "beir-nfcorpus"])

        assert args.dataset == "beir-nfcorpus"

    def test_custom_corpus(self):
        """Test custom corpus path."""
        args = parse_args(["-c", "/path/to/corpus.jsonl"])

        assert args.corpus == "/path/to/corpus.jsonl"

    def test_top_k(self):
        """Test top-k setting."""
        args = parse_args(["-k", "20"])

        assert args.top_k == 20

    def test_top_k_long_form(self):
        """Test --top-k long form."""
        args = parse_args(["--top-k", "50"])

        assert args.top_k == 50

    def test_stats_flag(self):
        """Test --stats flag."""
        args = parse_args(["--stats"])

        assert args.stats is True

    def test_no_rich_flag(self):
        """Test --no-rich flag."""
        args = parse_args(["--no-rich"])

        assert args.no_rich is True

    def test_combined_args(self):
        """Test multiple arguments combined."""
        args = parse_args([
            "-q", "neural networks",
            "-d", "beir-nfcorpus",
            "-k", "5",
            "--no-rich"
        ])

        assert args.query == "neural networks"
        assert args.dataset == "beir-nfcorpus"
        assert args.top_k == 5
        assert args.no_rich is True

    def test_mode_vector(self):
        """Test vector mode selection."""
        args = parse_args(["-m", "vector"])
        assert args.mode == "vector"

    def test_mode_hybrid(self):
        """Test hybrid mode selection."""
        args = parse_args(["--mode", "hybrid"])
        assert args.mode == "hybrid"

    def test_model_flag(self):
        """Test embedding model flag."""
        args = parse_args(["--model", "sentence-transformers/all-mpnet-base-v2"])
        assert args.model == "sentence-transformers/all-mpnet-base-v2"

    def test_alpha_flag(self):
        """Test alpha weight for hybrid search."""
        args = parse_args(["--alpha", "0.7"])
        assert args.alpha == 0.7

    def test_format_jsonl(self):
        """Test format flag for JSONL."""
        args = parse_args(["-f", "jsonl"])
        assert args.format == "jsonl"

    def test_format_pdf(self):
        """Test format flag for PDF."""
        args = parse_args(["--format", "pdf"])
        assert args.format == "pdf"

    def test_format_pdf_dir(self):
        """Test format flag for PDF directory."""
        args = parse_args(["-f", "pdf_dir"])
        assert args.format == "pdf_dir"

    def test_combined_vector_args(self):
        """Test vector search arguments combined."""
        args = parse_args([
            "-m", "vector",
            "--model", "all-MiniLM-L6-v2",
            "-c", "/path/to/corpus.jsonl",
            "-k", "20"
        ])

        assert args.mode == "vector"
        assert args.model == "all-MiniLM-L6-v2"
        assert args.corpus == "/path/to/corpus.jsonl"
        assert args.top_k == 20

    def test_combined_hybrid_args(self):
        """Test hybrid search arguments combined."""
        args = parse_args([
            "-m", "hybrid",
            "--alpha", "0.3",
            "-q", "semantic search"
        ])

        assert args.mode == "hybrid"
        assert args.alpha == 0.3
        assert args.query == "semantic search"


# ============================================================================
# Snippet Generation Tests
# ============================================================================

class TestCreateSnippet:
    """Tests for snippet generation."""

    def test_basic_snippet(self):
        """Test basic snippet creation."""
        content = "The quick brown fox jumps over the lazy dog"
        snippet = create_snippet(content, ["fox"], max_length=50)

        assert "fox" in snippet.lower()
        assert len(snippet) <= 60  # Some buffer for ellipsis

    def test_snippet_with_context(self):
        """Test that snippet includes context around match."""
        content = "Introduction to machine learning algorithms and their applications in data science."
        snippet = create_snippet(content, ["machine"], max_length=80)

        assert "machine" in snippet.lower()

    def test_snippet_no_match(self):
        """Test snippet when query term not found."""
        content = "The quick brown fox jumps over the lazy dog"
        snippet = create_snippet(content, ["elephant"], max_length=50)

        # Should return start of content
        assert snippet.startswith("The") or snippet.startswith("...")

    def test_empty_content(self):
        """Test snippet with empty content."""
        snippet = create_snippet("", ["term"], max_length=50)

        assert snippet == ""

    def test_empty_query_terms(self):
        """Test snippet with empty query terms."""
        content = "Some content here"
        snippet = create_snippet(content, [], max_length=50)

        assert "Some" in snippet or "content" in snippet

    def test_snippet_truncation(self):
        """Test that long content is truncated."""
        content = "A" * 500
        snippet = create_snippet(content, ["A"], max_length=100)

        assert len(snippet) <= 110  # Max length + ellipsis buffer

    def test_snippet_with_ellipsis(self):
        """Test that ellipsis is added appropriately."""
        content = "Start " + "word " * 100 + " End"
        snippet = create_snippet(content, ["word"], max_length=50)

        # Should have ellipsis since we're truncating
        assert "..." in snippet

    def test_multiple_query_terms(self):
        """Test snippet with multiple query terms."""
        content = "Machine learning and deep learning are related fields."
        snippet = create_snippet(content, ["machine", "learning"], max_length=100)

        assert "machine" in snippet.lower() or "learning" in snippet.lower()


# ============================================================================
# SearchConfig Tests
# ============================================================================

class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SearchConfig()

        assert config.dataset == "beir-scifact"
        assert config.corpus_path is None
        assert config.corpus_format is None
        assert config.top_k == 10
        assert config.snippet_length == 200
        assert config.use_rich is True
        assert config.mode == "bm25"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.alpha == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            dataset="beir-nfcorpus",
            corpus_path="/path/to/corpus.jsonl",
            top_k=20,
            snippet_length=300,
            use_rich=False
        )

        assert config.dataset == "beir-nfcorpus"
        assert config.corpus_path == "/path/to/corpus.jsonl"
        assert config.top_k == 20
        assert config.snippet_length == 300
        assert config.use_rich is False

    def test_vector_mode_config(self):
        """Test vector search mode configuration."""
        config = SearchConfig(
            mode="vector",
            embedding_model="sentence-transformers/all-mpnet-base-v2"
        )

        assert config.mode == "vector"
        assert config.embedding_model == "sentence-transformers/all-mpnet-base-v2"

    def test_hybrid_mode_config(self):
        """Test hybrid search mode configuration."""
        config = SearchConfig(
            mode="hybrid",
            alpha=0.7
        )

        assert config.mode == "hybrid"
        assert config.alpha == 0.7

    def test_corpus_format_config(self):
        """Test corpus format configuration."""
        config = SearchConfig(
            corpus_path="/path/to/docs",
            corpus_format="pdf_dir"
        )

        assert config.corpus_path == "/path/to/docs"
        assert config.corpus_format == "pdf_dir"


# ============================================================================
# PlainConsole Tests
# ============================================================================

class TestPlainConsole:
    """Tests for PlainConsole fallback class."""

    def test_print_string(self, capsys):
        """Test printing a plain string."""
        console = PlainConsole()
        console.print("Hello world")
        captured = capsys.readouterr()
        assert "Hello world" in captured.out

    def test_print_with_rich_markup(self, capsys):
        """Test that Rich markup is stripped."""
        console = PlainConsole()
        console.print("[bold]Bold text[/bold] and [red]red text[/red]")
        captured = capsys.readouterr()
        assert "Bold text" in captured.out
        assert "red text" in captured.out
        assert "[bold]" not in captured.out
        assert "[/bold]" not in captured.out

    def test_print_empty(self, capsys):
        """Test printing empty string."""
        console = PlainConsole()
        console.print("")
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_status_context_manager(self, capsys):
        """Test status context manager."""
        console = PlainConsole()
        with console.status("Loading..."):
            pass
        captured = capsys.readouterr()
        assert "Loading..." in captured.out

    def test_print_with_nested_brackets(self, capsys):
        """Test that nested brackets are handled."""
        console = PlainConsole()
        console.print("Result: [score: 0.5]")
        captured = capsys.readouterr()
        # Should strip Rich-like markup but this is not Rich markup
        assert "Result:" in captured.out


# ============================================================================
# VajraSearchCLI Tests
# ============================================================================

class TestVajraSearchCLI:
    """Tests for VajraSearchCLI class."""

    @pytest.fixture
    def sample_corpus(self):
        """Create sample corpus for testing."""
        return create_sample_corpus()

    @pytest.fixture
    def cli_with_corpus(self, sample_corpus):
        """Create CLI instance with sample corpus loaded."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus

        # Import VajraSearchOptimized and build index with sparse+eager for speed
        from vajra_bm25 import VajraSearchOptimized
        cli.bm25_engine = VajraSearchOptimized(
            sample_corpus,
            use_sparse=True,
            use_eager=True,
            cache_size=100
        )

        return cli

    def test_cli_initialization(self):
        """Test CLI initialization."""
        config = SearchConfig(use_rich=False)
        cli = VajraSearchCLI(config)

        assert cli.config == config
        assert cli.bm25_engine is None
        assert cli.vector_engine is None
        assert cli.corpus is None
        assert cli.last_query is None

    def test_search_basic(self, cli_with_corpus):
        """Test basic search functionality."""
        results = cli_with_corpus.search("category theory")

        assert len(results) > 0
        assert all(hasattr(r, 'document') for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_search_empty_query(self, cli_with_corpus):
        """Test search with empty query."""
        results = cli_with_corpus.search("")

        assert len(results) == 0

    def test_search_whitespace_query(self, cli_with_corpus):
        """Test search with whitespace-only query."""
        results = cli_with_corpus.search("   ")

        assert len(results) == 0

    def test_search_updates_last_query(self, cli_with_corpus):
        """Test that search updates last_query."""
        cli_with_corpus.search("functors morphisms")

        assert cli_with_corpus.last_query == "functors morphisms"
        assert len(cli_with_corpus.last_query_terms) > 0

    def test_search_respects_top_k(self, cli_with_corpus):
        """Test that search respects top_k config."""
        cli_with_corpus.config.top_k = 3
        results = cli_with_corpus.search("category")

        assert len(results) <= 3

    def test_show_stats(self, cli_with_corpus, capsys):
        """Test stats display."""
        cli_with_corpus.show_stats()
        captured = capsys.readouterr()

        assert "Documents" in captured.out
        assert "10" in captured.out  # Sample corpus has 10 docs

    def test_explain_without_query(self, cli_with_corpus, capsys):
        """Test explain without previous query."""
        cli_with_corpus.explain("doc1")
        captured = capsys.readouterr()

        assert "No previous query" in captured.out

    def test_explain_with_query(self, cli_with_corpus, capsys):
        """Test explain after a search."""
        cli_with_corpus.search("category functors")
        cli_with_corpus.explain("doc1")
        captured = capsys.readouterr()

        # Should show some score breakdown
        assert "doc1" in captured.out or "category" in captured.out.lower()

    def test_search_no_engine(self, capsys):
        """Test search without engine loaded."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = create_sample_corpus()
        # Don't set engine

        results = cli.search("test query")
        captured = capsys.readouterr()

        assert len(results) == 0
        assert "Search engine not loaded" in captured.out

    def test_display_results_plain_empty(self, capsys):
        """Test plain text display with no results."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = create_sample_corpus()
        cli.last_query_terms = ["test"]

        cli._display_results_plain([], "no matches", 1.0)
        captured = capsys.readouterr()

        assert "No results found for: no matches" in captured.out

    def test_display_results_plain_with_results(self, sample_corpus, capsys):
        """Test plain text display with results."""
        from vajra_bm25 import SearchResult

        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus
        cli.last_query_terms = ["category"]

        # Create mock results
        doc = sample_corpus.documents[0]
        results = [SearchResult(document=doc, score=5.0, rank=1)]

        cli._display_results_plain(results, "category", 1.23)
        captured = capsys.readouterr()

        assert "Results for: category" in captured.out
        assert "5.0000" in captured.out
        assert "1.23" in captured.out

    def test_show_stats_no_corpus(self, capsys):
        """Test show_stats when no corpus is loaded."""
        config = SearchConfig(use_rich=False)
        cli = VajraSearchCLI(config)

        cli.show_stats()
        captured = capsys.readouterr()

        assert "No corpus loaded" in captured.out

    def test_show_stats_with_engine(self, sample_corpus, capsys):
        """Test show_stats when engine is loaded."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus

        # Set up engine properly
        from vajra_bm25 import VajraSearchOptimized
        cli.engine = VajraSearchOptimized(
            sample_corpus,
            use_sparse=True,
            use_eager=True,
            cache_size=100
        )

        cli.show_stats()
        captured = capsys.readouterr()

        assert "Documents" in captured.out
        assert "10" in captured.out

    def test_show_welcome_plain(self, sample_corpus, capsys):
        """Test welcome message in plain mode."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus

        cli._show_welcome()
        captured = capsys.readouterr()

        assert "Vajra Search Engine" in captured.out
        assert "BM25" in captured.out

    def test_show_welcome_hybrid_mode(self, sample_corpus, capsys):
        """Test welcome message shows hybrid alpha."""
        config = SearchConfig(use_rich=False, mode="hybrid", alpha=0.7)
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus

        cli._show_welcome()
        captured = capsys.readouterr()

        assert "HYBRID" in captured.out

    def test_show_help_plain(self, capsys):
        """Test help message in plain mode."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)

        cli._show_help()
        captured = capsys.readouterr()

        assert "Commands" in captured.out
        assert ":help" in captured.out
        assert ":quit" in captured.out


class TestCLICommands:
    """Tests for CLI REPL commands."""

    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        config = SearchConfig(use_rich=False, mode="bm25")
        cli = VajraSearchCLI(config)
        cli.corpus = create_sample_corpus()

        from vajra_bm25 import VajraSearchOptimized
        cli.bm25_engine = VajraSearchOptimized(
            cli.corpus,
            use_sparse=True,
            use_eager=True,
            cache_size=100
        )

        return cli

    def test_handle_quit(self, cli):
        """Test :quit command."""
        result = cli._handle_command(":quit")
        assert result is False

    def test_handle_quit_short(self, cli):
        """Test :q command."""
        result = cli._handle_command(":q")
        assert result is False

    def test_handle_help(self, cli, capsys):
        """Test :help command."""
        result = cli._handle_command(":help")

        assert result is True
        captured = capsys.readouterr()
        assert "Commands" in captured.out

    def test_handle_help_short(self, cli, capsys):
        """Test :h command."""
        result = cli._handle_command(":h")

        assert result is True
        captured = capsys.readouterr()
        assert "Commands" in captured.out

    def test_handle_stats(self, cli, capsys):
        """Test :stats command."""
        result = cli._handle_command(":stats")

        assert result is True
        captured = capsys.readouterr()
        assert "Documents" in captured.out

    def test_handle_top_k(self, cli, capsys):
        """Test :top N command."""
        result = cli._handle_command(":top 25")

        assert result is True
        assert cli.config.top_k == 25

        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_handle_top_k_invalid(self, cli, capsys):
        """Test :top with invalid number."""
        original_k = cli.config.top_k
        result = cli._handle_command(":top abc")

        assert result is True
        assert cli.config.top_k == original_k  # Unchanged

        captured = capsys.readouterr()
        assert "Invalid" in captured.out

    def test_handle_unknown_command(self, cli, capsys):
        """Test unknown command."""
        result = cli._handle_command(":foobar")

        assert result is True
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_handle_exit(self, cli):
        """Test :exit command."""
        result = cli._handle_command(":exit")
        assert result is False

    def test_handle_stats_short(self, cli, capsys):
        """Test :s command."""
        result = cli._handle_command(":s")

        assert result is True
        captured = capsys.readouterr()
        assert "Documents" in captured.out

    def test_handle_k_command(self, cli, capsys):
        """Test :k N command as alias for :top N."""
        result = cli._handle_command(":k 15")

        assert result is True
        assert cli.config.top_k == 15

    def test_handle_question_mark_help(self, cli, capsys):
        """Test :? command for help."""
        result = cli._handle_command(":?")

        assert result is True
        captured = capsys.readouterr()
        assert "Commands" in captured.out

    def test_handle_top_negative(self, cli, capsys):
        """Test :top with negative number."""
        original_k = cli.config.top_k
        result = cli._handle_command(":top -5")

        assert result is True
        captured = capsys.readouterr()
        assert "must be at least 1" in captured.out
        assert cli.config.top_k == original_k  # Unchanged

    def test_handle_top_zero(self, cli, capsys):
        """Test :top with zero."""
        original_k = cli.config.top_k
        result = cli._handle_command(":top 0")

        assert result is True
        captured = capsys.readouterr()
        assert "must be at least 1" in captured.out
        assert cli.config.top_k == original_k  # Unchanged

    def test_handle_explain_no_arg(self, cli, capsys):
        """Test :explain without document ID."""
        result = cli._handle_command(":explain")

        assert result is True
        captured = capsys.readouterr()
        assert "Usage: :explain" in captured.out

    def test_handle_explain_short(self, cli, capsys):
        """Test :e shorthand for explain."""
        result = cli._handle_command(":e")

        assert result is True
        captured = capsys.readouterr()
        assert "Usage: :explain" in captured.out

    def test_handle_explain_with_query(self, cli, capsys):
        """Test :explain after a search."""
        cli.search("category functors")
        result = cli._handle_command(":e doc1")

        assert result is True
        captured = capsys.readouterr()
        # Should show score breakdown
        assert "doc1" in captured.out or "category" in captured.out.lower()


class TestCLIWithMockedBEIR:
    """Tests that mock BEIR loading."""

    def test_load_beir_not_installed(self):
        """Test error message when BEIR not installed."""
        config = SearchConfig(use_rich=False)
        cli = VajraSearchCLI(config)

        # Mock the import to fail
        with patch.dict('sys.modules', {'beir': None}):
            with pytest.raises(SystemExit):
                from vajra_bm25.cli import load_beir_dataset
                from vajra_bm25.cli import PlainConsole
                load_beir_dataset("scifact", PlainConsole())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
