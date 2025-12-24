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
        assert config.top_k == 10
        assert config.snippet_length == 200
        assert config.use_rich is True

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
        config = SearchConfig(use_rich=False)
        cli = VajraSearchCLI(config)
        cli.corpus = sample_corpus

        # Import VajraSearchOptimized and build index with sparse+eager for speed
        from vajra_bm25 import VajraSearchOptimized
        cli.engine = VajraSearchOptimized(
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
        assert cli.engine is None
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


class TestCLICommands:
    """Tests for CLI REPL commands."""

    @pytest.fixture
    def cli(self):
        """Create CLI instance for testing."""
        config = SearchConfig(use_rich=False)
        cli = VajraSearchCLI(config)
        cli.corpus = create_sample_corpus()

        from vajra_bm25 import VajraSearchOptimized
        cli.engine = VajraSearchOptimized(
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
