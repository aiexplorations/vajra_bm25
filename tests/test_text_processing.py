"""
Tests for text processing utilities.

Tests cover:
- Tokenization
- Stop word removal
- Token hashing and uniqueness
- Full preprocessing pipeline
"""

import pytest

from vajra_bm25.text_processing import (
    Token,
    tokenize,
    normalize_token,
    remove_stop_words,
    get_unique_terms,
    preprocess_text,
    STOP_WORDS,
)


class TestToken:
    """Tests for Token dataclass."""

    def test_token_creation(self):
        """Test Token creation."""
        token = Token(text="hello", position=0)

        assert token.text == "hello"
        assert token.position == 0

    def test_token_is_immutable(self):
        """Test that Token is frozen."""
        token = Token(text="hello", position=0)

        with pytest.raises(Exception):  # FrozenInstanceError
            token.text = "world"

    def test_token_hash(self):
        """Test Token hashing (based on text)."""
        t1 = Token(text="hello", position=0)
        t2 = Token(text="hello", position=5)
        t3 = Token(text="world", position=0)

        # Same text should have same hash
        assert hash(t1) == hash(t2)

        # Different text should (likely) have different hash
        assert hash(t1) != hash(t3)

    def test_token_in_set(self):
        """Test tokens in set (based on hash)."""
        t1 = Token(text="hello", position=0)
        t2 = Token(text="hello", position=5)
        t3 = Token(text="world", position=0)

        # Note: Token equality is based on all fields (frozen dataclass)
        # but hash is based on text only, so set behavior may vary
        token_set = {t1, t2, t3}

        # Should contain unique objects
        assert len(token_set) >= 2


class TestTokenization:
    """Tests for tokenization."""

    def test_basic_tokenization(self):
        """Test basic tokenization."""
        tokens = tokenize("Hello World")

        assert len(tokens) == 2
        assert tokens[0].text == "hello"  # Lowercased
        assert tokens[1].text == "world"

    def test_tokenization_with_punctuation(self):
        """Test that punctuation is removed."""
        tokens = tokenize("Hello, World! How are you?")

        texts = [t.text for t in tokens]
        assert "hello" in texts
        assert "world" in texts
        assert "how" in texts

    def test_tokenization_preserves_numbers(self):
        """Test that numbers are preserved."""
        tokens = tokenize("There are 42 items")

        texts = [t.text for t in tokens]
        assert "42" in texts

    def test_tokenization_positions(self):
        """Test that token positions are correct."""
        tokens = tokenize("one two three")

        assert tokens[0].position == 0
        assert tokens[1].position == 1
        assert tokens[2].position == 2

    def test_tokenization_empty_string(self):
        """Test tokenization of empty string."""
        tokens = tokenize("")
        assert tokens == []


class TestNormalization:
    """Tests for token normalization."""

    def test_normalize_token(self):
        """Test token normalization."""
        token = Token(text="hello", position=0)
        normalized = normalize_token(token)

        assert normalized == "hello"


class TestStopWordRemoval:
    """Tests for stop word removal."""

    def test_remove_stop_words(self):
        """Test that stop words are removed."""
        tokens = tokenize("the quick brown fox jumps over the lazy dog")
        filtered = remove_stop_words(tokens)

        texts = [t.text for t in filtered]

        assert "quick" in texts
        assert "brown" in texts
        assert "fox" in texts
        assert "the" not in texts

    def test_stop_words_list(self):
        """Test that common stop words are in the list."""
        assert "the" in STOP_WORDS
        assert "a" in STOP_WORDS
        assert "and" in STOP_WORDS
        assert "is" in STOP_WORDS

    def test_remove_stop_words_preserves_content_words(self):
        """Test that content words are preserved."""
        tokens = tokenize("category theory uses functors")
        filtered = remove_stop_words(tokens)

        texts = [t.text for t in filtered]

        assert "category" in texts
        assert "theory" in texts
        assert "functors" in texts


class TestGetUniqueTerms:
    """Tests for unique term extraction."""

    def test_get_unique_terms(self):
        """Test extracting unique terms."""
        tokens = tokenize("hello world hello")
        unique = get_unique_terms(tokens)

        assert unique == {"hello", "world"}

    def test_get_unique_terms_empty(self):
        """Test unique terms from empty list."""
        unique = get_unique_terms([])
        assert unique == set()


class TestPreprocessText:
    """Tests for full preprocessing pipeline."""

    def test_preprocess_with_stopwords(self):
        """Test preprocessing with stop word removal."""
        terms = preprocess_text("The quick brown fox")

        assert "quick" in terms
        assert "brown" in terms
        assert "fox" in terms
        assert "the" not in terms

    def test_preprocess_without_stopwords(self):
        """Test preprocessing without stop word removal."""
        terms = preprocess_text("The quick brown fox", remove_stopwords=False)

        assert "the" in terms
        assert "quick" in terms

    def test_preprocess_empty_string(self):
        """Test preprocessing empty string."""
        terms = preprocess_text("")
        assert terms == []

    def test_preprocess_only_stopwords(self):
        """Test preprocessing text with only stop words."""
        terms = preprocess_text("the and is a")
        assert terms == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
