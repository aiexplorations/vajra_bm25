"""
Text processing utilities

Tokenization and normalization are morphisms:
- Text -> List[Token]
- Token -> NormalizedToken

These morphisms preserve information while transforming representation.
"""

import re
from typing import List, Set
from dataclasses import dataclass

from vajra_bm25.logging_config import get_logger

# Initialize logger for this module
logger = get_logger("text_processing")


# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'they', 'them', 'their',
    'have', 'but', 'not', 'or', 'can', 'been', 'which', 'than', 'when'
}


@dataclass(frozen=True)
class Token:
    """
    A token (word) extracted from text.

    Immutable - tokens are objects in a category.
    """
    text: str
    position: int  # Position in original text

    def __hash__(self):
        return hash(self.text)


_WORD_PATTERN = re.compile(r'\b[a-z0-9]+\b')


def tokenize(text: str) -> List[Token]:
    """
    Tokenize text into words.

    Morphism: Text -> List[Token]

    Preserves word boundaries and order.
    """
    # Convert to lowercase
    text = text.lower()

    # Split on non-alphanumeric characters
    words = re.findall(r'\b[a-z0-9]+\b', text)

    # Create tokens with positions
    tokens = [Token(text=word, position=i) for i, word in enumerate(words)]

    return tokens


def normalize_token(token: Token) -> str:
    """
    Normalize a token.

    Morphism: Token -> NormalizedString

    Could include stemming, lemmatization, etc.
    For now, just lowercase (already done in tokenization).
    """
    return token.text


def remove_stop_words(tokens: List[Token]) -> List[Token]:
    """
    Filter out stop words.

    Morphism: List[Token] -> List[Token]

    Preserves structure while removing noise.
    """
    return [token for token in tokens if token.text not in STOP_WORDS]


def get_unique_terms(tokens: List[Token]) -> Set[str]:
    """
    Extract unique terms from tokens.

    Morphism: List[Token] -> Set[Term]
    """
    return {normalize_token(token) for token in tokens}


def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Complete preprocessing pipeline.

    Composition of morphisms:
    Text -> List[Token] -> List[Token] -> List[String]

    This is categorical composition!
    """
    # Fast path: same result as tokenize -> remove_stop_words -> normalize_token, without
    # allocating a Token object per word (the dominant cost of index builds).
    words = _WORD_PATTERN.findall(text.lower())
    if remove_stopwords:
        return [w for w in words if w not in STOP_WORDS]
    return words


# Hyphenated alphanumeric identifiers such as V-110, P-400A/B or DNV-ST-F101. The optional
# "/X" covers paired equipment (P-400A/B).
_TAG_PATTERN = re.compile(r'\b[a-z0-9]+(?:-[a-z0-9]+)+(?:/[a-z])?\b')


def tag_aware_analyzer(text: str) -> List[str]:
    """
    Analyzer that also indexes equipment tags as single terms.

    Morphism: Text -> List[Term]

    Returns the default terms from preprocess_text (so V-110 still yields "v" and
    "110"), followed by the joined form of every tag: "v110". Paired tags expand
    to both members: P-400A/B yields "p400a" and "p400b". A query for "V-110" or
    "V110" then matches a rare, specific term instead of two common ones.

    Pass it as VajraSearchOptimized(corpus, analyzer=tag_aware_analyzer). The same
    analyzer is applied to documents and queries.
    """
    terms = preprocess_text(text)

    for match in _TAG_PATTERN.finditer(text.lower()):
        tag = match.group()
        # Needs a letter and a digit: skips dates (2026-11-19) and plain hyphenated words
        if not (any(ch.isdigit() for ch in tag) and any(ch.isalpha() for ch in tag)):
            continue

        base, _, pair = tag.partition('/')
        joined = base.replace('-', '')
        terms.append(joined)

        # P-400A/B -> p400a and p400b: the letter after the slash replaces the last letter
        if pair and joined[-1].isalpha():
            terms.append(joined[:-1] + pair)

    return terms


if __name__ == "__main__":
    # Test preprocessing
    sample_text = """Category theory is a general theory of mathematical structures.
    Functors are structure-preserving maps between categories."""

    logger.info("Original text:")
    logger.info(sample_text)

    logger.info("Tokens:")
    tokens = tokenize(sample_text)
    logger.info([t.text for t in tokens])

    logger.info("After removing stop words:")
    filtered = remove_stop_words(tokens)
    logger.info([t.text for t in filtered])

    logger.info("Preprocessed terms:")
    terms = preprocess_text(sample_text)
    logger.info(terms)

    logger.info("Unique terms:")
    logger.info(get_unique_terms(tokenize(sample_text)))
