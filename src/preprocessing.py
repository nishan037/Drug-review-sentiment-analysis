"""Text and label preprocessing shared by training and inference."""

from __future__ import annotations

import html
import re
from numbers import Real

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LABEL_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}
NEGATIONS = {"no", "nor", "not", "never", "neither", "cannot"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS.difference(NEGATIONS))
_STEMMER = PorterStemmer()
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]+")


def validate_text(text: object) -> str:
    """Return stripped text or raise a clear error for invalid input."""
    if not isinstance(text, str):
        raise TypeError("Review text must be a string.")
    text = text.strip()
    if not text:
        raise ValueError("Review text cannot be empty.")
    return text


def clean_text(text: object) -> str:
    """Lowercase, remove markup/punctuation, filter stop words, and stem.

    This follows the academic notebook's ML cleaning approach while retaining
    common negations because they materially change sentiment.
    """
    value = html.unescape(validate_text(text)).lower()
    value = _TAG_RE.sub(" ", value)
    tokens = _NON_ALPHA_RE.sub(" ", value).split()
    return " ".join(_STEMMER.stem(token) for token in tokens if token not in STOP_WORDS)


def rating_to_label(rating: object) -> int:
    """Map a 1–10 rating to Negative (0), Neutral (1), or Positive (2)."""
    if isinstance(rating, bool) or not isinstance(rating, Real):
        raise TypeError("Rating must be a number between 1 and 10.")
    value = float(rating)
    if not 1 <= value <= 10:
        raise ValueError("Rating must be between 1 and 10.")
    if value <= 4:
        return 0
    if value <= 6:
        return 1
    return 2
