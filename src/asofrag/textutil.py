"""Text normalisation and tokenisation shared by the corpus, index and reader."""

from __future__ import annotations

import re
import unicodedata

_QUOTE_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z$0-9])")

# Function words plus words that appear in nearly every handbook clause and so carry no signal.
STOPWORDS = frozenset(
    """
    a an the of to in on at for and or but is are was were be been being by with as from that
    this these those it its what which who whom how many much when where why does do did can
    could may might must shall should will would per each any every within after before than
    there their they them up into out above below more most least also not no if so such only
    then has have had about under between during without employee employees company staff
    handbook policy get gets one
    """.split()
)


def normalize_text(text: str) -> str:
    """NFKC-normalise, straighten quotes and dashes, and collapse all whitespace to one space."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP)
    return _WS.sub(" ", text).strip()


def stem(word: str) -> str:
    """A small suffix stripper: enough to equate 'days'/'day' and 'reimbursed'/'reimburses'."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 4 and word.endswith("ly"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, nothing removed."""
    return _TOKEN.findall(text.lower())


def content_words(text: str) -> list[str]:
    """Stemmed tokens with stopwords and single characters removed, in order of appearance."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Split normalised clause text into sentences. Every piece is a substring of the input."""
    return [s for s in _SENTENCE_BREAK.split(text) if s]
