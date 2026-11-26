"""Work out the as-of date of a question.

Order of precedence: an explicit date (``--as-of``), then the first ISO (``2025-03-12``) or
day-first / month-first long-form date in the question text, then a reference date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

DEFAULT_REFERENCE_DATE = date(2026, 1, 1)

_MONTHS = {
    name: i
    for i, names in enumerate(
        [
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ],
        start=1,
    )
    for name in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE
)
_MDY = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ResolvedDate:
    value: date
    source: Literal["explicit", "question", "reference"]


def _safe(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[date, tuple[int, int]]]:
    """Every valid date in ``text`` with its character span, in order of appearance."""
    found: list[tuple[date, tuple[int, int]]] = []
    for m in _ISO.finditer(text):
        d = _safe(int(m[1]), int(m[2]), int(m[3]))
        if d:
            found.append((d, m.span()))
    for m in _DMY.finditer(text):
        d = _safe(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
        if d:
            found.append((d, m.span()))
    for m in _MDY.finditer(text):
        d = _safe(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))
        if d:
            found.append((d, m.span()))
    return sorted(found, key=lambda x: x[1][0])


def strip_dates(text: str) -> str:
    """``text`` with every date expression removed, so dates do not count as content words."""
    spans = sorted(span for _, span in find_dates(text))
    out, pos = [], 0
    for start, end in spans:
        if start < pos:
            continue
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


def resolve_as_of(
    question: str,
    explicit: date | None = None,
    reference: date = DEFAULT_REFERENCE_DATE,
) -> ResolvedDate:
    if explicit is not None:
        return ResolvedDate(explicit, "explicit")
    found = find_dates(question)
    if found:
        return ResolvedDate(found[0][0], "question")
    return ResolvedDate(reference, "reference")
