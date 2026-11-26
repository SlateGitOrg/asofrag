"""Load handbook versions from ``handbook/v<YYYY-MM-DD>.md``.

Each clause is a ``###`` heading carrying a stable anchor such as ``{#LEAVE-3.2}``; the ``##``
heading above it names the section. The body runs to the next heading and is normalised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .textutil import normalize_text

_FILE = re.compile(r"^v(\d{4}-\d{2}-\d{2})\.md$")
_SECTION = re.compile(r"^##\s+(.*?)\s*$")
_CLAUSE = re.compile(r"^###\s+(.*?)\s*\{#([A-Za-z0-9][A-Za-z0-9._-]*)\}\s*$")
_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_NUMBERING = re.compile(r"^\d+[.)]?\s+")


class CorpusError(ValueError):
    """The handbook directory or one of its files is malformed."""


@dataclass(frozen=True)
class Clause:
    clause_id: str
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class Handbook:
    effective: date
    clauses: tuple[Clause, ...]


def parse_handbook(markdown: str, effective: date) -> Handbook:
    """Parse one handbook edition."""
    section = ""
    clauses: list[Clause] = []
    seen: set[str] = set()
    current: tuple[str, str, str] | None = None
    body: list[str] = []

    def close() -> None:
        nonlocal current, body
        if current is None:
            return
        clause_id, title, sec = current
        text = normalize_text(" ".join(body))
        if not text:
            raise CorpusError(f"{effective}: clause {clause_id} has no text")
        clauses.append(Clause(clause_id, title, sec, text))
        current, body = None, []

    for line in markdown.splitlines():
        if line.startswith("###"):
            close()
            m = _CLAUSE.match(line)
            if not m:
                raise CorpusError(f"{effective}: clause heading without an anchor: {line!r}")
            title, clause_id = normalize_text(m.group(1)), m.group(2)
            if clause_id in seen:
                raise CorpusError(f"{effective}: duplicate clause id {clause_id}")
            seen.add(clause_id)
            current = (clause_id, title, section)
        elif line.startswith("#"):
            close()
            m = _SECTION.match(line) if line.startswith("##") else None
            if m:
                name = _ANCHOR.sub("", m.group(1))
                section = normalize_text(_NUMBERING.sub("", name))
        elif current is not None:
            body.append(line)
    close()
    return Handbook(effective, tuple(clauses))


def load_handbooks(directory: str | Path) -> list[Handbook]:
    """Load every ``v<date>.md`` in ``directory``, oldest first."""
    root = Path(directory)
    books: list[Handbook] = []
    for path in sorted(root.glob("v*.md")):
        m = _FILE.match(path.name)
        if not m:
            raise CorpusError(f"unexpected handbook file name: {path.name}")
        effective = date.fromisoformat(m.group(1))
        books.append(parse_handbook(path.read_text(encoding="utf-8"), effective))
    if not books:
        raise CorpusError(f"no handbook versions found in {root}")
    books.sort(key=lambda h: h.effective)
    return books
