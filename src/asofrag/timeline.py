"""Clause-version timeline: diff consecutive handbook editions clause by clause.

A clause whose normalised text changes starts a new version (``effective_from`` is the edition
date) and closes the previous one (``effective_to`` is the same date, exclusive). A clause that
is missing from a later edition is repealed and its last version is closed the same way. A clause
that only moves to another section keeps its single timeline; the move is recorded separately.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace
from datetime import date

from .corpus import Handbook


@dataclass(frozen=True)
class ClauseVersion:
    clause_id: str
    effective_from: date
    effective_to: date | None
    title: str
    text: str
    section: str

    @property
    def version(self) -> str:
        return self.effective_from.isoformat()

    @property
    def key(self) -> str:
        return f"{self.clause_id}@{self.version}"

    def in_force(self, as_of: date) -> bool:
        """Effective on ``effective_from`` itself, out of force on ``effective_to``."""
        if as_of < self.effective_from:
            return False
        return self.effective_to is None or as_of < self.effective_to


@dataclass(frozen=True)
class Move:
    clause_id: str
    effective: date
    from_section: str
    to_section: str


class Timeline:
    def __init__(self, versions: list[ClauseVersion], moves: list[Move]):
        self.versions = sorted(versions, key=lambda v: (v.clause_id, v.effective_from))
        self.moves = moves
        self._by_key = {v.key: v for v in self.versions}

    def clause_ids(self) -> list[str]:
        return sorted({v.clause_id for v in self.versions})

    def versions_of(self, clause_id: str) -> list[ClauseVersion]:
        return [v for v in self.versions if v.clause_id == clause_id]

    def moves_of(self, clause_id: str) -> list[Move]:
        return [m for m in self.moves if m.clause_id == clause_id]

    def get(self, clause_id: str, version: str) -> ClauseVersion | None:
        return self._by_key.get(f"{clause_id}@{version}")

    def in_force(self, as_of: date) -> list[ClauseVersion]:
        """Every clause-version in force on ``as_of``."""
        return [v for v in self.versions if v.in_force(as_of)]

    def is_repealed(self, clause_id: str) -> bool:
        versions = self.versions_of(clause_id)
        return bool(versions) and versions[-1].effective_to is not None


def build_timeline(handbooks: list[Handbook]) -> Timeline:
    """Diff the editions, oldest first, into clause-versions."""
    closed: list[ClauseVersion] = []
    moves: list[Move] = []
    open_versions: dict[str, ClauseVersion] = {}
    current_section: dict[str, str] = {}

    for book in sorted(handbooks, key=lambda h: h.effective):
        present = {c.clause_id: c for c in book.clauses}
        for clause in book.clauses:
            live = open_versions.get(clause.clause_id)
            if live is not None and live.text == clause.text:
                if current_section[clause.clause_id] != clause.section:
                    moves.append(
                        Move(
                            clause.clause_id,
                            book.effective,
                            current_section[clause.clause_id],
                            clause.section,
                        )
                    )
                    current_section[clause.clause_id] = clause.section
                continue
            if live is not None:
                closed.append(replace(live, effective_to=book.effective))
            open_versions[clause.clause_id] = ClauseVersion(
                clause.clause_id,
                book.effective,
                None,
                clause.title,
                clause.text,
                clause.section,
            )
            current_section[clause.clause_id] = clause.section
        for clause_id in [c for c in open_versions if c not in present]:
            closed.append(replace(open_versions.pop(clause_id), effective_to=book.effective))
            current_section.pop(clause_id)

    return Timeline(closed + list(open_versions.values()), moves)


def word_diff(old: str, new: str) -> str:
    """Word-level diff in git's style: ``[-removed-]`` and ``{+added+}``."""
    a, b = old.split(), new.split()
    out: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            out.extend(a[i1:i2])
            continue
        if i2 > i1:
            out.append("[-" + " ".join(a[i1:i2]) + "-]")
        if j2 > j1:
            out.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(out)


def render_timeline(timeline: Timeline, clause_id: str) -> str:
    """The text printed by ``asofrag timeline <clause>``."""
    versions = timeline.versions_of(clause_id)
    if not versions:
        raise KeyError(clause_id)
    lines = [f"{clause_id}  {versions[-1].title}"]
    for n, v in enumerate(versions, start=1):
        end = v.effective_to.isoformat() if v.effective_to else "(no end)"
        lines.append(f"  v{n}  {v.effective_from.isoformat()} -> {end}")
        if n == 1:
            lines.append(f"      {v.text}")
        else:
            lines.append(f"      diff vs v{n - 1}: {word_diff(versions[n - 2].text, v.text)}")
    if timeline.is_repealed(clause_id):
        lines.append(f"  repealed: no longer in the handbook from {versions[-1].effective_to}")
    for m in timeline.moves_of(clause_id):
        lines.append(f"  moved: {m.from_section} -> {m.to_section} on {m.effective.isoformat()}")
    return "\n".join(lines)
