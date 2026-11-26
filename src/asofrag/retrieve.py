"""Retrieval modes.

``asof``   keeps only clause-versions in force on the date, then ranks them.
``naive``  is the ablation: it ranks every version with no date filter, then applies a recency
           boost. Walking down the fused ranking, a hit is tied with the run's leader when both its
           BM25 score and its embedding cosine are within 2% of the leader's. Those are the
           pre-fusion similarity scores; the fused RRF value is not used because it only reflects
           rank positions. Each tied run is reordered newest first, as many production indexes do
           to prefer fresh content.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from .dates import strip_dates
from .index import ClauseIndex, Scored
from .timeline import ClauseVersion, Timeline

Mode = Literal["asof", "naive"]
RECENCY_TIE = 0.02
DEFAULT_K = 5


@dataclass(frozen=True)
class Hit:
    version: ClauseVersion
    score: float
    rank: int


def apply_recency_boost(
    ranked: list[Scored], versions: list[ClauseVersion], tie: float = RECENCY_TIE
) -> list[Scored]:
    """Split a fused ranking into runs whose BM25 and cosine are both within ``tie`` of the run
    leader's, and put each run's newest clause-version first."""
    out: list[Scored] = []
    i = 0
    while i < len(ranked):
        lead = ranked[i]
        j = i + 1
        while (
            j < len(ranked)
            and ranked[j].bm25 >= lead.bm25 * (1.0 - tie)
            and ranked[j].cosine >= lead.cosine * (1.0 - tie)
        ):
            j += 1
        out.extend(
            sorted(
                ranked[i:j],
                key=lambda h: (
                    -versions[h.doc].effective_from.toordinal(),
                    versions[h.doc].clause_id,
                ),
            )
        )
        i = j
    return out


class Retriever:
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.index = ClauseIndex(timeline.versions)
        self._position = {v.key: i for i, v in enumerate(timeline.versions)}

    def retrieve(self, question: str, as_of: date, mode: Mode, k: int = DEFAULT_K) -> list[Hit]:
        query = strip_dates(question)
        if mode == "asof":
            candidates = {self._position[v.key] for v in self.timeline.in_force(as_of)}
            scored = self.index.search(query, candidates)
            ranked = [(self.index.versions[s.doc], s.score) for s in scored]
        elif mode == "naive":
            scored = apply_recency_boost(self.index.search(query), self.index.versions)
            ranked = [(self.index.versions[s.doc], s.score) for s in scored]
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")
        return [Hit(v, s, r) for r, (v, s) in enumerate(ranked[:k], start=1)]
