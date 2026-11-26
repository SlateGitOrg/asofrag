import pytest

from asofrag.index import Scored
from asofrag.retrieve import apply_recency_boost
from asofrag.timeline import ClauseVersion
from conftest import d

QUESTION = "How many weeks of paid parental leave follow the birth or adoption of a child?"


def test_asof_mode_only_returns_versions_in_force(retriever):
    for when in ("2024-03-01", "2025-05-31", "2025-06-01", "2026-04-01"):
        hits = retriever.retrieve(QUESTION, d(when), "asof")
        assert hits and all(h.version.in_force(d(when)) for h in hits)
    assert retriever.retrieve(QUESTION, d("2025-05-31"), "asof")[0].version.key == (
        "LEAVE-3.2@2024-01-01"
    )
    assert retriever.retrieve(QUESTION, d("2025-06-01"), "asof")[0].version.key == (
        "LEAVE-3.2@2025-06-01"
    )


def test_naive_mode_prefers_the_newest_of_near_tied_versions(retriever):
    hits = retriever.retrieve(QUESTION, d("2025-03-12"), "naive")
    assert [h.version.key for h in hits[:2]] == ["LEAVE-3.2@2025-06-01", "LEAVE-3.2@2024-01-01"]
    assert not hits[0].version.in_force(d("2025-03-12"))


def test_no_clause_in_force_before_the_first_edition(retriever):
    assert retriever.retrieve(QUESTION, d("2023-12-31"), "asof") == []


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        retriever.retrieve(QUESTION, d("2025-03-12"), "fuzzy")  # type: ignore[arg-type]


def _v(clause: str, start: str) -> ClauseVersion:
    return ClauseVersion(clause, d(start), None, "t", "x", "S")


def test_recency_boost_ties_on_pre_fusion_scores_not_on_rank_position():
    versions = [_v("A", "2024-01-01"), _v("A", "2025-01-01"), _v("B", "2026-01-01")]
    # fused scores (the second field) are ignored; docs 0 and 1 have near-equal BM25 and cosine
    ranked = [
        Scored(0, 0.0328, bm25=4.00, cosine=0.990),
        Scored(1, 0.0323, bm25=3.95, cosine=0.985),
        Scored(2, 0.0320, bm25=3.00, cosine=0.985),  # BM25 25% below the leader: not tied
    ]
    got = [versions[s.doc].key for s in apply_recency_boost(ranked, versions)]
    assert got == ["A@2025-01-01", "A@2024-01-01", "B@2026-01-01"]


def test_adjacent_ranks_with_dissimilar_scores_are_not_tied():
    versions = [_v("A", "2024-01-01"), _v("A", "2025-01-01")]
    ranked = [Scored(0, 0.0328, 4.0, 0.99), Scored(1, 0.0323, 2.0, 0.99)]  # rank 1 and 2
    assert [s.doc for s in apply_recency_boost(ranked, versions)] == [0, 1]
