import math

import numpy as np
import pytest

from asofrag.index import BM25, ClauseIndex, HashingEmbedder, rrf

DOCS = [
    ["leave", "week", "leave"],
    ["week", "pay"],
    ["pay", "pay", "rate", "rate"],
]


def test_bm25_matches_hand_computed_scores():
    # N = 3, avgdl = 3. idf(leave) = ln(1 + 2.5/1.5) = ln(8/3); idf(pay) = ln(1 + 1.5/2.5) = ln(1.6)
    # score = idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len / 3))
    bm25 = BM25(DOCS)
    query = ["leave", "pay"]
    assert bm25.idf("leave") == pytest.approx(math.log(8 / 3), abs=1e-12)
    assert bm25.score(query, 0) == pytest.approx(math.log(8 / 3) * 2 * 2.2 / 3.2, abs=1e-9)
    assert bm25.score(query, 1) == pytest.approx(math.log(1.6) * 2.2 / 1.9, abs=1e-9)
    assert bm25.score(query, 2) == pytest.approx(math.log(1.6) * 2 * 2.2 / 3.5, abs=1e-9)
    # the same numbers, written out
    assert bm25.score(query, 0) == pytest.approx(1.3486402228911236, abs=1e-9)
    assert bm25.score(query, 1) == pytest.approx(0.5442147286003255, abs=1e-9)
    assert bm25.score(query, 2) == pytest.approx(0.5908617053374963, abs=1e-9)


def test_bm25_ignores_unknown_terms_and_repeated_query_terms():
    bm25 = BM25(DOCS)
    assert bm25.score(["zzz"], 0) == 0.0
    assert bm25.score(["leave", "leave"], 0) == bm25.score(["leave"], 0)


def test_rrf_matches_hand_computed_order():
    # k = 60. doc1 = 1/61 + 1/62, doc3 = 1/63 + 1/61, doc2 = 1/62 + 1/63
    fused = rrf([[1, 2, 3], [3, 1, 2]])
    assert [doc for doc, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62, abs=1e-12)
    assert fused[1][1] == pytest.approx(1 / 61 + 1 / 63, abs=1e-12)
    assert fused[2][1] == pytest.approx(1 / 62 + 1 / 63, abs=1e-12)


def test_rrf_keeps_documents_found_by_only_one_ranker_and_breaks_ties_by_id():
    fused = rrf([[5, 9], [7]])
    assert [doc for doc, _ in fused] == [5, 7, 9]  # 5 and 7 tie at 1/61; lower id first


def test_hashing_embedder_is_unit_norm_deterministic_and_similarity_aware():
    emb = HashingEmbedder()
    a = emb.embed("Employees receive 12 weeks of paid parental leave.")
    b = emb.embed("Employees receive 16 weeks of paid parental leave.")
    c = emb.embed("Airport parking is reimbursed at the long-stay rate.")
    assert emb.embed("Employees receive 12 weeks of paid parental leave.").tolist() == a.tolist()
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-6)
    assert float(a @ b) > 0.9 > float(a @ c)
    with pytest.raises(ValueError):
        HashingEmbedder(dim=1000)


def test_clause_index_search_respects_the_candidate_pool(timeline):
    index = ClauseIndex(timeline.versions)
    everything = index.search("paid parental leave after the birth of a child")
    pool = {i for i, v in enumerate(index.versions) if v.key == "LEAVE-3.2@2025-06-01"}
    restricted = index.search("paid parental leave after the birth of a child", pool)
    assert [index.versions[s.doc].clause_id for s in everything[:2]] == ["LEAVE-3.2"] * 2
    assert [index.versions[s.doc].key for s in restricted] == ["LEAVE-3.2@2025-06-01"]
    assert index.search("anything", set()) == []
