"""Lexical and dense scoring over clause-versions, fused with reciprocal rank fusion.

* ``BM25`` is written from scratch (k1 = 1.2, b = 0.75, Lucene-style non-negative idf).
* ``HashingEmbedder`` maps character 3-5-grams into 2**14 signed buckets and l2-normalises.
* ``rrf`` fuses rankings with ``sum(1 / (k + rank))``, k = 60.

Collection statistics (idf, average length) are computed once over every clause-version, so a
clause-version scores the same whether or not other versions are filtered out around it.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from .textutil import content_words, normalize_text
from .timeline import ClauseVersion

BM25_K1 = 1.2
BM25_B = 0.75
EMBED_DIM = 2**14
NGRAM_RANGE = (3, 5)
RRF_K = 60


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.lengths = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = sum(self.lengths) / self.n if self.n else 0.0
        self.df: Counter[str] = Counter()
        for counts in self.tf:
            self.df.update(counts.keys())

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def score(self, query: Sequence[str], doc: int) -> float:
        counts, length = self.tf[doc], self.lengths[doc]
        total = 0.0
        for term in dict.fromkeys(query):
            f = counts.get(term, 0)
            if f == 0:
                continue
            norm = f + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
            total += self.idf(term) * f * (self.k1 + 1.0) / norm
        return total


class HashingEmbedder:
    """Signed feature hashing of character n-grams. No vocabulary, no training."""

    def __init__(self, dim: int = EMBED_DIM, ngrams: tuple[int, int] = NGRAM_RANGE):
        if dim & (dim - 1):
            raise ValueError("dim must be a power of two")
        self.dim, self.ngrams = dim, ngrams

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        s = f" {normalize_text(text).lower()} "
        lo, hi = self.ngrams
        for n in range(lo, hi + 1):
            for i in range(len(s) - n + 1):
                h = int.from_bytes(
                    hashlib.blake2b(s[i : i + n].encode("utf-8"), digest_size=8).digest(), "big"
                )
                vec[h & (self.dim - 1)] += 1.0 if (h >> 40) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion. Ties break on the lower document id, so output is deterministic."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Scored:
    """A fused hit with the two pre-fusion similarity scores it was built from."""

    doc: int
    score: float  # RRF value: depends only on rank positions
    bm25: float
    cosine: float


class ClauseIndex:
    """BM25 plus hashing embeddings over a fixed list of clause-versions."""

    def __init__(self, versions: Sequence[ClauseVersion]):
        self.versions = list(versions)
        texts = [f"{v.title}. {v.text}" for v in self.versions]
        self.bm25 = BM25([content_words(t) for t in texts])
        self.embedder = HashingEmbedder()
        self.matrix = (
            np.stack([self.embedder.embed(t) for t in texts])
            if texts
            else np.zeros((0, EMBED_DIM), dtype=np.float32)
        )

    def search(self, query: str, candidates: Collection[int] | None = None) -> list[Scored]:
        """Fused ranking of ``candidates`` (every document when omitted), best first."""
        pool = sorted(candidates) if candidates is not None else list(range(len(self.versions)))
        if not pool:
            return []
        q_tokens = content_words(query)
        lexical = [(d, self.bm25.score(q_tokens, d)) for d in pool]
        lexical_rank = [d for d, s in sorted(lexical, key=lambda x: (-x[1], x[0])) if s > 0.0]
        sims = self.matrix[pool] @ self.embedder.embed(query)
        dense = sorted(zip(pool, sims.tolist(), strict=True), key=lambda x: (-x[1], x[0]))
        dense_rank = [d for d, s in dense if s > 0.0]
        bm25 = dict(lexical)
        cosine = dict(zip(pool, sims.tolist(), strict=True))
        return [Scored(d, s, bm25[d], cosine[d]) for d, s in rrf([lexical_rank, dense_rank])]
