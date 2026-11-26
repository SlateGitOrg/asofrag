"""Model interface and the three implementations.

A model is text in, text out: ``complete(request)`` returns the raw reply, which the answer
pipeline parses and validates. ``ExtractiveModel`` is a deterministic offline reader used by the
tests and the default demo; ``OpenAICompatModel`` talks to any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx
import yaml
from pydantic import BaseModel

from .dates import strip_dates
from .textutil import content_words, split_sentences

ABSTAIN_BELOW = 0.35

_QUESTION_LINE = re.compile(r"^Question:\s*(.*)$", re.MULTILINE)
_BLOCK = re.compile(r"^\[([^\]@\s]+)@(\d{4}-\d{2}-\d{2})\]\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class ModelRequest:
    prompt: str
    question_id: str | None = None
    attempt: int = 1


class Model(Protocol):
    def complete(self, request: ModelRequest) -> str: ...


def _overlap(question_words: set[str], text: str) -> float:
    if not question_words:
        return 0.0
    return len(question_words & set(content_words(text))) / len(question_words)


class ExtractiveModel:
    """Pick the context block that shares the most question content words, answer with its
    best-overlapping sentence, quote that sentence and cite the block. Abstain when even the best
    block covers under 35% of the question's content words."""

    def __init__(self, abstain_below: float = ABSTAIN_BELOW):
        self.abstain_below = abstain_below

    def complete(self, request: ModelRequest) -> str:
        m = _QUESTION_LINE.search(request.prompt)
        question = strip_dates(m.group(1)) if m else ""
        words = set(content_words(question))
        blocks = [(m[1], m[2], m[3]) for m in _BLOCK.finditer(request.prompt)]

        best: tuple[float, int] | None = None
        for i, (_, _, text) in enumerate(blocks):
            score = _overlap(words, text)
            if best is None or score > best[0]:
                best = (score, i)
        if best is None or best[0] < self.abstain_below:
            return json.dumps(
                {
                    "answer": "",
                    "citations": [],
                    "abstain": True,
                    "reason": "No context block covers enough of the question.",
                }
            )
        clause_id, version, text = blocks[best[1]]
        sentences = split_sentences(text) or [text]
        sentence = max(sentences, key=lambda s: _overlap(words, s))  # first wins ties
        return json.dumps(
            {
                "answer": sentence,
                "citations": [{"clause_id": clause_id, "version": version, "quote": sentence}],
                "abstain": False,
                "reason": "",
            }
        )


class Fault(BaseModel):
    """One planned failure of a model's first attempt."""

    question: str
    attempt: int = 1
    kind: Literal["fabricated_quote", "unretrieved_citation", "superseded_citation"]
    expect: str
    citation: dict[str, str]


def load_faults(path: str | Path) -> list[Fault]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Fault.model_validate(item) for item in data["faults"]]


class FaultInjector:
    """Wrap a model and replace its first citation per the fault plan (question id, attempt)."""

    def __init__(self, inner: Model, faults: Sequence[Fault]):
        self.inner = inner
        self._plan = {(f.question, f.attempt): f for f in faults}

    def complete(self, request: ModelRequest) -> str:
        raw = self.inner.complete(request)
        fault = self._plan.get((request.question_id or "", request.attempt))
        if fault is None:
            return raw
        data = json.loads(raw)
        if data.get("abstain"):
            return raw
        data["citations"] = [dict(fault.citation)]
        return json.dumps(data)


class ConfigError(RuntimeError):
    """A required environment variable is missing."""


class OpenAICompatModel:
    """Chat-completions client for any OpenAI-compatible endpoint, configured from the
    environment: ASOFRAG_LLM_BASE_URL, ASOFRAG_LLM_API_KEY and ASOFRAG_LLM_MODEL."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key, self.model = api_key, model
        self.client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, client: httpx.Client | None = None) -> OpenAICompatModel:
        names = ("ASOFRAG_LLM_BASE_URL", "ASOFRAG_LLM_API_KEY", "ASOFRAG_LLM_MODEL")
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise ConfigError("missing environment variables: " + ", ".join(missing))
        base, key, model = (os.environ[n] for n in names)
        return cls(base, key, model, client=client)

    def complete(self, request: ModelRequest) -> str:
        response = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": request.prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
