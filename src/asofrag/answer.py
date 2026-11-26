"""Prompt, response schema, citation validator and the ask-validate-re-ask loop."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ValidationError

from .model import Model, ModelRequest
from .retrieve import Hit, Mode, Retriever
from .textutil import normalize_text
from .timeline import Timeline

OUT_OF_FORCE = "CITATION_OUT_OF_FORCE"
NOT_RETRIEVED = "CITATION_NOT_RETRIEVED"
QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
MALFORMED = "MALFORMED_RESPONSE"
NO_CITATION = "NO_CITATION"

SCHEMA_HINT = (
    '{"answer": str, "citations": [{"clause_id": str, "version": str, "quote": str}], '
    '"abstain": bool, "reason": str}'
)


class Citation(BaseModel):
    clause_id: str
    version: str
    quote: str

    @property
    def key(self) -> str:
        return f"{self.clause_id}@{self.version}"


class ModelAnswer(BaseModel):
    answer: str = ""
    citations: list[Citation] = []
    abstain: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Issue:
    code: str
    citation: str  # the cited "<clause_id>@<version>", empty for response-level issues

    def __str__(self) -> str:
        return f"{self.code} ({self.citation})" if self.citation else self.code


class ResponseError(ValueError):
    """The model reply is not JSON matching the answer schema."""


def parse_response(raw: str) -> ModelAnswer:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        return ModelAnswer.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ResponseError(str(exc)) from exc


def validate_citations(
    answer: ModelAnswer,
    *,
    timeline: Timeline,
    retrieved: Sequence[Hit],
    as_of: date | None,
    check_temporal: bool = True,
) -> list[Issue]:
    """Check each citation in order: out of force, not retrieved, quote not verbatim.

    A citation gets the first code that applies. Quotes are compared after whitespace
    normalisation, case-sensitively, against the cited clause-version's text.
    """
    if answer.abstain:
        return []
    if not answer.citations:
        return [Issue(NO_CITATION, "")]
    context = {h.version.key: h.version for h in retrieved}
    issues: list[Issue] = []
    for c in answer.citations:
        known = timeline.get(c.clause_id, c.version)
        if check_temporal and as_of is not None and known and not known.in_force(as_of):
            issues.append(Issue(OUT_OF_FORCE, c.key))
        elif c.key not in context:
            issues.append(Issue(NOT_RETRIEVED, c.key))
        else:
            quote = normalize_text(c.quote)
            if not quote or quote not in context[c.key].text:
                issues.append(Issue(QUOTE_NOT_FOUND, c.key))
    return issues


def build_prompt(question: str, hits: Sequence[Hit], as_of: date | None) -> str:
    lines = [
        "Answer the question using only the handbook clauses below.",
        f"Reply with JSON only: {SCHEMA_HINT}",
        "Cite only the labelled clauses and quote them verbatim. "
        "If they do not answer the question, set abstain to true and give a reason.",
    ]
    if as_of is not None:
        lines.append(f"As-of date: {as_of.isoformat()}")
    lines += ["", "Context:"]
    lines += [f"[{h.version.key}] {h.version.text}" for h in hits]
    lines += ["", f"Question: {question}"]
    return "\n".join(lines)


def reask_prompt(prompt: str, issues: Sequence[Issue]) -> str:
    codes = "; ".join(str(i) for i in issues)
    return (
        f"{prompt}\n\nYour previous reply was rejected: {codes}.\n"
        "Cite only the labelled clauses, and quote them verbatim. Reply with corrected JSON."
    )


@dataclass
class AnswerResult:
    question: str
    as_of: date
    mode: Mode
    answer: str
    citations: list[Citation]
    abstained: bool
    reason: str
    attempts: int
    first_attempt_issues: list[Issue] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


class AnswerPipeline:
    """Retrieve, ask, validate, re-ask once, otherwise abstain.

    ``asof`` mode filters retrieval by date, tells the model the date and rejects citations that
    are out of force. ``naive`` mode does none of those, so it behaves like a conventional RAG.
    """

    def __init__(self, retriever: Retriever, model: Model, mode: Mode = "asof", k: int = 5):
        self.retriever, self.model, self.mode, self.k = retriever, model, mode, k

    def ask(self, question: str, as_of: date, question_id: str | None = None) -> AnswerResult:
        temporal = self.mode == "asof"
        hits = self.retriever.retrieve(question, as_of, self.mode, self.k)

        def result(**kw) -> AnswerResult:
            return AnswerResult(question, as_of, self.mode, hits=hits, **kw)

        if not hits:
            return result(
                answer="",
                citations=[],
                abstained=True,
                reason=f"No clause was in force on {as_of.isoformat()}.",
                attempts=0,
            )

        prompt = build_prompt(question, hits, as_of if temporal else None)
        first_issues: list[Issue] = []
        issues: list[Issue] = []
        for attempt in (1, 2):
            raw = self.model.complete(ModelRequest(prompt, question_id, attempt))
            try:
                parsed = parse_response(raw)
                issues = validate_citations(
                    parsed,
                    timeline=self.retriever.timeline,
                    retrieved=hits,
                    as_of=as_of,
                    check_temporal=temporal,
                )
            except ResponseError:
                parsed, issues = None, [Issue(MALFORMED, "")]
            if attempt == 1:
                first_issues = issues
            if not issues and parsed is not None:
                return result(
                    answer=parsed.answer,
                    citations=parsed.citations,
                    abstained=parsed.abstain,
                    reason=parsed.reason,
                    attempts=attempt,
                    first_attempt_issues=first_issues,
                )
            prompt = reask_prompt(build_prompt(question, hits, as_of if temporal else None), issues)

        codes = ", ".join(sorted({i.code for i in issues}))
        return result(
            answer="",
            citations=[],
            abstained=True,
            reason=f"Citation validation failed after one re-ask: {codes}.",
            attempts=2,
            first_attempt_issues=first_issues,
        )
