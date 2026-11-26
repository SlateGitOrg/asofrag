import json

from asofrag.answer import (
    MALFORMED,
    NO_CITATION,
    NOT_RETRIEVED,
    OUT_OF_FORCE,
    QUOTE_NOT_FOUND,
    AnswerPipeline,
    Citation,
    ModelAnswer,
    build_prompt,
    parse_response,
    validate_citations,
)
from asofrag.retrieve import Hit
from conftest import d

QUESTION = "How many weeks of paid parental leave follow the birth or adoption of a child?"
OLD, NEW = "LEAVE-3.2@2024-01-01", "LEAVE-3.2@2025-06-01"
OLD_QUOTE = (
    "Employees receive 12 weeks of paid parental leave after the birth or adoption of a child."
)


def answer_with(*citations: Citation) -> ModelAnswer:
    return ModelAnswer(answer="x", citations=list(citations))


def cite(key: str, quote: str) -> Citation:
    clause_id, version = key.split("@")
    return Citation(clause_id=clause_id, version=version, quote=quote)


def issues(timeline, retriever, citation, as_of="2025-03-12", retrieved_as_of=None, temporal=True):
    hits = retriever.retrieve(QUESTION, d(retrieved_as_of or as_of), "asof")
    return validate_citations(
        answer_with(citation),
        timeline=timeline,
        retrieved=hits,
        as_of=d(as_of),
        check_temporal=temporal,
    )


def test_a_valid_citation_has_no_issues(timeline, retriever):
    assert issues(timeline, retriever, cite(OLD, OLD_QUOTE)) == []


def test_superseded_version_is_out_of_force_even_with_a_verbatim_quote(timeline, retriever):
    new_quote = OLD_QUOTE.replace("12", "16")
    # the 2025-06-01 version exists and its quote is verbatim, but it is not in force on 2025-03-12
    (issue,) = issues(timeline, retriever, cite(NEW, new_quote), retrieved_as_of="2025-07-01")
    assert (issue.code, issue.citation) == (OUT_OF_FORCE, NEW)


def test_out_of_force_is_reported_before_not_retrieved(timeline, retriever):
    # not retrieved either: the as-of filter kept it out of the context
    (issue,) = issues(timeline, retriever, cite(NEW, OLD_QUOTE))
    assert issue.code == OUT_OF_FORCE


def test_in_force_but_not_retrieved(timeline, retriever):
    (issue,) = issues(timeline, retriever, cite("CONDUCT-1.4@2024-01-01", "anything"))
    assert issue.code == NOT_RETRIEVED


def test_unknown_clause_version_counts_as_not_retrieved(timeline, retriever):
    (issue,) = issues(timeline, retriever, cite("NOPE-9.9@2024-01-01", "x"))
    assert issue.code == NOT_RETRIEVED


def test_quote_must_be_verbatim_but_whitespace_is_normalised(timeline, retriever):
    spaced = OLD_QUOTE.replace(" paid ", "\n  paid\t")
    assert issues(timeline, retriever, cite(OLD, spaced)) == []
    (issue,) = issues(timeline, retriever, cite(OLD, OLD_QUOTE.replace("12", "13")))
    assert issue.code == QUOTE_NOT_FOUND
    (empty,) = issues(timeline, retriever, cite(OLD, "  "))
    assert empty.code == QUOTE_NOT_FOUND


def test_temporal_check_can_be_switched_off_for_the_naive_baseline(timeline, retriever):
    new_quote = OLD_QUOTE.replace("12", "16")
    got = issues(
        timeline, retriever, cite(NEW, new_quote), retrieved_as_of="2025-07-01", temporal=False
    )
    assert got == []


def test_answer_without_citations_is_flagged_unless_it_abstains(timeline, retriever):
    hits = retriever.retrieve(QUESTION, d("2025-03-12"), "asof")
    kw = {"timeline": timeline, "retrieved": hits, "as_of": d("2025-03-12")}
    assert [i.code for i in validate_citations(ModelAnswer(answer="x"), **kw)] == [NO_CITATION]
    assert validate_citations(ModelAnswer(abstain=True, reason="unsure"), **kw) == []


def test_parse_response_accepts_fenced_json_and_rejects_garbage():
    body = '{"answer": "a", "citations": [], "abstain": true, "reason": "r"}'
    assert parse_response(f"```json\n{body}\n```").reason == "r"
    for bad in ("not json", '{"citations": "nope"}', "[]"):
        try:
            parse_response(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse")


class Scripted:
    """A model that returns canned replies in order and records every prompt."""

    def __init__(self, *replies: str):
        self.replies, self.prompts = list(replies), []

    def complete(self, request):
        self.prompts.append(request.prompt)
        return self.replies.pop(0)


def reply(key: str, quote: str) -> str:
    clause_id, version = key.split("@")
    c = {"clause_id": clause_id, "version": version, "quote": quote}
    return json.dumps({"answer": quote, "citations": [c], "abstain": False, "reason": ""})


def test_one_reask_carries_the_error_codes_and_can_recover(retriever):
    model = Scripted(reply(OLD, "made up words"), reply(OLD, OLD_QUOTE))
    result = AnswerPipeline(retriever, model).ask(QUESTION, d("2025-03-12"))
    assert (result.attempts, result.abstained) == (2, False)
    assert result.citations[0].key == OLD
    assert [i.code for i in result.first_attempt_issues] == [QUOTE_NOT_FOUND]
    assert QUOTE_NOT_FOUND in model.prompts[1] and OLD in model.prompts[1]
    assert QUOTE_NOT_FOUND not in model.prompts[0]


def test_two_failures_abstain_with_the_reason(retriever):
    model = Scripted(reply(OLD, "bad one"), "{ not json")
    result = AnswerPipeline(retriever, model).ask(QUESTION, d("2025-03-12"))
    assert result.abstained and result.attempts == 2 and result.citations == []
    assert result.reason == f"Citation validation failed after one re-ask: {MALFORMED}."


def test_no_clause_in_force_abstains_without_calling_the_model(retriever):
    model = Scripted()
    result = AnswerPipeline(retriever, model).ask(QUESTION, d("2023-06-01"))
    assert result.abstained and result.attempts == 0 and model.prompts == []
    assert "2023-06-01" in result.reason


def test_prompt_labels_every_block_and_naive_prompt_omits_the_date(retriever):
    hits = retriever.retrieve(QUESTION, d("2025-03-12"), "asof")
    prompt = build_prompt(QUESTION, hits, d("2025-03-12"))
    assert f"[{OLD}] {OLD_QUOTE}" in prompt
    assert "As-of date: 2025-03-12" in prompt
    assert "As-of date" not in build_prompt(QUESTION, hits, None)
    assert isinstance(hits[0], Hit)
