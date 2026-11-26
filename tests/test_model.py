import json

import httpx
import pytest

from asofrag.model import (
    ConfigError,
    ExtractiveModel,
    Fault,
    FaultInjector,
    ModelRequest,
    OpenAICompatModel,
)

PROMPT = (
    "Context:\n"
    "[GIFT-1.1@2024-01-01] Vendors may give gifts worth up to $50 a year. "
    "Anything above that is declined.\n"
    "[PARK-2.1@2024-01-01] Parking is free for visitors on weekdays.\n"
    "\nQuestion: {q}"
)


def ask(q: str) -> dict:
    return json.loads(ExtractiveModel().complete(ModelRequest(PROMPT.format(q=q))))


def test_extractive_model_quotes_the_best_sentence_of_the_best_block():
    out = ask("On 5 May 2025, what is the vendor gift limit?")
    assert out["abstain"] is False
    assert out["answer"] == "Vendors may give gifts worth up to $50 a year."
    assert out["citations"] == [
        {
            "clause_id": "GIFT-1.1",
            "version": "2024-01-01",
            "quote": "Vendors may give gifts worth up to $50 a year.",
        }
    ]


def test_extractive_model_abstains_below_a_third_of_question_words_covered():
    # content words: vendor, gift, pension  -> 2 of 3 covered (0.67): answers
    assert ask("vendor gift pension")["abstain"] is False
    # content words: vendor, pension, plan  -> 1 of 3 covered (0.333 < 0.35): abstains
    assert ask("vendor pension plan")["abstain"] is True
    # content words: vendor, pension, plan, rules, trust -> 1 of 5 (0.2): abstains
    assert ask("vendor pension plan rules trust")["abstain"] is True


def test_dates_in_the_question_do_not_dilute_the_overlap():
    # without date stripping, "12 march 2025" would add three content words and force an abstain
    assert ask("Vendor gift on 12 March 2025?")["abstain"] is False


def test_extractive_model_abstains_with_no_context():
    out = json.loads(ExtractiveModel().complete(ModelRequest("Question: vendor gift")))
    assert out["abstain"] is True and out["citations"] == []


def _fault() -> Fault:
    return Fault(
        question="Q1",
        kind="fabricated_quote",
        expect="QUOTE_NOT_FOUND",
        citation={"clause_id": "GIFT-1.1", "version": "2024-01-01", "quote": "invented"},
    )


def test_fault_injector_rewrites_only_the_planned_question_and_attempt():
    injector = FaultInjector(ExtractiveModel(), [_fault()])
    prompt = PROMPT.format(q="vendor gift limit")
    hit = json.loads(injector.complete(ModelRequest(prompt, "Q1", 1)))
    assert hit["citations"][0]["quote"] == "invented"
    assert hit["answer"] == "Vendors may give gifts worth up to $50 a year."
    for request in (ModelRequest(prompt, "Q1", 2), ModelRequest(prompt, "Q2", 1)):
        clean = json.loads(injector.complete(request))
        assert clean["citations"][0]["quote"] == "Vendors may give gifts worth up to $50 a year."


def test_openai_compatible_model_posts_chat_completions_and_returns_the_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"abstain": true}'}}]}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAICompatModel("http://llm.test/v1/", "k-123", "small-model", client=client)
    assert model.complete(ModelRequest("hello")) == '{"abstain": true}'
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["auth"] == "Bearer k-123"
    assert seen["body"]["model"] == "small-model"
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["body"]["temperature"] == 0


def test_openai_compatible_model_is_configured_from_the_environment(monkeypatch):
    for name in ("ASOFRAG_LLM_BASE_URL", "ASOFRAG_LLM_API_KEY", "ASOFRAG_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError, match="ASOFRAG_LLM_BASE_URL"):
        OpenAICompatModel.from_env()
    monkeypatch.setenv("ASOFRAG_LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("ASOFRAG_LLM_API_KEY", "k")
    monkeypatch.setenv("ASOFRAG_LLM_MODEL", "m")
    model = OpenAICompatModel.from_env()
    assert (model.url, model.model) == ("http://llm.test/v1/chat/completions", "m")
