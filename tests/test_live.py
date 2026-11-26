import os

import pytest

from asofrag.answer import AnswerPipeline
from asofrag.model import OpenAICompatModel
from conftest import d

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("ASOFRAG_LIVE") != "1", reason="set ASOFRAG_LIVE=1 to run"),
]


def test_live_model_answers_with_a_valid_in_force_citation(retriever):
    pipeline = AnswerPipeline(retriever, OpenAICompatModel.from_env())
    question = "How many weeks of paid parental leave follow the birth or adoption of a child?"
    result = pipeline.ask(question, d("2025-03-12"))
    assert result.abstained or all(c.key == "LEAVE-3.2@2024-01-01" for c in result.citations)
