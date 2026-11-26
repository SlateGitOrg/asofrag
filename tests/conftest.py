from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from asofrag.corpus import load_handbooks
from asofrag.evaluate import load_questions, run_mode
from asofrag.model import ExtractiveModel
from asofrag.retrieve import Retriever
from asofrag.seed import write_seed
from asofrag.timeline import build_timeline

REPO = Path(__file__).resolve().parents[1]


def d(s: str) -> date:
    return date.fromisoformat(s)


@pytest.fixture(scope="session")
def seeded(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("seeded")
    write_seed(root, seed=7)
    return root


@pytest.fixture(scope="session")
def timeline(seeded):
    return build_timeline(load_handbooks(seeded / "handbook"))


@pytest.fixture(scope="session")
def retriever(timeline):
    return Retriever(timeline)


@pytest.fixture(scope="session")
def questions(seeded):
    return load_questions(seeded / "evals" / "questions.yaml")


@pytest.fixture(scope="session")
def outcomes(questions, retriever):
    model = ExtractiveModel()
    return {mode: run_mode(questions, retriever, model, mode) for mode in ("asof", "naive")}
