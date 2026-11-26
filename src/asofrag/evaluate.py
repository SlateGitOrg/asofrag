"""Evaluation: score both retrieval modes against the clause-version timeline.

Metrics (computed per mode over the 60 questions):

recall@3
    Share of the answerable questions whose gold clause-version is among the first three
    retrieved clause-versions.
out-of-force citation rate
    Share of final citations whose clause-version was not in force on the as-of date. A citation
    is scored wrong for this even when its quote is verbatim.
quote groundedness
    Share of final citations whose quote is verbatim in the cited clause-version; a citation of a
    clause-version that does not exist counts as ungrounded. This ignores time on purpose, so
    groundedness and temporal correctness are measured separately. Both validators reject
    ungrounded quotes and the extractive reader only quotes real sentences, so 100% here is
    expected by construction; the metric matters for real models.
answer accuracy
    Share of answerable questions whose answer cites the gold clause-version and contains the gold
    key value (for example "16 weeks").
abstention precision / recall / accuracy
    Abstaining on a question whose gold is ``abstain`` is a true positive. Accuracy counts correct
    abstain-or-answer decisions over all questions.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from .answer import AnswerPipeline, AnswerResult
from .dates import resolve_as_of
from .model import ExtractiveModel, FaultInjector, Model, load_faults
from .retrieve import Mode, Retriever
from .textutil import normalize_text
from .timeline import Timeline

Category = Literal["before_change", "after_change", "unchanged", "repealed", "not_yet_introduced"]


class Question(BaseModel):
    id: str
    category: Category
    question: str
    as_of: date
    gold: str
    gold_value: str | None = None

    @property
    def answerable(self) -> bool:
        return self.gold != "abstain"


def load_questions(path: str | Path) -> list[Question]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [Question.model_validate(q) for q in data["questions"]]


@dataclass
class Outcome:
    question: Question
    resolved_as_of: date
    top3: list[str]
    result: AnswerResult

    @property
    def cited(self) -> list[str]:
        return [c.key for c in self.result.citations]

    @property
    def correct(self) -> bool:
        q = self.question
        if not q.answerable:
            return self.result.abstained
        return (
            not self.result.abstained
            and q.gold in self.cited
            and q.gold_value is not None
            and q.gold_value.lower() in self.result.answer.lower()
        )


def run_mode(
    questions: list[Question], retriever: Retriever, model: Model, mode: Mode, k: int = 5
) -> list[Outcome]:
    pipeline = AnswerPipeline(retriever, model, mode, k)
    outcomes = []
    for q in questions:
        # the date comes from the question text, exactly as `asofrag ask` would read it
        when = resolve_as_of(q.question).value
        result = pipeline.ask(q.question, when, question_id=q.id)
        outcomes.append(Outcome(q, when, [h.version.key for h in result.hits[:3]], result))
    return outcomes


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score(outcomes: list[Outcome], timeline: Timeline) -> dict:
    """Aggregate metrics and the subset counts quoted in the README."""
    answerable = [o for o in outcomes if o.question.answerable]
    gold_abstain = [o for o in outcomes if not o.question.answerable]

    citations = [(o, c) for o in outcomes for c in o.result.citations]
    out_of_force = grounded = 0
    for o, c in citations:
        version = timeline.get(c.clause_id, c.version)
        if version is None:
            # a fabricated clause reference stays in the denominator and counts as ungrounded;
            # it is not "out of force" because no such clause-version exists
            continue
        out_of_force += not version.in_force(o.resolved_as_of)
        grounded += bool(normalize_text(c.quote)) and normalize_text(c.quote) in version.text

    predicted_abstain = [o for o in outcomes if o.result.abstained]
    true_abstain = [o for o in predicted_abstain if not o.question.answerable]

    def subset(category: str) -> list[Outcome]:
        return [o for o in outcomes if o.question.category == category]

    before = subset("before_change")
    unchanged = subset("unchanged")
    return {
        "metrics": {
            "recall_at_3": _ratio(
                sum(o.question.gold in o.top3 for o in answerable), len(answerable)
            ),
            "out_of_force_citation_rate": _ratio(out_of_force, len(citations)),
            "quote_groundedness": _ratio(grounded, len(citations)),
            "answer_accuracy": _ratio(sum(o.correct for o in answerable), len(answerable)),
            "abstention_precision": _ratio(len(true_abstain), len(predicted_abstain)),
            "abstention_recall": _ratio(len(true_abstain), len(gold_abstain)),
            "abstention_accuracy": _ratio(sum(o.correct for o in outcomes), len(outcomes)),
        },
        "counts": {
            "questions": len(outcomes),
            "answerable": len(answerable),
            "gold_abstain": len(gold_abstain),
            "citations": len(citations),
            "out_of_force_citations": out_of_force,
            "abstained_on_gold_abstain": len(true_abstain),
            "abstained_on_answerable": len(predicted_abstain) - len(true_abstain),
            "before_change_questions": len(before),
            "before_change_citing_newer_version": sum(
                any(
                    (v := timeline.get(c.clause_id, c.version)) is not None
                    and v.effective_from > o.resolved_as_of
                    for c in o.result.citations
                )
                for o in before
            ),
            "unchanged_questions": len(unchanged),
            "unchanged_recall_hits": sum(o.question.gold in o.top3 for o in unchanged),
        },
    }


def run_fault_plan(
    questions: list[Question], retriever: Retriever, faults_path: str | Path, k: int = 5
) -> dict:
    """Run the as-of pipeline with the fault plan and report what the validator caught."""
    faults = load_faults(faults_path)
    by_id = {q.id: q for q in questions}
    injector = FaultInjector(ExtractiveModel(), faults)
    pipeline = AnswerPipeline(retriever, injector, "asof", k)
    rows, caught = [], Counter()
    for fault in faults:
        q = by_id[fault.question]
        when = resolve_as_of(q.question).value
        result = pipeline.ask(q.question, when, question_id=q.id)
        codes = [i.code for i in result.first_attempt_issues]
        caught.update(codes)
        outcome = Outcome(q, when, [], result)
        rows.append(
            {
                "question": q.id,
                "kind": fault.kind,
                "expected_code": fault.expect,
                "first_attempt_codes": codes,
                "attempts": result.attempts,
                "answered_correctly_after_reask": outcome.correct,
            }
        )
    return {
        "injected": len(faults),
        "codes_caught": dict(sorted(caught.items())),
        "all_codes_as_expected": all(
            r["first_attempt_codes"] == [r["expected_code"]] for r in rows
        ),
        "all_correct_after_reask": all(r["answered_correctly_after_reask"] for r in rows),
        "faults": rows,
    }


def run_ablation(
    timeline: Timeline,
    questions_path: str | Path,
    faults_path: str | Path,
    model: Model | None = None,
) -> dict:
    questions = load_questions(questions_path)
    retriever = Retriever(timeline)
    model = model or ExtractiveModel()
    report: dict = {"questions": len(questions), "context_blocks": 5, "modes": {}}
    for mode in ("asof", "naive"):
        report["modes"][mode] = score(run_mode(questions, retriever, model, mode), timeline)
    report["fault_plan"] = run_fault_plan(questions, retriever, faults_path)
    return report


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_markdown(report: dict) -> str:
    modes = report["modes"]
    lines = [
        "# Ablation: as-of retrieval vs naive recency-boosted retrieval",
        "",
        f"{report['questions']} questions, {report['context_blocks']} context blocks per question.",
        "",
        "| mode | recall@3 | out-of-force citation rate | quote groundedness | answer accuracy "
        "| abstention accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for name, label in (("naive", "naive"), ("asof", "as-of")):
        m = modes[name]["metrics"]
        lines.append(
            f"| {label} | {_pct(m['recall_at_3'])} | {_pct(m['out_of_force_citation_rate'])} "
            f"| {_pct(m['quote_groundedness'])} | {_pct(m['answer_accuracy'])} "
            f"| {_pct(m['abstention_accuracy'])} |"
        )
    lines += [
        "",
        "| mode | abstention precision | abstention recall |",
        "|---|---|---|",
    ]
    for name, label in (("naive", "naive"), ("asof", "as-of")):
        m = modes[name]["metrics"]
        lines.append(
            f"| {label} | {_pct(m['abstention_precision'])} | {_pct(m['abstention_recall'])} |"
        )
    lines += ["", "| count | naive | as-of |", "|---|---|---|"]
    rows = [
        ("before-change questions citing the newer (out-of-force) version", "before_change"),
        ("abstentions on the 12 abstain questions", "abstain"),
        ("abstentions on the 48 answerable questions", "wrong_abstain"),
        ("unchanged-clause questions with gold in top 3", "unchanged"),
    ]
    for label, tag in rows:
        cells = []
        for name in ("naive", "asof"):
            c = modes[name]["counts"]
            cells.append(
                {
                    "before_change": f"{c['before_change_citing_newer_version']}"
                    f" of {c['before_change_questions']}",
                    "abstain": f"{c['abstained_on_gold_abstain']} of {c['gold_abstain']}",
                    "wrong_abstain": f"{c['abstained_on_answerable']} of {c['answerable']}",
                    "unchanged": f"{c['unchanged_recall_hits']} of {c['unchanged_questions']}",
                }[tag]
            )
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    fp = report["fault_plan"]
    lines += [
        "",
        "## Fault plan (as-of mode, first attempts only)",
        "",
        f"{fp['injected']} faults injected; codes caught on the first attempt: "
        + ", ".join(f"{k} x{v}" for k, v in fp["codes_caught"].items())
        + ".",
        "",
        "| question | fault | expected code | codes seen | attempts | correct after re-ask |",
        "|---|---|---|---|---|---|",
    ]
    for r in fp["faults"]:
        lines.append(
            f"| {r['question']} | {r['kind']} | {r['expected_code']} "
            f"| {', '.join(r['first_attempt_codes'])} | {r['attempts']} "
            f"| {'yes' if r['answered_correctly_after_reask'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, out: str | Path) -> None:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "ablation.md").write_text(render_markdown(report), encoding="utf-8")
