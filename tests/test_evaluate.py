"""Acceptance tests: the ablation over the seeded handbook and the 60-question set."""

from collections import Counter

from asofrag.answer import AnswerResult, Citation
from asofrag.dates import resolve_as_of
from asofrag.evaluate import Outcome, run_ablation, score


def answerable(outs):
    return [o for o in outs if o.question.answerable]


def test_question_set_has_the_specified_composition(questions):
    assert len(questions) == 60
    assert Counter(q.category for q in questions) == {
        "before_change": 24,
        "after_change": 8,
        "unchanged": 16,
        "repealed": 6,
        "not_yet_introduced": 6,
    }
    assert sum(not q.answerable for q in questions) == 12
    assert all(q.gold_value for q in questions if q.answerable)


def test_every_question_carries_its_as_of_date_in_the_text(questions):
    for q in questions:
        got = resolve_as_of(q.question)
        assert (got.value, got.source) == (q.as_of, "question"), q.id


def test_gold_clause_versions_are_the_ones_in_force_on_the_as_of_date(questions, timeline):
    for q in filter(lambda q: q.answerable, questions):
        live = {v.clause_id: v.key for v in timeline.in_force(q.as_of)}
        assert live[q.gold.split("@")[0]] == q.gold, q.id


def test_asof_mode_has_no_out_of_force_citations_and_high_recall(outcomes, timeline):
    metrics = score(outcomes["asof"], timeline)
    assert metrics["metrics"]["out_of_force_citation_rate"] == 0.0
    assert metrics["counts"]["out_of_force_citations"] == 0
    assert len(answerable(outcomes["asof"])) == 48
    assert metrics["metrics"]["recall_at_3"] >= 0.95


def test_naive_mode_cites_the_newer_version_for_most_before_change_questions(outcomes, timeline):
    naive = score(outcomes["naive"], timeline)["counts"]
    assert naive["before_change_questions"] == 24
    assert naive["before_change_citing_newer_version"] >= 20
    assert score(outcomes["asof"], timeline)["counts"]["before_change_citing_newer_version"] == 0


def test_unchanged_clause_recall_is_identical_in_both_modes(outcomes, timeline):
    hits = {m: score(o, timeline)["counts"]["unchanged_recall_hits"] for m, o in outcomes.items()}
    assert hits["asof"] == hits["naive"] == 16


def test_asof_abstains_on_abstain_questions_and_rarely_on_answerable_ones(outcomes, timeline):
    counts = score(outcomes["asof"], timeline)["counts"]
    assert counts["gold_abstain"] == 12
    assert counts["abstained_on_gold_abstain"] >= 10
    assert counts["abstained_on_answerable"] <= 2


def test_naive_mode_answers_the_abstain_questions_from_repealed_or_future_clauses(
    outcomes, timeline
):
    counts = score(outcomes["naive"], timeline)["counts"]
    assert 12 - counts["abstained_on_gold_abstain"] >= 10


def test_groundedness_and_temporal_correctness_are_measured_separately(outcomes, timeline):
    naive = score(outcomes["naive"], timeline)["metrics"]
    # naive citations quote their clause verbatim, yet more than half are out of force
    assert naive["quote_groundedness"] == 1.0
    assert naive["out_of_force_citation_rate"] > 0.5
    assert (
        naive["answer_accuracy"] < score(outcomes["asof"], timeline)["metrics"]["answer_accuracy"]
    )


def test_asof_mode_answers_every_answerable_question_correctly(outcomes):
    assert all(o.correct for o in answerable(outcomes["asof"]))


def test_a_verbatim_quote_of_a_superseded_version_is_scored_wrong(timeline, questions):
    q = next(q for q in questions if q.category == "before_change")
    old = timeline.get(q.gold.split("@")[0], q.gold.split("@")[1])
    newer = timeline.versions_of(old.clause_id)[-1]
    sentence = newer.text.split(". ")[0] + "."
    result = AnswerResult(
        q.question, q.as_of, "naive", sentence,
        [Citation(clause_id=newer.clause_id, version=newer.version, quote=sentence)],
        False, "", 1,
    )  # fmt: skip
    outcome = Outcome(q, q.as_of, [], result)
    scored = score([outcome], timeline)
    assert scored["metrics"]["quote_groundedness"] == 1.0
    assert scored["metrics"]["out_of_force_citation_rate"] == 1.0
    assert not outcome.correct


def test_fault_plan_is_caught_with_the_right_codes_and_recovers(seeded, timeline):
    report = run_ablation(timeline, seeded / "evals/questions.yaml", seeded / "evals/faults.yaml")
    plan = report["fault_plan"]
    assert plan["injected"] == 7
    assert plan["codes_caught"] == {
        "CITATION_NOT_RETRIEVED": 2,
        "CITATION_OUT_OF_FORCE": 2,
        "QUOTE_NOT_FOUND": 3,
    }
    assert plan["all_codes_as_expected"] is True
    assert plan["all_correct_after_reask"] is True
    assert {f["attempts"] for f in plan["faults"]} == {2}


def test_faults_only_touch_first_attempts_and_planned_questions(seeded):
    from asofrag.model import load_faults

    faults = load_faults(seeded / "evals/faults.yaml")
    assert {f.attempt for f in faults} == {1}
    assert len({f.question for f in faults}) == 7


def test_a_fabricated_clause_reference_counts_as_ungrounded(timeline, questions):
    q = next(q for q in questions if q.category == "unchanged")
    real_id, real_version = q.gold.split("@")
    real = timeline.get(real_id, real_version)
    quote = real.text.split(". ")[0] + "."

    def outcome(citation):
        result = AnswerResult(q.question, q.as_of, "naive", quote, [citation], False, "", 1)
        return Outcome(q, q.as_of, [], result)

    good = Citation(clause_id=real_id, version=real_version, quote=quote)
    fake = Citation(clause_id="LEAVE-9.9", version="2024-01-01", quote=quote)
    metrics = score([outcome(good), outcome(fake)], timeline)["metrics"]
    assert metrics["quote_groundedness"] == 0.5
    assert metrics["out_of_force_citation_rate"] == 0.0
