from asofrag.corpus import Clause, Handbook
from asofrag.timeline import build_timeline, render_timeline, word_diff
from conftest import d


def test_leave_3_2_has_two_versions_with_open_end(timeline):
    old, new = timeline.versions_of("LEAVE-3.2")
    assert (old.effective_from, old.effective_to) == (d("2024-01-01"), d("2025-06-01"))
    assert (new.effective_from, new.effective_to) == (d("2025-06-01"), None)
    assert "12 weeks" in old.text and "16 weeks" in new.text


def test_repealed_clause_final_interval_is_closed(timeline):
    (only,) = timeline.versions_of("EQUIP-8.5")
    assert only.effective_to == d("2025-01-01")
    assert timeline.is_repealed("EQUIP-8.5")
    assert not timeline.is_repealed("LEAVE-3.2")


def test_moved_clause_keeps_one_timeline(timeline):
    (only,) = timeline.versions_of("TRAVEL-6.5")
    assert only.effective_to is None
    (move,) = timeline.moves_of("TRAVEL-6.5")
    assert (move.from_section, move.to_section) == ("Travel", "Expenses")
    assert move.effective == d("2025-01-01")
    assert [m.clause_id for m in timeline.moves] == ["TRAVEL-6.5", "REMOTE-5.5"]


def test_unchanged_clause_has_single_open_version(timeline):
    (only,) = timeline.versions_of("CONDUCT-1.2")
    assert (only.effective_from, only.effective_to) == (d("2024-01-01"), None)


def test_in_force_boundary_is_inclusive_start_exclusive_end(timeline):
    old, new = timeline.versions_of("LEAVE-3.2")
    assert not new.in_force(d("2025-05-31"))
    assert new.in_force(d("2025-06-01"))
    assert old.in_force(d("2025-05-31"))
    assert not old.in_force(d("2025-06-01"))
    live = [v.key for v in timeline.in_force(d("2025-05-31")) if v.clause_id == "LEAVE-3.2"]
    assert live == ["LEAVE-3.2@2024-01-01"]


def test_timeline_recovers_every_injected_change(timeline):
    ids = timeline.clause_ids()
    amended = [c for c in ids if len(timeline.versions_of(c)) > 1]
    repealed = [c for c in ids if timeline.is_repealed(c)]
    introduced = [c for c in ids if timeline.versions_of(c)[0].effective_from > d("2024-01-01")]
    assert (len(ids), len(amended), len(repealed), len(introduced)) == (65, 14, 4, 5)
    assert len(timeline.moves) == 2
    assert len(timeline.in_force(d("2024-01-01"))) == 60
    assert len(timeline.in_force(d("2026-01-01"))) == 61


def test_amended_versions_differ_only_in_the_changed_value(timeline):
    for cid in ("LEAVE-3.2", "EXPENSE-4.1", "OVERTIME-7.1", "TRAVEL-6.2"):
        old, new = timeline.versions_of(cid)[:2]
        a, b = old.text.split(), new.text.split()
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b, strict=True)) == 1


def test_clause_that_returns_after_a_repeal_starts_a_new_version():
    def book(day, *texts):
        clauses = tuple(Clause(f"X-{i}", "t", "S", t) for i, t in enumerate(texts))
        return Handbook(d(day), clauses)

    tl = build_timeline([book("2024-01-01", "a"), book("2024-06-01"), book("2025-01-01", "a")])
    first, second = tl.versions_of("X-0")
    assert (first.effective_to, second.effective_from, second.effective_to) == (
        d("2024-06-01"),
        d("2025-01-01"),
        None,
    )
    assert tl.in_force(d("2024-08-01")) == []


def test_word_diff_marks_removed_and_added_words():
    assert word_diff("pays 12 weeks now", "pays 16 weeks now") == "pays [-12-] {+16+} weeks now"
    assert word_diff("same", "same") == "same"


def test_render_timeline_shows_dates_and_diff(timeline):
    out = render_timeline(timeline, "LEAVE-3.2")
    assert "v1  2024-01-01 -> 2025-06-01" in out
    assert "v2  2025-06-01 -> (no end)" in out
    assert "[-12-] {+16+}" in out
    assert "repealed" in render_timeline(timeline, "EQUIP-8.5")
    assert "moved: Travel -> Expenses on 2025-01-01" in render_timeline(timeline, "TRAVEL-6.5")
