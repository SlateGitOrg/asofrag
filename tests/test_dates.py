from asofrag.dates import DEFAULT_REFERENCE_DATE, find_dates, resolve_as_of, strip_dates
from conftest import d


def test_long_form_date_in_question_resolves():
    got = resolve_as_of("How many weeks of leave did I get on 12 March 2025?")
    assert (got.value, got.source) == (d("2025-03-12"), "question")


def test_other_date_styles_resolve():
    assert resolve_as_of("Rule on 2025-03-12?").value == d("2025-03-12")
    assert resolve_as_of("Rule on March 12, 2025?").value == d("2025-03-12")
    assert resolve_as_of("Rule on the 3rd of Sept 2024?").value == d("2024-09-03")


def test_explicit_as_of_overrides_the_question_date():
    got = resolve_as_of("What applied on 12 March 2025?", explicit=d("2024-01-15"))
    assert (got.value, got.source) == (d("2024-01-15"), "explicit")


def test_question_without_a_date_uses_the_reference_date():
    got = resolve_as_of("What is the per diem?")
    assert (got.value, got.source) == (DEFAULT_REFERENCE_DATE, "reference")
    assert resolve_as_of("What is the per diem?", reference=d("2025-07-04")).value == d(
        "2025-07-04"
    )


def test_impossible_dates_are_skipped_and_the_first_valid_date_wins():
    q = "Between 31 February 2025 and 2025-03-01 and 2025-04-01, what applied?"
    assert resolve_as_of(q).value == d("2025-03-01")
    assert [x for x, _ in find_dates(q)] == [d("2025-03-01"), d("2025-04-01")]


def test_strip_dates_removes_only_the_date_text():
    assert strip_dates("On 12 March 2025, what applied?") == "On , what applied?"
    assert strip_dates("No date here") == "No date here"
