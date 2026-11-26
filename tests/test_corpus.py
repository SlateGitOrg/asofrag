import pytest

from asofrag.corpus import CorpusError, load_handbooks, parse_handbook
from asofrag.textutil import normalize_text
from conftest import d

SAMPLE = """# Employee Handbook

Preface text that belongs to no clause.

## 3. Leave

### Parental leave {#LEAVE-3.2}

Employees receive 12 weeks
of paid   leave.

### Sick leave {#LEAVE-3.3}

Ten days.

## 4. Expenses {#S-EXP}

### Per diem {#EXPENSE-4.1}

The per diem is “$55”.
"""


def test_parse_sections_anchors_and_normalised_text():
    book = parse_handbook(SAMPLE, d("2024-01-01"))
    assert [c.clause_id for c in book.clauses] == ["LEAVE-3.2", "LEAVE-3.3", "EXPENSE-4.1"]
    first = book.clauses[0]
    assert (first.title, first.section) == ("Parental leave", "Leave")
    assert first.text == "Employees receive 12 weeks of paid leave."
    # numbering and the {#...} anchor are stripped from section names
    assert book.clauses[2].section == "Expenses"
    # curly quotes become straight so a model's straight-quoted citation still matches
    assert book.clauses[2].text == 'The per diem is "$55".'


def test_normalize_text_collapses_whitespace_and_dashes():
    assert normalize_text("a — b \n c") == "a - b c"


@pytest.mark.parametrize(
    "markdown, message",
    [
        ("## S\n\n### No anchor here\n\ntext\n", "without an anchor"),
        ("## S\n\n### A {#X-1}\n\ntext\n\n### B {#X-1}\n\ntext\n", "duplicate clause id"),
        ("## S\n\n### A {#X-1}\n\n### B {#X-2}\n\ntext\n", "has no text"),
    ],
)
def test_malformed_handbooks_are_rejected(markdown, message):
    with pytest.raises(CorpusError, match=message):
        parse_handbook(markdown, d("2024-01-01"))


def test_load_handbooks_orders_by_date_and_rejects_empty_dirs(tmp_path):
    with pytest.raises(CorpusError, match="no handbook versions"):
        load_handbooks(tmp_path)
    for name in ("v2025-01-01.md", "v2024-01-01.md"):
        (tmp_path / name).write_text("## S\n\n### A {#X-1}\n\ntext\n", encoding="utf-8")
    assert [b.effective for b in load_handbooks(tmp_path)] == [d("2024-01-01"), d("2025-01-01")]
