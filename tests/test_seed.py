from pathlib import Path

from asofrag.corpus import load_handbooks
from asofrag.evaluate import run_ablation, write_reports
from asofrag.seed import write_seed
from asofrag.timeline import build_timeline
from conftest import REPO


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_same_seed_writes_byte_identical_files_and_reports(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        write_seed(root, seed=7)
        tl = build_timeline(load_handbooks(root / "handbook"))
        report = run_ablation(tl, root / "evals/questions.yaml", root / "evals/faults.yaml")
        write_reports(report, root / "reports")
    assert tree(a) == tree(b)
    assert len(tree(a)) == 5 + 2 + 2


def test_a_different_seed_changes_the_questions_but_not_the_handbook(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_seed(a, seed=7)
    write_seed(b, seed=8)
    ta, tb = tree(a), tree(b)
    assert {k: v for k, v in ta.items() if k.startswith("handbook")} == {
        k: v for k, v in tb.items() if k.startswith("handbook")
    }
    assert ta["evals/questions.yaml"] != tb["evals/questions.yaml"]


def test_each_edition_has_about_sixty_clauses_in_eight_sections(seeded):
    sizes, sections = [], set()
    for book in load_handbooks(seeded / "handbook"):
        sizes.append(len(book.clauses))
        sections |= {c.section for c in book.clauses}
        assert len({c.section for c in book.clauses}) == 8
    assert sizes == [60, 60, 60, 61, 61]
    assert sections == {
        "Conduct", "Holidays", "Leave", "Expenses", "Remote work", "Travel", "Overtime",
        "Equipment",
    }  # fmt: skip


def test_committed_sample_data_matches_a_fresh_seed(tmp_path):
    write_seed(tmp_path, seed=7)
    fresh = tree(tmp_path)
    committed = tree(REPO)
    for name, content in fresh.items():
        assert committed[name] == content, f"{name} is stale: re-run `asofrag seed`"


def test_committed_reports_match_a_fresh_evaluation(tmp_path):
    write_seed(tmp_path, seed=7)
    tl = build_timeline(load_handbooks(tmp_path / "handbook"))
    report = run_ablation(tl, tmp_path / "evals/questions.yaml", tmp_path / "evals/faults.yaml")
    write_reports(report, tmp_path / "reports")
    for name in ("ablation.md", "ablation.json"):
        assert (REPO / "reports" / name).read_bytes() == (tmp_path / "reports" / name).read_bytes()
