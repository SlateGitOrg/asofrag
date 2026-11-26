import json

from typer.testing import CliRunner

from asofrag.cli import app

runner = CliRunner()
Q = (
    "On 12 March 2025, how many weeks of paid parental leave follow "
    "the birth or adoption of a child?"
)


def test_seed_then_timeline_ask_and_eval(tmp_path):
    out = runner.invoke(app, ["seed", "--out", str(tmp_path)])
    assert out.exit_code == 0 and "60 questions" in out.output
    hb = str(tmp_path / "handbook")

    tl = runner.invoke(app, ["timeline", "LEAVE-3.2", "--handbook", hb])
    assert tl.exit_code == 0
    assert "v1  2024-01-01 -> 2025-06-01" in tl.output and "[-12-] {+16+}" in tl.output

    asked = runner.invoke(app, ["ask", Q, "--handbook", hb])
    assert asked.exit_code == 0
    assert "as-of date: 2025-03-12 (question)" in asked.output
    assert "[LEAVE-3.2@2024-01-01]" in asked.output and "12 weeks" in asked.output

    naive = runner.invoke(app, ["ask", Q, "--handbook", hb, "--mode", "naive"])
    assert "[LEAVE-3.2@2025-06-01]" in naive.output

    ev = runner.invoke(
        app,
        [
            "eval", "--handbook", hb,
            "--questions", str(tmp_path / "evals/questions.yaml"),
            "--faults", str(tmp_path / "evals/faults.yaml"),
            "--out", str(tmp_path / "reports"),
        ],
    )  # fmt: skip
    assert ev.exit_code == 0
    report = json.loads((tmp_path / "reports/ablation.json").read_text())
    assert report["modes"]["asof"]["metrics"]["out_of_force_citation_rate"] == 0.0


def test_ask_date_precedence_and_reference_date(tmp_path):
    runner.invoke(app, ["seed", "--out", str(tmp_path)])
    hb = str(tmp_path / "handbook")
    explicit = runner.invoke(app, ["ask", Q, "--handbook", hb, "--as-of", "2025-07-01"])
    assert "as-of date: 2025-07-01 (explicit)" in explicit.output
    assert "[LEAVE-3.2@2025-06-01]" in explicit.output
    bare = "How many weeks of paid parental leave follow the birth or adoption of a child?"
    ref = runner.invoke(app, ["ask", bare, "--handbook", hb, "--reference-date", "2024-02-01"])
    assert "as-of date: 2024-02-01 (reference)" in ref.output
    assert "[LEAVE-3.2@2024-01-01]" in ref.output


def test_ask_abstains_when_nothing_in_force_matches(tmp_path):
    runner.invoke(app, ["seed", "--out", str(tmp_path)])
    hb = str(tmp_path / "handbook")
    question = "Does the company pay for a gym or fitness class membership?"
    res = runner.invoke(app, ["ask", question, "--handbook", hb, "--as-of", "2024-05-01"])
    assert "answer: ABSTAINED" in res.output


def test_errors_exit_non_zero(tmp_path):
    runner.invoke(app, ["seed", "--out", str(tmp_path)])
    hb = str(tmp_path / "handbook")
    assert runner.invoke(app, ["timeline", "NOPE-1.1", "--handbook", hb]).exit_code == 1
    assert runner.invoke(app, ["timeline", "LEAVE-3.2", "--handbook", str(tmp_path)]).exit_code == 2
    assert runner.invoke(app, ["ask", "q", "--handbook", hb, "--mode", "fuzzy"]).exit_code == 2
    assert runner.invoke(app, ["ask", "q", "--handbook", hb, "--as-of", "soon"]).exit_code == 2
    assert runner.invoke(app, ["ask", "q", "--handbook", hb, "--model", "gpt"]).exit_code == 2
