"""Command line: ``asofrag seed | timeline | ask | eval``."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from .answer import AnswerPipeline
from .corpus import CorpusError, load_handbooks
from .dates import DEFAULT_REFERENCE_DATE, resolve_as_of
from .evaluate import run_ablation, write_reports
from .model import ConfigError, ExtractiveModel, Model, OpenAICompatModel
from .retrieve import Mode, Retriever
from .seed import DEFAULT_SEED, write_seed
from .timeline import Timeline, build_timeline, render_timeline

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console(highlight=False)

HandbookDir = typer.Option("handbook", "--handbook", help="Directory of v<date>.md editions.")


def _timeline(directory: Path) -> Timeline:
    try:
        return build_timeline(load_handbooks(directory))
    except CorpusError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _model(name: str) -> Model:
    if name == "extractive":
        return ExtractiveModel()
    if name == "openai":
        try:
            return OpenAICompatModel.from_env()
        except ConfigError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from exc
    typer.echo(f"error: unknown model {name!r} (use extractive or openai)", err=True)
    raise typer.Exit(2)


@app.command()
def seed(
    out: Path = typer.Option(Path("."), help="Directory to write handbook/ and evals/ into."),
    seed: int = typer.Option(DEFAULT_SEED, help="Seed for question dates and phrasing."),
) -> None:
    """Write five synthetic handbook editions, 60 questions and the fault plan."""
    counts = write_seed(out, seed)
    typer.echo(
        f"wrote {counts['editions']} handbook editions, {counts['questions']} questions "
        f"and {counts['faults']} planned faults to {out}"
    )


@app.command()
def timeline(
    clause_id: str = typer.Argument(..., help="Clause anchor, for example LEAVE-3.2."),
    handbook: Path = HandbookDir,
) -> None:
    """Print a clause's versions with their dates and a word-level diff."""
    tl = _timeline(handbook)
    try:
        typer.echo(render_timeline(tl, clause_id))
    except KeyError as exc:
        typer.echo(f"error: no clause {clause_id} in {handbook}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question, optionally containing a date."),
    as_of: str = typer.Option(
        None, "--as-of", help="ISO date; overrides any date in the question."
    ),
    reference_date: str = typer.Option(
        DEFAULT_REFERENCE_DATE.isoformat(),
        "--reference-date",
        envvar="ASOFRAG_REFERENCE_DATE",
        help="Used when neither --as-of nor the question gives a date.",
    ),
    mode: str = typer.Option("asof", help="asof or naive."),
    model: str = typer.Option("extractive", help="extractive (offline) or openai."),
    handbook: Path = HandbookDir,
) -> None:
    """Answer a question as of a date, with validated citations."""
    if mode not in ("asof", "naive"):
        typer.echo("error: --mode must be asof or naive", err=True)
        raise typer.Exit(2)
    try:
        explicit = date.fromisoformat(as_of) if as_of else None
        reference = date.fromisoformat(reference_date)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    resolved = resolve_as_of(question, explicit, reference)
    pipeline = AnswerPipeline(Retriever(_timeline(handbook)), _model(model), cast(Mode, mode))
    result = pipeline.ask(question, resolved.value)

    typer.echo(f"as-of date: {resolved.value.isoformat()} ({resolved.source})")
    typer.echo(f"mode: {mode}")
    if result.abstained:
        typer.echo("answer: ABSTAINED")
        typer.echo(f"reason: {result.reason}")
    else:
        typer.echo(f"answer: {result.answer}")
        for c in result.citations:
            typer.echo(f'citation: [{c.key}] "{c.quote}"')
    typer.echo(f"attempts: {result.attempts}")


@app.command("eval")
def evaluate_cmd(
    handbook: Path = HandbookDir,
    questions: Path = typer.Option(Path("evals/questions.yaml"), help="Question set."),
    faults: Path = typer.Option(Path("evals/faults.yaml"), help="Fault plan."),
    out: Path = typer.Option(Path("reports"), help="Where to write ablation.md and .json."),
) -> None:
    """Run both retrieval modes over the question set and write the ablation report."""
    report = run_ablation(_timeline(handbook), questions, faults)
    write_reports(report, out)
    table = Table(title="asofrag ablation")
    for col in ("mode", "recall@3", "out-of-force", "groundedness", "accuracy", "abstention"):
        table.add_column(col)
    for name in ("naive", "asof"):
        m = report["modes"][name]["metrics"]
        table.add_row(
            name,
            *(
                f"{m[k]:.3f}"
                for k in (
                    "recall_at_3",
                    "out_of_force_citation_rate",
                    "quote_groundedness",
                    "answer_accuracy",
                    "abstention_accuracy",
                )
            ),
        )
    console.print(table)
    typer.echo(f"wrote {out / 'ablation.md'} and {out / 'ablation.json'}")
