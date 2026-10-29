"""Convert negative (👎) UI feedback entries into draft evaluation cases for curation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="feedback-to-eval",
    help="Convert negative (down/👎) feedback JSONL entries into eval/answer_cases.draft.jsonl.",
    add_completion=False,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEEDBACK_FILE = REPO_ROOT / "feedback" / "feedback.jsonl"
DEFAULT_DRAFT_FILE = REPO_ROOT / "eval" / "answer_cases.draft.jsonl"


def convert_feedback_to_draft_cases(
    feedback_file: Path = DEFAULT_FEEDBACK_FILE,
    out_file: Path = DEFAULT_DRAFT_FILE,
) -> list[dict[str, Any]]:
    """Read feedback.jsonl, select 👎 entries, and write draft AnswerCase JSONL."""
    if out_file.name == "answer_cases.jsonl":
        raise ValueError(
            "Refusing to overwrite curated answer_cases.jsonl; write to answer_cases.draft.jsonl instead."
        )

    if not feedback_file.is_file():
        return []

    draft_cases: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for line in feedback_file.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        entry = json.loads(raw)
        rating = str(entry.get("rating", "")).strip().lower()
        if rating not in {"down", "👎", "-1", "negative"}:
            continue
        query = str(entry.get("query", "")).strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        idx = len(draft_cases) + 1
        draft_cases.append(
            {
                "id": f"feedback-{idx:03d}",
                "query": query,
                "must_cite": [],
                "should_refuse": False,
                "tags": ["from-feedback"],
            }
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False) for case in draft_cases]
    out_file.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return draft_cases


@app.command()
def main(
    feedback_file: Annotated[
        Path,
        typer.Option("--feedback-file", help="Path to feedback.jsonl."),
    ] = DEFAULT_FEEDBACK_FILE,
    out: Annotated[
        Path,
        typer.Option("--out", help="Path to draft output JSONL (never answer_cases.jsonl)."),
    ] = DEFAULT_DRAFT_FILE,
) -> None:
    """Convert negative feedback entries into draft evaluation cases."""
    try:
        cases = convert_feedback_to_draft_cases(feedback_file=feedback_file, out_file=out)
        typer.echo(f"Wrote {len(cases)} draft cases to {out}")
    except Exception as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()

