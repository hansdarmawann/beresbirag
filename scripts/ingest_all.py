"""CLI script to ingest all 4 Berean Standard Bible (BSB) corpora into Qdrant."""

from __future__ import annotations

from typing import Literal

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from beresbirag.config import get_settings
from beresbirag.ingestion.pipeline import IngestionPipeline, IngestionStats
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager

app = typer.Typer(
    name="ingest-all",
    help="Ingest Berean Standard Bible (BSB) USJ, Outlines PDF, Topical Index, and Concordance into Qdrant.",
    add_completion=False,
)
console = Console()


def run_ingestion(
    *,
    recreate: bool = False,
    sources: str = "scripture,book_outline,topical_index,concordance",
    books: str | None = None,
    max_topics: int | None = None,
    max_concordance: int | None = None,
    batch_size: int = 64,
    qdrant_mode: Literal["memory", "local", "remote"] | None = None,
    deterministic_embeddings: bool = False,
    show_progress: bool = True,
) -> tuple[QdrantManager, IngestionStats]:
    """Configure QdrantManager and execute IngestionPipeline."""
    base_settings = get_settings()
    settings = (
        base_settings.model_copy(update={"qdrant_mode": qdrant_mode})
        if qdrant_mode is not None
        else base_settings
    )

    if deterministic_embeddings:
        manager = QdrantManager(
            settings=settings,
            dense_embeddings=DeterministicHashEmbeddings(size=384),
            sparse_embeddings=DeterministicBM25SparseEmbeddings(),
            dense_vector_size=384,
        )
    else:
        manager = QdrantManager(settings=settings)

    pipeline = IngestionPipeline(
        qdrant_manager=manager,
        settings=settings,
        console=console,
    )
    stats = pipeline.run(
        recreate=recreate,
        sources=sources,
        books=books,
        max_topics=max_topics,
        max_concordance=max_concordance,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return manager, stats


@app.command()
def main(
    recreate: bool = typer.Option(
        False,
        "--recreate",
        help="Drop and recreate the Qdrant collection before ingesting.",
    ),
    sources: str = typer.Option(
        "scripture,book_outline,topical_index,concordance",
        "--sources",
        help="Comma-separated list of BSB sources to index.",
    ),
    books: str | None = typer.Option(
        None,
        "--books",
        help="Optional comma-separated USJ book codes (e.g., GEN,EXO,JHN,REV).",
    ),
    max_topics: int | None = typer.Option(
        None,
        "--max-topics",
        help="Optional cap on number of Topical Index entries to ingest.",
    ),
    max_concordance: int | None = typer.Option(
        None,
        "--max-concordance",
        help="Optional cap on number of Concordance headwords to ingest.",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        help="Upsert batch size for Qdrant.",
    ),
    qdrant_mode: str | None = typer.Option(
        None,
        "--qdrant-mode",
        help="Override QDRANT_MODE ('memory', 'local', or 'remote').",
    ),
    deterministic_embeddings: bool = typer.Option(
        False,
        "--deterministic-embeddings",
        help="Use fast deterministic hash/BM25 embeddings (offline smoke runs).",
    ),
) -> None:
    """Execute unified BSB multi-corpus ingestion into Qdrant."""
    mode_override: Literal["memory", "local", "remote"] | None = None
    if qdrant_mode is not None:
        normalized_mode = qdrant_mode.strip().lower()
        if normalized_mode not in {"memory", "local", "remote"}:
            raise typer.BadParameter("qdrant_mode must be 'memory', 'local', or 'remote'")
        mode_override = normalized_mode  # type: ignore[assignment]

    _, stats = run_ingestion(
        recreate=recreate,
        sources=sources,
        books=books,
        max_topics=max_topics,
        max_concordance=max_concordance,
        batch_size=batch_size,
        qdrant_mode=mode_override,
        deterministic_embeddings=deterministic_embeddings,
        show_progress=True,
    )

    table = Table(title="BSB Ingestion Summary", show_lines=True, box=box.ASCII)
    table.add_column("Source Type", style="cyan", min_width=18)
    table.add_column("Upserted Chunks", style="bold green", justify="right")

    for src_name, count in stats.counts_by_source.items():
        table.add_row(src_name, str(count))

    table.add_row("[bold]TOTAL UPSERTED[/bold]", f"[bold]{stats.total_upserted}[/bold]")
    console.print(table)
    console.print(
        Panel(
            f"[bold]Total Qdrant Collection Points:[/bold] {stats.collection_points}\n"
            f"[bold]Elapsed Time:[/bold] {stats.elapsed_seconds:.2f}s",
            title="Qdrant Index Status",
            border_style="green",
            box=box.ASCII,
        )
    )


if __name__ == "__main__":
    app()

