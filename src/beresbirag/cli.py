"""Command-line interface for Berean Standard Bible RAG using Typer and Rich."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from beresbirag.config import Settings, get_settings
from beresbirag.diagnostics import run_diagnostics
from beresbirag.llm.factory import (
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
    _resolve_api_key,
)
from beresbirag.llm.rag_chain import BSBRAGPipeline
from beresbirag.models.schemas import RetrievalResult
from beresbirag.vectorstore.qdrant_manager import CollectionConfigError

# Configuration errors surfaced to the user with exit code 2.
CONFIG_ERRORS: tuple[type[Exception], ...] = (
    MissingAPIKeyError,
    ProviderModelMismatchError,
    ProviderUnavailableError,
    CollectionConfigError,
)
CONFIG_ERROR_EXIT_CODE = 2

app = typer.Typer(
    name="beresbirag",
    help="Berean Standard Bible (BSB) RAG CLI: Grounded Scripture Search & QA.",
    add_completion=False,
)
console = Console()

_custom_retriever: Any | None = None
_custom_pipeline_factory: Callable[..., BSBRAGPipeline] | None = None


def set_custom_retriever(retriever: Any | None) -> None:
    """Set a custom retriever instance (used for testing or dependency injection)."""
    global _custom_retriever
    _custom_retriever = retriever


def set_custom_pipeline_factory(factory: Callable[..., BSBRAGPipeline] | None) -> None:
    """Set a custom pipeline factory callable (used for testing or dependency injection)."""
    global _custom_pipeline_factory
    _custom_pipeline_factory = factory


def _get_retriever(settings: Settings) -> Any:
    if _custom_retriever is not None:
        return _custom_retriever
    try:
        from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever

        return BSBHybridRetriever(settings=settings)
    except Exception as exc:
        console.print(
            f"[bold red]Error initializing BSBHybridRetriever:[/bold red] {exc}\n"
            "[yellow]Ensure Qdrant is running or an in-memory/test retriever is provided.[/yellow]"
        )
        raise typer.Exit(code=1) from exc


def _exit_with_config_error(exc: Exception) -> typer.Exit:
    console.print(f"[bold red]Configuration error:[/bold red] [red]{escape(str(exc))}[/red]")
    return typer.Exit(code=CONFIG_ERROR_EXIT_CODE)


def _render_retrieval_table(
    results: list[RetrievalResult], title: str = "Retrieved BSB Chunks"
) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="cyan", width=4, justify="right")
    table.add_column("Reference", style="bold green", min_width=18)
    table.add_column("Source Type", style="magenta", width=14)
    table.add_column("Score", style="yellow", width=8, justify="right")
    table.add_column("Section / Heading", style="blue", min_width=20)
    table.add_column("Text Preview", style="dim", min_width=30)

    for res in results:
        meta = res.chunk.metadata
        ref = meta.canonical_ref or res.chunk.chunk_id
        src_type = (
            meta.source_type.value
            if hasattr(meta.source_type, "value")
            else str(meta.source_type)
        )
        score_val = (
            f"{res.rerank_score:.3f}"
            if res.rerank_score is not None
            else f"{res.score:.3f}"
        )
        heading = meta.section_heading or meta.outline_path or "-"
        preview = res.chunk.text.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        table.add_row(
            str(res.rank), ref, src_type, score_val, heading, preview
        )

    console.print(table)


@app.command("retrieve")
def retrieve_command(
    question: str = typer.Argument(
        ..., help="The Bible passage reference, topic, or search query to retrieve"
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", min=1, max=50, help="Number of top chunks to retrieve (default: profile)"
    ),
) -> None:
    """Retrieve ranked Berean Standard Bible chunks without invoking an LLM."""
    settings = get_settings()
    retriever = _get_retriever(settings)

    try:
        with console.status("[bold cyan]Searching Berean Standard Bible corpus..."):
            plan, results = retriever.retrieve(question, top_k=top_k if top_k is not None else settings.retrieval_top_k)
    except CollectionConfigError as exc:
        raise _exit_with_config_error(exc) from exc

    intent_label = (
        getattr(plan, "intent", None)
        if plan is not None
        else "hybrid_general"
    )
    console.print(
        Panel(
            f"[bold]Query:[/bold] {question}\n"
            f"[bold]Intent:[/bold] {intent_label}\n"
            f"[bold]Returned Chunks:[/bold] {len(results)}",
            title="BSB Retrieval Summary",
            border_style="cyan",
        )
    )

    if not results:
        console.print("[yellow]No relevant chunks found for this query.[/yellow]")
        return

    _render_retrieval_table(results)


@app.command("query")
def query_command(
    question: str = typer.Argument(
        ..., help="The Bible question or passage inquiry to answer"
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider: 'openai', 'anthropic', or 'ollama' (default: LLM_PROVIDER)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model name (e.g. gpt-4o-mini, claude-haiku-5-5; default: provider default)",
    ),
    effort: str | None = typer.Option(
        None,
        "--effort",
        "-e",
        help="Reasoning effort: 'minimal', 'low', 'medium', 'high' (default: LLM_EFFORT)",
    ),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", min=1, max=50, help="Number of context chunks to retrieve (default: profile)"
    ),
) -> None:
    """Answer questions grounded strictly in the Berean Standard Bible using LCEL."""
    settings = get_settings()
    retriever = _get_retriever(settings)

    try:
        if _custom_pipeline_factory is not None:
            pipeline = _custom_pipeline_factory(
                retriever=retriever,
                provider=provider,
                model_name=model,
                effort=effort,
            )
        else:
            pipeline = BSBRAGPipeline(
                retriever=retriever,
                settings=settings,
                provider=provider,  # type: ignore[arg-type]
                model_name=model,
                effort=effort,  # type: ignore[arg-type]
            )
            # Preserve the standard CLI's configuration-error contract without
            # constructing a model. Fast paths and retrieval remain key-free.
            if not settings.fast_path_enabled and pipeline.provider in {"openai", "anthropic"}:
                if settings.offline_mode:
                    raise ProviderUnavailableError("offline mode: use --provider ollama")
                _resolve_api_key(pipeline.provider, None, settings)

        with console.status("[bold green]Querying BSB RAG Pipeline..."):
            response = pipeline.ask(question, top_k=top_k)
    except CONFIG_ERRORS as exc:
        raise _exit_with_config_error(exc) from exc

    console.print(
        Panel(
            Markdown(response.answer),
            title=escape(
                f"BSB Answer [{response.provider} / {response.model} | effort: {response.effort} | {response.answer_mode}]"
            ),
            border_style="green",
        )
    )

    if response.citations:
        cit_table = Table(title="Citations & Grounding Sources", show_lines=True)
        cit_table.add_column("#", style="cyan", width=4, justify="right")
        cit_table.add_column("Canonical Ref", style="bold green", min_width=20)
        cit_table.add_column("Source Type", style="magenta", width=14)
        cit_table.add_column("Score", style="yellow", width=8, justify="right")
        cit_table.add_column("Chunk ID", style="dim", min_width=25)

        for idx, cit in enumerate(response.citations, start=1):
            src_val = (
                cit.source_type.value
                if hasattr(cit.source_type, "value")
                else str(cit.source_type)
            )
            cit_table.add_row(
                str(idx),
                cit.canonical_ref,
                src_val,
                f"{cit.score:.3f}",
                cit.chunk_id,
            )
        console.print(cit_table)


@app.command("doctor")
def doctor_command() -> None:
    """Check hardware, local model caches, Qdrant, and configured LLM availability."""
    checks = run_diagnostics(get_settings())
    table = Table(title="BSB diagnostics")
    for column in ("Check", "Status", "Detail"):
        table.add_column(column)
    for check in checks:
        table.add_row(check.name, check.status, escape(check.detail))
    console.print(table)
    if any(check.status == "fail" for check in checks):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
