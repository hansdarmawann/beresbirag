"""Multi-corpus retrieval benchmark suite (Recall@1, Recall@5, MRR) for BSB RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from beresbirag.config import get_settings
from beresbirag.ingestion.pipeline import IngestionPipeline
from beresbirag.models.schemas import RetrievalResult, SourceType
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager

app = typer.Typer(
    name="evaluate-retrieval",
    help="Run the 15-query Berean Standard Bible (BSB) multi-corpus retrieval benchmark.",
    add_completion=False,
)
console = Console()


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """Single ground-truth retrieval benchmark query and relevance criteria."""

    case_id: str
    category: str
    query: str
    expected_source: SourceType
    expected_verse_ref: str | None = None
    expected_book_code: str | None = None
    expected_topic: str | None = None
    expected_concordance_word: str | None = None
    expected_text_substring: str | None = None

    def matches(self, result: RetrievalResult) -> bool:
        """Return True if a retrieved chunk satisfies all ground-truth criteria."""
        meta = result.chunk.metadata
        if meta.source_type != self.expected_source:
            return False

        if self.expected_verse_ref is not None:
            verse_refs_upper = {r.upper() for r in meta.verse_refs}
            if self.expected_verse_ref.upper() not in verse_refs_upper:
                return False

        if self.expected_book_code is not None:
            if (meta.book_code or "").upper() != self.expected_book_code.upper():
                return False

        if self.expected_topic is not None:
            if (meta.topic or "").strip().lower() != self.expected_topic.strip().lower():
                return False

        if self.expected_concordance_word is not None:
            if (
                meta.concordance_word or ""
            ).strip().lower() != self.expected_concordance_word.strip().lower():
                return False

        if self.expected_text_substring is not None:
            if self.expected_text_substring.lower() not in result.chunk.text.lower():
                return False

        return True


BENCHMARK_QUERIES: tuple[BenchmarkCase, ...] = (
    # 1-5: Exact Scripture passage queries
    BenchmarkCase(
        case_id="Q01",
        category="Exact Scripture",
        query="Genesis 1:1-3",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="GEN 1:1",
        expected_text_substring="In the beginning God created the heavens and the earth",
    ),
    BenchmarkCase(
        case_id="Q02",
        category="Exact Scripture",
        query="John 1:1",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="JHN 1:1",
        expected_text_substring="In the beginning was the Word",
    ),
    BenchmarkCase(
        case_id="Q03",
        category="Exact Scripture",
        query="Revelation 13:18",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="REV 13:18",
        expected_text_substring="666",
    ),
    BenchmarkCase(
        case_id="Q04",
        category="Exact Scripture",
        query="John 3:16",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="JHN 3:16",
        expected_text_substring="For God so loved the world",
    ),
    BenchmarkCase(
        case_id="Q05",
        category="Exact Scripture",
        query="Revelation 21:1-4",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="REV 21:1",
        expected_text_substring="new heaven and a new earth",
    ),
    # 6-8: Footnote & Cross-reference queries
    BenchmarkCase(
        case_id="Q06",
        category="Footnote & Cross-Ref",
        query="Genesis 1:3",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="GEN 1:3",
        expected_text_substring="Cited in 2 Corinthians 4:6",
    ),
    BenchmarkCase(
        case_id="Q07",
        category="Footnote & Cross-Ref",
        query="Genesis 1:5",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="GEN 1:5",
        expected_text_substring="Literally day one",
    ),
    BenchmarkCase(
        case_id="Q08",
        category="Footnote & Cross-Ref",
        query="What cross-references are listed in Genesis 1:1?",
        expected_source=SourceType.SCRIPTURE,
        expected_verse_ref="GEN 1:1",
        expected_text_substring="John 1:1",
    ),
    # 9-11: Book Outline queries
    BenchmarkCase(
        case_id="Q09",
        category="Book Outline",
        query="Genesis outline",
        expected_source=SourceType.BOOK_OUTLINE,
        expected_book_code="GEN",
        expected_text_substring="God and the World",
    ),
    BenchmarkCase(
        case_id="Q10",
        category="Book Outline",
        query="Exodus main divisions outline",
        expected_source=SourceType.BOOK_OUTLINE,
        expected_book_code="EXO",
        expected_text_substring="Exodus",
    ),
    BenchmarkCase(
        case_id="Q11",
        category="Book Outline",
        query="John book outline and structure",
        expected_source=SourceType.BOOK_OUTLINE,
        expected_book_code="JHN",
        expected_text_substring="John",
    ),
    # 12-13: Topical Index queries
    BenchmarkCase(
        case_id="Q12",
        category="Topical Index",
        query="Topic: 666",
        expected_source=SourceType.TOPICAL_INDEX,
        expected_topic="666",
        expected_text_substring="Revelation 13:16",
    ),
    BenchmarkCase(
        case_id="Q13",
        category="Topical Index",
        query="Topical index for 10 Commandments",
        expected_source=SourceType.TOPICAL_INDEX,
        expected_topic="10 commandments",
    ),
    # 14-15: Concordance queries
    BenchmarkCase(
        case_id="Q14",
        category="Concordance",
        query="Occurrences of 12",
        expected_source=SourceType.CONCORDANCE,
        expected_concordance_word="12",
        expected_text_substring="26",
    ),
    BenchmarkCase(
        case_id="Q15",
        category="Concordance",
        query="Occurrences of 10",
        expected_source=SourceType.CONCORDANCE,
        expected_concordance_word="10",
    ),
)


@dataclass(slots=True)
class CaseEvaluationResult:
    """Evaluation outcome for a single BenchmarkCase."""

    case: BenchmarkCase
    hit_rank: int | None
    top_chunk_id: str | None
    top_canonical_ref: str | None

    @property
    def hit_at_1(self) -> bool:
        return self.hit_rank == 1

    @property
    def hit_at_5(self) -> bool:
        return self.hit_rank is not None and self.hit_rank <= 5

    @property
    def reciprocal_rank(self) -> float:
        if self.hit_rank is None or self.hit_rank <= 0:
            return 0.0
        return 1.0 / float(self.hit_rank)


@dataclass(slots=True)
class BenchmarkReport:
    """Aggregate benchmark metrics across all evaluated cases."""

    results: list[CaseEvaluationResult] = field(default_factory=list)
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0


def evaluate_benchmark(
    retriever: BSBHybridRetriever,
    cases: tuple[BenchmarkCase, ...] | list[BenchmarkCase] = BENCHMARK_QUERIES,
    top_k: int = 5,
) -> BenchmarkReport:
    """Run all benchmark queries against `retriever` and compute Recall@1, Recall@5, and MRR."""
    eval_results: list[CaseEvaluationResult] = []
    effective_k = max(top_k, 5)

    for case in cases:
        _plan, retrieved = retriever.retrieve(case.query, top_k=effective_k)
        hit_rank: int | None = None
        for idx, item in enumerate(retrieved, start=1):
            if case.matches(item):
                hit_rank = idx
                break

        top_chunk_id = retrieved[0].chunk.chunk_id if retrieved else None
        top_ref = (
            (retrieved[0].chunk.metadata.canonical_ref or top_chunk_id)
            if retrieved
            else None
        )
        eval_results.append(
            CaseEvaluationResult(
                case=case,
                hit_rank=hit_rank,
                top_chunk_id=top_chunk_id,
                top_canonical_ref=top_ref,
            )
        )

    total = len(eval_results) or 1
    recall_at_1 = sum(1 for r in eval_results if r.hit_at_1) / total
    recall_at_5 = sum(1 for r in eval_results if r.hit_at_5) / total
    mrr = sum(r.reciprocal_rank for r in eval_results) / total

    return BenchmarkReport(
        results=eval_results,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        mrr=mrr,
    )


def build_in_memory_benchmark_retriever(
    *,
    books: str = "GEN,JHN,REV",
    max_topics: int = 15,
    max_concordance: int = 15,
    deterministic_embeddings: bool = True,
) -> BSBHybridRetriever:
    """Index a multi-source slice of the real BSB files into in-memory Qdrant and return a retriever."""
    settings = get_settings().model_copy(
        update={
            "qdrant_mode": "memory",
            "qdrant_collection_name": "bsb_eval_benchmark",
            "reranker_enabled": False,
        }
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
    pipeline.run(
        recreate=True,
        books=books,
        max_topics=max_topics,
        max_concordance=max_concordance,
        show_progress=False,
    )
    reranker = PassageReranker(settings=settings, enabled=False)
    return BSBHybridRetriever(
        qdrant_manager=manager,
        reranker=reranker,
        settings=settings,
    )


def _ascii_safe(text: str) -> str:
    return (
        text.replace("\u2013", "-")
        .replace("\u2014", "--")
        .encode("ascii", errors="replace")
        .decode("ascii")
    )


def render_benchmark_report(report: BenchmarkReport) -> None:
    """Print the per-query benchmark table and summary metrics using Rich."""
    table = Table(
        title="BSB 15-Query Multi-Corpus Retrieval Benchmark",
        show_lines=True,
        box=box.ASCII,
    )
    table.add_column("ID", style="cyan", width=5)
    table.add_column("Category", style="magenta", min_width=16)
    table.add_column("Query", style="bold white", min_width=26)
    table.add_column("Rank", style="yellow", justify="right", width=6)
    table.add_column("Hit@1", justify="center", width=7)
    table.add_column("Hit@5", justify="center", width=7)
    table.add_column("Top Retrieved Reference", style="green", min_width=24)

    for res in report.results:
        rank_str = str(res.hit_rank) if res.hit_rank is not None else "MISS"
        h1 = "[bold green]YES[/bold green]" if res.hit_at_1 else "[red]NO[/red]"
        h5 = "[bold green]YES[/bold green]" if res.hit_at_5 else "[red]NO[/red]"
        table.add_row(
            res.case.case_id,
            res.case.category,
            _ascii_safe(res.case.query),
            rank_str,
            h1,
            h5,
            _ascii_safe(res.top_canonical_ref or "-"),
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Queries Evaluated:[/bold] {len(report.results)}\n"
            f"[bold cyan]Recall@1:[/bold cyan] {report.recall_at_1:.2%}\n"
            f"[bold green]Recall@5:[/bold green] {report.recall_at_5:.2%}\n"
            f"[bold yellow]MRR (Mean Reciprocal Rank):[/bold yellow] {report.mrr:.4f}",
            title="Retrieval Benchmark Summary",
            border_style="green" if report.recall_at_5 >= 0.90 else "yellow",
            box=box.ASCII,
        )
    )


@app.command()
def main(
    in_memory_slice: bool = typer.Option(
        True,
        "--in-memory-slice/--use-existing-collection",
        help="Build an in-memory multi-source BSB slice automatically (default) or query existing Qdrant.",
    ),
    books: str = typer.Option(
        "GEN,JHN,REV",
        "--books",
        help="Comma-separated USJ book codes to index when --in-memory-slice is active.",
    ),
    max_topics: int = typer.Option(
        15,
        "--max-topics",
        help="Number of Topical Index topics to index when --in-memory-slice is active.",
    ),
    max_concordance: int = typer.Option(
        15,
        "--max-concordance",
        help="Number of Concordance entries to index when --in-memory-slice is active.",
    ),
    qdrant_mode: str | None = typer.Option(
        None,
        "--qdrant-mode",
        help="Optional QDRANT_MODE override when using an existing collection.",
    ),
    min_recall_at_5: float = typer.Option(
        0.90,
        "--min-recall-at-5",
        help="Minimum Recall@5 threshold required to exit with code 0.",
    ),
) -> None:
    """Execute the 15-query BSB retrieval evaluation suite and report Recall@1, Recall@5, and MRR."""
    if in_memory_slice:
        retriever = build_in_memory_benchmark_retriever(
            books=books,
            max_topics=max_topics,
            max_concordance=max_concordance,
            deterministic_embeddings=True,
        )
    else:
        base_settings = get_settings()
        if qdrant_mode is not None:
            mode_val: Literal["memory", "local", "remote"] = qdrant_mode.strip().lower()  # type: ignore[assignment]
            settings = base_settings.model_copy(update={"qdrant_mode": mode_val})
        else:
            settings = base_settings
        retriever = BSBHybridRetriever(settings=settings)

    report = evaluate_benchmark(retriever, cases=BENCHMARK_QUERIES, top_k=5)
    render_benchmark_report(report)

    if report.recall_at_5 < min_recall_at_5:
        console.print(
            f"[bold red]Benchmark Recall@5 ({report.recall_at_5:.2%}) is below threshold ({min_recall_at_5:.2%}).[/bold red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

