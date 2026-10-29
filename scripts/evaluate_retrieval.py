"""Multi-corpus retrieval benchmark suite (Recall@1, Recall@5, MRR, coverage) for BSB RAG.

Runs the JSONL case suite (./eval/retrieval_cases.jsonl) by default, or the original 15
in-code queries with --legacy. --verify-cases proves every case is satisfiable by the corpus.
"""

from __future__ import annotations

import json
import zlib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from beresbirag.config import Settings, get_settings
from beresbirag.evaluation import (
    RetrievalCase,
    build_corpus_index,
    context_coverage,
    has_false_exact_match,
    hit_rank,
    load_answer_cases,
    load_retrieval_cases,
    mrr,
    percentile,
    recall_at_k,
    verify_cases,
)
from beresbirag.ingestion.pipeline import IngestionPipeline
from beresbirag.models.schemas import ChunkDocument, RetrievalResult, SourceType
from beresbirag.retrieval.hybrid_retriever import BSBHybridRetriever, _topic_variants
from beresbirag.retrieval.query_router import route_query
from beresbirag.retrieval.reference_parser import BSB_VERSE_COUNTS
from beresbirag.retrieval.reranker import PassageReranker
from beresbirag.vectorstore.embeddings import (
    DeterministicBM25SparseEmbeddings,
    DeterministicHashEmbeddings,
)
from beresbirag.vectorstore.qdrant_manager import QdrantManager

app = typer.Typer(
    name="evaluate-retrieval",
    help="Run the Berean Standard Bible (BSB) multi-corpus retrieval benchmark.",
    add_completion=False,
)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "eval" / "retrieval_cases.jsonl"
DEFAULT_ANSWER_CASES = REPO_ROOT / "eval" / "answer_cases.jsonl"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"


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


# ---------------------------------------------------------------------------
# JSONL case suite
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CaseOutcome:
    """Evaluation outcome for one RetrievalCase."""

    case: RetrievalCase
    hit_rank: int | None
    coverage: float | None
    latency_s: float
    false_exact: bool | None
    top_canonical_ref: str | None
    k: int

    @property
    def passed(self) -> bool:
        if self.case.unanswerable:
            return not self.false_exact
        return self.hit_rank is not None and self.hit_rank <= self.k


@dataclass(slots=True)
class GroupStats:
    """Aggregate metrics for a group of case outcomes."""

    n: int
    answerable: int
    recall_at_1: float | None
    recall_at_5: float | None
    mrr: float | None
    mean_coverage: float | None
    pass_rate: float
    p50_latency_s: float | None
    p95_latency_s: float | None


def summarize(outcomes: Sequence[CaseOutcome]) -> GroupStats:
    """Recall/MRR over answerable cases; pass rate over all (unanswerable = no false exact)."""
    answerable = [o for o in outcomes if not o.case.unanswerable]
    ranks = [o.hit_rank for o in answerable]
    coverages = [o.coverage for o in answerable if o.coverage is not None]
    latencies = [o.latency_s for o in outcomes]
    return GroupStats(
        n=len(outcomes),
        answerable=len(answerable),
        recall_at_1=recall_at_k(ranks, 1) if answerable else None,
        recall_at_5=recall_at_k(ranks, 5) if answerable else None,
        mrr=mrr(ranks) if answerable else None,
        mean_coverage=sum(coverages) / len(coverages) if coverages else None,
        pass_rate=sum(o.passed for o in outcomes) / len(outcomes) if outcomes else 0.0,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
    )


def evaluate_cases(
    retriever: BSBHybridRetriever, cases: Sequence[RetrievalCase], top_k: int = 5
) -> list[CaseOutcome]:
    """Retrieve every case and score it (hit rank, coverage@k, false exact matches, latency)."""
    outcomes: list[CaseOutcome] = []
    k = max(top_k, 5)
    for case in cases:
        start = perf_counter()
        plan, results = retriever.retrieve(case.query, top_k=k)
        latency = perf_counter() - start
        outcomes.append(
            CaseOutcome(
                case=case,
                hit_rank=None if case.unanswerable else hit_rank(case, results),
                coverage=None if case.unanswerable else context_coverage(case, results[:k]),
                latency_s=latency,
                false_exact=has_false_exact_match(plan, results) if case.unanswerable else None,
                top_canonical_ref=(
                    results[0].chunk.metadata.canonical_ref or results[0].chunk.chunk_id
                    if results
                    else None
                ),
                k=k,
            )
        )
    return outcomes


def slice_requirements(cases: Sequence[RetrievalCase]) -> tuple[set[str], set[str], set[str]]:
    """Books, topics and concordance words the case suite needs indexed."""
    books: set[str] = set()
    topics: set[str] = set()
    words: set[str] = set()
    for case in cases:
        plan = route_query(case.query)
        exp = case.expected
        books.update(plan.book_codes)
        books.update(ref.split(" ", 1)[0].upper() for ref in [*exp.verse_refs_any, *exp.verse_refs_all])
        if exp.book_code:
            books.add(exp.book_code.upper())
        for topic in (exp.topic, plan.topic_keyword):
            if topic:
                topics.update(_topic_variants(topic))
        for word in (exp.concordance_word, plan.concordance_word):
            if word:
                words.add(word.lower())
    return books & set(BSB_VERSE_COUNTS), topics, words


def _sampled(name: str, rate: float) -> bool:
    """Deterministic pseudo-random distractor sampling, stable across runs and machines."""
    return rate > 0 and zlib.crc32(name.encode("utf-8")) % 10_000 < rate * 10_000


def build_case_slice_retriever(
    cases: Sequence[RetrievalCase],
    *,
    embeddings: Literal["hash", "real"] = "hash",
    reranker: bool = False,
    distractor_rate: float = 0.1,
    extra_books: str | None = None,
) -> BSBHybridRetriever:
    """Index the books/topics/words the cases need plus sampled distractors into memory.

    All 66 book outlines are indexed. Scripture covers every book any case references.
    About `distractor_rate` of the remaining topics and concordance headwords are added so
    exact lookups are not the only candidates.
    """
    settings = get_settings().model_copy(
        update={
            "qdrant_mode": "memory",
            "qdrant_collection_name": "bsb_eval_cases",
            "reranker_enabled": reranker,
        }
    )
    if embeddings == "hash":
        manager = QdrantManager(
            settings=settings,
            dense_embeddings=DeterministicHashEmbeddings(size=384),
            sparse_embeddings=DeterministicBM25SparseEmbeddings(),
            dense_vector_size=384,
        )
    else:
        manager = QdrantManager(settings=settings)

    books, topics, words = slice_requirements(cases)
    if extra_books:
        books.update(code.strip().upper() for code in extra_books.split(",") if code.strip())
    pipeline = IngestionPipeline(qdrant_manager=manager, settings=settings, console=console)
    manager.ensure_collection(recreate=True)

    def _chunks() -> Iterator[ChunkDocument]:
        yield from pipeline.iter_outline_chunks()
        yield from pipeline.iter_scripture_chunks(book_filter=books)
        for chunk in pipeline.iter_topical_chunks():
            topic = chunk.metadata.topic or ""
            if topic in topics or _sampled(topic, distractor_rate):
                yield chunk
        for chunk in pipeline.iter_concordance_chunks():
            word = chunk.metadata.concordance_word or ""
            if word in words or _sampled(word, distractor_rate):
                yield chunk

    start = perf_counter()
    with console.status(f"Indexing case slice ({embeddings} embeddings, {len(books)} books)..."):
        total = manager.upsert_chunks(_chunks(), batch_size=64)
    console.print(f"Indexed {total} chunks in {perf_counter() - start:.1f}s ({sorted(books)}).")
    return BSBHybridRetriever(
        qdrant_manager=manager,
        reranker=PassageReranker(settings=settings, enabled=reranker),
        settings=settings,
    )


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def _stats_table(title: str, groups: dict[str, GroupStats]) -> Table:
    table = Table(title=title, box=box.ASCII, show_lines=False)
    for name, justify in (
        ("Category", "left"), ("N", "right"), ("R@1", "right"), ("R@5", "right"),
        ("MRR", "right"), ("Coverage", "right"), ("Pass", "right"),
        ("p50 s", "right"), ("p95 s", "right"),
    ):  # fmt: skip
        table.add_column(name, justify=justify)  # type: ignore[arg-type]
    for name, st in groups.items():
        table.add_row(
            name,
            str(st.n),
            _fmt(st.recall_at_1),
            _fmt(st.recall_at_5),
            _fmt(st.mrr, pct=False),
            _fmt(st.mean_coverage),
            _fmt(st.pass_rate),
            _fmt(st.p50_latency_s, pct=False),
            _fmt(st.p95_latency_s, pct=False),
        )
    return table


def group_outcomes(outcomes: Sequence[CaseOutcome]) -> tuple[dict[str, GroupStats], GroupStats]:
    """Per-category stats plus an overall row (non-gap cases only)."""
    by_category: dict[str, list[CaseOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_category[outcome.case.category].append(outcome)
    return {cat: summarize(items) for cat, items in by_category.items()}, summarize(outcomes)


def render_case_report(outcomes: Sequence[CaseOutcome]) -> dict[str, Any]:
    """Print per-category tables (known-gap cases separately) and return the summary dict."""
    core = [o for o in outcomes if not o.case.known_gap]
    gaps = [o for o in outcomes if o.case.known_gap]
    categories, overall = group_outcomes(core)
    console.print(_stats_table("BSB Retrieval Evaluation (per category)", {**categories, "OVERALL": overall}))

    misses = [o for o in core if not o.passed]
    if misses:
        miss_table = Table(title="Failed cases", box=box.ASCII)
        for column in ("ID", "Category", "Query", "Rank", "Coverage", "Top retrieved"):
            miss_table.add_column(column)
        for o in misses:
            miss_table.add_row(
                o.case.id,
                o.case.category,
                _ascii_safe(o.case.query),
                "FALSE-EXACT" if o.false_exact else str(o.hit_rank or "MISS"),
                _fmt(o.coverage),
                _ascii_safe(o.top_canonical_ref or "-"),
            )
        console.print(miss_table)

    gap_categories: dict[str, GroupStats] = {}
    if gaps:
        gap_categories, gap_overall = group_outcomes(gaps)
        gap_categories["KNOWN-GAP TOTAL"] = gap_overall
        console.print(_stats_table("Known gaps (excluded from thresholds)", gap_categories))
        for o in gaps:
            console.print(
                f"  {o.case.id} [{', '.join(o.case.tags)}] {_ascii_safe(o.case.query)} -> "
                f"{'PASS' if o.passed else 'MISS'} (top: {_ascii_safe(o.top_canonical_ref or '-')})"
            )
    return {
        "overall": asdict(overall),
        "categories": {name: asdict(st) for name, st in categories.items()},
        "known_gap": {name: asdict(st) for name, st in gap_categories.items()},
    }


def outcome_to_dict(outcome: CaseOutcome) -> dict[str, Any]:
    return {
        "id": outcome.case.id,
        "category": outcome.case.category,
        "query": outcome.case.query,
        "tags": outcome.case.tags,
        "unanswerable": outcome.case.unanswerable,
        "hit_rank": outcome.hit_rank,
        "coverage": outcome.coverage,
        "false_exact": outcome.false_exact,
        "passed": outcome.passed,
        "latency_s": round(outcome.latency_s, 4),
        "top_canonical_ref": outcome.top_canonical_ref,
    }


def legacy_report_to_dict(report: BenchmarkReport) -> dict[str, Any]:
    """JSON-serializable form of the legacy 15-query BenchmarkReport."""
    return {
        "recall_at_1": report.recall_at_1,
        "recall_at_5": report.recall_at_5,
        "mrr": report.mrr,
        "results": [
            {
                "case_id": r.case.case_id,
                "category": r.case.category,
                "query": r.case.query,
                "hit_rank": r.hit_rank,
                "hit_at_1": r.hit_at_1,
                "hit_at_5": r.hit_at_5,
                "top_chunk_id": r.top_chunk_id,
                "top_canonical_ref": r.top_canonical_ref,
            }
            for r in report.results
        ],
    }


def write_report(payload: dict[str, Any], path: Path | None, prefix: str) -> Path:
    """Write a JSON report (default ./eval/reports/<prefix>-<timestamp>.json)."""
    target = path or REPORTS_DIR / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.print(f"Report written to {target}")
    return target


def run_verification(cases_path: Path, answer_cases_path: Path | None, settings: Settings) -> int:
    """Scan the real corpus once and print every unsatisfiable expectation; return exit code."""
    retrieval_cases = load_retrieval_cases(cases_path)
    answer_cases = (
        load_answer_cases(answer_cases_path)
        if answer_cases_path is not None and answer_cases_path.is_file()
        else []
    )
    with console.status("Scanning the BSB corpus for ground-truth verification..."):
        corpus = build_corpus_index(retrieval_cases, answer_cases, settings=settings)
    issues = verify_cases(retrieval_cases, answer_cases, corpus)

    counts: dict[str, int] = defaultdict(int)
    for case in retrieval_cases:
        counts[f"{case.category}{' (known-gap)' if case.known_gap else ''}"] += 1
    table = Table(title="Retrieval cases per category", box=box.ASCII)
    table.add_column("Category")
    table.add_column("N", justify="right")
    for name, n in sorted(counts.items()):
        table.add_row(name, str(n))
    console.print(table)
    console.print(
        f"Verified {len(retrieval_cases)} retrieval and {len(answer_cases)} answer cases against "
        f"{len(corpus.topics)} topics and {len(corpus.concordance_words)} concordance headwords."
    )
    if issues:
        for issue in issues:
            console.print(f"[red]{issue.case_id}[/red]: {issue.message}")
        failing = sorted({issue.case_id for issue in issues})
        console.print(f"[bold red]{len(failing)} case(s) unsatisfiable: {', '.join(failing)}[/bold red]")
        return 1
    console.print("[bold green]All cases are satisfiable by the real BSB corpus.[/bold green]")
    return 0


def _existing_collection_settings(qdrant_mode: str | None, reranker: bool) -> Settings:
    update: dict[str, Any] = {"reranker_enabled": reranker}
    if qdrant_mode is not None:
        update["qdrant_mode"] = qdrant_mode.strip().lower()
    return get_settings().model_copy(update=update)


@app.command()
def main(
    cases: Annotated[Path, typer.Option("--cases", help="JSONL retrieval cases.")] = DEFAULT_CASES,
    legacy: bool = typer.Option(False, "--legacy", help="Run the original 15 in-code queries."),
    embeddings: str = typer.Option(
        "hash", "--embeddings", help="hash (deterministic, offline) or real (configured models)."
    ),
    reranker: str | None = typer.Option(
        None, "--reranker", help="on|off (default: on with --embeddings real, else off)."
    ),
    in_memory_slice: bool = typer.Option(
        True,
        "--in-memory-slice/--use-existing-collection",
        help="Build an in-memory BSB slice (default) or query the configured Qdrant collection.",
    ),
    categories: str | None = typer.Option(
        None, "--categories", help="Comma-separated categories to evaluate (case suite only)."
    ),
    verify: bool = typer.Option(
        False, "--verify-cases", help="Verify every case against the real corpus and exit."
    ),
    answer_cases: Annotated[
        Path, typer.Option("--answer-cases", help="Answer cases also checked by --verify-cases.")
    ] = DEFAULT_ANSWER_CASES,
    report_json: Annotated[
        Path | None,
        typer.Option(
            "--report-json", help="JSON report path (default ./eval/reports/retrieval-<ts>.json)."
        ),
    ] = None,
    top_k: int = typer.Option(5, "--top-k", help="Results retrieved per query (>= 5)."),
    distractor_rate: float = typer.Option(
        0.1, "--distractor-rate", help="Fraction of other topics/headwords indexed as distractors."
    ),
    books: str = typer.Option(
        "GEN,JHN,REV",
        "--books",
        help="Legacy slice: book codes. Case suite: extra books added to the case-derived set.",
    ),
    max_topics: int = typer.Option(15, "--max-topics", help="Legacy slice: topics indexed."),
    max_concordance: int = typer.Option(
        15, "--max-concordance", help="Legacy slice: concordance entries indexed."
    ),
    qdrant_mode: str | None = typer.Option(
        None, "--qdrant-mode", help="Optional QDRANT_MODE override with --use-existing-collection."
    ),
    min_recall_at_5: float | None = typer.Option(
        None,
        "--min-recall-at-5",
        help="Fail below this Recall@5 on non-gap answerable cases (default 0.90 with --legacy).",
    ),
) -> None:
    """Evaluate BSB retrieval quality (Recall@1/@5, MRR, coverage, latency) per category."""
    embeddings = embeddings.strip().lower()
    if embeddings not in {"hash", "real"}:
        raise typer.BadParameter("--embeddings must be 'hash' or 'real'")
    if reranker is not None and reranker.strip().lower() not in {"on", "off"}:
        raise typer.BadParameter("--reranker must be 'on' or 'off'")
    use_reranker = (
        reranker.strip().lower() == "on"
        if reranker is not None
        else (embeddings == "real" or not in_memory_slice)
    )

    if verify:
        raise typer.Exit(code=run_verification(cases, answer_cases, get_settings()))

    config = {
        "embeddings": embeddings if in_memory_slice else "configured",
        "reranker": use_reranker,
        "collection": "in-memory-slice" if in_memory_slice else "existing",
        "top_k": top_k,
    }
    if legacy:
        if in_memory_slice:
            retriever = build_in_memory_benchmark_retriever(
                books=books,
                max_topics=max_topics,
                max_concordance=max_concordance,
                deterministic_embeddings=embeddings == "hash",
            )
        else:
            retriever = BSBHybridRetriever(
                settings=_existing_collection_settings(qdrant_mode, use_reranker)
            )
        report = evaluate_benchmark(retriever, cases=BENCHMARK_QUERIES, top_k=5)
        render_benchmark_report(report)
        write_report(
            {"mode": "legacy", "config": config, **legacy_report_to_dict(report)},
            report_json,
            "retrieval-legacy",
        )
        threshold = 0.90 if min_recall_at_5 is None else min_recall_at_5
        if report.recall_at_5 < threshold:
            console.print(
                f"[bold red]Benchmark Recall@5 ({report.recall_at_5:.2%}) is below threshold ({threshold:.2%}).[/bold red]"
            )
            raise typer.Exit(code=1)
        return

    suite = load_retrieval_cases(cases)
    if categories:
        wanted = {c.strip() for c in categories.split(",") if c.strip()}
        suite = [case for case in suite if case.category in wanted]
        if not suite:
            raise typer.BadParameter(f"No cases in categories {sorted(wanted)}")
    if in_memory_slice:
        retriever = build_case_slice_retriever(
            suite,
            embeddings=embeddings,  # type: ignore[arg-type]
            reranker=use_reranker,
            distractor_rate=distractor_rate,
            extra_books=books if books != "GEN,JHN,REV" else None,
        )
    else:
        retriever = BSBHybridRetriever(
            settings=_existing_collection_settings(qdrant_mode, use_reranker)
        )

    outcomes = evaluate_cases(retriever, suite, top_k=top_k)
    summary = render_case_report(outcomes)
    write_report(
        {
            "mode": "cases",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "cases_file": str(cases),
            "config": {**config, "distractor_rate": distractor_rate},
            **summary,
            "cases": [outcome_to_dict(o) for o in outcomes],
        },
        report_json,
        "retrieval",
    )

    if min_recall_at_5 is not None:
        gated = [o for o in outcomes if not o.case.known_gap and not o.case.unanswerable]
        recall5 = recall_at_k([o.hit_rank for o in gated], 5)
        if recall5 < min_recall_at_5:
            console.print(
                f"[bold red]Recall@5 ({recall5:.2%}) on non-gap cases is below threshold "
                f"({min_recall_at_5:.2%}).[/bold red]"
            )
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
