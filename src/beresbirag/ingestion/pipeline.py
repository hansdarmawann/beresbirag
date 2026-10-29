"""Unified Berean Standard Bible (BSB) multi-corpus ingestion pipeline."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from beresbirag.config import Settings, get_settings, resolve_data_file
from beresbirag.ingestion.concordance_parser import iter_concordance_chunks
from beresbirag.ingestion.pdf_outline_parser import BookOutlineIndex
from beresbirag.ingestion.topical_parser import iter_topical_chunks
from beresbirag.ingestion.usj_parser import iter_usj_scripture_chunks
from beresbirag.models.schemas import ChunkDocument, SourceType
from beresbirag.vectorstore.qdrant_manager import QdrantManager

DEFAULT_SOURCES: tuple[SourceType, ...] = (
    SourceType.BOOK_OUTLINE,
    SourceType.SCRIPTURE,
    SourceType.TOPICAL_INDEX,
    SourceType.CONCORDANCE,
)


def parse_source_list(
    sources: Iterable[SourceType | str] | str | None,
) -> list[SourceType]:
    """Normalize a comma-separated string or iterable of source names into SourceType enums."""
    if sources is None:
        return list(DEFAULT_SOURCES)
    if isinstance(sources, str):
        raw_items = [part.strip().lower() for part in sources.split(",") if part.strip()]
    else:
        raw_items = []
        for item in sources:
            if isinstance(item, SourceType):
                raw_items.append(item.value)
            else:
                for sub in str(item).split(","):
                    if sub.strip():
                        raw_items.append(sub.strip().lower())
    if not raw_items:
        return list(DEFAULT_SOURCES)

    resolved: list[SourceType] = []
    for raw in raw_items:
        st = SourceType(raw)
        if st not in resolved:
            resolved.append(st)
    return resolved


def parse_book_list(books: Iterable[str] | str | None) -> set[str] | None:
    """Normalize a comma-separated string or iterable of USJ book codes into an uppercase set."""
    if books is None:
        return None
    if isinstance(books, str):
        codes = {part.strip().upper() for part in books.split(",") if part.strip()}
    else:
        codes = set()
        for item in books:
            for sub in str(item).split(","):
                if sub.strip():
                    codes.add(sub.strip().upper())
    return codes or None


@dataclass(slots=True)
class IngestionStats:
    """Summary counts and timing from an IngestionPipeline run."""

    counts_by_source: dict[str, int] = field(default_factory=dict)
    total_upserted: int = 0
    collection_points: int = 0
    elapsed_seconds: float = 0.0

    @property
    def book_outline_count(self) -> int:
        return self.counts_by_source.get(SourceType.BOOK_OUTLINE.value, 0)

    @property
    def scripture_count(self) -> int:
        return self.counts_by_source.get(SourceType.SCRIPTURE.value, 0)

    @property
    def topical_index_count(self) -> int:
        return self.counts_by_source.get(SourceType.TOPICAL_INDEX.value, 0)

    @property
    def concordance_count(self) -> int:
        return self.counts_by_source.get(SourceType.CONCORDANCE.value, 0)


class IngestionPipeline:
    """Unified ingestion orchestrator across all 4 Berean Standard Bible (BSB) corpora."""

    def __init__(
        self,
        qdrant_manager: QdrantManager | None = None,
        settings: Settings | None = None,
        usj_zip_path: str | Path | None = None,
        outlines_pdf_path: str | Path | None = None,
        topical_xlsx_path: str | Path | None = None,
        concordance_xlsx_path: str | Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.settings = settings or (
            qdrant_manager.settings if qdrant_manager is not None else get_settings()
        )
        self.qdrant_manager = qdrant_manager or QdrantManager(settings=self.settings)
        # Resolve missing paths from the injected settings so parsers never fall back to the
        # global get_settings().
        self.usj_zip_path: str | Path = (
            usj_zip_path if usj_zip_path is not None else self.settings.usj_zip_file
        )
        self.outlines_pdf_path: str | Path = (
            outlines_pdf_path if outlines_pdf_path is not None else self.settings.outlines_pdf_file
        )
        self.topical_xlsx_path: str | Path = (
            topical_xlsx_path if topical_xlsx_path is not None else self.settings.topical_xlsx_file
        )
        self.concordance_xlsx_path: str | Path = (
            concordance_xlsx_path
            if concordance_xlsx_path is not None
            else self.settings.concordance_xlsx_file
        )
        self.console = console or Console()
        self._outline_index: BookOutlineIndex | None = None

    def load_outline_index(self) -> BookOutlineIndex:
        """Load and cache the hierarchical BookOutlineIndex from the BSB Outlines PDF."""
        if self._outline_index is None:
            self._outline_index = BookOutlineIndex.from_pdf(self.outlines_pdf_path)
        return self._outline_index

    def iter_outline_chunks(
        self,
        book_filter: set[str] | None = None,
        filter_outlines_by_book: bool = False,
    ) -> Iterator[ChunkDocument]:
        """Yield book outline chunks from the BSB Book Outlines PDF."""
        index = self.load_outline_index()
        for chunk in index.to_chunk_documents():
            if (
                filter_outlines_by_book
                and book_filter is not None
                and chunk.metadata.book_code not in book_filter
            ):
                continue
            yield chunk

    def iter_scripture_chunks(
        self,
        book_filter: set[str] | None = None,
        max_verses_per_chunk: int = 10,
        verse_overlap: int = 2,
    ) -> Iterator[ChunkDocument]:
        """Stream pericope-aligned USJ scripture chunks enriched with outline breadcrumbs."""
        outline_index = self.load_outline_index()
        yield from iter_usj_scripture_chunks(
            zip_path=self.usj_zip_path,
            outline_index=outline_index,
            book_filter=book_filter,
            max_verses_per_chunk=max_verses_per_chunk,
            verse_overlap=verse_overlap,
        )

    def iter_topical_chunks(
        self,
        max_topics: int | None = None,
        max_full_verses_per_chunk: int = 15,
        max_chunks_per_topic: int = 3,
    ) -> Iterator[ChunkDocument]:
        """Stream bounded topical index chunks from bsb_topical_index.xlsx."""
        yield from iter_topical_chunks(
            xlsx_path=self.topical_xlsx_path,
            max_topics=max_topics,
            max_full_verses_per_chunk=max_full_verses_per_chunk,
            max_chunks_per_topic=max_chunks_per_topic,
        )

    def iter_concordance_chunks(
        self,
        max_concordance: int | None = None,
        max_snippets: int = 25,
    ) -> Iterator[ChunkDocument]:
        """Stream bounded concordance headword chunks from bsb_concordance.xlsx."""
        yield from iter_concordance_chunks(
            xlsx_path=self.concordance_xlsx_path,
            max_entries=max_concordance,
            max_snippets=max_snippets,
        )

    def collect_chunks(
        self,
        *,
        sources: Iterable[SourceType | str] | str | None = None,
        books: Iterable[str] | str | None = None,
        max_topics: int | None = None,
        max_concordance: int | None = None,
        max_verses_per_chunk: int = 10,
        verse_overlap: int = 2,
        filter_outlines_by_book: bool = False,
    ) -> list[ChunkDocument]:
        """Collect all chunks matching the requested filters into memory."""
        active_sources = parse_source_list(sources)
        book_filter = parse_book_list(books)
        chunks: list[ChunkDocument] = []

        if SourceType.BOOK_OUTLINE in active_sources:
            chunks.extend(
                self.iter_outline_chunks(
                    book_filter=book_filter,
                    filter_outlines_by_book=filter_outlines_by_book,
                )
            )
        if SourceType.SCRIPTURE in active_sources:
            chunks.extend(
                self.iter_scripture_chunks(
                    book_filter=book_filter,
                    max_verses_per_chunk=max_verses_per_chunk,
                    verse_overlap=verse_overlap,
                )
            )
        if SourceType.TOPICAL_INDEX in active_sources:
            chunks.extend(self.iter_topical_chunks(max_topics=max_topics))
        if SourceType.CONCORDANCE in active_sources:
            chunks.extend(self.iter_concordance_chunks(max_concordance=max_concordance))

        return chunks

    def run(
        self,
        *,
        recreate: bool = False,
        sources: Iterable[SourceType | str] | str | None = None,
        books: Iterable[str] | str | None = None,
        max_topics: int | None = None,
        max_concordance: int | None = None,
        batch_size: int = 64,
        max_verses_per_chunk: int = 10,
        verse_overlap: int = 2,
        filter_outlines_by_book: bool = False,
        show_progress: bool = True,
        corpus_manifest: dict[str, str] | None = None,
    ) -> IngestionStats:
        """Stream and upsert BSB chunks into Qdrant with Rich progress tracking."""
        start_time = perf_counter()
        active_sources = parse_source_list(sources)
        book_filter = parse_book_list(books)

        self.qdrant_manager.ensure_collection(recreate=recreate)

        counts_by_source: dict[str, int] = {st.value: 0 for st in active_sources}
        total_upserted = 0

        source_generators: list[tuple[SourceType, str, Iterator[ChunkDocument]]] = []
        if SourceType.BOOK_OUTLINE in active_sources:
            source_generators.append(
                (
                    SourceType.BOOK_OUTLINE,
                    "BSB Book Outlines (PDF)",
                    self.iter_outline_chunks(
                        book_filter=book_filter,
                        filter_outlines_by_book=filter_outlines_by_book,
                    ),
                )
            )
        if SourceType.SCRIPTURE in active_sources:
            source_generators.append(
                (
                    SourceType.SCRIPTURE,
                    "BSB Scripture (USJ 3.1)",
                    self.iter_scripture_chunks(
                        book_filter=book_filter,
                        max_verses_per_chunk=max_verses_per_chunk,
                        verse_overlap=verse_overlap,
                    ),
                )
            )
        if SourceType.TOPICAL_INDEX in active_sources:
            source_generators.append(
                (
                    SourceType.TOPICAL_INDEX,
                    "BSB Topical Index (XLSX)",
                    self.iter_topical_chunks(max_topics=max_topics),
                )
            )
        if SourceType.CONCORDANCE in active_sources:
            source_generators.append(
                (
                    SourceType.CONCORDANCE,
                    "BSB Concordance (XLSX)",
                    self.iter_concordance_chunks(max_concordance=max_concordance),
                )
            )

        if show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="bold green"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
                transient=False,
            ) as progress:
                for source_type, label, chunk_iter in source_generators:
                    task_id = progress.add_task(label, total=None)
                    upserted_for_source = self._upsert_with_progress(
                        chunk_iter,
                        batch_size=batch_size,
                        progress=progress,
                        task_id=task_id,
                    )
                    counts_by_source[source_type.value] = upserted_for_source
                    total_upserted += upserted_for_source
        else:
            for source_type, _label, chunk_iter in source_generators:
                upserted_for_source = self._upsert_with_progress(
                    chunk_iter,
                    batch_size=batch_size,
                )
                counts_by_source[source_type.value] = upserted_for_source
                total_upserted += upserted_for_source

        resolved_manifest = (
            corpus_manifest if corpus_manifest is not None else self._load_default_manifest_hashes()
        )
        self.qdrant_manager.write_index_metadata(
            max_verses_per_chunk=max_verses_per_chunk,
            verse_overlap=verse_overlap,
            corpus_manifest=resolved_manifest,
            recreate=recreate,
        )

        elapsed = perf_counter() - start_time
        return IngestionStats(
            counts_by_source=counts_by_source,
            total_upserted=total_upserted,
            collection_points=self.qdrant_manager.count_points(),
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _load_default_manifest_hashes() -> dict[str, str]:
        manifest_path = resolve_data_file("data/corpus_manifest.json")
        if not manifest_path.is_file():
            manifest_path = Path("data/corpus_manifest.json")
        if not manifest_path.is_file():
            return {}
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = data.get("files", []) if isinstance(data, dict) else data
            return {
                str(entry["name"]): str(entry["sha256"])
                for entry in files
                if isinstance(entry, dict) and entry.get("name") and entry.get("sha256")
            }
        except Exception:
            return {}

    def _upsert_with_progress(
        self,
        chunk_iter: Iterable[ChunkDocument],
        *,
        batch_size: int,
        progress: Progress | None = None,
        task_id: int | None = None,
    ) -> int:
        iterator = iter(chunk_iter)
        total = 0
        while True:
            batch = list(itertools.islice(iterator, batch_size))
            if not batch:
                break
            upserted = self.qdrant_manager.upsert_chunks(batch, batch_size=batch_size)
            total += upserted
            if progress is not None and task_id is not None:
                progress.update(TaskID(task_id), advance=len(batch))
        return total
