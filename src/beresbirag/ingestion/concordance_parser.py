"""Stream bounded headword summaries from the BSB Concordance workbook."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

from beresbirag.config import get_settings, resolve_data_file
from beresbirag.ingestion.xlsx_stream import _slug, _verse_keys, iter_xlsx_rows
from beresbirag.models.schemas import ChunkDocument, ChunkMetadata, SourceType

_ENTRY = re.compile(r"^(.*?)\s*\((\d+)\s+Occurrences?\)$", re.IGNORECASE)


def _entry_chunk(
    word: str, total: int, books: Counter[str], snippets: list[tuple[str, str]], refs: list[str]
) -> ChunkDocument:
    chunk_id = f"concordance:{_slug(word)}:0"
    lines = [
        f'[Source: BSB Concordance | Headword: "{word}" | Total Occurrences: {total}]',
        "Book Distribution: " + ", ".join(f"{book} ({count})" for book, count in books.items()),
        "Representative Occurrences & Contexts:",
        *(f"- [{ref}] {context}" for ref, context in snippets),
        "Verse Citations: " + "; ".join(refs),
    ]
    return ChunkDocument(
        chunk_id=chunk_id,
        text="\n".join(lines),
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            source_type=SourceType.CONCORDANCE,
            canonical_ref=f"Concordance: {word} ({total} Occurrences)",
            concordance_word=word.lower(),
            concordance_occurrences=total,
            verse_refs=_verse_keys([ref for ref, _ in snippets] + refs),
        ),
    )


def iter_concordance_chunks(
    xlsx_path: str | Path | None = None, max_entries: int | None = None, max_snippets: int = 25
) -> Iterator[ChunkDocument]:
    """Emit one summary per entry; counters cover all rows, samples remain bounded."""
    if max_snippets < 0:
        raise ValueError("max_snippets must be nonnegative")
    if max_entries is not None and max_entries <= 0:
        return
    path = (
        resolve_data_file(xlsx_path)
        if xlsx_path is not None
        else get_settings().concordance_xlsx_file
    )
    word = ""
    total = completed = 0
    books: Counter[str] = Counter()
    snippets: list[tuple[str, str]] = []
    refs: list[str] = []
    with closing(iter_xlsx_rows(path)) as rows:
        for _, row in rows:
            if row.get("E") == "0" and row.get("G", "").strip():
                if word:
                    yield _entry_chunk(word, total, books, snippets, refs)
                    completed += 1
                    if max_entries is not None and completed >= max_entries:
                        return
                entry = row["G"].strip()
                match = _ENTRY.fullmatch(entry)
                word = match[1].strip() if match else entry
                total = int(match[2]) if match else int(row.get("F") or 0)
                books, snippets, refs = Counter(), [], []
            elif word and row.get("H", "").strip():
                ref = row["H"].strip()
                book = row.get("B", "").strip() or ref.rsplit(" ", 1)[0]
                books[book] += 1
                if len(refs) < 80 and ref not in refs:
                    refs.append(ref)
                context = row.get("I", "").strip()
                if context and len(snippets) < max_snippets:
                    snippets.append((ref, context))
        if word:
            yield _entry_chunk(word, total, books, snippets, refs)
